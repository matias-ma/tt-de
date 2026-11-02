"""Direct, observation-only optimization of squared Chebyshev TT densities.

No signal pilot is constructed. Alternative losses and Sobolev penalties
share the same coefficient representation and exact physical normalization.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.special import roots_legendre

from .cheb_gram import chebyshev_gram_matrix
from .gaussian_convolution import _transfer
from .marginal_regularization import marginal_curvature, marginal_curvature_geometry
from .mode_schedule import mask_modes, mode_masks, prepare_mode_schedule
from .sobolev import derivative_grams
from .tt_density import rebuild_tt


def save_tt(path, model):
    """Atomically save only compact Chebyshev coefficients, losslessly compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, domain=np.asarray(model.domain),
                            **{f'signal_{i}': np.asarray(c.coeffs) for i, c in enumerate(model.cores)})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_tt(path):
    with np.load(path) as data:
        domain = data['domain']
        cores = tuple(ChebyshevCoeffCore(jnp.asarray(data[f'signal_{i}']), tuple(box))
                      for i, box in enumerate(domain))
    return ChebTT(cores=cores, domain=jnp.asarray(domain))


def write_record(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def chebyshev_features(points, domain, sizes):
    points, domain = np.asarray(points), np.asarray(domain)
    if points.ndim != 2 or points.shape[1] != len(domain):
        raise ValueError('Observations must have shape (count, dimensions)')
    if np.any(points < domain[:, 0]) or np.any(points > domain[:, 1]):
        raise ValueError('Noise-free observations must lie inside the box')
    return tuple(np.polynomial.chebyshev.chebvander(
        (2*points[:, i]-sum(box))/(box[1]-box[0]), n-1)
        for i, (box, n) in enumerate(zip(domain, sizes, strict=True)))


def amplitude(cores, features):
    state = jnp.ones((len(features[0]), 1))
    for core, basis in zip(cores, features, strict=True):
        state = jnp.einsum('bi,ijn,bn->bj', state, core, basis, optimize='optimal')
    return state[:, 0]


def integrated_energies(cores, grams, first, second):
    """Return norm, H1, axiswise H2 and full Hessian Frobenius energy.

    The full Hessian includes each mixed derivative twice. Matrices already
    include physical length scales. These are quadratic forms in the amplitude.
    """
    norm = jnp.ones((1, 1, 1))
    slope = axis = hessian = jnp.zeros_like(norm)
    for c, g, d1, d2 in zip(cores, grams, first, second, strict=True):
        transfer = lambda state, matrix, core=c: _transfer(state, core, matrix[None])
        hessian = transfer(hessian, g)+2*transfer(slope, d1)+transfer(norm, d2)
        axis = transfer(axis, g)+transfer(norm, d2)
        slope = transfer(slope, g)+transfer(norm, d1)
        norm = transfer(norm, g)
    return tuple(x[0, 0, 0] for x in (norm, slope, axis, hessian))


def conditional_log_densities(cores, features, grams):
    """All exact one-coordinate conditional log densities, without a pilot."""
    count = len(features[0])
    left = [jnp.ones((count, 1))]
    for core, basis in zip(cores, features, strict=True):
        left.append(jnp.einsum('bi,ijn,bn->bj', left[-1], core, basis, optimize='optimal'))
    right = [None]*(len(cores)+1)
    right[-1] = jnp.ones((count, 1))
    for k in range(len(cores)-1, -1, -1):
        right[k] = jnp.einsum('ijn,bn,bj->bi', cores[k], features[k], right[k+1], optimize='optimal')
    numerator = jnp.log(left[-1][:, 0]**2+1e-300)
    values = []
    for k, (c, g) in enumerate(zip(cores, grams, strict=True)):
        local = jnp.einsum('bi,ijn,bj->bn', left[k], c, right[k+1], optimize='optimal')
        denominator = jnp.einsum('bn,nm,bm->b', local, g, local)
        values.append(numerator-jnp.log(denominator))
    return jnp.stack(values, axis=1)


def fourth_integral_4d(cores, bases, weights):
    """Exact tensor Gauss integration of amplitude^4 in four dimensions.

    With n Chebyshev coefficients, 2*n-1 nodes suffice in each coordinate.
    Only a pair-by-pair amplitude matrix is formed; no persistent large cache.
    """
    if len(cores) != 4:
        raise ValueError('Exact quartic quadrature is currently restricted to four dimensions')
    v0 = bases[0]@cores[0][0].T
    v3 = bases[3]@cores[3][:, 0, :].T
    left = jnp.einsum('xi,ijn,yn->xyj', v0, cores[1], bases[1], optimize='optimal')
    right = jnp.einsum('ijn,xn,yj->ixy', cores[2], bases[2], v3, optimize='optimal')
    left = left.reshape(-1, cores[1].shape[1])
    right = right.reshape(cores[2].shape[0], -1)
    left *= jnp.outer(weights[0], weights[1]).reshape(-1, 1)**.25
    right *= jnp.outer(weights[2], weights[3]).reshape(1, -1)**.25
    return jnp.sum((left@right)**4)


class DirectTrainingProblem:
    """Learn TT coefficients with independent loss and optimizer geometry.

    ``strength`` scales the statistical penalty. ``precondition_strength``
    scales the per-core Sobolev whitening metric; None preserves the old
    coupling to strength. Changing the latter leaves the represented
    density and objective unchanged, but changes optimizer coordinates.
    """

    def __init__(self, initial, *, strength=1e-4, regularizer='axis_h2',
                 objective='nll', conditional_weight=1., length_scales=None, marginal_strength=0., precondition_strength=None):
        if regularizer not in ('h1', 'axis_h2', 'hessian'):
            raise ValueError('Unknown Sobolev penalty')
        if objective not in ('nll', 'logpen', 'composite', 'l2') or not np.isfinite(strength) or strength < 0:
            raise ValueError('Unknown objective or negative regularization')
        if not np.isfinite(marginal_strength) or marginal_strength < 0:
            raise ValueError('marginal_strength must be finite and nonnegative')
        metric_strength = strength if precondition_strength is None else precondition_strength
        if not np.isfinite(metric_strength) or metric_strength < 0:
            raise ValueError('precondition_strength must be finite and nonnegative')
        self.precondition_strength = float(metric_strength)
        self.marginal_strength = float(marginal_strength)
        self.initial, self.strength = initial, float(strength)
        self.regularizer, self.objective = regularizer, objective
        self.conditional_weight = float(conditional_weight)
        self.domain = np.asarray(initial.domain)
        self.sizes = tuple(c.coeffs.shape[-1] for c in initial.cores)
        scales = .05*np.ptp(self.domain, axis=1) if length_scales is None else np.asarray(length_scales)
        g = tuple(np.asarray(chebyshev_gram_matrix(n, box)) for n, box in zip(self.sizes, self.domain, strict=True))
        first = tuple(np.asarray(x)*s**2 for x, s in zip(derivative_grams(initial, 1), scales, strict=True))
        second = tuple(np.asarray(x)*s**4 for x, s in zip(derivative_grams(initial, 2), scales, strict=True))
        curvature = first if regularizer == 'h1' else second
        self.factors = tuple(np.linalg.cholesky(a+self.precondition_strength*b) for a, b in zip(g, curvature, strict=True))
        self.inverse = tuple(np.linalg.inv(x) for x in self.factors)
        self.grams, self.first, self.second = (
            tuple(jnp.asarray(inv@matrix@inv.T) for inv, matrix in zip(self.inverse, matrices, strict=True))
            for matrices in (g, first, second))
        self.marginal_geometry = marginal_curvature_geometry(self.sizes, self.domain, self.inverse, scales) if self.marginal_strength else None
        self.params = tuple(jnp.asarray(c.coeffs)@f for c, f in zip(initial.cores, self.factors, strict=True))
        if objective == 'l2':
            if len(initial.cores) != 4:
                raise ValueError('The exact quadratic-density score currently supports 4D')
            bases, weights = [], []
            for n, box, inv in zip(self.sizes, self.domain, self.inverse, strict=True):
                x, w = roots_legendre(2*n-1)
                bases.append(jnp.asarray(np.polynomial.chebyshev.chebvander(x, n-1)@inv.T))
                weights.append(jnp.asarray(w*(box[1]-box[0])/2))
            self.quad_bases, self.quad_weights = tuple(bases), tuple(weights)
        self.logs = jax.jit(self._logs)
        self.energies = jax.jit(lambda p: integrated_energies(p, self.grams, self.first, self.second))

    def features(self, observations):
        return tuple(jnp.asarray(a@inv.T) for a, inv in zip(
            chebyshev_features(observations, self.domain, self.sizes), self.inverse, strict=True))

    def model(self, params):
        return rebuild_tt(self.initial, tuple(c@inv for c, inv in zip(params, self.inverse, strict=True)))

    def _logs(self, params, features):
        norm = integrated_energies(params, self.grams, self.first, self.second)[0]
        return jnp.log(amplitude(params, features)**2+1e-300)-jnp.log(norm)

    def marginal_penalty(self, params):
        if not self.marginal_strength:
            return jnp.asarray(0.)
        return self.marginal_strength*marginal_curvature(params, self.grams, self.marginal_geometry)

    def loss(self, params, features):
        norm, h1, axis, hessian = integrated_energies(params, self.grams, self.first, self.second)
        value = amplitude(params, features)**2/norm
        if self.objective == 'l2':
            base = fourth_integral_4d(params, self.quad_bases, self.quad_weights)/norm**2-2*jnp.mean(value)
        else:
            base = -jnp.mean(jnp.log(value+1e-300))
            if self.objective == 'composite':
                base -= self.conditional_weight*jnp.mean(conditional_log_densities(params, features, self.grams))
        penalty = {'h1': h1, 'axis_h2': axis, 'hessian': hessian}[self.regularizer]/norm
        return base+(jnp.log1p(self.strength*penalty) if self.objective == 'logpen' else self.strength*penalty)+self.marginal_penalty(params)

    def fit_adam(self, train, valid, output, *, steps=4000, batch_size=1024,
                 learning_rate=.001, batch_seed=0, gradient_clip=1., checkpoint_every=200, validation_batch_size=4096,
                 mode_schedule=None, selection_start=0):
        """Adam with optional nested mode activation, retaining optimizer state.

        Schedule entries are (first update, active coefficient counts).
        Inactive parameters, gradients and updates are zero. Lower-triangular
        Cholesky whitening makes this exactly a low-degree Chebyshev subspace.
        The global learning-rate schedule and batch stream never restart.
        ``selection_start`` restricts which checkpoints can be retained; all
        checkpoints are still recorded. Defaults preserve existing training.
        """
        stages = prepare_mode_schedule(self.sizes, steps, mode_schedule)
        steps = int(steps)
        if not isinstance(selection_start, (int, np.integer)) or not 0 <= selection_start <= steps:
            raise ValueError('selection_start must be an update count between zero and steps')
        selection_start = int(selection_start)
        masks_by_stage = tuple(mode_masks(self.sizes, counts) for _, counts in stages)
        stage = 0
        masks = masks_by_stage[0]
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        training, validation = self.features(train), self.features(valid)
        schedule = optax.cosine_decay_schedule(learning_rate, steps, alpha=.1)
        transformations = ([] if gradient_clip is None else [optax.clip_by_global_norm(gradient_clip)])
        optimizer = optax.chain(*transformations, optax.adam(schedule))
        params = mask_modes(self.params, masks) if mode_schedule is not None else self.params
        state = optimizer.init(params)

        @jax.jit
        def step(p, state, key, masks):
            indices = jax.random.randint(key, (batch_size,), 0, len(train))
            batch = tuple(x[indices] for x in training)
            loss, gradient = jax.value_and_grad(self.loss)(p, batch)
            if mode_schedule is not None:
                gradient = mask_modes(gradient, masks)
            updates, state = optimizer.update(gradient, state, p)
            if mode_schedule is not None:
                updates = mask_modes(updates, masks)
            updated = optax.apply_updates(p, updates)
            return mask_modes(updated, masks) if mode_schedule is not None else updated, state, loss

        best_score, best_step, history = np.inf, 0, []
        key = jax.random.PRNGKey(batch_seed)
        start, last_loss = time.monotonic(), None
        for iteration in range(steps+1):
            if stage+1 < len(stages) and iteration == stages[stage+1][0]:
                stage += 1
                masks = masks_by_stage[stage]
            if iteration % checkpoint_every == 0 or iteration == steps or iteration in [s[0] for s in stages]:
                logs = np.concatenate([np.asarray(self.logs(params, tuple(a[j:j+validation_batch_size] for a in validation)))
                                       for j in range(0, len(valid), validation_batch_size)])
                energies = np.asarray(self.energies(params), dtype=float)
                if not np.isfinite(logs).all() or not np.isfinite(energies).all() or energies[0] <= 0:
                    raise FloatingPointError('Nonfinite direct-training checkpoint')
                nll = -float(logs.mean())
                row = {'step': iteration, 'validation_nll': nll,
                       'roughness_h1': float(energies[1]/energies[0]),
                       'roughness_axis_h2': float(energies[2]/energies[0]),
                       'roughness_hessian': float(energies[3]/energies[0]),
                       'batch_objective': last_loss, 'seconds': time.monotonic()-start,
                       'active_modes': list(stages[stage][1]), 'selection_eligible': iteration >= selection_start}
                if mode_schedule is not None:
                    row['inactive_coefficient_max'] = max(float(np.max(np.abs(np.asarray(c@inv)[..., active:]), initial=0.))
                        for c, inv, active in zip(params, self.inverse, stages[stage][1], strict=True))
                history.append(row)
                if iteration >= selection_start and nll < best_score:
                    best_score, best_step = nll, iteration
                    save_tt(output/'fit.npz', self.model(params))
                    np.save(output/'validation.npy', logs)
                write_record(output/'progress.json', {'history': history, 'best_step': best_step if np.isfinite(best_score) else None,
                                                        'best_nll': best_score if np.isfinite(best_score) else None})
                print(output.name, row, flush=True)
            if iteration == steps:
                break
            key, subkey = jax.random.split(key)
            params, state, loss = step(params, state, subkey, masks)
            last_loss = float(loss)
            if not np.isfinite(last_loss):
                raise FloatingPointError('Nonfinite direct-training objective')
        record = {'history': history, 'best_step': best_step, 'best_nll': best_score,
                  'seconds': time.monotonic()-start, 'steps': steps, 'batch_seed': batch_seed,
                  'objective': self.objective, 'regularizer': self.regularizer, 'strength': self.strength,
                  'marginal_strength': self.marginal_strength, 'precondition_strength': self.precondition_strength,
                  'mode_schedule': [[start, list(counts)] for start, counts in stages],
                  'selection_start': int(selection_start),
                  'conditional_weight': self.conditional_weight, 'training_count': len(train),
                  'selection': 'minimum held-out joint NLL checkpoint', 'uses_signal_pilot': False}
        save_tt(output/'last.npz', self.model(params))
        write_record(output/'result.json', record)
        return load_tt(output/'fit.npz'), record


class NoisyDirectTrainingProblem(DirectTrainingProblem):
    """The same coefficient objective with a prescribed convolution operator.

    The operator is fixed before fitting and supplies log_density(model, x).
    Only signal TT coefficients are optimized; no signal pilot is constructed.
    """

    def __init__(self, initial, forward, **kwargs):
        super().__init__(initial, **kwargs)
        if self.objective not in ('nll', 'logpen'):
            raise ValueError('Noisy direct fitting currently supports likelihood objectives')
        self.forward = forward

    def features(self, observations):
        observations = np.asarray(observations)
        if observations.ndim != 2 or observations.shape[1] != len(self.domain) or not np.isfinite(observations).all():
            raise ValueError('Invalid noisy observations')
        return (jnp.asarray(observations),)

    def _logs(self, params, features):
        return self.forward.log_density(self.model(params), features[0])

    def loss(self, params, features):
        norm, h1, axis, hessian = self.energies(params)
        penalty = {'h1': h1, 'axis_h2': axis, 'hessian': hessian}[self.regularizer]/norm
        regularizer = jnp.log1p(self.strength*penalty) if self.objective == 'logpen' else self.strength*penalty
        return -jnp.mean(self._logs(params, features))+regularizer+self.marginal_penalty(params)

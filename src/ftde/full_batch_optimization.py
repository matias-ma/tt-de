"""Full-data L-BFGS refinement of the original squared Chebyshev TT likelihood.

Every function and gradient evaluation uses the same complete training set.
Chunking controls memory without changing the objective or its gradient.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from .cheb_gram import chebyshev_gram_matrix
from .tt_convolution import TTConvolution
from .tt_density import log_density_tt, rebuild_tt


def fit_noisy_tt_lbfgs(initial, train, valid, kernel, penalty, *, strength=1e-4,
                        iterations=100, batch_size=256, checkpoint_every=5, callback=None):
    """Refine a TT by a deterministic likelihood objective and line search.

    Returns the best validation-NLL-plus-penalty checkpoint and diagnostics.
    Caller supplies the initializer; no KDE, mixture or surrogate density is
    constructed. Training and validation observations are kept separate.
    Pass ``kernel=None`` for the noise-free squared-TT likelihood.
    """
    if not len(train) or not len(valid) or iterations < 1 or batch_size < 1 or strength < 0 or checkpoint_every < 1:
        raise ValueError('nonempty splits and positive iteration/batch sizes required')
    if kernel is None:
        box = np.asarray(initial.domain)
        if not all(np.all((x >= box[:, 0]) & (x <= box[:, 1])) for x in (train, valid)):
            raise ValueError('Noise-free observations must lie inside the TT domain')
        log_density = log_density_tt
    else:
        log_density = TTConvolution.from_tt(initial, kernel).prepare(tolerance=1e-14).log_density
    factors = [np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(c.coeffs.shape[-1], c.domain))
                 +strength*w*np.asarray(d)) for c, w, d in zip(initial.cores, penalty.weights, penalty.derivatives, strict=True)]
    inverse = tuple(jnp.asarray(np.linalg.inv(f)) for f in factors)
    flat_initial, unravel = ravel_pytree(tuple(c.coeffs@f for c, f in zip(initial.cores, factors, strict=True)))

    def model(flat):
        return rebuild_tt(initial, tuple(c@f for c, f in zip(unravel(flat), inverse, strict=True)))

    train = np.asarray(train)
    pad = (-len(train)) % batch_size
    # Masked entries must still be valid evaluation points: extrapolation at
    # zero can overflow for a shifted box before its zero mask is applied.
    batches = jnp.asarray(np.pad(train, ((0, pad), (0, 0)), mode='edge')).reshape(-1, batch_size, train.shape[1])
    masks = jnp.asarray(np.pad(np.ones(len(train)), (0, pad))).reshape(-1, batch_size)

    @jax.jit
    def full_value_gradient(flat):
        def accumulate(state, batch):
            x, mask = batch
            value, gradient = jax.value_and_grad(lambda p: -jnp.sum(log_density(model(p), x)*mask)/len(train))(flat)
            return (state[0]+value, state[1]+gradient), None
        (value, gradient), _ = jax.lax.scan(accumulate, (jnp.array(0.), jnp.zeros_like(flat)), (batches, masks))
        regularizer, regularizer_gradient = jax.value_and_grad(lambda p: strength*penalty(model(p)))(flat)
        return value+regularizer, gradient+regularizer_gradient

    evaluate = jax.jit(lambda flat, x: log_density(model(flat), x))
    roughness = jax.jit(lambda flat: penalty(model(flat)))
    start = time.monotonic()
    evaluations, history = [], []
    best = {'score': np.inf, 'flat': flat_initial, 'step': 0, 'logs': None}
    count = [0]

    def objective(flat):
        value, gradient = full_value_gradient(jnp.asarray(flat))
        value, gradient = float(value), np.asarray(gradient, dtype=float)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite deterministic likelihood or gradient')
        evaluations.append({'value': value, 'gradient_norm': float(np.linalg.norm(gradient)),
                            'seconds': time.monotonic()-start})
        return value, gradient

    def checkpoint(flat, step):
        logs = np.concatenate([np.asarray(evaluate(jnp.asarray(flat), jnp.asarray(valid[s:s+batch_size])))
                               for s in range(0, len(valid), batch_size)])
        energy = float(roughness(jnp.asarray(flat)))
        score = -float(logs.mean())+strength*energy
        row = {'step': step, 'nll': -float(logs.mean()), 'roughness': energy, 'score': score,
               'seconds': time.monotonic()-start, 'function_evaluations': len(evaluations)}
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite validation likelihood')
        history.append(row)
        if score < best['score']:
            best.update(score=score, flat=jnp.asarray(flat), step=step, logs=logs)
        if callback:
            callback(row)

    def accepted(flat):
        count[0] += 1
        if count[0] % checkpoint_every == 0:
            checkpoint(flat, count[0])

    checkpoint(flat_initial, 0)
    result = minimize(objective, np.asarray(flat_initial), jac=True, method='L-BFGS-B', callback=accepted,
                      options={'maxiter': iterations, 'ftol': 1e-12, 'gtol': 1e-6, 'maxls': 30, 'maxcor': 15})
    if history[-1]['step'] != result.nit:
        checkpoint(result.x, int(result.nit))
    record = {'history': history, 'evaluations': evaluations, 'best_step': best['step'],
              'best_score': best['score'], 'best_nll': -float(best['logs'].mean()),
              'iterations': int(result.nit), 'success': bool(result.success), 'message': str(result.message),
              'seconds': time.monotonic()-start, 'training_count': len(train), 'batch_size': batch_size,
              'objective': 'full training NLL plus amplitude H2; batch size controls memory only'}
    return model(best['flat']), model(jnp.asarray(result.x)), best['logs'], record

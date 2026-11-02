"""Canonical TT sweeps with convex, sign-preserving core likelihood updates.

The global shape objective is NLL + log(1 + lambda * Sobolev_energy).
Fixing the other cores and rescaling the active core gives the convex
subproblem c.T Q c - 2 sum_i w_i log(abs(v_i.T c)) within its current sign
sector. Damped Newton-CG preserves the signs at all training observations.
All parameters remain coefficients of one Chebyshev functional TT.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from .cheb_gram import chebyshev_gram_matrix
from .direct_training import (
    amplitude,
    chebyshev_features,
    integrated_energies,
    load_tt,
    save_tt,
    write_record,
)
from .gaussian_convolution import _transfer
from .sobolev import derivative_grams
from .tt_density import rebuild_tt


def shift_center(cores, index, direction):
    """Change only the physical-L2 TT gauge by an exact reduced QR."""
    c = np.asarray(cores[index])
    a, b, n = c.shape
    if direction == 1:
        q, r = np.linalg.qr(c.transpose(0, 2, 1).reshape(a*n, b))
        cores[index] = jnp.asarray(q.reshape(a, n, -1).transpose(0, 2, 1))
        cores[index+1] = jnp.einsum('ab,bcn->acn', jnp.asarray(r), cores[index+1])
    else:
        q, r = np.linalg.qr(c.reshape(a, b*n).T)
        cores[index] = jnp.asarray(q.T.reshape(-1, b, n))
        cores[index-1] = jnp.einsum('ian,ab->ibn', cores[index-1], jnp.asarray(r.T))


def moment_environment(cores, first, second, indices, *, reverse=False):
    states = tuple(jnp.ones((1, 1, 1)) if k == 0 else jnp.zeros((1, 1, 1)) for k in range(4))
    for k in indices:
        c = cores[k].transpose(1, 0, 2) if reverse else cores[k]
        gram = jnp.eye(c.shape[-1])
        transfer = lambda x, matrix, core=c: _transfer(x, core, matrix[None])
        norm, slope, axis, hessian = states
        states = (transfer(norm, gram), transfer(slope, gram)+transfer(norm, first[k]),
                  transfer(axis, gram)+transfer(norm, second[k]),
                  transfer(hessian, gram)+2*transfer(slope, first[k])+transfer(norm, second[k]))
    return tuple(x[0] for x in states)


def local_quadratic_terms(cores, first, second, index, strength, regularizer):
    left = moment_environment(cores, first, second, range(index))
    right = moment_environment(cores, first, second, range(len(cores)-1, index, -1), reverse=True)
    identity = jnp.eye(cores[index].shape[-1])
    terms = [(left[0], right[0], identity)]
    p = {'h1': 1, 'axis_h2': 2, 'hessian': 3}[regularizer]
    local = first[index] if regularizer == 'h1' else second[index]
    terms += [(left[p], right[0], strength*identity),
              (left[0], right[p], strength*identity),
              (left[0], right[0], strength*local)]
    if regularizer == 'hessian':
        terms += [(left[1], right[1], 2*strength*identity),
                  (left[1], right[0], 2*strength*first[index]),
                  (left[0], right[1], 2*strength*first[index])]
    return tuple(jnp.stack([term[k] for term in terms]) for k in range(3))


@jax.jit
def quadratic_action(core, terms):
    left, right, modes = terms
    return jnp.einsum('tac,tbd,tnm,cdm->abn', left, right, modes, core, optimize='optimal')


@jax.jit
def core_values(core, left, basis, right):
    return jnp.einsum('bi,ijn,bn,bj->b', left, core, basis, right, optimize='optimal')


@jax.jit
def core_adjoint(values, left, basis, right):
    return jnp.einsum('b,bi,bn,bj->ijn', values, left, basis, right, optimize='optimal')


@jax.jit
def newton_direction(core, terms, left, basis, right, weights):
    values = core_values(core, left, basis, right)
    qc = quadratic_action(core, terms)
    gradient = 2*qc-2*core_adjoint(weights/values, left, basis, right)
    l, r, m = terms
    diagonal = jnp.einsum('ti,tj,tn->ijn', jnp.diagonal(l, axis1=1, axis2=2),
        jnp.diagonal(r, axis1=1, axis2=2), jnp.diagonal(m, axis1=1, axis2=2))
    diagonal = 2*diagonal+2*core_adjoint(weights/values**2, left**2, basis**2, right**2)
    diagonal = jnp.maximum(diagonal, 1e-12)

    def hessian(vector):
        projected = core_values(vector, left, basis, right)
        return 2*quadratic_action(vector, terms)+2*core_adjoint(weights*projected/values**2, left, basis, right)

    direction, _ = cg(hessian, -gradient, M=lambda v: v/diagonal, tol=1e-4, maxiter=60)
    residual = jnp.linalg.norm(hessian(direction)+gradient)/jnp.maximum(jnp.linalg.norm(gradient), 1e-30)
    descent = jnp.sum(direction*gradient) < 0
    direction = jnp.where(descent, direction, -gradient/diagonal)
    return direction, gradient, values, qc, residual


def minimize_core(core, terms, left, basis, right, weights, *, steps=5):
    records = []
    for iteration in range(steps):
        delta, gradient, values, qc, residual = newton_direction(core, terms, left, basis, right, weights)
        if not np.isfinite(np.asarray(delta)).all():
            raise FloatingPointError('Nonfinite core Newton direction')
        projected = core_values(delta, left, basis, right)
        qd = quadratic_action(delta, terms)
        energy = float(jnp.sum(core*qc))
        cross, curvature = float(jnp.sum(core*qd)), float(jnp.sum(delta*qd))
        before = energy-2*float(jnp.sum(weights*jnp.log(jnp.abs(values))))
        slope = float(jnp.sum(gradient*delta))
        limit = float(jnp.min(jnp.where(values*projected < 0, -values/projected, jnp.inf)))
        alpha = min(1., .99*limit)
        accepted = False
        for _ in range(30):
            new_values = values+alpha*projected
            candidate_energy = energy+2*alpha*cross+alpha**2*curvature
            after = candidate_energy-2*float(jnp.sum(weights*jnp.log(jnp.abs(new_values))))
            if np.isfinite(after) and after <= before+1e-4*alpha*slope+1e-12:
                accepted = True
                break
            alpha *= .5
        if not accepted:
            break
        if np.any(np.asarray(values*new_values) <= 0):
            raise FloatingPointError('Core update crossed a training amplitude zero')
        core = core+alpha*delta
        records.append({'newton_step': iteration, 'before': before, 'after': after,
                        'step_length': alpha, 'cg_residual': float(residual)})
        if abs(after-before) < 1e-9:
            break
    energy = float(jnp.sum(core*quadratic_action(core, terms)))
    if not np.isfinite(energy) or energy <= 0:
        raise FloatingPointError('Invalid penalized norm after core optimization')
    return core/np.sqrt(energy), records


def fit_core_sweeps(initial, train, valid, output, *, strength=1e-3, regularizer='hessian',
                    sweeps=5, newton_steps=5, sample_weights=None):
    if regularizer not in ('h1', 'axis_h2', 'hessian') or strength < 0:
        raise ValueError('Invalid core-sweep regularizer')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    domain = np.asarray(initial.domain)
    sizes = tuple(c.coeffs.shape[-1] for c in initial.cores)
    gram = tuple(np.asarray(chebyshev_gram_matrix(n, box)) for n, box in zip(sizes, domain, strict=True))
    factors = tuple(np.linalg.cholesky(g) for g in gram)
    inverse = tuple(np.linalg.inv(f) for f in factors)
    cores = [jnp.asarray(c.coeffs)@f for c, f in zip(initial.cores, factors, strict=True)]
    for k in range(len(cores)-1, 0, -1):
        shift_center(cores, k, -1)
    scales = .05*np.ptp(domain, axis=1)
    first, second = (tuple(jnp.asarray(inv@np.asarray(d)@inv.T*s**(2*order))
                          for inv, d, s in zip(inverse, derivative_grams(initial, order), scales, strict=True))
                     for order in (1, 2))
    identity = tuple(jnp.eye(n) for n in sizes)
    training, validation = (tuple(jnp.asarray(b@inv.T) for b, inv in zip(
        chebyshev_features(data, domain, sizes), inverse, strict=True)) for data in (train, valid))
    weights = np.ones(len(train)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != (len(train),) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('Invalid observation weights')
    weights = jnp.asarray(weights/weights.sum())
    penalty_index = {'h1': 1, 'axis_h2': 2, 'hessian': 3}[regularizer]
    evaluate = jax.jit(lambda c, b: amplitude(c, b))
    energies = jax.jit(lambda c: integrated_energies(c, identity, first, second))
    start, history, updates = time.monotonic(), [], []
    best_nll, best_step = np.inf, 0

    def model():
        return rebuild_tt(initial, tuple(c@inv for c, inv in zip(cores, inverse, strict=True)))

    def checkpoint(sweep):
        nonlocal best_nll, best_step
        e = np.asarray(energies(tuple(cores)), dtype=float)
        tr = np.concatenate([np.asarray(evaluate(tuple(cores), tuple(b[j:j+4096] for b in training)))
                             for j in range(0, len(train), 4096)])
        va = np.asarray(evaluate(tuple(cores), validation))
        logs = np.log(va**2+1e-300)-np.log(e[0])
        objective = np.log(e[0]+strength*e[penalty_index])-2*float(np.dot(np.asarray(weights), np.log(np.abs(tr))))
        nll = -float(logs.mean())
        row = {'step': sweep, 'validation_nll': nll, 'training_objective': objective,
               'roughness_hessian': float(e[3]/e[0]), 'roughness_axis_h2': float(e[2]/e[0]),
               'roughness_h1': float(e[1]/e[0]), 'seconds': time.monotonic()-start}
        if not np.isfinite(list(row.values())).all():
            raise FloatingPointError('Nonfinite sweeping checkpoint')
        if history and objective > history[-1]['training_objective']+1e-7:
            raise FloatingPointError('Sweeping increased the full training objective')
        history.append(row)
        if nll < best_nll:
            best_nll, best_step = nll, sweep
            save_tt(output/'fit.npz', model())
            np.save(output/'validation.npy', logs)
        write_record(output/'progress.json', {'history': history, 'best_step': best_step, 'best_nll': best_nll})
        print(output.name, row, flush=True)

    checkpoint(0)
    for sweep in range(1, sweeps+1):
        direction = 1 if sweep % 2 else -1
        indices = range(len(cores)) if direction == 1 else range(len(cores)-1, -1, -1)
        for index in indices:
            left = jnp.ones((len(train), 1))
            for k in range(index):
                left = jnp.einsum('bi,ijn,bn->bj', left, cores[k], training[k], optimize='optimal')
            right = jnp.ones((len(train), 1))
            for k in range(len(cores)-1, index, -1):
                right = jnp.einsum('ijn,bn,bj->bi', cores[k], training[k], right, optimize='optimal')
            terms = local_quadratic_terms(cores, first, second, index, strength, regularizer)
            metric = np.eye(sizes[index])+strength*np.asarray(first[index] if regularizer == 'h1' else second[index])
            f = np.linalg.cholesky(metric)
            inv = np.linalg.inv(f)
            terms = (terms[0], terms[1], jnp.einsum('an,tnm,bm->tab', jnp.asarray(inv), terms[2], jnp.asarray(inv)))
            core = cores[index]@f
            energy = float(jnp.sum(core*quadratic_action(core, terms)))
            core /= np.sqrt(energy)
            core, record = minimize_core(core, terms, left, training[index]@inv.T, right, weights, steps=newton_steps)
            cores[index] = core@inv
            updates.append({'sweep': sweep, 'core': index, 'updates': record})
            if 0 <= index+direction < len(cores):
                shift_center(cores, index, direction)
        checkpoint(sweep)
        write_record(output/'core_updates.json', updates)
    save_tt(output/'last.npz', model())
    result = {'history': history, 'best_step': best_step, 'best_nll': best_nll,
              'objective': 'NLL + log(1 + lambda * normalized amplitude Sobolev energy)',
              'strength': strength, 'regularizer': regularizer, 'sweeps': sweeps,
              'newton_steps': newton_steps, 'seconds': time.monotonic()-start,
              'training_count': len(train), 'uses_signal_pilot': False,
              'selection': 'minimum held-out joint NLL checkpoint'}
    write_record(output/'result.json', result)
    return load_tt(output/'fit.npz'), result

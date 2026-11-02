"""Deterministic full-data optimization of the direct Chebyshev TT objectives."""
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from .direct_training import load_tt, save_tt, write_record


def fit_lbfgs(problem, train, valid, output, *, iterations=300, batch_size=2000, checkpoint_every=10, validation_batch_size=4096):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    flat, unravel = ravel_pytree(problem.params)
    training, validation = problem.features(train), problem.features(valid)
    count = len(train)//batch_size
    if count == 0:
        batch_size, count = len(train), 1
    batches = tuple(x[:count*batch_size].reshape(count, batch_size, -1) for x in training)
    remainder = tuple(x[count*batch_size:] for x in training)

    @jax.jit
    def value_gradient(vector):
        def add(state, features):
            val, grad = jax.value_and_grad(lambda v: problem.loss(unravel(v), features))(vector)
            return (state[0]+val*batch_size/len(train), state[1]+grad*batch_size/len(train)), None
        (val, grad), _ = jax.lax.scan(add, (jnp.array(0.), jnp.zeros_like(vector)), batches)
        if len(remainder[0]):
            v, g = jax.value_and_grad(lambda v: problem.loss(unravel(v), remainder))(vector)
            val, grad = val+v*len(remainder[0])/len(train), grad+g*len(remainder[0])/len(train)
        return val, grad

    start = time.monotonic()
    history, evaluations = [], []
    best_nll, best_step, step = np.inf, 0, 0

    def objective(vector):
        v, g = value_gradient(jnp.asarray(vector))
        v, g = float(v), np.asarray(g, dtype=float)
        if not np.isfinite(v) or not np.isfinite(g).all():
            raise FloatingPointError('Nonfinite full-data likelihood or gradient')
        evaluations.append({'value': v, 'gradient_norm': float(np.linalg.norm(g))})
        return v, g

    def checkpoint(vector):
        nonlocal best_nll, best_step
        params = unravel(jnp.asarray(vector))
        logs = np.concatenate([np.asarray(problem.logs(params, tuple(b[i:i+validation_batch_size] for b in validation)))
                               for i in range(0, len(valid), validation_batch_size)])
        e = np.asarray(problem.energies(params))
        nll = -float(logs.mean())
        row = {'step': step, 'validation_nll': nll, 'roughness_h1': float(e[1]/e[0]),
               'roughness_axis_h2': float(e[2]/e[0]), 'roughness_hessian': float(e[3]/e[0]),
               'training_objective': evaluations[-1]['value'] if evaluations else float(value_gradient(flat)[0]),
               'seconds': time.monotonic()-start}
        history.append(row)
        if nll < best_nll:
            best_nll, best_step = nll, step
            save_tt(output/'fit.npz', problem.model(params))
            np.save(output/'validation.npy', logs)
        write_record(output/'progress.json', {'history': history, 'best_step': best_step, 'best_nll': best_nll})
        print(output.name, row, flush=True)

    def accepted(vector):
        nonlocal step
        step += 1
        if step % checkpoint_every == 0:
            checkpoint(vector)

    checkpoint(flat)
    result = minimize(objective, np.asarray(flat), jac=True, method='L-BFGS-B', callback=accepted,
                      options={'maxiter': iterations, 'maxls': 30, 'maxcor': 15, 'ftol': 1e-12, 'gtol': 1e-6})
    if history[-1]['step'] != step:
        checkpoint(result.x)
    save_tt(output/'last.npz', problem.model(unravel(jnp.asarray(result.x))))
    record = {'history': history, 'evaluations': evaluations, 'best_step': best_step, 'best_nll': best_nll,
              'iterations': int(result.nit), 'success': bool(result.success), 'message': str(result.message),
              'seconds': time.monotonic()-start, 'training_count': len(train), 'uses_signal_pilot': False,
              'objective': problem.objective, 'regularizer': problem.regularizer, 'strength': problem.strength,
              'precondition_strength': problem.precondition_strength,
              'marginal_strength': problem.marginal_strength,
              'selection': 'minimum held-out joint NLL checkpoint'}
    write_record(output/'result.json', record)
    return load_tt(output/'fit.npz'), record

"""Learn TT coefficients with likelihood, a Fourier kernel score, or both."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .direct_training import save_tt, write_record
from .kernel_score import (
    FourierKernelScore,
    empirical_characteristic,
    gaussian_mixture_characteristic,
)


def fit_scored(problem, train, valid, output, frequencies, validation_frequencies, *,
               noise_weights, noise_covariances, objective='nll', kernel_weight=100.,
               steps=4000, batch_size=256, frequency_batch=32, learning_rate=.001,
               checkpoint_every=400, validation_batch=128, seed=0):
    if objective not in ('nll', 'mmd', 'hybrid') or kernel_weight <= 0:
        raise ValueError('Invalid score configuration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    training, validation = problem.features(train), problem.features(valid)
    val_score = FourierKernelScore(problem, validation_frequencies,
        noise_characteristic=gaussian_mixture_characteristic(validation_frequencies, noise_weights, noise_covariances))
    target_valid = empirical_characteristic(valid, validation_frequencies)
    if objective != 'nll':
        score = FourierKernelScore(problem, frequencies,
            noise_characteristic=gaussian_mixture_characteristic(frequencies, noise_weights, noise_covariances))
        target_train = jnp.asarray(empirical_characteristic(train, frequencies))
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(
        optax.cosine_decay_schedule(learning_rate, steps, alpha=.1)))
    params = problem.params
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, key):
        data_key, frequency_key = jax.random.split(key)
        indices = jax.random.randint(data_key, (batch_size,), 0, len(train))
        features = tuple(a[indices] for a in training)
        fi = jax.random.randint(frequency_key, (frequency_batch,), 0, len(frequencies))
        def loss(p):
            e = problem.energies(p)
            penalty = e[{'h1': 1, 'axis_h2': 2, 'hessian': 3}[problem.regularizer]]/e[0]
            value = problem.strength*penalty
            if objective != 'mmd':
                value -= jnp.mean(problem.logs(p, features))
            if objective != 'nll':
                value += kernel_weight*score.discrepancy(p, target_train, fi)
            return value
        value, gradient = jax.value_and_grad(loss)(params)
        updates, state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, updates), state, value

    key = jax.random.PRNGKey(seed)
    best_nll, best_kernel = np.inf, np.inf
    best_step = kernel_step = 0
    history = []
    last_loss = None
    for iteration in range(steps+1):
        if iteration % checkpoint_every == 0 or iteration == steps:
            logs = np.concatenate([np.asarray(problem.logs(params, tuple(a[k:k+validation_batch] for a in validation)))
                                   for k in range(0, len(valid), validation_batch)])
            prediction = val_score.predict(params, batch_size=32)
            mmd = float(np.mean(np.abs(prediction-target_valid)**2))
            e = np.asarray(problem.energies(params))
            nll = -float(logs.mean())
            if not np.isfinite(logs).all() or not np.isfinite(mmd) or not np.isfinite(e).all() or e[0] <= 0:
                raise FloatingPointError('Invalid scored-training checkpoint')
            row = {'step': iteration, 'validation_nll': nll, 'validation_mmd': mmd,
                   'roughness_hessian': float(e[3]/e[0]), 'batch_objective': last_loss,
                   'seconds': time.monotonic()-start}
            history.append(row)
            if nll < best_nll:
                best_nll, best_step = nll, iteration
                save_tt(output/'fit.npz', problem.model(params))
                np.savez_compressed(output/'validation.npz', logs=logs, characteristic=prediction)
            if mmd < best_kernel:
                best_kernel, kernel_step = mmd, iteration
                save_tt(output/'kernel_fit.npz', problem.model(params))
                np.savez_compressed(output/'kernel_validation.npz', logs=logs, characteristic=prediction)
            write_record(output/'progress.json', {'history': history})
            print(output.name, row, flush=True)
        if iteration < steps:
            key, subkey = jax.random.split(key)
            params, state, loss = step(params, state, subkey)
            last_loss = float(loss)
            if not np.isfinite(last_loss):
                raise FloatingPointError('Nonfinite training objective')
    record = {'history': history, 'best_nll': best_nll, 'best_step': best_step,
              'best_kernel': best_kernel, 'kernel_step': kernel_step,
              'objective': objective, 'kernel_weight': kernel_weight, 'steps': steps,
              'strength': problem.strength, 'regularizer': problem.regularizer,
              'seconds': time.monotonic()-start, 'uses_signal_pilot': False,
              'checkpoint_rules': 'separate minimum held-out NLL and minimum held-out kernel score',
              'validation_quadrature_orders': val_score.orders}
    write_record(output/'result.json', record)
    return record

"""Fit the positive-core Chebyshev density TT using observations only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from refresh_density_fits import read_model

from ftde.positive_chebyshev import (
    PositiveChebyshevTT,
    expand_positive_modes,
    initialize_positive_tt,
)
from ftde.positive_tt_fit import PositiveTTForward, make_core_solver, sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--modes', type=int, default=49)
    parser.add_argument('--subdivisions', type=int, default=16)
    parser.add_argument('--width', type=float, default=.4)
    parser.add_argument('--sweeps', type=int, default=30)
    parser.add_argument('--inner-steps', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--barrier', type=float, default=1e-3)
    parser.add_argument('--barrier-final', type=float, default=1e-5)
    parser.add_argument('--strength', type=float, default=1e-7)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['data_sha256'] = hashlib.sha256((args.base/'data.npz').read_bytes()).hexdigest()
    config['model'] = 'direct Chebyshev density TT, positive polynomial cores, stochastic core integrals'
    config['solver_version'] = 'qr_whitened_newton_v3'
    config['selection'] = 'minimum noisy validation NLL; no truth in fitting or selection'
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text()) != config:
        raise ValueError('Choose a fresh output directory for different settings')
    if (out/'result.json').exists():
        print('Already complete:', out, flush=True)
        return
    (out/'experiment.json').write_text(json.dumps(config, indent=2)+'\n')
    start = time.monotonic()
    data = np.load(args.base/'data.npz')
    train, valid = data['train'], data['valid']
    domain = np.asarray(read_model(args.base/'selected.npz').domain)
    initial = PositiveChebyshevTT.load(args.initial) if args.initial else initialize_positive_tt(
        train, domain, rank=args.rank, modes=args.modes, subdivisions=args.subdivisions,
        seed=args.seed, width=args.width)
    if args.initial:
        if max(initial.tt.ranks) != args.rank or initial.subdivisions != args.subdivisions:
            raise ValueError('Initial model rank/subdivision settings do not match')
        initial = expand_positive_modes(initial, args.modes)
    initial.save(out/'init.npz')
    settings = json.loads((args.base/'settings.json').read_text())
    noisy = settings.get('noise') != 'free'
    kernel = read_model(args.base/'kernel.npz') if noisy else None
    observations = np.concatenate([train, valid])
    observation_domain = np.stack([observations.min(axis=0)-.02, observations.max(axis=0)+.02], axis=1)
    if not noisy and np.any((observations < domain[:, 0]) | (observations > domain[:, 1])):
        raise ValueError('Observations outside density support')
    forward = PositiveTTForward(initial, observation_domain, kernel)
    params = forward.parameters(initial)
    # Strictly interior phase-I initialization, also for resumed checkpoints.
    # The affine move toward constant normalized cores preserves row masses.
    interior = []
    for u, coord in zip(params, forward.coordinates, strict=True):
        minimum = float(jnp.min(u@jnp.asarray(coord.bounds.T)))
        fraction = max(0., (1e-5-minimum)/(1-minimum)) if minimum < 1e-5 else 0.
        interior.append((1-fraction)*u+fraction)
    params = tuple(interior)
    forward.model(params).save(out/'init.npz')
    solvers = [make_core_solver(c, b-a) for c, (a, b) in zip(forward.coordinates, domain, strict=True)]
    evaluate = jax.jit(forward.log_density)
    best_nll, history, best_step = np.inf, [], 0
    rng = np.random.default_rng(args.seed)
    order, position = rng.permutation(len(train)), 0
    for iteration in range(args.sweeps+1):
        logs = np.concatenate([np.asarray(evaluate(params, jnp.asarray(valid[s:s+1024]))) for s in range(0, len(valid), 1024)])
        nll = -float(logs.mean())
        model = forward.model(params)
        audit = model.validate()
        if not np.isfinite(nll):
            raise FloatingPointError('Invalid validation likelihood')
        if nll < best_nll:
            best_nll, best_step = nll, iteration
            model.save(out/'fit.npz')
            np.save(out/'validation.npy', logs)
        record = {'sweep': iteration, 'nll': nll, 'seconds': time.monotonic()-start,
                  'min_certificate': min(audit['minimum_core_certificates'])}
        history.append(record)
        (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(out.name, record, flush=True)
        if iteration == args.sweeps:
            break
        count = min(args.batch_size, len(train))
        if position+count > len(train):
            order, position = rng.permutation(len(train)), 0
        batch = train[order[position:position+count]]
        position += count
        features = [f(jnp.asarray(batch[:, k])) for k, f in enumerate(forward.features)]
        mu = args.barrier*(args.barrier_final/args.barrier)**min(1., iteration/max(.7*args.sweeps, 1.))
        params, diagnostic = sweep(params, features, solvers, barrier=mu, strength=args.strength,
                                   inner_steps=args.inner_steps, reverse=bool(iteration % 2))
        record['barrier'] = mu
        record['core_updates'] = diagnostic
    forward.model(params).save(out/'last.npz')
    result = {'best_nll': best_nll, 'best_sweep': best_step, 'seconds': time.monotonic()-start,
              'history': history, 'certificate': PositiveChebyshevTT.load(out/'fit.npz').audit(),
              'device': jax.devices()[0].device_kind}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', out, best_nll, best_step, flush=True)


if __name__ == '__main__':
    main()

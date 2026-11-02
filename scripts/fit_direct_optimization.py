"""Direct TT optimization from a near-uniform start; no fitted signal pilots.

This training script never imports or evaluates the signal truth. The saved
noise specification, observations and box are the complete statistical input.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from refresh_density_fits import fit_candidate, read_model, save_model

from ftde.linear_tt_convolution import LinearTTConvolution
from ftde.signed_tt_least_squares import SignedTTLeastSquares4D
from ftde.simple_initialization import init_tt_constant
from ftde.sobolev import AmplitudeSobolevPenalty


def record(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare_problem(base, root):
    root.mkdir(parents=True, exist_ok=True)
    if not (root/'previous_figures.json').exists():
        record(root/'previous_figures.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in Path('figs').rglob('*.png')})
    data = np.load(base/'data.npz')
    # Only the old box is reused, never the old fitted coefficients.
    domain = np.load(base/'init.npz')['domain']
    dim = len(domain)
    if not (base/'kernel.npz').exists():
        if json.loads((base/'settings.json').read_text()).get('noise') != 'free':
            raise FileNotFoundError(f'Missing prescribed noise kernel: {base}')
        info = {'base': str(base), 'data_sha256': hashlib.sha256((base/'data.npz').read_bytes()).hexdigest(),
                'domain': domain.tolist(), 'noise': 'free', 'curvature_scales': (.05*np.ptp(domain, axis=1)).tolist(),
                'initialization': 'constant plus small random coefficients; no fitted pilot',
                'selection': 'held-out observations only; truth reserved for final evaluation'}
        if (root/'problem.json').exists() and json.loads((root/'problem.json').read_text()) != info:
            raise ValueError('Problem changed; use a separate output directory')
        record(root/'problem.json', info)
        return data, domain, None, None
    # Prescribed noise from the earlier saved observation-only study.
    noise_path = Path(f'artifacts/swirl_improvement/swirl_mixture_dim{dim}/noise.npz')
    saved_noise = np.load(noise_path)
    covs, weights = saved_noise['covariances'], saved_noise['weights']
    info = {'base': str(base), 'data_sha256': hashlib.sha256((base/'data.npz').read_bytes()).hexdigest(),
            'kernel_sha256': hashlib.sha256((base/'kernel.npz').read_bytes()).hexdigest(),
            'domain': domain.tolist(), 'noise_source': str(noise_path),
            'initialization': 'constant plus small random coefficients; no fitted pilot',
            'selection': 'noisy validation only; truth reserved for final evaluation'}
    if (root/'problem.json').exists() and json.loads((root/'problem.json').read_text()) != info:
        raise ValueError('Problem changed; use a separate output directory')
    record(root/'problem.json', info)
    return data, domain, covs, weights


def signed(args, data, domain, covs, weights):
    geometry_file = args.root/f'geometry_n{args.modes}_q{args.order}.npz'
    start = time.monotonic()
    if geometry_file.exists():
        a = np.load(geometry_file)
        geometry = SignedTTLeastSquares4D(domain,
            tuple(jnp.asarray(a[f'transform_{k}']) for k in range(4)),
            tuple(jnp.asarray(a[f'moment_{k}']) for k in range(4)),
            tuple(jnp.asarray(a[f'derivative_{k}']) for k in range(4)),
            jnp.asarray(a['left_grams']), jnp.asarray(a['right_grams']),
            jnp.asarray(a['overlap_weights']), args.order)
        train_term, valid_term = jnp.asarray(a['train_term']), jnp.asarray(a['valid_term'])
    else:
        print('PREPARE empirical objective coefficients', flush=True)
        if covs is None:
            geometry = SignedTTLeastSquares4D.without_noise(domain, args.modes)
            train_term = geometry.empirical_term_without_noise(data['train'])
            valid_term = geometry.empirical_term_without_noise(data['valid'])
        else:
            geometry = SignedTTLeastSquares4D.from_noise(domain, args.modes, covs, weights, order=args.order)
            left, right = geometry.initial_factors(args.rank, args.seed)
            template = geometry.model(left, right)
            kernel = read_model(args.base/'kernel.npz')
            # Safe interval: kernel lower + signal upper, kernel upper + signal lower.
            observation_domain = np.stack([np.asarray(kernel.domain)[:, 0]+domain[:, 1],
                                           np.asarray(kernel.domain)[:, 1]+domain[:, 0]], axis=1)
            operator = LinearTTConvolution.from_tt(template, kernel, observation_domain)
            train_term = geometry.empirical_term(operator, data['train'])
            valid_term = geometry.empirical_term(operator, data['valid'])
        np.savez_compressed(geometry_file, **{f'transform_{k}': geometry.transforms[k] for k in range(4)},
                 **{f'moment_{k}': geometry.moments[k] for k in range(4)},
                 **{f'derivative_{k}': geometry.derivatives[k] for k in range(4)},
                 left_grams=geometry.left_grams, right_grams=geometry.right_grams,
                 overlap_weights=geometry.overlap_weights, train_term=train_term, valid_term=valid_term)
    print('PREPARED seconds', time.monotonic()-start, flush=True)
    terms = jax.jit(geometry.terms)
    penalty_bases = geometry.penalty_bases(args.penalty_order) if args.negative_strength else None
    if penalty_bases is not None:
        @jax.jit
        def negative_energy(left, right):
            weighted = (penalty_bases[0]@left)@(right@penalty_bases[1].T)
            return jnp.sum(jnp.minimum(weighted, 0)**2)
    ml, mr = geometry.pair_moments
    for strength in args.strengths:
        suffix = f'_neg{args.negative_strength:g}' if args.negative_strength else ''
        out = args.root/f'signed_n{args.modes}_r{args.rank}_l{strength:g}_s{args.seed}{suffix}'
        out.mkdir(exist_ok=True)
        config = {'method': 'signed_als', 'modes': args.modes, 'rank': args.rank, 'seed': args.seed,
                  'strength': strength, 'sweeps': args.sweeps, 'order': args.order,
                  'ridge': 1e-10, 'selection': 'minimum unpenalized noisy validation least-squares risk'}
        if covs is None:
            config['selection'] = 'minimum unpenalized noise-free validation least-squares risk'
        if args.negative_strength:
            config.update(negative_strength=args.negative_strength, penalty_order=args.penalty_order)
        if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
            raise ValueError('Run settings changed')
        record(out/'config.json', config)
        if (out/'result.json').exists():
            continue
        left, right = geometry.initial_factors(args.rank, args.seed)
        save_model(out/'init.npz', geometry.model(left, right))
        first = geometry.make_half_solver(strength=strength, negative_strength=args.negative_strength, penalty_bases=penalty_bases)
        second = geometry.make_half_solver(strength=strength, reverse=True,
                    negative_strength=args.negative_strength, penalty_bases=penalty_bases)
        best_score, best_step, history, residual = np.inf, 0, [], 0.
        start = time.monotonic()
        previous_objective = np.inf
        for sweep in range(args.sweeps+1):
            train_risk, roughness = map(float, terms(left, right, train_term))
            valid_risk = float(terms(left, right, valid_term)[0])
            mass = float(ml@left@right@mr)
            objective = train_risk+strength*roughness+1e-10*float(jnp.sum((left.T@left)*(right@right.T)))
            negative = float(negative_energy(left, right)) if args.negative_strength else 0.
            objective += args.negative_strength*negative
            row = {'sweep': sweep, 'train_risk': train_risk, 'valid_risk': valid_risk, 'roughness': roughness,
                   'objective': objective, 'mass': mass, 'cg_relative_residual': residual,
                   'negative_squared_integral': negative,
                   'seconds': time.monotonic()-start}
            history.append(row)
            print(out.name, row, flush=True)
            if not np.isfinite(objective) or abs(mass-1) > 1e-7:
                raise FloatingPointError('Invalid objective or lost mass constraint')
            if objective > previous_objective+1e-7:
                raise FloatingPointError('Constrained ALS increased the training objective')
            previous_objective = objective
            if valid_risk < best_score:
                best_score, best_step = valid_risk, sweep
                save_model(out/'fit.npz', geometry.model(left, right))
                np.savez(out/'factors.npz', left=left, right=right)
            if sweep == args.sweeps:
                break
            left, right, r1 = first(left, right, train_term)
            reverse_left, reverse_right, r2 = second(right.T, left.T, train_term.T)
            left, right = reverse_right.T, reverse_left.T
            residual = max(float(r1), float(r2))
        save_model(out/'last.npz', geometry.model(left, right))
        record(out/'result.json', {'best_step': best_step, 'best_score': best_score, 'history': history,
                                  'seconds': time.monotonic()-start, 'device': jax.devices()[0].device_kind})


def squared(args, data, domain, covs, weights):
    kernel = read_model(args.base/'kernel.npz') if covs is not None else None
    for strength in args.strengths:
        out = args.root/f'squared_n{args.modes}_r{args.rank}_l{strength:g}_b{args.batch}_s{args.seed}'
        out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            saved = json.loads((out/'result.json').read_text())
            expected = {'strength': strength, 'steps': args.steps, 'lr': args.lr,
                        'batch_size': args.batch, 'seed': args.seed, 'checkpoint_rule': 'regularized',
                        'preconditioning': 'sobolev', 'microbatches': max(1, args.batch//256)}
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Completed squared fit has different settings: {out}')
            # In particular, never rewrite an already frozen initializer.
            continue
        initial = init_tt_constant(domain, rank=args.rank, modes=args.modes, seed=args.seed)
        save_model(out/'init.npz', initial)
        penalty = AmplitudeSobolevPenalty.from_tt(initial, order=2,
                        length_scales=(.05*np.ptp(domain, axis=1) if covs is None else
                                       np.sqrt(np.einsum('c,cii->i', weights, covs))))
        fit_candidate(initial, data['train'], data['valid'], out, kernel=kernel,
                      penalty=penalty, strength=strength, steps=args.steps, lr=args.lr,
                      batch_size=args.batch, checkpoint_rule='regularized', seed=args.seed,
                      preconditioning='sobolev', microbatches=max(1, args.batch//256))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('artifacts/swirl/swirl_mixture_dim4'))
    parser.add_argument('--root', type=Path, default=Path('artifacts/direct_optimization/swirl4'))
    parser.add_argument('--method', choices=['signed', 'squared'], default='signed')
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--modes', type=int, default=25)
    parser.add_argument('--order', type=int, default=65)
    parser.add_argument('--sweeps', type=int, default=20)
    parser.add_argument('--strengths', type=float, nargs='+', default=[1e-6, 1e-5, 1e-4, 1e-3])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--negative-strength', type=float, default=0.)
    parser.add_argument('--penalty-order', type=int, default=65)
    arguments = parser.parse_args()
    values = prepare_problem(arguments.base, arguments.root)
    (signed if arguments.method == 'signed' else squared)(arguments, *values)

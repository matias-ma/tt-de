"""Fresh 4D/10D/20D Hessian fits; freeze observation-only choices before evaluation.

Run inside one GPU allocation using run_noise_diagnosis_gpu.sh. Each fit has
its own process, so JAX memory is released between cases and recipes.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ftde.benchmark_defaults import DIMENSIONS, recipes
from ftde.direct_training import write_record

ROOT = Path('artifacts/dimension_gallery_v9')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_name(family, dim, noise):
    return f'{family}{dim}_{noise}'


def setup(family, dim, noise):
    """Reuse observations, never the historical KDE/pilot initialization."""
    from gl_common import data_domain, robust_domain
    from objective_study import noise_parameters

    from ftde.direct_training import save_tt

    folder = ROOT/case_name(family, dim, noise)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        protocol = read(folder/'protocol.json')
        if digest(protocol['data']) != protocol['data_sha256']:
            raise RuntimeError('Saved observations changed')
        return folder, protocol
    old = Path('artifacts/swirl' if family == 'swirl' else 'artifacts/figure_refresh')/f'{family}_{noise}_dim{dim}'
    weights, cov = noise_parameters(case_name(family, dim, noise), dim)
    source = old/'data.npz'
    if source.exists():
        with np.load(old/'init.npz') as data:
            domain = data['domain']
    else:
        # Synthetic samplers generate observations; density values are not used.
        if family == 'swirl':
            from swirl_fits import observations
            train, valid = (observations(dim, noise, n, seed) for n, seed in [(100000, 0), (4096, 2)])
        else:
            from ftde.ginzburg_landau import sample_ginzburg_landau
            def observations(n, seed):
                x = np.asarray(sample_ginzburg_landau(seed, n, dim))
                if noise != 'free':
                    rng = np.random.default_rng(10000+seed)
                    labels = rng.choice(len(weights), n, p=weights)
                    x = x+np.einsum('nij,nj->ni', np.linalg.cholesky(cov)[labels], rng.normal(size=x.shape))
                return x
            train, valid = observations(100000, 0), observations(4096, 2)
        source = folder/'data.npz'
        np.savez_compressed(source, train=train, valid=valid)
        domain = np.asarray(data_domain(train) if noise == 'free' else robust_domain(train))
    with np.load(source) as data:
        train, valid = data['train'], data['valid']
    if noise == 'free' and not np.all((valid >= domain[:, 0]) & (valid <= domain[:, 1])):
        raise ValueError('Validation outside observation-derived support')
    kernel_path = old/'kernel.npz'
    if noise != 'free' and not kernel_path.exists():
        from ftde.noise_kernel import build_paired_gaussian_mixture_noise_kernel_tt
        kernel_path = folder/'kernel.npz'
        save_tt(kernel_path, build_paired_gaussian_mixture_noise_kernel_tt(cov, weights, domain))
    protocol = {'family': family, 'dim': dim, 'noise': noise, 'data': str(source), 'data_sha256': digest(source),
        'domain': domain.tolist(), 'kernel': str(kernel_path) if noise != 'free' else None,
        'noise_weights': weights.tolist(), 'noise_covariances': cov.tolist(),
        'training_count': len(train), 'validation_count': len(valid), 'recipes': recipes(family, dim, noise != 'free'),
        'initialization': 'near-uniform, perturbation 0.01, seed 0; no learned signal pilot',
        'selection': 'minimum exact marginal curvature within one paired SE of minimum held-out NLL' if noise != 'free' else 'minimum held-out NLL checkpoint',
        'validation_clusters': 200 if family == 'gl' else None, 'truth_used_for_selection': False}
    write_record(folder/'protocol.json', protocol)
    return folder, protocol


def fit(family, dim, noise, name):
    import jax

    from ftde.core_sweeps import fit_core_sweeps
    from ftde.direct_training import (
        DirectTrainingProblem,
        NoisyDirectTrainingProblem,
        load_tt,
    )
    from ftde.simple_initialization import init_tt_constant
    from ftde.tt_convolution import TTConvolution

    folder, protocol = setup(family, dim, noise)
    out = folder/name
    config = protocol['recipes'][name].copy()
    out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        return
    if (folder/'selection.json').exists():
        raise RuntimeError('Case frozen; use a new study root')
    with np.load(protocol['data']) as data:
        train, valid = data['train'], data['valid']
    parent = config['parent']
    initial = load_tt(folder/parent/'fit.npz') if parent else init_tt_constant(
        np.array(protocol['domain']), rank=config['rank'], modes=config['modes'], seed=config['seed'])
    if parent:
        config['parent_sha256'] = digest(folder/parent/'fit.npz')
    if config.get('expand_rank'):
        from refine_cryoem_gallery import expand_tt
        initial = expand_tt(initial, config['modes'], config['rank'], jitter=config['jitter'], seed=config['seed'])
    if 'marginal_budget' in config:
        calibration = DirectTrainingProblem(initial, strength=0., marginal_strength=1.)
        curvature = float(jax.jit(calibration.marginal_penalty)(calibration.params))
        if curvature <= 0:
            raise ValueError('Positive curvature needed for marginal calibration')
        config['source_curvature'] = curvature
        config['marginal_strength'] = config['marginal_budget']/curvature
    if (out/'config.json').exists() and read(out/'config.json') != json.loads(json.dumps(config)):
        raise RuntimeError('Configuration differs from existing fit')
    write_record(out/'config.json', config)
    if config['optimizer'] == 'core_sweeps':
        fit_core_sweeps(initial, train, valid, out, regularizer='hessian', strength=config['strength'],
                        sweeps=config['sweeps'], newton_steps=5)
        return
    kwargs = {'regularizer': 'hessian', 'strength': config['strength'], 'marginal_strength': config['marginal_strength']}
    if noise == 'free':
        problem = DirectTrainingProblem(initial, **kwargs)
    else:
        box = np.stack([np.minimum(train.min(0), valid.min(0))-.02,
                        np.maximum(train.max(0), valid.max(0))+.02], axis=1)
        forward = TTConvolution.from_tt(initial, load_tt(protocol['kernel'])).prepare(box, tolerance=1e-14)
        problem = NoisyDirectTrainingProblem(initial, forward, **kwargs)
    problem.fit_adam(train, valid, out, steps=config['steps'], batch_size=config['batch_size'],
        learning_rate=config['learning_rate'], mode_schedule=config['mode_schedule'],
        selection_start=config['selection_start'], validation_batch_size=64 if dim == 20 else 128,
        microbatches=config.get('microbatches', 1))


def select(family, dim, noise):
    import jax

    from ftde.direct_training import DirectTrainingProblem, load_tt
    from ftde.noisy_model_selection import select_by_paired_noisy_nll
    folder, protocol = setup(family, dim, noise)
    if (folder/'selection.json').exists():
        return
    names = list(protocol['recipes'])
    logs, roughness, files = [], [], {}
    for name in names:
        out = folder/name
        if not (out/'result.json').exists():
            raise RuntimeError(f'Incomplete fit: {out}')
        logs.append(np.load(out/'validation.npy'))
        p = DirectTrainingProblem(load_tt(out/'fit.npz'), strength=0., marginal_strength=1.)
        roughness.append(float(jax.jit(p.marginal_penalty)(p.params)))
        for path in [out/'fit.npz', out/'config.json', out/'result.json', out/'validation.npy']:
            files[str(path)] = digest(path)
        jax.clear_caches()
    cluster_ids = np.arange(len(logs[0])) % 200 if family == 'gl' else None
    complexity = roughness if noise != 'free' else np.zeros(len(logs))
    if protocol.get('selection_complexity') == 'rank_then_marginal':
        complexity = [protocol['recipes'][name]['rank']+r/(1+r) for name, r in zip(names, roughness, strict=True)]
    chosen, detail = select_by_paired_noisy_nll(np.stack(logs), complexity, cluster_ids=cluster_ids)
    for path in [Path(protocol['data']), folder/'protocol.json'] + ([Path(protocol['kernel'])] if noise != 'free' else []):
        files[str(path)] = digest(path)
    minimum = int(np.argmin([-x.mean() for x in logs]))
    write_record(folder/'selection.json', {'selected': names[chosen], 'minimum_nll': names[minimum],
        'names': names, 'details': detail, 'files': files, 'truth_evaluated': False})
    print('FROZEN', folder.name, names[chosen], flush=True)


def driver(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    specs = [(f, d, n) for d in args.dimensions for f in args.families for n in ('free', 'mixture')]
    full_matrix = set(args.families) == {'gl', 'swirl'} and set(args.dimensions) == set(DIMENSIONS)
    scope = '_'.join(args.families)+'_'+'_'.join(map(str, args.dimensions))
    plan = {'cases': [case_name(*x) for x in specs], 'defaults': 'ftde.benchmark_defaults',
                'note': 'Truth is evaluated only after observation-only case selection is frozen.'}
    write_record(ROOT/('plan.json' if full_matrix else f'plan_{scope}.json'), plan)
    for family, dim, noise in specs:
        case = case_name(family, dim, noise)
        for name in recipes(family, dim, noise != 'free'):
            command = [sys.executable, '-u', __file__, '--action', 'fit', '--family', family,
                       '--dim', str(dim), '--noise', noise, '--name', name]
            print('RUN', case, name, flush=True)
            with (ROOT/'logs'/f'{case}_{name}.log').open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, '-u', __file__, '--action', 'select', '--family', family,
                        '--dim', str(dim), '--noise', noise], check=True)
        with (ROOT/'logs'/f'{case}_evaluation.log').open('a') as log:
            subprocess.run([sys.executable, '-u', 'scripts/plot_dimension_gallery.py', case],
                            stdout=log, stderr=subprocess.STDOUT, check=True)
    if full_matrix:
        subprocess.run([sys.executable, '-u', 'scripts/run_noisy_dimension_capacity.py'], check=True)
        subprocess.run([sys.executable, 'scripts/plot_dimension_gallery.py', '--publish'], check=True)
        write_record(ROOT/'complete.json', plan)
    else:
        write_record(ROOT/f'complete_{scope}.json', plan)
        print('Subset complete; figures staged. Full gallery publication requires all twelve cases.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['run', 'fit', 'select'], default='run')
    parser.add_argument('--families', nargs='+', choices=['gl', 'swirl'], default=['gl', 'swirl'])
    parser.add_argument('--dimensions', nargs='+', type=int, choices=DIMENSIONS, default=list(DIMENSIONS))
    parser.add_argument('--family', choices=['gl', 'swirl'])
    parser.add_argument('--dim', type=int, choices=DIMENSIONS)
    parser.add_argument('--noise', choices=['free', 'mixture'])
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.action == 'run':
        driver(args)
    elif args.action == 'fit':
        fit(args.family, args.dim, args.noise, args.name)
    else:
        select(args.family, args.dim, args.noise)

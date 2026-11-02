"""Tucker counterparts of the dimensional gallery; TT remains the default.

Run in the chebdGE environment on one GPU. Each fit/evaluation is a fresh
process. Selection uses observations only; results never replace TT files.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ftde.direct_training import load_tt, save_tt, write_record

ROOT = Path('artifacts/tucker_gallery')
CASES = [('gl', 4, 5), ('swirl', 4, 8), ('gl', 8, 3), ('swirl', 6, 4)]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_folder(case):
    """Resolve explicitly published follow-ups, retaining frozen baselines."""
    root = Path('artifacts/tucker_gallery')
    path = root/'current.json'
    return Path(read(path).get(case, root/case)) if path.exists() else root/case


def setup(family, dim, noise, rank, steps):
    from gl_common import data_domain, robust_domain
    from objective_study import noise_parameters

    case = f'{family}{dim}_{noise}'
    folder = ROOT/case
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        protocol = read(folder/'protocol.json')
        if (protocol['rank'], protocol['steps']) != (rank, steps) or digest(protocol['data']) != protocol['data_sha256']:
            raise ValueError('Existing protocol differs; use a new --root')
        return folder, protocol
    old = Path('artifacts/dimension_gallery_v9')/case/'protocol.json'
    if old.exists():
        source = read(old)
        domain, data_path, kernel = np.array(source['domain']), Path(source['data']), source['kernel']
        weights, cov = np.array(source['noise_weights']), np.array(source['noise_covariances'])
    else:
        weights, cov = noise_parameters(case, dim)
        if family == 'swirl':
            from swirl_fits import observations
        else:
            from ftde.ginzburg_landau import sample_ginzburg_landau
            def observations(d, noise, count, seed):
                x = np.asarray(sample_ginzburg_landau(seed, count, d))
                if noise != 'free':
                    rng = np.random.default_rng(10000+seed)
                    labels = rng.choice(len(weights), count, p=weights)
                    x = x+np.einsum('nij,nj->ni', np.linalg.cholesky(cov)[labels], rng.normal(size=x.shape))
                return x
        train, valid = (observations(dim, noise, n, seed) for n, seed in [(100000, 0), (4096, 2)])
        domain = np.asarray(data_domain(train) if noise == 'free' else robust_domain(train))
        data_path = folder/'data.npz'
        np.savez_compressed(data_path, train=train, valid=valid)
        kernel = None
        if noise != 'free':
            from ftde.noise_kernel import build_paired_gaussian_mixture_noise_kernel_tt
            kernel = str(folder/'kernel.npz')
            save_tt(kernel, build_paired_gaussian_mixture_noise_kernel_tt(cov, weights, domain))
    protocol = {'family': family, 'dim': dim, 'noise': noise, 'rank': rank, 'steps': steps,
                'decomposition': 'tucker', 'domain': domain.tolist(), 'data': str(data_path), 'data_sha256': digest(data_path),
                'kernel': kernel, 'noise_weights': weights.tolist(), 'noise_covariances': cov.tolist(),
                'modes': 25 if family == 'gl' else 49, 'seed': 0, 'perturbation': .01,
                'initialization': 'near-uniform native Tucker; no fitted pilot',
                'optimizer': 'Adam (TT canonical core sweeps are decomposition-specific)',
                'selection': 'minimum held-out NLL checkpoint and recipe', 'truth_used_for_selection': False,
                'recipes': ['strong'] if noise == 'free' else ['strong', 'curriculum']}
    write_record(folder/'protocol.json', protocol)
    return folder, protocol


def fit(args):
    import fcntl
    folder = ROOT/f'{args.family}{args.dim}_{args.noise}'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/f'.{args.recipe}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _fit(args)


def _fit(args):
    from ftde import TuckerConvolution, TuckerTrainingProblem, init_tucker_constant
    folder, p = setup(args.family, args.dim, args.noise, args.rank, args.steps)
    out = folder/args.recipe
    noisy = p['noise'] != 'free'
    curriculum = args.recipe == 'curriculum'
    strength = 1e-5 if curriculum else .001
    if not noisy and p['family'] == 'swirl':
        strength = .01 if p['dim'] == 4 else .003
    strength = strength if args.strength is None else args.strength
    if (out/'result.json').exists():
        previous = read(out/'result.json')
        if previous['strength'] != strength or previous['objective'] != args.objective:
            raise ValueError('Existing fit has a different objective/penalty; use a new --root')
        return
    if (folder/'selection.json').exists():
        raise ValueError('Case frozen; use a new --root')
    with np.load(p['data']) as data:
        train, valid = data['train'], data['valid']
    initial = init_tucker_constant(p['domain'], rank=p['rank'], modes=p['modes'])
    forward = None
    if noisy:
        obs = np.concatenate([train, valid])
        observation_domain = np.stack([obs.min(axis=0)-.02, obs.max(axis=0)+.02], axis=1)
        forward = TuckerConvolution.from_tucker(initial, load_tt(p['kernel'])).prepare(observation_domain)
    problem = TuckerTrainingProblem(initial, forward=forward, strength=strength, objective=args.objective)
    steps = p['steps']
    schedule = [(0, 9 if p['family'] == 'gl' else 13), (steps//4, 17 if p['family'] == 'gl' else 25), (steps//2, p['modes'])] if curriculum else None
    batch = 256 if noisy else 1024
    micro = args.microbatches if noisy else 1
    out.mkdir(exist_ok=True)
    write_record(out/'config.json', p | {'strength': strength, 'objective': args.objective, 'recipe': args.recipe, 'microbatches': micro,
                                       'convolution_contraction': 'exact coordinate-pair elimination' if noisy else None,
                                       'core_entries': int(initial.core.size), 'parameter_count': int(initial.core.size+sum(f.core.coeffs.size for f in initial.factors))})
    problem.fit_adam(train, valid, out, steps=steps, batch_size=batch, learning_rate=.001,
                     checkpoint_every=max(1, steps//20), validation_batch_size=16 if noisy else 256,
                     mode_schedule=schedule, selection_start=steps//2 if curriculum else 0, microbatches=micro)


def freeze(folder):
    import fcntl
    with (folder/'.selection.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _freeze(folder)


def _freeze(folder):
    if (folder/'selection.json').exists():
        return
    p = read(folder/'protocol.json')
    scores = {r: read(folder/r/'result.json')['best_nll'] for r in p['recipes']}
    selected = min(scores, key=scores.get)
    files = {str(folder/r/name): digest(folder/r/name) for r in p['recipes'] for name in ['fit.npz', 'result.json', 'config.json', 'validation.npy']}
    files[p['data']] = digest(p['data'])
    files[str(folder/'protocol.json')] = digest(folder/'protocol.json')
    if p['kernel']:
        files[p['kernel']] = digest(p['kernel'])
    write_record(folder/'selection.json', {'selected': selected, 'scores': scores, 'files': files, 'truth_evaluated': False})


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['run', 'fit', 'select'], default='run')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--family', choices=['gl', 'swirl'])
    parser.add_argument('--dim', type=int, default=4)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--noise', choices=['free', 'mixture'])
    parser.add_argument('--recipe', choices=['strong', 'curriculum'], default='strong')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--microbatches', type=int, default=1)
    parser.add_argument('--strength', type=float, help='Override the recipe Hessian penalty')
    parser.add_argument('--objective', choices=['nll', 'logpen'], default='nll')
    args = parser.parse_args()
    ROOT = args.root
    if args.action != 'run' and (args.family is None or args.noise is None):
        parser.error('--family and --noise are required for fit/select')
    if args.family and args.rank is None and args.action != 'select':
        defaults = {(f, d): r for f, d, r in CASES}
        args.rank = defaults.get((args.family, args.dim))
        if args.rank is None:
            parser.error('Specify --rank for dimensions outside the standard Tucker cases')
    if args.steps < 4 and args.noise != 'free':
        parser.error('The degree curriculum requires at least four steps')
    if args.action == 'fit':
        fit(args)
    elif args.action == 'select':
        freeze(ROOT/f'{args.family}{args.dim}_{args.noise}')
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT/'logs').mkdir(exist_ok=True)
        specs = CASES if args.family is None else [(args.family, args.dim, args.rank)]
        for family, dim, rank in specs:
            for noise in ([args.noise] if args.noise else ['free', 'mixture']):
                case = f'{family}{dim}_{noise}'
                for recipe in (['strong'] if noise == 'free' else ['strong', 'curriculum']):
                    command = [sys.executable, '-u', __file__, '--action', 'fit', '--root', str(ROOT), '--family', family,
                               '--dim', str(dim), '--rank', str(rank), '--noise', noise, '--recipe', recipe,
                               '--steps', str(args.steps), '--microbatches', str(args.microbatches)]
                    command += ['--objective', args.objective]
                    if args.strength is not None:
                        command += ['--strength', str(args.strength)]
                    print('RUN', case, recipe, flush=True)
                    with (ROOT/'logs'/f'{case}_{recipe}.log').open('a') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                freeze(ROOT/case)
                published = ROOT.resolve() == Path('artifacts/tucker_gallery').resolve()
                plot_folder = current_folder(case) if published else ROOT/case
                with (ROOT/'logs'/f'{case}_plots.log').open('a') as log:
                    subprocess.run([sys.executable, '-u', 'scripts/plot_tucker_gallery.py', str(plot_folder)], stdout=log, stderr=subprocess.STDOUT, check=True)
                if published and plot_folder != ROOT/case:
                    directory = 'swirl' if family == 'swirl' else 'ginzburg_landau'
                    prefix = f'tucker_{"noise_free" if noise == "free" else "noisy"}_dim{dim}'
                    for suffix in ['marg_01', 'marg_23', 'convergence']:
                        name = f'{prefix}_{suffix}.png'
                        shutil.copy2(plot_folder.parent/'figs'/directory/name, Path('figs')/directory/name)
        scope = 'all' if args.family is None and args.noise is None else '_'.join(str(x) for x in (args.family, args.dim, args.noise) if x is not None)
        write_record(ROOT/f'complete_{scope}.json', {'cases': specs, 'steps': args.steps, 'noise': args.noise or 'both'})
        if ROOT.resolve() == Path('artifacts/tucker_gallery').resolve():
            subprocess.run([sys.executable, 'scripts/summarize_tucker_gallery.py'], check=True)


if __name__ == '__main__':
    main()

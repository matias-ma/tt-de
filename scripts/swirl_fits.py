"""Fits for the continuous swirl-pretzel benchmark. Selection sees observations only.

The target (src/ftde/swirl.py) is a thin tube around one smooth path, the
continuous analogue of the cryo-EM conformational path, so any beads in a
recovered ridge are estimator artifacts. The recipe reuses the gallery code:

1. Base fit from a KDE-initialized TT (rank --init-rank, 25 modes, 64 starts):
   noise-free uses the GL noise-free recipe (unregularized NLL, lr 1e-3,
   batch 2048); noisy uses the GL warm-up (amplitude H^2 3e-5 through the
   prescribed TT convolution).
2. Capacity continuations, as in the current cryo-EM gallery: expand to each
   --ranks value with --modes Chebyshev modes (expand_tt), then continue.
   Noisy candidates use amplitude H^2 strengths --strengths, Sobolev
   preconditioning and regularized checkpoints (docs/cryoem_stability.md).
3. Selection: noise-free takes the minimum validation NLL (GL rule); noisy
   takes paired noisy one-SE, then minimum common amplitude H^2 (cryo-EM rule).

The clean density is never evaluated here; see swirl_figures.py.
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from gl_common import data_domain, robust_domain
from refine_cryoem_gallery import expand_tt
from refresh_density_fits import fit_candidate, read_model, save_model

from ftde.init import init_tt_from_kde
from ftde.kde import select_gaussian_kde_bandwidth
from ftde.noise_kernel import build_paired_gaussian_mixture_noise_kernel_tt
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.swirl import sample_swirl
from ftde.tt_density import log_density_tt

# Three-component noise mixture, "a sum of covariances": a narrow isotropic
# component, an anisotropic diagonal one, and a broad one with alternating
# +-0.6 correlations within adjacent coordinate pairs.
NOISE_WEIGHTS = np.array([.5, .3, .2])
NOISE_STD_ISOTROPIC = .22
NOISE_STDS_DIAGONAL = (.45, .25)
NOISE_STD_CORRELATED, NOISE_RHO = .5, .6


def noise_covariances(dim):
    covs = np.zeros((3, dim, dim))
    covs[0] = np.eye(dim)*NOISE_STD_ISOTROPIC**2
    covs[1] = np.diag(np.tile(np.square(NOISE_STDS_DIAGONAL), dim//2))
    for m in range(dim//2):
        rho = NOISE_RHO*(-1)**m
        covs[2][2*m:2*m+2, 2*m:2*m+2] = NOISE_STD_CORRELATED**2*np.array([[1, rho], [rho, 1]])
    return covs


def observations(dim, noise, count, seed):
    clean = sample_swirl(seed, count, dim)
    if noise == 'free':
        return clean
    rng = np.random.default_rng(10_000+seed)
    covs = noise_covariances(dim)
    comp = rng.choice(len(NOISE_WEIGHTS), count, p=NOISE_WEIGHTS)
    factors = np.linalg.cholesky(covs)
    return clean+np.einsum('nij,nj->ni', factors[comp], rng.normal(size=(count, dim)))


def microbatches(rank, dim):
    """Gradient-accumulation chunks for one 256-row batch (same gradient).

    Backward memory of the TT convolution grows like rank^2 * dim; rank 32 in
    10D fits in one chunk on an 80 GB GPU shared with other runs.
    """
    load = rank**2*dim
    return 1 if load <= 32**2*10 else 4 if load <= 64**2*10 else 8


def initialize(train, valid, domain, bandwidth, rank, out, seed=0):
    """KDE initialization; broaden only the initializer if its pivot search collapses.

    A thin tube in many dimensions can make the LSCV KDE so concentrated that
    the uniform pivot search finds a rank-one, essentially empty TT. Only the
    data are used: the attempt is accepted when all ranks exceed one and every
    training and validation observation has a finite initial log density
    (unconvolved fits must score the observations they train on).
    """
    attempts = []
    ladder = [1., 2., 4.]
    widths = [np.asarray(bandwidth)*f for f in ladder]
    widths.append(np.maximum(bandwidth, .25*np.ptp(domain, axis=1)))  # cryo-EM naive-baseline rule
    for label, width in zip([*map(str, ladder), 'quarter-box'], widths, strict=True):
        print('Initializing Chebyshev TT, bandwidth factor', label, flush=True)
        init = init_tt_from_kde(jnp.asarray(train), max_rank=rank, domain=jnp.asarray(domain),
            n_coeffs=(25,), bandwidth=jnp.asarray(width), n_starts=64, gd_steps=10, random_seed=seed)
        logs = np.concatenate([np.asarray(log_density_tt(init, jnp.asarray(x[s:s+4096])))
                               for x in (valid, train) for s in range(0, len(x), 4096)])
        ok = min(init.ranks) > 1 and bool(np.all(np.isfinite(logs) & (logs > -700)))
        attempts.append({'bandwidth_factor': label, 'ranks': list(init.ranks), 'accepted': ok,
                         'minimum_log_density': float(np.min(logs))})
        if ok:
            break
    else:
        raise FloatingPointError(f'KDE initialization failed: {attempts}')
    (out/'initialization.json').write_text(json.dumps({'attempts': attempts,
        'bandwidth': np.asarray(width).tolist()}, indent=2)+'\n')
    save_model(out/'init.npz', init)
    return init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--noise', choices=['free', 'mixture'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-train', type=int, default=100_000)
    parser.add_argument('--n-valid', type=int, default=4096)
    parser.add_argument('--init-rank', type=int, default=16)
    parser.add_argument('--ranks', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--modes', type=int, default=49)
    parser.add_argument('--strengths', type=float, nargs='+', default=[1e-4, 3e-4])
    parser.add_argument('--base-steps', type=int, default=8000)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--train-seed', type=int, default=0, help='validation seed is 2')
    parser.add_argument('--init-seed', type=int, default=0, help='seed of the KDE subsample and pivot search')
    args = parser.parse_args()
    out, d = args.output, args.dim
    if d % 2:
        raise ValueError('the paired noise mixture needs an even dimension')
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings['family'] = 'swirl'
    for key in ('train_seed', 'init_seed'):
        if settings[key] == 0:
            settings.pop(key)  # keep earlier saved runs valid
    if args.train_seed == 2:
        raise ValueError('train seed 2 is the validation seed')
    if (out/'settings.json').exists() and json.loads((out/'settings.json').read_text()) != settings:
        raise ValueError('Settings differ from saved run; choose a new output directory')
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    if (out/'selection.json').exists():
        print('Completed case already saved:', out, flush=True)
        return
    noisy = args.noise != 'free'
    train = observations(d, args.noise, args.n_train, args.train_seed)
    valid = observations(d, args.noise, args.n_valid, 2)
    np.savez(out/'data.npz', train=train, valid=valid, shift=np.zeros(d), scale=np.ones(d))
    domain = np.asarray(robust_domain(jnp.asarray(train)) if noisy else data_domain(jnp.asarray(train)))
    if not noisy and not np.all((valid >= domain[:, 0]) & (valid <= domain[:, 1])):
        raise ValueError('Noise-free validation lies outside training-derived support')
    if (out/'bandwidth.npy').exists():
        bandwidth = np.load(out/'bandwidth.npy')
    else:
        print('Selecting KDE bandwidth', flush=True)
        bandwidth = np.asarray(select_gaussian_kde_bandwidth(jnp.asarray(train)))
        np.save(out/'bandwidth.npy', bandwidth)
    if (out/'init.npz').exists():
        init = read_model(out/'init.npz')
    else:
        init = initialize(train, valid, domain, bandwidth, args.init_rank, out, seed=args.init_seed)
    print('Initialized ranks', init.ranks, flush=True)
    kernel, scales = None, None
    if noisy:
        if (out/'kernel.npz').exists():
            kernel = read_model(out/'kernel.npz')
        else:
            kernel = build_paired_gaussian_mixture_noise_kernel_tt(noise_covariances(d), NOISE_WEIGHTS, domain)
            save_model(out/'kernel.npz', kernel)
        (out/'noise_model.json').write_text(json.dumps({'weights': NOISE_WEIGHTS.tolist(),
            'isotropic_std': NOISE_STD_ISOTROPIC, 'diagonal_stds': NOISE_STDS_DIAGONAL,
            'correlated_std': NOISE_STD_CORRELATED, 'correlation': NOISE_RHO,
            'kernel_ranks': list(kernel.ranks)}, indent=2)+'\n')
        scales = np.array([m.moments()[1] for m in noise_marginals_from_tt(kernel)])
        base = fit_candidate(init, train, valid, out/'base', kernel=kernel,
            penalty=AmplitudeSobolevPenalty.from_tt(init, order=2, length_scales=scales),
            strength=3e-5, steps=args.base_steps)
    else:
        base = fit_candidate(init, train, valid, out/'base', steps=args.base_steps, lr=.001, batch_size=2048)
    names, fits = ['base'], [base]
    for rank in args.ranks:
        start = expand_tt(base, args.modes, rank)
        for strength in (args.strengths if noisy else [0.]):
            name = f'rank{rank}' + (f'_h2_{strength:g}' if noisy else '')
            if noisy:
                penalty = AmplitudeSobolevPenalty.from_tt(start, order=2, length_scales=scales)
                fit = fit_candidate(start, train, valid, out/name, kernel=kernel, penalty=penalty,
                    strength=strength, steps=args.steps, lr=2e-4, batch_size=256,
                    checkpoint_rule='regularized', preconditioning='sobolev',
                    microbatches=microbatches(rank, d))
            else:
                fit = fit_candidate(start, train, valid, out/name, steps=args.steps, lr=2e-4, batch_size=1024)
            names.append(name)
            fits.append(fit)
    logs = np.stack([np.load(out/n/'validation.npy') for n in names])
    if noisy:
        roughness = [float(AmplitudeSobolevPenalty.from_tt(f, order=2, length_scales=scales)(f)) for f in fits]
        chosen, detail = select_by_paired_noisy_nll(logs, roughness)
        rule = 'paired noisy one-SE, then minimum common amplitude H2'
    else:
        roughness, detail = None, None
        chosen = int(np.argmax(logs.mean(1)))
        rule = 'minimum validation NLL'
    save_model(out/'selected.npz', fits[chosen])
    (out/'selection.json').write_text(json.dumps({'selected': names[chosen], 'names': names,
        'validation_nll': (-logs.mean(1)).tolist(), 'roughness': roughness, 'selection': detail,
        'rule': rule}, indent=2)+'\n')
    print('SELECTED', names[chosen], flush=True)


if __name__ == '__main__':
    main()

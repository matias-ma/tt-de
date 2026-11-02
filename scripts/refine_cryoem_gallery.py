"""Data-only cryo-EM resolution and prescribed-noise sensitivity experiments."""

import argparse
import json
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from refresh_density_fits import fit_candidate, read_model, save_model
from scipy.cluster.vq import kmeans2
from scipy.fft import dct

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.noise_kernel import gaussian_noise_domain
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.sobolev import AmplitudeSobolevPenalty


def expand_tt(model, modes, rank, *, jitter=1e-3, seed=0):
    """Increase capacity while preserving the density before small rank perturbations."""
    rng = np.random.default_rng(seed)
    arrays, inverse = [], []
    for core in model.cores:
        if modes < core.coeffs.shape[-1]:
            raise ValueError('Expansion cannot remove existing modes')
        coeff = np.pad(np.asarray(core.coeffs), ((0, 0), (0, 0), (0, modes-core.coeffs.shape[-1])))
        chol = np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(modes, core.domain)))
        arrays.append(coeff@chol)
        inverse.append(np.linalg.inv(chol))
    # Fix the TT gauge in physical L2 coordinates before adding new directions.
    for k in range(len(arrays)-1):
        left, right, n = arrays[k].shape
        q, r = np.linalg.qr(arrays[k].transpose(0, 2, 1).reshape(left*n, right))
        arrays[k] = q.reshape(left, n, q.shape[1]).transpose(0, 2, 1)
        arrays[k+1] = np.einsum('ab,bcn->acn', r, arrays[k+1])
    arrays[-1] /= np.linalg.norm(arrays[-1])
    expanded = []
    d = len(arrays)
    for k, a in enumerate(arrays):
        shape = (1 if k == 0 else rank, 1 if k == d-1 else rank, modes)
        if shape[0] < a.shape[0] or shape[1] < a.shape[1]:
            raise ValueError('Expansion cannot remove existing ranks')
        b = rng.normal(size=shape)*jitter*np.linalg.norm(a)/np.sqrt(np.prod(shape))
        b[:a.shape[0], :a.shape[1], :] = a
        expanded.append(ChebyshevCoeffCore(jnp.asarray(b@inverse[k]), model.cores[k].domain))
    return ChebTT(cores=tuple(expanded), domain=model.domain, ranks=(rank,)*(d-1))


def diagonal_mixture_kernel(variances, weights, signal_domain, modes=513):
    """Represent every Gaussian component on the SAME Chebyshev domain."""
    variances, weights = np.asarray(variances), np.asarray(weights)
    if variances.ndim != 2 or not np.all(np.isfinite(variances) & (variances > 0)):
        raise ValueError('Mixture variances must be a positive finite component-by-axis matrix')
    if weights.shape != (len(variances),) or not np.all(np.isfinite(weights) & (weights >= 0)) or weights.sum() <= 0:
        raise ValueError('Mixture weights must be finite, nonnegative and have positive sum')
    weights = weights/weights.sum()
    components, d = variances.shape
    domain = np.asarray(gaussian_noise_domain(jnp.diag(jnp.asarray(variances.max(0))), signal_domain))
    cores = []
    for k, (a, b) in enumerate(domain):
        t = np.cos(np.pi*(np.arange(modes)+.5)/modes)
        x = a+(t+1)*(b-a)/2
        values = np.exp(-x[:, None]**2/(2*variances[:, k]))/np.sqrt(2*np.pi*variances[:, k])
        coefficients = dct(values, type=2, axis=0)/modes
        coefficients[0] *= .5
        probe = np.linspace(a, b, 4097)
        analytic = np.exp(-probe[:, None]**2/(2*variances[:, k]))/np.sqrt(2*np.pi*variances[:, k])
        represented = np.polynomial.chebyshev.chebval((2*probe-a-b)/(b-a), coefficients).T
        if np.max(np.abs(represented-analytic)) > 1e-11*np.max(analytic):
            raise FloatingPointError('Mixture kernel needs more Chebyshev modes')
        if d == 1:
            block = (coefficients@weights)[None, None]
        else:
            block = np.zeros((1 if k == 0 else components, 1 if k == d-1 else components, modes))
            for c in range(components):
                block[0 if k == 0 else c, 0 if k == d-1 else c] = coefficients[:, c]*(weights[c] if k == 0 else 1.)
        cores.append(ChebyshevCoeffCore(jnp.asarray(block), tuple(domain[k])))
    return ChebTT(cores=tuple(cores), domain=jnp.asarray(domain), ranks=(components,)*(d-1))


def mixture_noise(base, metadata, output, components=4):
    """Cluster supplied noise variances, without using positions or labels."""
    data = np.load(base/'data.npz')
    split = np.load(base/'split.npz')
    supplied = np.load(metadata)
    for part in ('train', 'valid'):
        standardized = (supplied['points'][split[f'{part}_idx']]-data['shift'])/data['scale']
        np.testing.assert_allclose(standardized, data[part], atol=1e-12, rtol=1e-12,
                                   err_msg='Noise metadata and fitted observations are misaligned')
    raw = supplied['variances'][split['train_idx']]/data['scale']**2
    features = np.log(raw)
    center, spread = features.mean(0), np.maximum(features.std(0), 1e-12)
    centers, labels = kmeans2((features-center)/spread, components, minit='++', seed=0, iter=60)
    weights = np.bincount(labels, minlength=components)/len(labels)
    if np.any(weights == 0):
        raise ValueError('Noise clustering produced an empty component; reduce the component count')
    variances = np.stack([raw[labels == k].mean(0) for k in range(components)])
    template = read_model(base/'selected.npz')
    kernel = diagonal_mixture_kernel(variances, weights, template.domain)
    save_model(output/'mixture_kernel.npz', kernel)
    (output/'mixture_noise.json').write_text(json.dumps({'weights': weights.tolist(),
        'variances_standardized': variances.tolist(), 'clusters': components,
        'source': str(metadata), 'uses_observation_positions': False,
        'feature_center': center.tolist(), 'feature_scale': spread.tolist(),
        'cluster_centers': centers.tolist()}, indent=2)+'\n')
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--noise-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--modes', type=int, default=49)
    parser.add_argument('--rank', type=int, default=12)
    parser.add_argument('--strengths', type=float, nargs='+', default=[3e-5])
    parser.add_argument('--correct-projection', action='store_true',
                        help='Replace historical path precision-submatrix noise by marginal noise')
    args = parser.parse_args()
    out, base = args.output, args.base
    out.mkdir(parents=True, exist_ok=True)
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if (out/'experiment.json').exists():
        previous = json.loads((out/'experiment.json').read_text())
        previous.setdefault('correct_projection', False)
        if previous != configuration:
            raise ValueError('Experiment configuration differs; use a new output directory')
    (out/'experiment.json').write_text(json.dumps(configuration, indent=2)+'\n')
    data = np.load(base/'data.npz')
    train, valid = data['train'], data['valid']
    original, kernel = read_model(base/'selected.npz'), read_model(base/'kernel.npz')
    if args.correct_projection:
        split = np.load(base/'split.npz')
        variances = np.load(args.noise_metadata)['variances'][split['train_idx']].mean(0)/data['scale']**2
        kernel = diagonal_mixture_kernel(variances[None], np.ones(1), original.domain)
        save_model(out/'corrected_kernel.npz', kernel)
    mixed = mixture_noise(base, args.noise_metadata, out)
    scales = np.array([m.moments()[1] for m in noise_marginals_from_tt(kernel)])
    prior_selection = json.loads((base/'selection.json').read_text())['selected']
    if args.correct_projection:
        fit_candidate(original, train, valid, out/'baseline_rechecked', kernel=kernel, steps=0)
        shutil.copy2(out/'baseline_rechecked/validation.npy', out/'baseline_validation.npy')
    else:
        shutil.copy2(base/prior_selection/'validation.npy', out/'baseline_validation.npy')
    fits, scores = [original], [np.load(out/'baseline_validation.npy')]
    penalties = [float(AmplitudeSobolevPenalty.from_tt(original, order=2, length_scales=scales)(original))]
    names = ['baseline']
    candidates = [(f'resolution{args.modes}_rank{max(original.ranks)}', max(original.ranks), kernel, 3e-5)]
    for strength in args.strengths:
        for label, known_kernel in [('resolution', kernel), ('mixture', mixed)]:
            candidates.append((f'{label}{args.modes}_rank{args.rank}_h2_{strength:g}',
                               args.rank, known_kernel, strength))
    for name, rank, known_kernel, strength in candidates:
        init = expand_tt(original, args.modes, rank)
        save_model(out/f'{name}_init.npz', init)
        penalty = AmplitudeSobolevPenalty.from_tt(init, order=2, length_scales=scales)
        fitted = fit_candidate(init, train, valid, out/name, kernel=known_kernel,
            penalty=penalty, strength=strength, steps=args.steps, lr=.0002, batch_size=256,
            checkpoint_rule='regularized')
        fits.append(fitted)
        scores.append(np.load(out/name/'validation.npy'))
        penalties.append(float(penalty(fitted)))
        names.append(name)
        print('FINISHED', name, 'curvature', penalties[-1], flush=True)
    selected, detail = select_by_paired_noisy_nll(np.stack(scores), penalties)
    save_model(out/'selected.npz', fits[selected])
    (out/'selection.json').write_text(json.dumps({'selected': names[selected], 'names': names,
        'curvature': penalties, 'selection': detail,
        'rule': 'paired noisy one-SE, then minimum physical amplitude curvature'}, indent=2)+'\n')
    print('SELECTED', names[selected], flush=True)


if __name__ == '__main__':
    main()

"""Label-informed clean-density oracle for the cryo-EM gallery (validation only).

The simulation assigns every image to one of a finite set of conformational
states. For state k with noisy latent points z (supplied per-image noise
covariances N_i), the clean latent distribution is estimated by moments:

    mean_k  = mean(z)
    clean_k = PSD part of [cov(z) - mean(N_i)]

and the oracle clean density is the mixture sum_k pi_k N(mean_k, clean_k).
A continuous counterfactual joins consecutive state means by straight
segments, with the same interpolated transverse covariance and state weights.
It is what a smooth "pretzel" with this path and width would look like.

Labels enter only after all fitting and selection. This script never alters
fits; it compares frozen gallery densities against both reference densities.

Stage 1 (needs ~10 GB RAM; run inside the allocation):
    python scripts/audit_cryoem_oracle.py extract
Stage 2 (figures and numbers):
    python scripts/audit_cryoem_oracle.py plot
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import numpy as np
from refresh_density_figures import draw_panels, histogram
from refresh_density_fits import read_model

from ftde.marginals import squared_pair_grid

ROOT = Path('artifacts/cryoem_oracle')
CASES = {4: Path('artifacts/cryoem_continuity/gallery_dim4'),
         10: Path('artifacts/cryoem_stability/gallery_dim10'),
         20: Path('artifacts/cryoem_stability/gallery_dim20')}
PAIRS = {4: [(0, 1), (2, 3)], 10: [(0, 1), (2, 3), (8, 9)], 20: [(0, 1), (2, 3), (8, 9)]}


def extract():
    import joblib
    from cryoem_cache import marginal_noise_covariances
    ROOT.mkdir(parents=True, exist_ok=True)
    embedding = joblib.load('cryoem_data/cryoem_test.joblib')
    assignment = np.asarray(joblib.load('cryoem_data/noise1/simulation_info.joblib')['image_assignment'])
    for dim in CASES:
        precision = np.asarray(embedding['cov_zs'][f'{dim}_noreg'])
        covariance = marginal_noise_covariances(precision, dim)
        source = np.asarray(embedding['zs'][f'{dim}_noreg'])[:, :dim]
        cached = np.load(f'cryoem_data/cache/dim{dim}.npz')['points']
        # Same two candidate masks as audit_cryoem_noise.py.
        for mask_covariance in (np.linalg.inv(precision[:, :dim, :dim]), covariance):
            trace = np.trace(mask_covariance, axis1=1, axis2=2)
            indices = np.flatnonzero(trace < np.percentile(trace, 99))
            if np.array_equal(cached, source[indices]):
                break
        else:
            raise ValueError(f'dim{dim}: cached observations do not match a known mask')
        points, labels, noise = cached, assignment[indices], covariance[indices]
        states = int(labels.max())+1
        counts = np.bincount(labels, minlength=states)
        means = np.stack([points[labels == k].mean(0) for k in range(states)])
        within = np.stack([np.cov(points[labels == k].T) for k in range(states)])
        mean_noise = np.stack([noise[labels == k].mean(0) for k in range(states)])
        clean = []
        for s, n in zip(within, mean_noise, strict=True):
            w, v = np.linalg.eigh(s-n)
            clean.append((v*np.maximum(w, 0.))@v.T)
        np.savez(ROOT/f'oracle_dim{dim}.npz', counts=counts, means=means, within=within,
                 mean_noise=mean_noise, clean=np.stack(clean), labels=labels, indices=indices)
        ratio = np.sqrt(np.diagonal(within, axis1=1, axis2=2)/np.diagonal(mean_noise, axis1=1, axis2=2))
        print(f'dim{dim}: {states} states, counts {counts.min()}-{counts.max()}, '
              f'median within/noise std ratio {np.median(ratio):.3f}', flush=True)


def mixture_pair_grid(weights, means, covs, i, j, xs, ys):
    """Rows y, columns x; exact pair marginal of a Gaussian mixture."""
    xx, yy = np.meshgrid(xs, ys)
    out = np.zeros_like(xx)
    for w, m, c in zip(weights, means, covs, strict=True):
        c2 = c[np.ix_([i, j], [i, j])]+1e-9*np.eye(2)*np.trace(c)
        inv, det = np.linalg.inv(c2), np.linalg.det(c2)
        dx, dy = xx-m[i], yy-m[j]
        q = inv[0, 0]*dx**2+2*inv[0, 1]*dx*dy+inv[1, 1]*dy**2
        out += w*np.exp(-.5*q)/(2*np.pi*np.sqrt(det))
    return out


def continuous_counterfactual(counts, means, clean, per_segment=40):
    """Densely interpolate the ordered state path; conserve each state's weight."""
    weights, centers, covs = [], [], []
    p = counts/counts.sum()
    for k in range(len(means)-1):
        for t in (np.arange(per_segment)+.5)/per_segment:
            weights.append(((1-t)*p[k]+t*p[k+1])/per_segment)
            centers.append((1-t)*means[k]+t*means[k+1])
            covs.append((1-t)*clean[k]+t*clean[k+1])
    weights = np.asarray(weights)
    return weights/weights.sum(), np.asarray(centers), np.asarray(covs)


def relative_l2(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def plot(figures):
    summary = {}
    for dim, case in CASES.items():
        oracle = np.load(ROOT/f'oracle_dim{dim}.npz')
        data = np.load(case/'data.npz')
        shift, scale, train = data['shift'], data['scale'], data['train']
        fit, naive = read_model(case/'selected.npz'), read_model(case/'naive/fit.npz')
        counts, means, clean, within = (oracle[k] for k in ('counts', 'means', 'clean', 'within'))
        weights = counts/counts.sum()
        cw, cm, cc = continuous_counterfactual(counts, means, clean)
        # Consecutive state separation, in units of supplied noise std.
        noise_std = np.sqrt(np.diagonal(oracle['mean_noise'], axis1=1, axis2=2).mean(0))
        spacing = np.linalg.norm(np.diff(means, axis=0)/noise_std, axis=1)
        clean_std = np.sqrt(np.diagonal(clean, axis1=1, axis2=2))/noise_std
        record = {'states': len(counts), 'median_step_over_noise_std': float(np.median(spacing)),
                  'median_clean_std_over_noise_std': float(np.median(clean_std)),
                  'median_within_std_over_noise_std': float(np.median(
                      np.sqrt(np.diagonal(within, axis1=1, axis2=2))/noise_std)), 'pairs': {}}
        for i, j in PAIRS[dim]:
            box = np.asarray(fit.domain)[[i, j]]
            zx, zy = np.linspace(*box[0], 240), np.linspace(*box[1], 240)
            xs, ys = zx*scale[i]+shift[i], zy*scale[j]+shift[j]
            jac = scale[i]*scale[j]
            deconv = squared_pair_grid(fit, i, j, zx, zy)/jac
            observed = squared_pair_grid(naive, i, j, zx, zy)/jac
            discrete = mixture_pair_grid(weights, means, clean, i, j, xs, ys)
            smooth = mixture_pair_grid(cw, cm, cc, i, j, xs, ys)
            noisy_oracle = mixture_pair_grid(weights, means, within, i, j, xs, ys)
            extent = (xs[0], xs[-1], ys[0], ys[-1])
            hist = histogram(np.column_stack([train[:, i]*scale[i]+shift[i], train[:, j]*scale[j]+shift[j]]),
                             0, 1, extent)
            errors = {'deconvolved_vs_discrete_oracle': relative_l2(deconv, discrete),
                      'deconvolved_vs_continuous_path': relative_l2(deconv, smooth),
                      'continuous_path_vs_discrete_oracle': relative_l2(smooth, discrete),
                      'observed_fit_vs_oracle_convolved': relative_l2(observed, noisy_oracle)}
            record['pairs'][f'{i}{j}'] = errors
            footer = (f'Oracle: {len(counts)} simulated states · state mean and PSD part of (within-state covariance − '
                      'mean supplied noise covariance) · labels used only here, after fitting\n'
                      f'Median consecutive-state spacing {np.median(spacing):.2f} noise σ · median clean state σ '
                      f'{np.median(clean_std):.2f} noise σ · continuous path: linear interpolation of means, weights, covariances\n'
                      f'Relative L2 of deconvolved ft-de: vs discrete oracle {errors["deconvolved_vs_discrete_oracle"]:.3f}, '
                      f'vs continuous path {errors["deconvolved_vs_continuous_path"]:.3f} · observed ft-de vs oracle⊛noise '
                      f'{errors["observed_fit_vs_oracle_convolved"]:.3f} · gallery fit {case}')
            draw_panels([hist, noisy_oracle, discrete, smooth, deconv],
                        ['Histogram of data', 'Oracle ⊛ noise', 'Discrete-state oracle',
                         'Continuous-path counterfactual', 'Deconvolved ft-de'],
                        extent, i, j, f'Cryo-EM · {dim} dimensions · label oracle check',
                        footer, figures/f'cryoem_oracle_dim{dim}_marg_{i}{j}.png', [0, 0, 1, 1, 1])
        summary[f'dim{dim}'] = record
        print(dim, json.dumps(record, indent=1), flush=True)
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract', 'plot'])
    parser.add_argument('--figures', type=Path, default=ROOT/'figures')
    args = parser.parse_args()
    extract() if args.stage == 'extract' else plot(args.figures)


if __name__ == '__main__':
    main()

"""Evaluate and plot the sample-size sweep (post-selection diagnostics only).

``evaluate CASE``: exact full-dimensional relative L2 of the selected ft-de
density (GL: transfer quadrature, evaluate_gallery_fits.py; swirl: path
quadrature, swirl_figures.py) and of a Gaussian KDE baseline. The KDE is
fitted to the same training observations (noisy ones in noisy cases, so it
estimates the observation density, not the latent one). Its bandwidth is
the gallery's LSCV bandwidth, which is selected on a 1,200-point subsample
and is therefore rescaled by Scott's exponent (N / 1200)^(-1/(d+4)).
For the KDE, int kde^2 is an exact pairwise Gaussian sum. int p kde is
exact path quadrature for the swirl and Monte Carlo over 200,000 fresh
exact GL samples (standard error recorded) for GL.

``plot``: figs/convergence/convergence.png and artifacts/convergence/summary.json.
"""

import argparse
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path('artifacts/convergence')


def _gaussian_sum(a, b, width, block=(2048, 32768)):
    """sum_ij prod_k N(a_ik - b_jk; 0, width_k^2)."""
    ua, ub = jnp.asarray(a/width), jnp.asarray(b/width)
    na, nb = jnp.sum(ua**2, 1), jnp.sum(ub**2, 1)

    @jax.jit
    def tile(xa, xb, qa, qb):
        return jnp.sum(jnp.exp(-.5*jnp.maximum(qa[:, None]+qb[None, :]-2*xa@xb.T, 0.)))

    total = 0.
    for i in range(0, len(a), block[0]):
        for j in range(0, len(b), block[1]):
            total += float(tile(ua[i:i+block[0]], ub[j:j+block[1]], na[i:i+block[0]], nb[j:j+block[1]]))
    return total/np.prod(np.sqrt(2*np.pi)*width)


def kde_l2(case, settings):
    family = settings.get('family')
    dim = settings['dim']
    train = np.load(case/'data.npz')['train']
    n = len(train)
    bandwidth = np.load(case/'bandwidth.npy')*min(1., (n/1200)**(-1/(dim+4)))
    k2 = _gaussian_sum(train, train, np.sqrt(2)*bandwidth)/n**2
    if family == 'swirl':
        from ftde.swirl import TUBE_STD, curve, path_quadrature, swirl_squared_norm
        p2 = swirl_squared_norm(dim)
        t, w = path_quadrature(4000)
        g = jnp.asarray(curve(t, dim))
        var = jnp.asarray(TUBE_STD**2+bandwidth**2)

        @jax.jit
        def chunk(x):
            d2 = jnp.sum((x[:, None, :]-g[None])**2/var, -1)
            return jnp.sum(jnp.exp(-.5*d2)@jnp.asarray(w))
        pk = sum(float(chunk(jnp.asarray(train[s:s+512]))) for s in range(0, n, 512))
        pk /= n*np.prod(np.sqrt(2*np.pi*np.asarray(var)))
        pk_se = 0.
    else:
        from ftde.ginzburg_landau import (
            GL_BETA,
            ginzburg_landau_log_normalizer,
            sample_ginzburg_landau,
        )
        log_z = ginzburg_landau_log_normalizer(dim)
        p2 = float(np.exp(ginzburg_landau_log_normalizer(dim, beta=2*GL_BETA)-2*log_z))
        fresh = np.asarray(sample_ginzburg_landau(7, 200_000, dim))
        values = np.array([_gaussian_sum(fresh[s:s+2048], train, bandwidth)
                           for s in range(0, len(fresh), 2048)])
        # Chunk sums -> chunk means; standard error from per-chunk variation.
        sizes = np.array([len(fresh[s:s+2048]) for s in range(0, len(fresh), 2048)])
        means = values/(sizes*n)
        pk = float(np.sum(values)/(len(fresh)*n))
        pk_se = float(np.std(means, ddof=1)/np.sqrt(len(means)))
    squared = (k2-2*pk+p2)/p2
    return {'relative_l2': float(np.sqrt(max(squared, 0.))), 'relative_squared_l2': float(squared),
            'integral_kde_squared': float(k2), 'integral_p_kde': float(pk), 'integral_p_kde_standard_error': pk_se,
            'integral_p_squared': p2, 'relative_l2_standard_error': float(pk_se/p2/max(np.sqrt(max(squared, 1e-12)), 1e-6)),
            'bandwidth': bandwidth.tolist(), 'n_train': n}


def evaluate(case):
    settings = json.loads((case/'settings.json').read_text())
    if settings.get('family') == 'swirl':
        from refresh_density_fits import read_model
        from swirl_figures import cached_l2
        selection = json.loads((case/'selection.json').read_text())
        record = cached_l2(case, selection['selected'], read_model(case/'selected.npz'), settings['dim'])
        tt = record['relative_l2']
    else:
        from evaluate_gallery_fits import evaluate_case
        evaluate_case(case)
        tt = json.loads((case/'integrated_l2.json').read_text())['relative_l2']
    kde = kde_l2(case, settings)
    (case/'kde_l2.json').write_text(json.dumps(kde, indent=2)+'\n')
    print(case.name, 'ft-de', tt, 'KDE', kde['relative_l2'], flush=True)


def collect():
    rows = []
    for case in sorted(ROOT.glob('*_n*_s*')):
        m = re.match(r'(gl|swirl)_(free|mixture)_dim(\d+)_n(\d+)_s(\d+)$', case.name)
        if not m or not (case/'kde_l2.json').exists():
            continue
        family, noise, dim, n, seed = m.groups()
        if family == 'gl':
            tt = json.loads((case/'integrated_l2.json').read_text())['relative_l2']
        else:
            selection = json.loads((case/'selection.json').read_text())
            tt = json.loads((case/'integrated_l2.json').read_text())[selection['selected']]['relative_l2']
        kde = json.loads((case/'kde_l2.json').read_text())
        rows.append({'family': family, 'noise': noise, 'dim': int(dim), 'n_train': int(n), 'seed': int(seed),
                     'ftde_relative_l2': tt, 'kde_relative_l2': kde['relative_l2'],
                     'kde_relative_l2_standard_error': kde['relative_l2_standard_error']})
    return rows


def plot(figures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = collect()
    (ROOT/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    styles = {('free', 'ftde'): ('#1f5fbf', 'o', '-', 'ft-de, noise-free data'),
              ('mixture', 'ftde'): ('#dd5b00', 's', '-', 'deconvolved ft-de, noisy data'),
              ('free', 'kde'): ('#1f5fbf', 'o', ':', 'KDE, noise-free data'),
              ('mixture', 'kde'): ('#dd5b00', 's', ':', 'KDE of noisy data (not deconvolved)')}
    panels = [('gl', 10, 'Ginzburg–Landau · 10 dimensions'), ('swirl', 4, 'Swirl-pretzel · 4 dimensions')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained')
    for ax, (family, dim, title) in zip(axes, panels, strict=True):
        for (noise, method), (color, marker, line, label) in styles.items():
            sel = [r for r in rows if r['family'] == family and r['dim'] == dim and r['noise'] == noise]
            if not sel:
                continue
            key = 'ftde_relative_l2' if method == 'ftde' else 'kde_relative_l2'
            sizes = sorted({r['n_train'] for r in sel})
            mean = [np.mean([r[key] for r in sel if r['n_train'] == n]) for n in sizes]
            ax.plot(sizes, mean, line, color=color, marker=marker, lw=2, ms=6, label=label,
                    mfc='white' if method == 'kde' else color)
            for r in sel:
                ax.scatter(r['n_train'], r[key], color=color, s=12, alpha=.45, marker=marker)
        guide = np.array([1e3, 3e5])
        anchor = [r['ftde_relative_l2'] for r in rows if r['family'] == family and r['noise'] == 'free'
                  and r['n_train'] == 100000]
        if anchor:
            ax.plot(guide, np.mean(anchor)*(guide/1e5)**-.5, '--', color='#a1a1aa', lw=1, label=r'$\propto N^{-1/2}$')
        ax.set(xscale='log', yscale='log', xlabel='training observations N',
               ylabel='relative L2 error of the latent density', title=title)
        ax.grid(alpha=.2, which='both')
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Full-dimensional relative L2 versus sample size', fontsize=14, fontweight='bold')
    fig.supxlabel('Lines: mean over training seeds (GL: 2 seeds; swirl: 1); points: individual fits. '
                  'Noisy cases score against the clean latent density. Validation sets fixed (seed 2).',
                  fontsize=8, color='#52525b')
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures/'convergence.png', dpi=180)
    print('Saved', figures/'convergence.png', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['evaluate', 'plot'])
    parser.add_argument('cases', nargs='*', type=Path)
    parser.add_argument('--figures', type=Path, default=Path('figs/convergence'))
    args = parser.parse_args()
    if args.stage == 'evaluate':
        for case in args.cases:
            evaluate(case)
    else:
        plot(args.figures)


if __name__ == '__main__':
    main()

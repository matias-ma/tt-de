"""Analysis figures for the LaTeX note (post-selection diagnostics only).

1. fisher_spectrum.png: marginal Fisher-visibility eigenvalues and filter
   weights of the 10D noisy GL gallery fit.
2. nll_gap_vs_l2.png: for every searched swirl candidate, the paired
   held-out NLL gap to the true (observation) density against the latent
   relative L2 error.
3. h2_tradeoff.png: 10D noisy swirl, rank-32 continuations at increasing
   amplitude-H^2 strength: NLL gap, relative L2 and along-ridge ripple.
Writes note/figures/*.png and note/figures/numbers.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from swirl_fits import NOISE_WEIGHTS, noise_covariances

from ftde.swirl import swirl_convolved_log_density, swirl_log_density

OUT = Path('note/figures')
SWIRL = Path('artifacts/swirl')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'figure.facecolor': 'white'})


def oracle_logs(case):
    settings = json.loads((case/'settings.json').read_text())
    valid = np.load(case/'data.npz')['valid']
    if settings['noise'] == 'free':
        return swirl_log_density(valid)
    return swirl_convolved_log_density(valid, noise_covariances(settings['dim']), NOISE_WEIGHTS)


def candidate_table():
    rows, seen = [], set()
    for case in sorted(SWIRL.glob('swirl_*_dim*')):
        if not (case/'selection.json').exists() or not (case/'integrated_l2.json').exists():
            continue
        settings = json.loads((case/'settings.json').read_text())
        selection = json.loads((case/'selection.json').read_text())
        l2 = json.loads((case/'integrated_l2.json').read_text())
        ripple = json.loads((case/'ripple.json').read_text())['candidates'] if (case/'ripple.json').exists() else {}
        oracle = oracle_logs(case)
        for name in selection['names']:
            if name not in l2 or (case/name).resolve() in seen:  # shared with another case
                continue
            seen.add((case/name).resolve())
            gap = oracle-np.load(case/name/'validation.npy')
            rows.append({'case': case.name, 'dim': settings['dim'], 'noise': settings['noise'], 'candidate': name,
                         'selected': name == selection['selected'], 'nll_gap': float(gap.mean()),
                         'nll_gap_se': float(gap.std()/np.sqrt(len(gap))), 'relative_l2': l2[name]['relative_l2'],
                         'ripple_23': ripple.get(name, {}).get('23', {}).get('ripple_relative_error'),
                         'ripple_01': ripple.get(name, {}).get('01', {}).get('ripple_relative_error')})
    return rows


def fisher_spectrum():
    spectra = np.load('artifacts/figure_refresh/gl_mixture_dim10/fisher_spectra.npz')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    for k in range(10):
        lam = np.sort(spectra[f'axis_{k}'])[::-1]
        axes[0].semilogy(np.arange(1, len(lam)+1), np.maximum(lam, 1e-12), 'o-', ms=3, lw=1, alpha=.7)
        axes[1].plot(np.arange(1, len(lam)+1), .01/(.01+lam), 'o-', ms=3, lw=1, alpha=.7)
    axes[0].axhline(.01, color='k', ls=':', lw=1)
    axes[0].text(20, .013, r'visibility floor $\alpha=0.01$', fontsize=9)
    axes[0].set(xlabel='eigen-direction $m$', ylabel=r'observed/complete information $\lambda_m$',
                title='Marginal Fisher visibility (10D GL, mixture noise)', ylim=(1e-8, 1.5))
    axes[1].set(xlabel='eigen-direction $m$', ylabel=r'filter weight $\alpha/(\alpha+\lambda_m)$',
                title='Penalty weight per direction')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT/'fisher_spectrum.png', dpi=200)
    plt.close(fig)


def nll_gap_vs_l2(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    colors = {4: '#1f5fbf', 10: '#dd5b00', 20: '#2a9d4b'}
    for ax, noise, title in zip(axes, ('free', 'mixture'), ('Noise-free fits', 'Noisy fits (deconvolved)'), strict=True):
        for r in rows:
            if r['noise'] != noise:
                continue
            ax.errorbar(max(r['nll_gap'], 1e-4), r['relative_l2'], xerr=r['nll_gap_se'], fmt='*' if r['selected'] else 'o',
                        color=colors[r['dim']], ms=11 if r['selected'] else 5, alpha=.85, lw=.8)
        for dim, color in colors.items():
            ax.plot([], [], 'o', color=color, label=f'd = {dim}')
        ax.plot([], [], '*', color='k', ms=10, label='selected')
        ax.set(xscale='log', xlabel='held-out NLL minus true-density NLL (nats, ± s.e.)',
               ylabel='latent relative L2', title=title)
        ax.grid(alpha=.2, which='both')
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT/'nll_gap_vs_l2.png', dpi=200)
    plt.close(fig)


def h2_tradeoff(rows):
    wanted = {('swirl_mixture_dim10', 'rank32_h2_0.0001'): 1e-4, ('swirl_mixture_dim10', 'rank32_h2_0.0003'): 3e-4,
              ('swirl_mixture_dim10_h2sweep', 'rank32_h2_0.001'): 1e-3,
              ('swirl_mixture_dim10_h2sweep', 'rank32_h2_0.003'): 3e-3,
              ('swirl_mixture_dim10_h2sweep', 'rank32_h2_0.01'): 1e-2}
    sel = sorted(((wanted[r['case'], r['candidate']], r) for r in rows if (r['case'], r['candidate']) in wanted),
                 key=lambda item: item[0])
    if not sel:
        return []
    lam = [s for s, _ in sel]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    axes[0].errorbar(lam, [r['nll_gap'] for _, r in sel], yerr=[r['nll_gap_se'] for _, r in sel], fmt='o-', color='#1f5fbf')
    axes[0].set(ylabel='NLL gap to truth (nats)', title='Held-out noisy likelihood')
    axes[1].plot(lam, [r['relative_l2'] for _, r in sel], 'o-', color='#dd5b00')
    axes[1].set(ylabel='latent relative L2', title='Full-dimensional error')
    axes[2].plot(lam, [r['ripple_01'] for _, r in sel], 'o-', label='swirl (0,1)', color='#2a9d4b')
    axes[2].plot(lam, [r['ripple_23'] for _, r in sel], 's-', label='pretzel (2,3)', color='#7b2cbf')
    axes[2].set(ylabel='along-ridge ripple', title='Bead-like oscillation')
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.set(xscale='log', xlabel=r'amplitude $H^2$ strength $\lambda$')
        ax.grid(alpha=.2, which='both')
    fig.suptitle('10D noisy swirl-pretzel, rank-32 continuations of one base fit', fontweight='bold')
    fig.savefig(OUT/'h2_tradeoff.png', dpi=200)
    plt.close(fig)
    return [{'strength': s, **r} for s, r in sel]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fisher_spectrum()
    rows = candidate_table()
    nll_gap_vs_l2(rows)
    tradeoff = h2_tradeoff(rows)
    (OUT/'numbers.json').write_text(json.dumps({'candidates': rows, 'h2_tradeoff': tradeoff}, indent=2)+'\n')
    for r in rows:
        print(r['case'], r['candidate'], 'sel' if r['selected'] else '', round(r['nll_gap'], 4), round(r['nll_gap_se'], 4),
              round(r['relative_l2'], 3), r['ripple_01'] and round(r['ripple_01'], 3), r['ripple_23'] and round(r['ripple_23'], 3))


def init_seed_figure():
    """10D noisy swirl: rank-32, H^2 3e-4 fits from three chebdGE initializations."""
    from refresh_density_fits import read_model

    from ftde.marginals import squared_pair_grid
    from ftde.swirl import swirl_pair_marginal
    cases = [('swirl_mixture_dim10', 'init seed 0'), ('swirl_mixture_dim10_initseed1', 'init seed 1'),
             ('swirl_mixture_dim10_initseed2', 'init seed 2')]
    if not all((SWIRL/c/'rank32_h2_0.0003'/'fit.npz').exists() for c, _ in cases):
        return {}
    xs = np.linspace(-2.5, 2.5, 200)
    fits = [read_model(SWIRL/c/'rank32_h2_0.0003'/'fit.npz') for c, _ in cases]
    summary = {}
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.8), layout='constrained')
    for row, (i, j) in enumerate([(0, 1), (2, 3)]):
        truth = swirl_pair_marginal(10, i, j, xs, xs)
        grids = [squared_pair_grid(f, i, j, xs, xs) for f in fits]
        mean = np.mean(grids, axis=0)
        panels = [truth, *grids, mean]
        titles = ['truth', *[t for _, t in cases], 'average of the three']
        vmax = max(p.max() for p in panels)
        for ax, z, title in zip(axes[row], panels, titles, strict=True):
            ax.imshow(z, origin='lower', extent=(-2.5, 2.5, -2.5, 2.5), cmap='magma', vmin=0, vmax=vmax)
            err = '' if title == 'truth' else f'\n2D rel. L2 {np.linalg.norm(z-truth)/np.linalg.norm(truth):.3f}'
            ax.set_title(f'({i},{j}) {title}{err}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        summary[f'{i}{j}'] = {'members': [float(np.linalg.norm(g-truth)/np.linalg.norm(truth)) for g in grids],
                              'average': float(np.linalg.norm(mean-truth)/np.linalg.norm(truth))}
    fig.suptitle('10D noisy swirl-pretzel: same data and recipe (rank 32, $H^2$ 3e-4), three initializations',
                 fontweight='bold')
    fig.savefig(OUT/'init_seeds.png', dpi=170)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    main()
    print('init seeds', json.dumps(init_seed_figure()))

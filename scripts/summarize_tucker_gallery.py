"""Index completed Tucker runs without replacing the existing TT gallery."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_tucker_gallery import current_folder

from ftde.direct_training import write_record

ROOT = Path('artifacts/tucker_gallery')


def main():
    rows = []
    for path in sorted(ROOT.glob('*/evaluation.json')):
        folder = current_folder(path.parent.name)
        path = folder/'evaluation.json'
        p = json.loads((folder/'protocol.json').read_text())
        e = json.loads(path.read_text())
        s = json.loads((folder/'selection.json').read_text())
        r = json.loads((folder/s['selected']/'result.json').read_text())
        rows.append({'case': folder.name, 'family': p['family'], 'dimension': p['dim'], 'noise': p['noise'],
                     'rank': p['rank'], 'core_entries': p['rank']**p['dim'], 'relative_l2': e['relative_l2'],
                     'selected': s['selected'], 'validation_nll': r['best_nll'],
                     'relative_l2_standard_error': e.get('relative_l2_standard_error', 0.),
                     'selected_fit_seconds': r['seconds'], 'model': str(folder/s['selected']/'fit.npz')})
    write_record(ROOT/'summary.json', {'cases': rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    for family, ax in zip(['gl', 'swirl'], axes, strict=True):
        for noise, color in [('free', '#285bba'), ('mixture', '#c34e25')]:
            entries = sorted((r for r in rows if r['family'] == family and r['noise'] == noise), key=lambda r: r['dimension'])
            ax.errorbar([r['dimension'] for r in entries], [r['relative_l2'] for r in entries],
                        yerr=[r['relative_l2_standard_error'] for r in entries], fmt='o-', capsize=3,
                        color=color, label=f'Tucker {"clean" if noise == "free" else "noisy"}')
            tt_summary = json.loads(Path('artifacts/dimension_gallery_v9/summary.json').read_text())
            tt = [(r['dim'], r['noise_free' if noise == 'free' else 'noisy']) for r in tt_summary if r['family'] == family]
            ax.plot([r[0] for r in tt], [r[1] for r in tt], 'x--', alpha=.6, color=color, label=f'TT {"clean" if noise == "free" else "noisy"}')
        ax.set(title='Ginzburg–Landau' if family == 'gl' else 'Swirl', xlabel='Dimension', ylabel='Full latent relative L2')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle('Tucker and TT: feasible-rank experiments\nRanks and optimizer recipes differ; not a matched-capacity comparison', fontsize=12)
    fig.savefig('figs/convergence/tucker_dimensional_errors.png', dpi=165)
    plt.close(fig)
    lines = ['# Tucker density figures', '', 'TT remains the default. Native Tucker fits use the same squared-density likelihood and full-Hessian penalty.',
             'Mode ranks are limited by the dense core; see [protocol and implementation](../docs/tucker_density.md).', '',
             '| Benchmark | Dimension | Mode rank | Clean relative L2 | Noisy relative L2 |', '|---|---:|---:|---:|---:|']
    for family, dim in sorted({(r['family'], r['dimension']) for r in rows}):
        entries = {r['noise']: r for r in rows if (r['family'], r['dimension']) == (family, dim)}
        directory = 'ginzburg_landau' if family == 'gl' else 'swirl'
        cells = []
        for noise, prefix in [('free', 'noise_free'), ('mixture', 'noisy')]:
            r = entries.get(noise)
            error = f'{r["relative_l2"]:.4f}' if r else ''
            if r and r['relative_l2_standard_error']:
                error += f' ± {r["relative_l2_standard_error"]:.4f}'
            cells.append(f'[{error}]({directory}/tucker_{prefix}_dim{dim}_marg_01.png)' if r else 'pending')
        ranks = ' / '.join(str(entries[n]['rank']) for n in ['free', 'mixture'] if n in entries)
        lines.append(f'| {"GL" if family == "gl" else "Swirl"} | {dim} | {ranks} | '+ ' | '.join(cells)+' |')
    lines += ['', 'Each condition has two pair-marginal figures (`marg_01`, `marg_23`) and a convergence plot, with `tucker_` prefixes.',
              'Errors are unsquared full-dimensional latent L2, evaluated only after held-out selection.',
              '', '[Dimensional error comparison](convergence/tucker_dimensional_errors.png).',
              'Models, protocols, numerical checks and logs: `artifacts/tucker_gallery`.']
    if (ROOT/'current.json').exists():
        pointer = ROOT/'current_selection.json'
        report = json.loads(pointer.read_text())['report'] if pointer.exists() else 'artifacts/tucker_capacity/report.md'
        lines += ['', 'Ranks are listed as clean / noisy. Swirl uses the higher-rank follow-up; the original low-rank fits remain archived.',
                  'The ± values are numerical standard errors from randomized importance quadrature in 6D; pair marginals remain exact.',
                  f'[Rank and training comparison](../{report}).']
    if Path('figs/cryoem/tucker_dim4.png').exists():
        lines += ['', '[4D Cryo-EM marginals](cryoem/tucker_dim4.png) · [training](cryoem/tucker_training.png). No clean-density error is available.']
    Path('figs/TUCKER.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()

"""LaTeX table of the sample-size sweep (mean over training seeds) for the note."""

import json
from collections import defaultdict
from pathlib import Path


def _power(n):
    """1000 -> $10^{3}$, 3000 -> $3\\cdot10^{3}$."""
    exponent = len(str(n)) - 1
    lead = n // 10**exponent
    return f'${lead}\\cdot10^{{{exponent}}}$' if lead > 1 else f'$10^{{{exponent}}}$'


rows = json.loads(Path('artifacts/convergence/summary.json').read_text())
table = defaultdict(list)
for r in rows:
    table[r['family'], r['noise'], r['n_train']].append(r)
sizes = sorted({r['n_train'] for r in rows})
labels = [('gl', 'free', 'GL, noise-free'), ('gl', 'mixture', 'GL, mixture noise'),
          ('swirl', 'free', 'swirl, noise-free'), ('swirl', 'mixture', 'swirl, noisy')]
lines = [r'\begin{tabular}{l' + 'c'*len(sizes) + '}', r'\toprule',
         r'$N$ & ' + ' & '.join(_power(n) for n in sizes) + r' \\', r'\midrule']
for family, noise, label in labels:
    for key, name in (('ftde_relative_l2', 'ft-de'), ('kde_relative_l2', 'KDE')):
        cells = []
        for n in sizes:
            vals = [r[key] for r in table[family, noise, n]]
            cells.append(f'{sum(vals)/len(vals):.3f}' if vals else '--')
        lines.append(f'{label} ({name}) & ' + ' & '.join(cells) + r' \\')
    lines.append(r'\addlinespace')
lines[-1] = r'\bottomrule'
lines.append(r'\end{tabular}')
Path('note/convergence_table.tex').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

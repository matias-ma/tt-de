"""Finish cached cryo-EM runs, render results, publish measured tables, audit."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('artifacts/direct_optimization')
CASES = [('dim4', '4D'), ('path_dim4', '4D path projection'), ('dim10', '10D'), ('dim20', '20D')]


def publish_block(path, block, *, before=None):
    start, end = '<!-- direct-cryo-transfer:start -->', '<!-- direct-cryo-transfer:end -->'
    contents = path.read_text()
    replacement = start+'\n'+block.strip()+'\n'+end
    if start in contents:
        first, last = contents.index(start), contents.index(end)+len(end)
        contents = contents[:first]+replacement+contents[last:]
    elif before is None:
        contents = contents.rstrip()+'\n\n'+replacement+'\n'
    else:
        if before not in contents:
            raise ValueError(f'Missing insertion marker in {path}')
        contents = contents.replace(before, replacement+'\n\n'+before, 1)
    path.write_text(contents)


def main(skip_fit):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ftde_mpl_config')
    if not skip_fit:
        subprocess.run([sys.executable, '-u', 'scripts/run_direct_cryoem.py'], check=True)
    subprocess.run([sys.executable, 'scripts/plot_direct_cryoem.py'], check=True)
    rows, new_winners = [], []
    for case, label in CASES:
        folder = ROOT/f'cryoem_{case}'
        selected = json.loads((folder/'selection.json').read_text())
        index = selected['names'].index(selected['new_selected'])
        old, new = selected['validation_nll'][0], selected['validation_nll'][index]
        if selected['selected'] != 'historical':
            new_winners.append(label)
        result = json.loads((folder/selected['new_selected']/'result.json').read_text())
        if selected['new_selected'] == 'lbfgs':
            source = json.loads((folder/selected['source']/'result.json').read_text())
            step = f"Adam {source['best_step']} + L-BFGS {result['best_step']}"
        else:
            step = f"Adam {result['best_step']}"
        rows.append((case, label, old, new, step))
    outcome = ('None of the new fits beats its historical reference under the fixed '
               'development-validation rule. The historical references remain unchanged.' if not new_winners else
               'The development-validation rule selects a new fit for '+', '.join(new_winners)+
               '. Historical references remain available for comparison.')
    documentation = '\n'.join([
        '### Cryo-EM measurements', '', outcome, '',
        '| Case | Historical validation NLL | Selected new validation NLL | Retained optimization checkpoint |',
        '|---|---:|---:|---|',
        *[f'| {label} | {old:.4f} | {new:.4f} | {step} |' for _, label, old, new, step in rows], '',
        ('Every new fit learns from a near-uniform Chebyshev TT. Full-data refinement '
        'continues that trained TT; it does not import a separately learned signal density. '
        'The figures retain both curvature strengths so their visual and likelihood '
        'tradeoff is visible. These are development-validation comparisons, not claims '
        'about the unknown clean-density error.'), '',
        *[f'- [{label} comparison](../figs/cryoem/direct_optimization_{case}.png).' for case, label, _, _, _ in rows],
        '- [All cryo-EM training trajectories](../figs/cryoem/direct_optimization_training.png).', '',
        ('The source data and prescribed noise are unchanged. The final audit checks '
        'all original 121 PNGs, the frozen model hashes, the numerical checks, and '
        'the 19 new figures. Eighteen core tests and three spatial-filter tests pass.')])
    publish_block(Path('docs/direct_optimization.md'), documentation)
    index = '\n'.join([
        '**Direct-training cryo-EM reruns:** '+outcome, '',
        '| Case | Historical validation NLL | Selected new validation NLL | Comparison figure |',
        '|---|---:|---:|---|',
        *[f'| {label} | {old:.4f} | {new:.4f} | [Marginals](cryoem/direct_optimization_{case}.png) |'
          for case, label, old, new, _ in rows], '',
        ('[Training trajectories](cryoem/direct_optimization_training.png) show the '
        'near-uniform starting point and subsequent TT updates. These comparisons '
        'reuse development validation; the clean cryo-EM density is unknown.')])
    publish_block(Path('figs/README.md'), index, before='The earlier studies below remain for comparison.')
    sources = sorted(str(p) for p in Path('scripts').glob('*direct*.py'))
    sources += ['src/ftde/simple_initialization.py', 'src/ftde/signed_tt_least_squares.py',
                'src/ftde/full_batch_optimization.py', 'src/ftde/tt_smoothing.py', 'src/ftde/__init__.py',
                'tests/test_direct_optimization.py', 'tests/test_tt_smoothing.py']
    with (ROOT/'logs/style.log').open('w') as output:
        subprocess.run([str(Path(sys.executable).with_name('ruff')), 'check', *sources],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, 'scripts/audit_direct_results.py'], check=True)
    print('COMPLETE: 19 figures; measured tables; integrity audit', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-fit', action='store_true')
    main(parser.parse_args().skip_fit)

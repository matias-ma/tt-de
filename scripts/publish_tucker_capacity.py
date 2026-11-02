"""Publish observation-selected rank follow-ups while retaining baselines.

Run after the fits listed in SPECS finish. Model choice reads held-out NLL
only. Truth-based evaluation and figure generation follow the frozen choice.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from run_tucker_gallery import digest, freeze, read

from ftde.direct_training import write_record

ROOT = Path('artifacts/tucker_capacity')
BASE = Path('artifacts/tucker_gallery')
SPECS = {
    'swirl4_free': ['r8_control', 'r16', 'r24'],
    'swirl6_free': ['r8', 'r10'],
    'swirl4_mixture': ['r16_noisy'],
    'swirl6_mixture': ['r6_noisy'],
}


def main():
    chosen, comparisons = {}, {}
    for case, roots in SPECS.items():
        folders = [BASE/case]+[ROOT/r/case for r in roots]
        candidates = []
        for folder in folders:
            freeze(folder)
            s = read(folder/'selection.json')
            p = read(folder/'protocol.json')
            r = read(folder/s['selected']/'result.json')
            candidates.append({'folder': str(folder), 'rank': p['rank'], 'steps': r['steps'],
                               'objective': r['objective'], 'validation_nll': r['best_nll'],
                               'selection_sha256': digest(folder/'selection.json')})
        best = min(candidates, key=lambda c: c['validation_nll'])
        chosen[case], comparisons[case] = best['folder'], candidates
    record = {'selected': chosen, 'candidates': comparisons,
              'selection': 'minimum held-out NLL across rank/optimizer recipes', 'truth_used_for_selection': False}
    path = ROOT/'selection.json'
    if path.exists() and read(path) != record:
        raise RuntimeError('Capacity selection already frozen with different inputs')
    write_record(path, record)

    for case, source in chosen.items():
        folder = Path(source)
        subprocess.run([sys.executable, 'scripts/plot_tucker_gallery.py', source], check=True)
        if case.endswith('mixture'):
            from check_tucker_convolution import check
            check(folder)
        prefix = f'tucker_{"noise_free" if case.endswith("free") else "noisy"}_dim{read(folder/"protocol.json")["dim"]}'
        for suffix in ['marg_01', 'marg_23', 'convergence']:
            name = f'{prefix}_{suffix}.png'
            target = Path('figs/swirl')/name
            archive = ROOT/'baseline_figs'/name
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                shutil.copy2(target, archive)
            staged = folder.parent/'figs/swirl'/name
            if folder.parent != BASE:
                shutil.copy2(staged, target)
    if not (ROOT/'baseline_summary.json').exists():
        shutil.copy2(BASE/'summary.json', ROOT/'baseline_summary.json')
        shutil.copy2(BASE/'publication.json', ROOT/'baseline_publication.json')
        shutil.copy2('figs/TUCKER.md', ROOT/'baseline_index.md')
    write_record(BASE/'current.json', chosen)
    write_record(BASE/'current_selection.json', {'path': str(path), 'sha256': digest(path), 'report': str(ROOT/'report.md')})
    lines = ['# Higher-rank Tucker Swirl follow-up', '',
             'The initial ranks were too restrictive. A 12,000-update rank-8 control improves held-out NLL from 1.468 to 1.390; rank 16 with the same objective and update count reaches 1.239. Rank therefore matters beyond extra training time.', '',
             'Rank 24 in 4D and ranks 8/10 in 6D use 20,000 updates and the TT core-sweep log-penalty objective. Noisy candidates use 4,000 updates and both strong and degree-curriculum recipes. All starts use the same observation-only box, seed and near-uniform construction; no true density or clean pilot enters fitting.', '',
             '| Condition | Selected rank | Old relative L2 | New relative L2 | Old held-out NLL | New held-out NLL |',
             '|---|---:|---:|---:|---:|---:|']
    for case, source in chosen.items():
        folder = Path(source)
        p, e = read(folder/'protocol.json'), read(folder/'evaluation.json')
        old = read(BASE/case/'evaluation.json')
        scores = comparisons[case]
        best = min(scores, key=lambda c: c['validation_nll'])
        error = f'{e["relative_l2"]:.4f}'
        if e.get('relative_l2_standard_error'):
            error += f' ± {e["relative_l2_standard_error"]:.4f}'
        lines.append(f'| {case} | {p["rank"]} | {old["relative_l2"]:.4f} | {error} | {scores[0]["validation_nll"]:.4f} | {best["validation_nll"]:.4f} |')
    lines += ['', 'The 4D full-density error uses exact blocked polynomial quadrature. The 6D fourth moment uses eight independent scrambled Sobol importance estimates (1,048,576 total samples); ± denotes one numerical standard error. The cross integral is deterministic, refined path/Gaussian quadrature. Every displayed pair marginal is exact.',
              '', 'The noisy 6D recovery improves but remains poor in full-dimensional L2. These results do not establish a best attainable Tucker error. In 4D, the updated clean/noisy errors are close to the TT gallery (0.116 / 0.296).',
              '', 'Frozen candidate hashes and observation-only choices: [selection.json](selection.json). Original figures: [baseline_figs](baseline_figs). Published figures: [Tucker index](../../figs/TUCKER.md).',
              '', 'Each fit is reproduced with `scripts/run_tucker_gallery.py --action fit --root artifacts/tucker_capacity/ROOT --family swirl --dim D --rank R --noise free|mixture --steps N --objective nll|logpen`; noisy runs also use `--microbatches 4` and both `--recipe strong` and `--recipe curriculum`. Full per-run settings are in each frozen `config.json`. Publish with `scripts/publish_tucker_capacity.py` after fitting.',
              '', 'TT remains the default. Its fits and figures are unchanged.']
    if (ROOT/'pair_benchmark.json').exists():
        b = read(ROOT/'pair_benchmark.json')
        lines += ['', f'The exact coordinate-pair contraction is {b["speedup"]:.2f}× faster than sequential contraction in the rank-16, 49-mode, batch-16 real-kernel value/gradient benchmark. Maximum absolute gradient difference: {b["maximum_absolute_gradient_error"]:.2g}. These are warm microbenchmark timings on the allocated GPU; see [record](pair_benchmark.json).',
                  '', 'The prepared polynomial basis also uses an exact doubling recurrence, with value and endpoint-gradient regression tests. [Data equivalence checks](data_equivalence.json) confirm identical observations, domains, and noise covariances across the original and updated comparisons.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, 'scripts/summarize_tucker_gallery.py'], check=True)
    subprocess.run([sys.executable, 'scripts/audit_tucker_gallery.py'], check=True)
    subprocess.run([sys.executable, 'scripts/audit_dimension_gallery.py', '--publication'], check=True)
    write_record(ROOT/'publication.json', {'selection_sha256': digest(path), 'report_sha256': digest(ROOT/'report.md'),
                                         'gallery_manifest_sha256': digest(BASE/'publication.json')})


if __name__ == '__main__':
    main()

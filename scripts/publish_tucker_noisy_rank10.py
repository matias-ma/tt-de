"""Freeze, evaluate and publish the higher-rank 6D noisy comparison."""
import shutil
import subprocess
import sys
from pathlib import Path

from check_tucker_convolution import check
from run_tucker_gallery import digest, freeze, read

from ftde.direct_training import write_record

ROOT = Path('artifacts/tucker_noisy_rank10')
BASE = Path('artifacts/tucker_gallery')
CASE = 'swirl6_mixture'


def main():
    folder = ROOT/'r10'/CASE
    freeze(folder)
    s = read(folder/'selection.json')
    result = read(folder/s['selected']/'result.json')
    prior_path = Path('artifacts/tucker_capacity/selection.json')
    selection = read(prior_path)
    prior_sources = dict(selection['selected'])
    row = {'folder': str(folder), 'rank': 10, 'steps': result['steps'], 'objective': result['objective'],
           'validation_nll': result['best_nll'], 'selection_sha256': digest(folder/'selection.json')}
    selection['candidates'][CASE].append(row)
    winner = min(selection['candidates'][CASE], key=lambda c: c['validation_nll'])
    selection['selected'][CASE] = winner['folder']
    current = read(BASE/'current.json')
    if current not in (prior_sources, selection['selected']):
        raise RuntimeError('Published model sources changed during the rank-10 study')
    selection['prior_selection'] = {'path': str(prior_path), 'sha256': digest(prior_path)}
    path = ROOT/'selection.json'
    if path.exists() and read(path) != selection:
        raise RuntimeError('Rank-10 selection was already frozen differently')
    write_record(path, selection)
    subprocess.run([sys.executable, 'scripts/plot_tucker_gallery.py', str(folder)], check=True)
    check(folder)
    source = Path(winner['folder'])
    for suffix in ['marg_01', 'marg_23', 'convergence']:
        name = f'tucker_noisy_dim6_{suffix}.png'
        destination = Path('figs/swirl')/name
        archive = ROOT/'previous_figs'/name
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            shutil.copy2(destination, archive)
        shutil.copy2(source.parent/'figs/swirl'/name, destination)
    for name in ['summary.json', 'publication.json', 'current.json']:
        target = ROOT/f'previous_{name}'
        if not target.exists():
            shutil.copy2(BASE/name, target)
    write_record(BASE/'current.json', selection['selected'])
    write_record(BASE/'current_selection.json', {'path': str(path), 'sha256': digest(path), 'report': str(ROOT/'report.md')})
    old = Path('artifacts/tucker_capacity/r6_noisy')/CASE
    old_e, new_e = read(old/'evaluation.json'), read(folder/'evaluation.json')
    old_s = read(old/'selection.json')
    old_r = read(old/old_s['selected']/'result.json')
    lines = ['# Rank 10 for noisy 6D Tucker', '',
             'The earlier cutoff at rank 6 was a conservative run-budget choice, not a demonstrated hardware or mathematical limit. Rank 4 was the original baseline. Clean point evaluation reduces one core mode at each contraction; noisy evaluation applies quadratic moment matrices while retaining the full core.', '',
             'Both runs below use exactly the same observations, latent box, prescribed noise kernel, 49 polynomial modes, log-penalty objective and weak-penalty degree-curriculum recipe. The recipe was fixed from the previous held-out winner before rank-10 fitting. Rank 10 uses 8,000 updates instead of 4,000 and the same batch of 256 observations.', '',
             '| Noisy 6D fit | Core entries | Updates | Held-out NLL | Full latent relative L2 |', '|---|---:|---:|---:|---:|',
             f'| Rank 6 | {6**6:,} | 4,000 | {old_r["best_nll"]:.6f} | {old_e["relative_l2"]:.4f} ± {old_e["relative_l2_standard_error"]:.4f} |',
             f'| Rank 10 | {10**6:,} | 8,000 | {result["best_nll"]:.6f} | {new_e["relative_l2"]:.4f} ± {new_e["relative_l2_standard_error"]:.4f} |', '',
             f'The published noisy 6D model comes from `{winner["folder"]}`, selected by held-out NLL before evaluating the new fit against truth. The ± values are numerical integration standard errors, not statistical uncertainty of the fitted model.', '',
             'Rank 10 has a 7.63 MiB dense core, versus 0.36 MiB at rank 6. The expensive part is batched noisy intermediates and reverse-mode differentiation. Pair contraction work grows approximately as rank^(dimension+2), while its core-sized intermediate storage grows as batch × rank^dimension, times the remaining noise bond. Neither formula includes fixed polynomial-moment preparation costs.', '',
             'Independent kernel mixture components now use separate exact pair contractions. Contributions sharing the final output bond are summed; no approximation or rank truncation is introduced. Tests compare values and gradients with dense paired and sequential contractions, including four- and six-dimensional mixtures.', '',
             '[Profiling measurements](profiles) report warmed value-gradient times and compiled temporary storage for the actual prescribed kernel. These are not process peak-memory measurements or full Adam iteration timings.', '',
             '[Current figures](../../figs/TUCKER.md) · [Rank-10 figures](r10/figs/swirl) · [Previous rank-6 figures](previous_figs) · [Earlier rank comparison](../tucker_capacity/report.md)', '',
             'Reproduce on the allocated GPU with `scripts/run_tucker_noisy_rank10.py`, then `scripts/publish_tucker_noisy_rank10.py`. TT remains the default; its figures are unchanged.']
    lines += ['', '| Rank | Likelihood | Warm value + gradient (16 observations) | Compiled temporary storage |',
              '|---|---|---:|---:|']
    for rank in [6, 10]:
        profile = read(ROOT/f'profiles/r{rank}_b16_grouped.json')
        for label, measurement in profile['measurements'].items():
            lines.append(f'| {rank} | {label} | {1000*measurement["seconds"]:.2f} ms | {measurement["compiled_temporary_bytes"]/2**20:.1f} MiB |')
    lines += ['', 'The measured rank-10 full-batch training run used roughly 40 GiB of GPU allocation on an 80 GiB H100. That is distinct from the 32 GiB host-memory allocation and from the compiled-temporary figures above. Smaller microbatches trade runtime for GPU memory; the rank-10 limit was not a lack of mathematical support for convolution.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, 'scripts/summarize_tucker_gallery.py'], check=True)
    subprocess.run([sys.executable, 'scripts/audit_tucker_gallery.py'], check=True)
    subprocess.run([sys.executable, 'scripts/audit_dimension_gallery.py', '--publication'], check=True)
    write_record(ROOT/'publication.json', {'selection_sha256': digest(path), 'report_sha256': digest(ROOT/'report.md'),
                                         'gallery_manifest_sha256': digest(BASE/'publication.json')})


if __name__ == '__main__':
    main()

"""Evaluate frozen training-geometry fits and save diagnostic figures."""
import subprocess
import sys
from pathlib import Path

from run_marginal_regularization import CASES_RUN, read
from run_training_geometry import ROOT, digest

from ftde.direct_training import write_record


def audit():
    frozen = read(ROOT/'frozen.json')
    for path, sha in frozen['files'].items():
        if digest(path) != sha:
            raise RuntimeError(f'Frozen input changed: {path}')
    tails = []
    for case in frozen['cases'].values():
        for info in case['models'].values():
            result = info['result']
            if result['best_step'] < 2000 or result['history'][-1]['step'] != 4000:
                raise RuntimeError('Mismatched selection window or training budget')
            tails.extend(row['inactive_coefficient_max'] for row in result['history']
                         if 'inactive_coefficient_max' in row)
    if max(tails) > 1e-10:
        raise RuntimeError('Mode constraint failed')
    old_path = Path('artifacts/smooth_initialization_v6/swirl4_mixture/default_seed1/result.json')
    old_frozen = read(old_path.parents[2]/'frozen.json')
    if digest(old_path) != old_frozen['files'][str(old_path)]:
        raise RuntimeError('Previous seed1 control changed')
    old = {h['step']: h for h in read(old_path)['history']}
    new = read(ROOT/'swirl4_mixture/baseline_seed1/result.json')['history']
    write_record(ROOT/'implementation_audit.json', {'verified_files': len(frozen['files']),
        'largest_locked_physical_coefficient': max(tails),
        'unrestricted_seed1_nll_max_difference_from_previous_implementation':
            max(abs(h['validation_nll']-old[h['step']]['validation_nll']) for h in new),
        'truth_evaluation_not_yet_run': not (ROOT/'evaluation_complete.json').exists()})


def main():
    import evaluate_marginal_regularization as evaluation
    evaluation.ROOT = ROOT
    if len(sys.argv) > 1:
        if set(read(ROOT/'smooth_selection.json')['cases']) != set(CASES_RUN):
            raise RuntimeError('Freeze both cases through the main driver before evaluating')
        evaluation.CASES_RUN = (sys.argv[1],)
        evaluation.evaluate()
        return
    audit()
    evaluation.freeze_smooth_choices()
    for case in CASES_RUN:
        subprocess.run([sys.executable, '-u', __file__, case], check=True)
    from plot_training_geometry import plot
    plot()


if __name__ == '__main__':
    main()

"""Finish full-data refinement, then audit, freeze, evaluate, and plot."""

import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path('artifacts/direct_optimization')


def require_time(minutes):
    """Avoid losing an entire unsaved optimizer run at an allocation boundary."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return
    description = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    match = re.search(r'\bEndTime=(\S+)', description)
    if match and match[1] != 'Unknown':
        end = datetime.datetime.fromisoformat(match[1]).astimezone()
        remaining = (end-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/60
        print('ALLOCATION minutes remaining', round(remaining, 1), flush=True)
        if remaining < minutes:
            raise RuntimeError(f'Resume this driver in an allocation with at least {minutes} minutes; completed fits are saved')


required = [root/'swirl4/signed_n49_r32_l1e-05_s0_neg100/result.json',
            root/'swirl10/squared_n49_r32_l0.0001_b256_s0/result.json',
            root/'swirl4/squared_n25_r16_l0.0001_b256_s0_lbfgs/result.json']
for path in required:
    if not path.exists():
        raise RuntimeError(f'Finish the observation-only controls before evaluation: {path}')
selection = json.loads((root/'swirl4/selection_q65.json').read_text())
if 'signed_n49_r32_l1e-05_s0_neg100' not in selection['names']:
    raise RuntimeError('Final clipped noisy-validation selection is incomplete')
for name, command in [
    ('noise_free_controls', ['scripts/run_direct_noise_free.py']),
    ('free10_adam', ['scripts/fit_direct_optimization.py', '--base', 'artifacts/swirl/swirl_free_dim10',
                    '--root', 'artifacts/direct_optimization/swirl_free10', '--method', 'squared',
                    '--modes', '49', '--rank', '32', '--strengths', '.0001', '--steps', '4000']),
    ('free10_rank64', ['scripts/fit_direct_optimization.py', '--base', 'artifacts/swirl/swirl_free_dim10',
                     '--root', 'artifacts/direct_optimization/swirl_free10', '--method', 'squared',
                     '--modes', '49', '--rank', '64', '--strengths', '.0001', '--steps', '4000']),
    ('free10_lbfgs', ['scripts/refine_direct_lbfgs.py', '--base', 'artifacts/swirl/swirl_free_dim10',
                    '--root', 'artifacts/direct_optimization/swirl_free10',
                    '--source', 'squared_n49_r32_l0.0001_b256_s0', '--iterations', '40']),
    ('lbfgs_n49', ['scripts/refine_direct_lbfgs.py', '--source', 'squared_n49_r32_l0.0001_b256_s0',
                   '--iterations', '40']),
    ('validation_complete', ['scripts/score_direct_optimization.py', '--order', '97']),
    ('validation_free', ['scripts/evaluate_direct_noise_free.py', '--stage', 'score']),
    ('numerical_checks', ['scripts/audit_direct_numerics.py']),
    ('evaluation4', ['scripts/evaluate_direct_optimization.py', '--dim', '4']),
    ('evaluation10', ['scripts/evaluate_direct_optimization.py', '--dim', '10']),
    ('evaluation_free10', ['scripts/evaluate_direct_optimization.py', '--dim', '10', '--noise-free']),
    ('evaluation_free', ['scripts/evaluate_direct_noise_free.py', '--stage', 'evaluate']),
    ('figures', ['scripts/plot_direct_optimization.py']),
    ('figures_free', ['scripts/plot_direct_noise_free.py']),
]:
    if name == 'lbfgs_n49' and not (root/'swirl4/squared_n49_r32_l0.0001_b256_s0_lbfgs/result.json').exists():
        require_time(30)
    if name == 'free10_lbfgs' and not (root/'swirl_free10/squared_n49_r32_l0.0001_b256_s0_lbfgs/result.json').exists():
        require_time(10)
    print('START', name, flush=True)
    with (root/'logs'/f'{name}.log').open('a') as stream:
        subprocess.run([sys.executable, '-u', *command], stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)

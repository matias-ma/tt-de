"""Run the observation-only optimizer controls sequentially on one allocation."""

import subprocess
import sys
from pathlib import Path

root = Path('artifacts/direct_optimization/logs')
tasks = [
    ('negative_penalty', ['scripts/fit_direct_optimization.py', '--negative-strength', '1',
                         '--strengths', '0.00001', '0.0001', '--sweeps', '40']),
    ('squared_n49', ['scripts/fit_direct_optimization.py', '--method', 'squared', '--modes', '49',
                    '--rank', '32', '--strengths', '.0001', '--steps', '4000', '--batch', '256']),
    ('lbfgs_n25', ['scripts/refine_direct_lbfgs.py', '--iterations', '80']),
    ('validation_followup', ['scripts/score_direct_optimization.py']),
]
for name, arguments in tasks:
    print('START', name, flush=True)
    with (root/f'{name}.log').open('a') as stream:
        subprocess.run([sys.executable, '-u', *arguments], stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)

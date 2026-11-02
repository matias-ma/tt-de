"""Further capacity and dimension controls; observations only."""

import subprocess
import sys
from pathlib import Path

root = Path('artifacts/direct_optimization/logs')
tasks = [
    ('negative_n49', ['scripts/fit_direct_optimization.py', '--modes', '49', '--rank', '32', '--order', '81',
                     '--negative-strength', '1', '--strengths', '.00001', '.0001', '--sweeps', '60']),
    ('negative_strong_n49', ['scripts/fit_direct_optimization.py', '--modes', '49', '--rank', '32', '--order', '81',
                           '--negative-strength', '100', '--strengths', '.00001', '--sweeps', '60']),
    ('swirl10_squared', ['scripts/fit_direct_optimization.py', '--base', 'artifacts/swirl/swirl_mixture_dim10',
                        '--root', 'artifacts/direct_optimization/swirl10', '--method', 'squared',
                        '--modes', '49', '--rank', '32', '--strengths', '.0001', '--steps', '4000']),
    ('validation_capacity', ['scripts/score_direct_optimization.py']),
]
for name, arguments in tasks:
    print('START', name, flush=True)
    with (root/f'{name}.log').open('a') as stream:
        subprocess.run([sys.executable, '-u', *arguments], stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', name, flush=True)

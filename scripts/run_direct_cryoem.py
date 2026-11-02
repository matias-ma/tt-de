"""Run the observation-only transfer of direct TT optimization to cryo-EM."""

import subprocess
import sys
from pathlib import Path

root = Path('artifacts/direct_optimization/logs')
for case in ('dim4', 'path_dim4', 'dim10', 'dim20'):
    print('START', case, flush=True)
    with (root/f'cryoem_{case}.log').open('a') as stream:
        subprocess.run([sys.executable, '-u', 'scripts/transfer_direct_cryoem.py', '--case', case],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', case, flush=True)

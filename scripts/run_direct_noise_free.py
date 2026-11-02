"""Noise-free controls from elementary starts; no signal truth is imported."""

import fcntl
import subprocess
import sys
from pathlib import Path

base = 'artifacts/swirl/swirl_free_dim4'
root = 'artifacts/direct_optimization/swirl_free4'
common = ['scripts/fit_direct_optimization.py', '--base', base, '--root', root]
# A concurrent launch in the existing allocation may finish these controls
# before the main driver reaches them. Serialize drivers and reuse results.
tasks = []
for modes, rank, strengths, seed in [
    (25, 16, ['.0001'], 0), (49, 32, ['0', '.00001', '.0001'], 0),
    (49, 32, ['.0001'], 1), (65, 32, ['.0001'], 0), (49, 64, ['.0001'], 0),
    (49, 32, ['.001'], 0),
]:
    tasks.append((f'free_squared_n{modes}_r{rank}_s{seed}', common+[
        '--method', 'squared', '--modes', str(modes), '--rank', str(rank), '--strengths', *strengths,
        '--seed', str(seed), '--steps', '4000', '--batch', '1024']))
for modes, rank in [(25, 16), (49, 32), (49, 64)]:
    tasks.append((f'free_signed_n{modes}_r{rank}', common+['--modes', str(modes), '--rank', str(rank),
        '--strengths', '.000001', '.00001', '.0001', '--sweeps', '30']))
tasks.append(('free_signed_penalty', common+['--modes', '49', '--rank', '32', '--strengths', '.00001',
    '--negative-strength', '1', '--sweeps', '30']))
tasks.append(('free_lbfgs', ['scripts/refine_direct_lbfgs.py', '--base', base, '--root', root,
    '--source', 'squared_n49_r32_l0.0001_b1024_s0', '--iterations', '100']))
tasks.append(('free_lbfgs_smoother', ['scripts/refine_direct_lbfgs.py', '--base', base, '--root', root,
    '--source', 'squared_n49_r32_l0.001_b1024_s0', '--strength', '.001', '--iterations', '100']))
with Path('artifacts/direct_optimization/noise_free.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for name, arguments in tasks:
        print('START', name, flush=True)
        with (Path('artifacts/direct_optimization/logs')/f'{name}.log').open('a') as stream:
            subprocess.run([sys.executable, '-u', *arguments], stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('DONE', name, flush=True)

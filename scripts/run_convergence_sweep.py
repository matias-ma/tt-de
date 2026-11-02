"""Sample-size sweep: relative L2 of the unchanged estimators versus N.

Runs, one srun step per case inside an existing allocation:

- GL, 10 dimensions, noise-free and Gaussian-mixture noise (scale 1.5), the
  exact gallery recipe of refresh_density_fits.py, training seeds 0 and 1;
- swirl-pretzel, 4 dimensions, noise-free and noisy, swirl_fits.py with a
  single rank-32 continuation (the gallery also tries rank 64).

Validation sets are fixed (seed 2, as in the gallery), so selection always
uses the same number of held-out observations. The N = 100,000, seed-0 GL
cases are the gallery fits themselves (linked, not refitted). After each fit
the exact full-dimensional L2 is computed; see convergence_summary.py.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SIZES = (1000, 3000, 10000, 30000, 100000, 300000)


def cases():
    for seed in (0, 1):
        for n in SIZES:
            for noise in ('free', 'mixture'):
                yield (f'gl_{noise}_dim10_n{n}_s{seed}', 'refresh_density_fits.py',
                       ['--family', 'gl', '--dim', '10', '--noise', noise, '--n-train', str(n),
                        '--train-seed', str(seed)])
        if seed == 0:
            for n in SIZES:
                for noise in ('free', 'mixture'):
                    yield (f'swirl_{noise}_dim4_n{n}_s0', 'swirl_fits.py',
                           ['--dim', '4', '--noise', noise, '--n-train', str(n), '--ranks', '32'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--root', type=Path, default=Path('artifacts/convergence'))
    parser.add_argument('--mem', default='7G')
    parser.add_argument('--overlap', action='store_true', help='share the allocation with another running step')
    args = parser.parse_args()
    (args.root/'logs').mkdir(parents=True, exist_ok=True)
    for noise in ('free', 'mixture'):
        link = args.root/f'gl_{noise}_dim10_n100000_s0'
        if not link.exists():
            link.symlink_to(Path(f'artifacts/figure_refresh/gl_{noise}_dim10').resolve())
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    env.setdefault('CHEBTTGE_JAX_CACHE_DIR', f'/tmp/ftde-jax-{os.getuid()}')
    prefix = ['srun', f'--jobid={args.job_id}', '--ntasks=1', '--cpus-per-task=4', f'--mem={args.mem}']
    if args.overlap:
        prefix.append('--overlap')
    prefix += [sys.executable, '-u']
    for name, script, flags in cases():
        case = args.root/name
        print('START', name, time.ctime(), flush=True)
        with (args.root/'logs'/f'{name}.log').open('a') as log:
            fit = subprocess.run(prefix+[f'scripts/{script}', *flags, '--output', str(case)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if fit.returncode == 0:
                subprocess.run(prefix+['scripts/convergence_summary.py', 'evaluate', str(case)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        print('DONE', name, 'exit', fit.returncode, time.ctime(), flush=True)


if __name__ == '__main__':
    main()

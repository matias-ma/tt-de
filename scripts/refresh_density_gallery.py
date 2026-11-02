"""Run the complete gallery in one existing Slurm allocation, one case at a time."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/figure_refresh'))
    parser.add_argument('--figures', type=Path, default=Path('artifacts/figure_refresh/gallery'))
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    logs = args.artifacts/'logs'
    logs.mkdir(exist_ok=True)
    cases = [(f'gl_{noise}_dim{d}', ['--family', 'gl', '--dim', str(d), '--noise', noise])
             for noise in ('free', 'paired', 'mixture') for d in (6, 10, 14, 20)]
    cases += [(f'cryoem_dim{d}', ['--family', 'cryoem', '--dim', str(d)]) for d in (4, 10, 20)]
    cases += [('cryoem_path_dim4', ['--family', 'cryoem', '--dim', '4', '--path'])]
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    # Keep recompilable GPU executables off the small home-directory quota.
    # All srun steps reuse this node-local cache in the existing allocation.
    env.setdefault('CHEBTTGE_JAX_CACHE_DIR', f'/tmp/ftde-jax-{os.getuid()}')
    env.setdefault('CHEBTTGE_JAX_CACHE_MIN_COMPILE_SECS', '1')
    prefix = ['srun', f'--jobid={args.job_id}', '--ntasks=1', '--cpus-per-task=4', sys.executable, '-u']
    results = []
    for name, flags in cases:
        case = args.artifacts/name
        print('START', name, time.ctime(), flush=True)
        with (logs/f'{name}.log').open('a') as log:
            fit = subprocess.run(prefix+['scripts/refresh_density_fits.py', *flags, '--output', str(case)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            plotted = None
            if fit.returncode == 0:
                plotted = subprocess.run(prefix+['scripts/refresh_density_figures.py', str(case),
                    '--figures', str(args.figures)], env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        results.append({'case': name, 'fit_exit': fit.returncode,
                        'plot_exit': None if plotted is None else plotted.returncode})
        (args.artifacts/'run_status.json').write_text(json.dumps(results, indent=2)+'\n')
        print('DONE', results[-1], time.ctime(), flush=True)
    if any(r['fit_exit'] or r['plot_exit'] != 0 for r in results):
        raise SystemExit('Some cases failed; inspect run_status.json and per-case logs')


if __name__ == '__main__':
    main()

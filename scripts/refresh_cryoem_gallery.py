"""Refine cached cryo-EM fits and redraw the flat gallery in one GPU allocation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--baseline', type=Path, default=Path('artifacts/figure_refresh'))
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/cryoem_refinement'))
    parser.add_argument('--figures', type=Path, default=Path('figs'))
    parser.add_argument('--cases', nargs='+', choices=['dim4', 'dim10', 'dim20', 'path_dim4'],
                        default=['dim4', 'dim10', 'dim20', 'path_dim4'])
    parser.add_argument('--polish-cases', nargs='*', choices=['dim4', 'dim10', 'dim20', 'path_dim4'],
                        default=['dim10', 'dim20'])
    args = parser.parse_args()
    root = args.artifacts
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CHEBTTGE_JAX_CACHE_DIR=f'/tmp/ftde-jax-{os.getuid()}',
               CHEBTTGE_JAX_CACHE_MIN_COMPILE_SECS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')

    def run(script, arguments, logfile):
        command = ['srun', f'--jobid={args.job_id}', '--ntasks=1', '--cpus-per-task=4',
                   sys.executable, '-u', f'scripts/{script}', *map(str, arguments)]
        with logfile.open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    run('audit_cryoem_noise.py', ['--output', root], root/'noise_audit.log')
    summary_path = root/'run_summary.json'
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in args.cases:
        experiment, case = root/f'{name}_rank20', root/f'gallery_{name}'
        print('Refining', name, flush=True)
        arguments = ['--base', args.baseline/f'cryoem_{name}', '--noise-metadata', root/f'noise_{name}.npz',
                     '--output', experiment, '--steps', '4000', '--rank', '20', '--strengths', '0.00003', '0.0003']
        if name.startswith('path'):
            arguments.append('--correct-projection')
        run('refine_cryoem_gallery.py', arguments, root/f'{name}_rank20.log')
        if name in args.polish_cases:
            run('polish_cryoem_refinement.py', [experiment], root/f'{name}_continuation.log')
        run('evaluate_cryoem_noise_kernel.py', [experiment], root/f'{name}_noise_check.log')
        run('publish_cryoem_refinement.py', ['--experiment', experiment, '--output', case], root/f'{name}_publication.log')
        run('refresh_density_figures.py', [case, '--figures', root/'gallery'], root/f'{name}_plots.log')
        summaries[name] = json.loads((case/'refinement.json').read_text())
        summary_path.write_text(json.dumps(summaries, indent=2)+'\n')
        # Each staged file was produced successfully before publication.
        destination = args.figures/'cryoem'
        destination.mkdir(parents=True, exist_ok=True)
        for figure in (root/'gallery/cryoem').glob('*.png'):
            shutil.copy2(figure, destination/figure.name)
        print('Published', name, flush=True)


if __name__ == '__main__':
    main()

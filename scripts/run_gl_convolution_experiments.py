"""Driver: rerun the validation GL-convolution configs with the improved
(preconditioned + prepared + cosine-LR) recipe, plus a rank / n-coeffs /
n-train scaling sweep and one explicit --naive-steps 0 example. Toy GL only.

Each config is a separate subprocess (own JAX process) so a crash in one
config cannot corrupt another's compiled cache. Results land as JSON/NPZ in
--directory; scaling_summary.json indexes every scaling run for plotting.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

VALIDATION = [
    dict(name='gl8_gaussian', dim=8, noise='gaussian'),
    dict(name='gl8_mixture', dim=8, noise='mixture', compare_spectral=True),
    dict(name='gl8_paired', dim=8, noise='paired', kernel_rank=12),
    dict(name='gl20_mixture', dim=20, noise='mixture', n_coeffs=21, steps=8000),
]

BASELINE = dict(dim=8, noise='mixture', n_train=20000, rank=8, n_coeffs=17, steps=6000)


def scaling_runs():
    runs = []
    for rank in (4, 6, 8, 12):
        runs.append({**BASELINE, 'rank': rank, 'name': f'scaling_rank_{rank}', 'sweep': 'rank', 'value': rank})
    for n_coeffs in (13, 17, 21, 25):
        runs.append({**BASELINE, 'n_coeffs': n_coeffs, 'name': f'scaling_ncoeffs_{n_coeffs}',
                     'sweep': 'n_coeffs', 'value': n_coeffs})
    for n_train in (2000, 5000, 20000, 50000, 100000):
        runs.append({**BASELINE, 'n_train': n_train, 'name': f'scaling_ntrain_{n_train}',
                     'sweep': 'n_train', 'value': n_train})
    runs.append({**BASELINE, 'name': 'scaling_naive_steps_0', 'sweep': 'naive_steps',
                 'value': 0, 'naive_steps': 0})
    return runs


FLAG_MAP = {
    'dim': '--dim', 'noise': '--noise', 'n_train': '--n-train', 'rank': '--rank',
    'n_coeffs': '--n-coeffs', 'steps': '--steps', 'naive_steps': '--naive-steps',
    'deconv_steps': '--deconv-steps', 'kernel_rank': '--kernel-rank',
}
FLAG_ACTIONS = {'compare_spectral': '--compare-spectral'}


def run_one(config, directory):
    output = directory/f"{config['name']}.json"
    args = [sys.executable, str(HERE/'benchmark_gl_convolution.py'), '--output', str(output)]
    for key, flag in FLAG_MAP.items():
        if key in config:
            args += [flag, str(config[key])]
    for key, flag in FLAG_ACTIONS.items():
        if config.get(key):
            args.append(flag)
    print(f"=== running {config['name']}: {' '.join(args)} ===", flush=True)
    tic = time.monotonic()
    result = subprocess.run(args, cwd=HERE)
    elapsed = time.monotonic()-tic
    status = 'ok' if result.returncode == 0 else f'FAILED (rc={result.returncode})'
    print(f"=== {config['name']} {status} in {elapsed:.1f}s ===", flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/gl_convolution_scaling'))
    parser.add_argument('--skip-validation', action='store_true')
    parser.add_argument('--skip-scaling', action='store_true')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)

    summary = []
    if not args.skip_validation:
        for config in VALIDATION:
            ok = run_one(config, args.directory)
            summary.append({**config, 'ok': ok, 'group': 'validation'})
    if not args.skip_scaling:
        for config in scaling_runs():
            ok = run_one(config, args.directory)
            summary.append({**config, 'ok': ok, 'group': 'scaling'})

    (args.directory/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    n_failed = sum(not s['ok'] for s in summary)
    print(f"\n=== done: {len(summary)-n_failed}/{len(summary)} succeeded ===")
    if n_failed:
        sys.exit(1)


if __name__ == '__main__':
    main()

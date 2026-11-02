"""Evaluate and publish once every noisy-data choice has been frozen."""
import subprocess
import sys

from noise_diagnosis import ROOT, SIGMAS


def call(script, argument):
    print(script, argument, flush=True)
    with (ROOT/f'{script.removesuffix(".py")}_{argument}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', f'scripts/{script}', argument], stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    call('evaluate_noise_diagnosis.py', 'freeze')
    subprocess.run([sys.executable, '-u', 'scripts/check_noisy_stability.py'], check=True)
    for case in ['swirl4_mixture', 'gl6_mixture', 'gl6_free', 'gl6_wide_free', *[f'gaussian_{s:g}' for s in SIGMAS]]:
        call('evaluate_noise_diagnosis.py', case)
    for part in ['accuracy', 'optimization', 'strength', 'box', 'swirl4_mixture', 'gl6_mixture']:
        call('plot_noise_diagnosis.py', part)

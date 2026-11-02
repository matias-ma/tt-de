"""Sequential GPU controls; all choices are frozen before truth evaluation."""
import subprocess
import sys

from noise_diagnosis import ROOT, SIGMAS

from ftde.direct_training import write_record


def launch(case, name, *args):
    with (ROOT/f'{case}_{name}.log').open('a') as log:
        command = [sys.executable, '-u', 'scripts/noise_diagnosis.py', case, '--name', name, *map(str, args)]
        print('RUN', ' '.join(command), flush=True)
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    write_record(ROOT/'plan.json', {'objective': 'nll', 'regularizer': 'hessian', 'strength': .001,
        'new_seed_restarts': {'swirl4_mixture': [0, 1, 2], 'gl6_mixture': [1]},
        'continuations': '4000 lower-step Adam updates from old NLL fit, same data and penalty',
        'gaussian_noise_sigmas': list(SIGMAS), 'gaussian_common_box_source': 'existing noisy-only box rounded outwards',
        'gaussian_capacity': [32, 49], 'steps': 4000, 'batch': 256,
        'selection': 'minimum observed validation NLL; all fits frozen before latent evaluation',
        'storage_budget_MiB': 256})
    for case, seeds in [('swirl4_mixture', [0, 1, 2]), ('gl6_mixture', [1])]:
        for seed in seeds:
            launch(case, f'seed{seed}', '--seed', seed)
        capacity = 'r32_n49' if case.startswith('swirl') else 'r10_n25'
        baseline = f'artifacts/objective_v3/{case}/nll_{capacity}_l0.001_w100/fit.npz'
        launch(case, 'longer_adam', '--resume', baseline, '--lr', .0003)
    subprocess.run([sys.executable, 'scripts/noise_diagnosis.py', 'generate'], check=True)
    for sigma in SIGMAS:
        launch(f'gaussian_{sigma:g}', 'seed0')

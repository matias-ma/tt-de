"""Optimizer and box-width controls, specified before any truth evaluation."""
import subprocess
import sys

from noise_diagnosis import ROOT
from run_noise_diagnosis import launch

from ftde.direct_training import write_record

if __name__ == '__main__':
    write_record(ROOT/'extension_plan.json', {
        'full_data_lbfgs': {'swirl4_mixture': 30, 'gl6_mixture': 30},
        'lbfgs_initialization': 'original 4000-update noisy NLL fit, no clean input',
        'box_controls': ['gl6_free', 'gl6_wide_free'],
        'wide_box_resolution_control': '49 modes, chosen to offset the roughly 1.7x wider box (25 * 1.7 rounded up to existing mode grid)',
        'box_control_note': 'Noise-free GL observations only for these separately labeled controls; never passed to a noisy fit.',
        'all_selection': 'observed validation NLL only'})
    for case in ['gl6_free', 'gl6_wide_free']:
        launch(case, 'seed0')
    launch('gl6_wide_free', 'modes49', '--modes', 49)
    for case in ['gl6_mixture', 'swirl4_mixture']:
        capacity = 'r32_n49' if case.startswith('swirl') else 'r10_n25'
        baseline = f'artifacts/objective_v3/{case}/nll_{capacity}_l0.001_w100/fit.npz'
        launch(case, 'lbfgs', '--resume', baseline, '--lbfgs', '--steps', 30)
    subprocess.run([sys.executable, '-u', 'scripts/finish_noise_diagnosis.py'], check=True)

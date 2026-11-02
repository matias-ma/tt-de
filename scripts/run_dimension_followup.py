"""Observation-selected penalty transfer check for clean 20D Swirl.

This development follow-up follows inspection of the initial matrix's
baseline truth diagnostics. New choices use observations only and are frozen
before evaluating the additional candidates' truth errors.
"""
import argparse
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import run_dimension_gallery as fitting

from ftde.direct_training import write_record

BASE = fitting.ROOT
ROOT = Path('artifacts/dimension_followup_v10')
CASE = 'swirl20_free'


def setup(capacity=False):
    folder = ROOT/CASE
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'protocol.json').exists():
        return fitting.read(folder/'protocol.json')
    source = (Path('artifacts/dimension_followup_v10') if capacity else BASE)/CASE
    from plot_dimension_gallery import verify
    source_selection = verify(source)
    source_name = source_selection['selected']
    protocol = copy.deepcopy(fitting.read(source/'protocol.json'))
    original = fitting.read(source/source_name/'config.json')
    protocol['recipes'] = {'hessian': original}
    if capacity:
        protocol['recipes']['rank64'] = original | {'rank': 64}
    else:
        for strength in (.001, .0003, .0001):
            protocol['recipes'][f'hessian_{strength:g}'] = original | {'strength': strength}
    protocol['selection'] = 'minimum held-out NLL across recipes and checkpoints; no latent-error selection'
    protocol['development_followup'] = True
    protocol['baseline_truth_previously_inspected'] = True
    protocol['baseline_selection_sha256'] = fitting.digest(source/'selection.json')
    protocol['baseline_source'] = str(source/source_name)
    protocol['motivation'] = ('Compare rank 64 with the selected rank-32 fit at fixed penalty and budget.' if capacity else
                             'Check fixed Hessian strength across dimensions; fixed rank, basis and initialization.')
    (folder/'hessian').mkdir(exist_ok=True)
    for name in ('config.json', 'result.json', 'fit.npz', 'validation.npy'):
        shutil.copy2(source/source_name/name, folder/'hessian'/name)
    write_record(folder/'protocol.json', protocol)
    return protocol


def publish():
    import plot_dimension_gallery as plotting
    plotting.ROOT, plotting.STAGE = BASE, BASE/'figs'
    folder = ROOT/CASE
    plotting.verify(folder)
    if not (folder/'plot_checks.json').exists():
        raise RuntimeError('Follow-up has not been evaluated and plotted')
    manifest = BASE/'publication_overrides.json'
    overrides = fitting.read(manifest) if manifest.exists() else {}
    write_record(manifest, overrides | {CASE: str(folder)})
    if (BASE/'complete.json').exists():
        plotting.publish()


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--capacity', action='store_true')
    args = parser.parse_args()
    if args.capacity:
        ROOT = Path('artifacts/dimension_capacity_v11')
    protocol = setup(args.capacity)
    extra = ['--capacity'] if args.capacity else []
    fitting.ROOT = ROOT
    if args.name:
        fitting.fit('swirl', 20, 'free', args.name)
    elif args.evaluate:
        import plot_dimension_gallery as plotting
        plotting.ROOT, plotting.STAGE = ROOT, ROOT/'figs'
        plotting.plot(CASE)
    elif args.publish:
        publish()
    else:
        (ROOT/'logs').mkdir(exist_ok=True)
        for name in protocol['recipes']:
            if name == 'hessian':
                continue
            with (ROOT/'logs'/f'{name}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', __file__, '--name', name, *extra],
                                stdout=log, stderr=subprocess.STDOUT, check=True)
        fitting.select('swirl', 20, 'free')
        subprocess.run([sys.executable, '-u', __file__, '--evaluate', *extra], check=True)
        publish()


if __name__ == '__main__':
    main()

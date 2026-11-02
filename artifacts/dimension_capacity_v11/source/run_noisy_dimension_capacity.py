"""Matched rank-32/rank-64 continuation of the frozen noisy 20D Swirl fit.

No clean model or clean observations enter initialization or model selection.
The fixed budget is declared before the noisy 20D baseline's truth evaluation.
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
ROOT = Path('artifacts/noisy_dimension_capacity_v12')
CASE = 'swirl20_mixture'
PLAN = {'case': CASE, 'ranks': [32, 64], 'steps': 1000, 'learning_rate': .0001,
        'effective_batch_size': 256, 'microbatches': [2, 8], 'jitter': .001, 'seed': 0,
        'source': 'observation-selected noisy baseline; never use the clean fit',
        'penalties': 'retain the source amplitude and marginal penalties',
        'initialization': 'physical-L2 canonicalize both; perturb added ranks only',
        'selection': 'lowest rank, then marginal curvature, within one paired SE of minimum noisy validation NLL',
        'truth_evaluated_after_selection': True}


def plan():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/'plan.json'
    if path.exists():
        if fitting.read(path) != PLAN:
            raise RuntimeError('Capacity protocol changed')
        return
    write_record(path, PLAN)


def setup():
    folder = ROOT/CASE
    folder.mkdir(exist_ok=True)
    if (folder/'protocol.json').exists():
        return fitting.read(folder/'protocol.json')
    from plot_dimension_gallery import verify
    selection = verify(BASE/CASE)
    source = BASE/CASE/selection['selected']
    protocol = copy.deepcopy(fitting.read(BASE/CASE/'protocol.json'))
    original = fitting.read(source/'config.json')
    protocol['recipes'] = {'source': original}
    for rank, chunks in zip(PLAN['ranks'], PLAN['microbatches'], strict=True):
        config = original | {'rank': rank, 'microbatches': chunks, 'parent': 'source',
            'steps': PLAN['steps'], 'learning_rate': PLAN['learning_rate'], 'expand_rank': True,
            'jitter': PLAN['jitter'], 'seed': 0, 'mode_schedule': None, 'selection_start': 0}
        for key in ('marginal_budget', 'source_curvature', 'parent_sha256'):
            config.pop(key, None)
        protocol['recipes'][f'continue_rank{rank}'] = config
    protocol['selection'] = PLAN['selection']
    protocol['selection_complexity'] = 'rank_then_marginal'
    protocol['baseline_source'] = str(source)
    protocol['baseline_selection_sha256'] = fitting.digest(BASE/CASE/'selection.json')
    protocol['plan_sha256'] = fitting.digest(ROOT/'plan.json')
    (folder/'source').mkdir(exist_ok=True)
    for name in ('config.json', 'result.json', 'fit.npz', 'validation.npy'):
        shutil.copy2(source/name, folder/'source'/name)
    write_record(folder/'protocol.json', protocol)
    return protocol


def publish():
    import plot_dimension_gallery as plotting
    plotting.ROOT, plotting.STAGE = BASE, BASE/'figs'
    folder = ROOT/CASE
    plotting.verify(folder)
    if not (folder/'plot_checks.json').exists():
        raise RuntimeError('Evaluate and plot the capacity study before publication')
    manifest = BASE/'publication_overrides.json'
    previous = fitting.read(manifest) if manifest.exists() else {}
    write_record(manifest, previous | {CASE: str(folder)})
    if (BASE/'complete.json').exists():
        plotting.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--name')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    plan()
    if args.plan:
        return
    protocol = setup()
    fitting.ROOT = ROOT
    if args.name:
        fitting.fit('swirl', 20, 'mixture', args.name)
    elif args.evaluate:
        import plot_dimension_gallery as plotting
        plotting.ROOT, plotting.STAGE = ROOT, ROOT/'figs'
        plotting.plot(CASE)
    elif args.publish:
        publish()
    else:
        (ROOT/'logs').mkdir(exist_ok=True)
        for name in protocol['recipes']:
            if name == 'source':
                continue
            with (ROOT/'logs'/f'{name}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', __file__, '--name', name],
                                stdout=log, stderr=subprocess.STDOUT, check=True)
        fitting.select('swirl', 20, 'mixture')
        subprocess.run([sys.executable, '-u', __file__, '--evaluate'], check=True)
        publish()


if __name__ == '__main__':
    main()

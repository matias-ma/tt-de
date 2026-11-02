"""Observation-only 4D Swirl follow-up; freeze all choices before truth evaluation."""
import argparse
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
from run_tucker_gallery import digest, freeze, read

from ftde.direct_training import load_tt, write_record

ROOT = Path('artifacts/tucker_4d_tuning')
BASE = Path('artifacts/tucker_gallery')


def protocol_path():
    for name in ['protocol_v3.json', 'protocol_v2.json', 'protocol.json']:
        path = ROOT/name
        if path.exists():
            return path
    return ROOT/'protocol.json'


def setup():
    path = protocol_path()
    if path.exists():
        return read(path)
    prior = read(BASE/'current_selection.json')
    sources = read(BASE/'current.json')
    candidates = {}
    for strength in [.003, .01, .03]:
        candidates[f'clean_r32_h{strength:g}'] = {'noise': 'free', 'rank': 32, 'strength': strength,
                                                   'steps': 20000, 'batch_size': 1024, 'learning_rate': .001}
    for strength in [1e-5, 1e-4, 1e-3]:
        candidates[f'noisy_r24_h{strength:g}'] = {'noise': 'mixture', 'rank': 24, 'strength': strength,
                                                   'steps': 12000, 'batch_size': 256, 'learning_rate': .001}
    protocol = {'prior_selection': prior, 'prior_sources': sources, 'candidates': candidates,
                    'selection': 'minimum held-out NLL including the prior candidates; truth only after final freeze',
                    'truth_used_for_selection': False, 'baseline_truth_previously_inspected': True,
                    'refinement': {'clean': {'steps': 5000, 'batch_size': 4096, 'learning_rate': .0001},
                                    'noisy': {'steps': 4000, 'batch_size': 1024, 'learning_rate': .0001,
                                               'marginal_strengths': [0., .0003, .001]}},
                    'refinement_parent': 'minimum held-out NLL across scratch fits and the current published fit, separately by condition'}
    ROOT.mkdir(parents=True, exist_ok=True)
    write_record(path, protocol)
    return protocol


def fit(name):
    from ftde import (
        TuckerConvolution,
        TuckerTrainingProblem,
        init_tucker_constant,
        load_tucker,
    )
    from ftde.tucker_operations import expand_tucker_rank
    study = setup()
    specs = study['candidates'] if name in study['candidates'] else read(ROOT/'refinements.json')
    spec = specs[name]
    case = f'swirl4_{spec["noise"]}'
    folder, out = ROOT/name/case, ROOT/name/case/'fit'
    if (folder/'selection.json').exists():
        return
    p = copy.deepcopy(read(Path(study['prior_sources'][case])/'protocol.json'))
    p.update(rank=spec['rank'], steps=spec['steps'], recipes=['fit'],
             optimizer='Adam with cosine decay', initialization='near-uniform native Tucker',
             tuning_protocol=str(protocol_path()), tuning_protocol_sha256=digest(protocol_path()))
    parent = spec.get('parent')
    if parent:
        if digest(parent) != spec['parent_sha256']:
            raise RuntimeError('Refinement parent changed')
        initial = load_tucker(parent)
        initial = expand_tucker_rank(initial, spec['rank'])
        p.update(initialization='observation-selected native Tucker checkpoint', parent=parent,
                 parent_sha256=spec['parent_sha256'])
    else:
        initial = init_tucker_constant(p['domain'], rank=p['rank'], modes=p['modes'])
    if digest(p['data']) != p['data_sha256']:
        raise RuntimeError('Observations changed')
    with np.load(p['data']) as data:
        train, valid = data['train'], data['valid']
    noisy = spec['noise'] == 'mixture'
    forward = None
    if noisy:
        obs = np.concatenate([train, valid])
        box = np.stack([obs.min(axis=0)-.02, obs.max(axis=0)+.02], axis=1)
        forward = TuckerConvolution.from_tucker(initial, load_tt(p['kernel'])).prepare(box)
    problem = TuckerTrainingProblem(initial, forward=forward, strength=spec['strength'], objective='logpen',
                                    marginal_strength=spec.get('marginal_strength', 0.))
    schedule = [(0, 13), (spec['steps']//4, 25), (spec['steps']//2, 49)] if noisy and not parent else None
    out.mkdir(parents=True, exist_ok=True)
    write_record(folder/'protocol.json', p)
    config = p | spec | {'objective': 'logpen', 'microbatches': 4 if noisy else 1,
                            'core_entries': int(initial.core.size),
                            'parameter_count': int(initial.core.size+sum(f.core.coeffs.size for f in initial.factors))}
    write_record(out/'config.json', config)
    if not (out/'result.json').exists():
        problem.fit_adam(train, valid, out, steps=spec['steps'], batch_size=spec['batch_size'],
                         learning_rate=spec['learning_rate'], checkpoint_every=spec['steps']//20,
                         validation_batch_size=16 if noisy else 256, mode_schedule=schedule,
                         selection_start=spec['steps']//2 if schedule else 0, microbatches=config['microbatches'])
    freeze(folder)


def refinement_specs():
    study = setup()
    path = ROOT/'refinements.json'
    if path.exists():
        return read(path)
    specs = {}
    for noise in ['free', 'mixture']:
        case = f'swirl4_{noise}'
        folders = [Path(study['prior_sources'][case])]
        folders += [ROOT/name/case for name, spec in study['candidates'].items() if spec['noise'] == noise]
        def score(folder):
            s = read(folder/'selection.json')
            return read(folder/s['selected']/'result.json')['best_nll']
        source = min(folders, key=score)
        s = read(source/'selection.json')
        config = read(source/s['selected']/'config.json')
        parent = source/s['selected']/'fit.npz'
        common = {'noise': noise, 'rank': config['rank'], 'strength': config['strength'],
                      'parent': str(parent), 'parent_sha256': digest(parent)}
        if noise == 'free':
            specs['clean_refine'] = common | study['refinement']['clean']
        else:
            recipe = dict(study['refinement']['noisy'])
            for marginal in recipe.pop('marginal_strengths'):
                specs[f'noisy_refine_m{marginal:g}'] = common | recipe | {'marginal_strength': marginal}
    write_record(path, specs)
    return specs


def select():
    import jax

    from ftde import TuckerTrainingProblem, load_tucker
    from ftde.noisy_model_selection import select_by_paired_noisy_nll
    study = setup()
    path = ROOT/'selection.json'
    if path.exists():
        frozen = read(path)
        if (frozen['tuning_protocol_sha256'] != digest(protocol_path()) or
                frozen['refinements_sha256'] != digest(ROOT/'refinements.json')):
            raise RuntimeError('Existing selection uses a different tuning protocol')
        for candidates in frozen['candidates'].values():
            for row in candidates:
                if digest(Path(row['folder'])/'selection.json') != row['selection_sha256']:
                    raise RuntimeError('Frozen candidate selection changed')
        return
    prior = study['prior_selection']
    if digest(prior['path']) != prior['sha256']:
        raise RuntimeError('Prior selection changed')
    selection = read(prior['path'])
    for name, spec in (study['candidates'] | refinement_specs()).items():
        case = f'swirl4_{spec["noise"]}'
        folder = ROOT/name/case
        s = read(folder/'selection.json')
        r = read(folder/s['selected']/'result.json')
        selection['candidates'][case].append({'folder': str(folder), 'rank': spec['rank'], 'steps': r['steps'],
            'objective': r['objective'], 'validation_nll': r['best_nll'], 'selection_sha256': digest(folder/'selection.json')})
    for case in ['swirl4_free', 'swirl4_mixture']:
        selection['selected'][case] = min(selection['candidates'][case], key=lambda r: r['validation_nll'])['folder']
    case = 'swirl4_mixture'
    selection['minimum_nll_control'] = {case: selection['selected'][case]}
    logs, roughness = [], []
    for row in selection['candidates'][case]:
        folder = Path(row['folder'])
        fit = folder/read(folder/'selection.json')['selected']
        logs.append(np.load(fit/'validation.npy'))
        problem = TuckerTrainingProblem(load_tucker(fit/'fit.npz'), strength=0., marginal_strength=1.)
        roughness.append(float(jax.jit(problem.marginal_penalty)(problem.params)))
        jax.clear_caches()
    chosen, details = select_by_paired_noisy_nll(np.stack(logs), roughness)
    selection['selected'][case] = selection['candidates'][case][chosen]['folder']
    selection['selection_rules'] = {case: {'rule': 'paired_one_standard_error', 'details': details}}
    selection['selection'] = 'minimum held-out NLL, except noisy 4D uses TT-matched paired one-SE marginal-smoothness selection'
    selection['prior_selection'] = {'path': prior['path'], 'sha256': prior['sha256']}
    selection['tuning_protocol'] = str(protocol_path())
    selection['tuning_protocol_sha256'] = digest(protocol_path())
    selection['refinements_sha256'] = digest(ROOT/'refinements.json')
    write_record(path, selection)
    print('FROZEN', selection['selected'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name')
    args = parser.parse_args()
    study = setup()
    if args.name:
        fit(args.name)
        return
    (ROOT/'logs').mkdir(exist_ok=True)
    for stage in [lambda: study['candidates'], refinement_specs]:
        for name in stage():
            print('RUN', name, flush=True)
            with (ROOT/'logs'/f'{name}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', __file__, '--name', name], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    select()


if __name__ == '__main__':
    main()

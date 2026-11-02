"""Freeze cross-study choices, then evaluate all three noisy follow-ups."""
import hashlib

import evaluate_marginal_regularization as evaluation
import numpy as np
import plot_marginal_regularization as plots
from run_calibrated_marginals import ROOT as CALIBRATED_ROOT
from run_marginal_regularization import CASES_RUN, read
from run_marginal_regularization import ROOT as REGULARIZATION_ROOT
from run_smooth_initialization import ROOT as INITIALIZATION_ROOT

from ftde.direct_training import write_record
from ftde.noisy_model_selection import select_by_paired_noisy_nll

ROOTS = {'regularization': REGULARIZATION_ROOT, 'initialization': INITIALIZATION_ROOT,
         'calibrated': CALIBRATED_ROOT}


def freeze_overall():
    path = CALIBRATED_ROOT/'overall_selection.json'
    sources = {}
    for root in ROOTS.values():
        read(root/'frozen.json')
    for root in ROOTS.values():
        evaluation.ROOT = root
        evaluation.freeze_smooth_choices()
        for p in [root/'frozen.json', root/'smooth_selection.json']:
            sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    if path.exists():
        assert read(path)['sources'] == sources
        return
    cases = {}
    for case in CASES_RUN:
        models, logs, roughness, marginal_candidates = {}, [], [], []
        for study, root in ROOTS.items():
            frozen = read(root/'frozen.json')['cases'][case]
            smooth = read(root/'smooth_selection.json')['cases'][case]
            rough = dict(zip(smooth['names'], [r['roughness'] for r in smooth['detail']['candidates']], strict=True))
            for name, info in frozen['models'].items():
                label = study+':'+name
                if info['config'].get('marginal_strength', 0.) > 0 and info['result']['best_step'] > 0:
                    marginal_candidates.append(label)
                models[label] = {'study': study, 'name': name, 'path': info['path'],
                                 'validation_nll': info['validation_nll'], 'marginal_curvature': rough[name]}
                from pathlib import Path
                logs.append(evaluation.validation_logs(Path(info['path']).parent))
                roughness.append(rough[name])
        clusters = np.arange(len(logs[0])) % 200 if case.startswith('gl') else None
        chosen, detail = select_by_paired_noisy_nll(np.stack(logs), roughness, cluster_ids=clusters)
        cases[case] = {'models': models, 'primary': min(models, key=lambda n: models[n]['validation_nll']),
                       'smooth': list(models)[chosen], 'smooth_selection': detail,
                       'best_marginal': min(marginal_candidates, key=lambda n: models[n]['validation_nll'])}
    write_record(path, {'cases': cases, 'sources': sources, 'truth_evaluation_not_yet_run': True})
    print('OVERALL FROZEN', {c: (v['primary'], v['smooth']) for c, v in cases.items()}, flush=True)


if __name__ == '__main__':
    # All studies must be finished before any new latent truth evaluation.
    freeze_overall()
    for root in ROOTS.values():
        evaluation.ROOT = root
        evaluation.evaluate()
    plots.ROOT = REGULARIZATION_ROOT
    plots.plot()
    for root, stem in [(INITIALIZATION_ROOT, 'smooth_initialization'),
                       (CALIBRATED_ROOT, 'calibrated_marginals')]:
        plots.ROOT, plots.STEM = root, stem
        def models(case, root=root):
            info = read(root/'frozen.json')['cases'][case]
            smooth = read(root/'smooth_selection.json')['cases'][case]['selected']
            extra = 'lowpass_seed0' if root == INITIALIZATION_ROOT else 'marginal_moderate'
            return list(dict.fromkeys(['original', info['selected'], smooth, extra]))
        plots.chosen_models = models
        plots.plot()
        if root == INITIALIZATION_ROOT:
            plots.plot_initialization_training()

    from summarize_noisy_followup import summarize
    summarize()

"""Freeze likelihood and kernel-score selections before evaluating any truth."""
import argparse
import hashlib
import json

import numpy as np
from objective_study import setup

from ftde.direct_training import write_record
from ftde.kernel_score import draw_frequencies
from ftde.noisy_model_selection import select_by_paired_noisy_nll


def freeze(case):
    folder, _, train, valid, _, _, _ = setup(case)
    paths = sorted(p for p in folder.iterdir() if (p/'result.json').exists())
    frequencies = draw_frequencies(train, 512, 2702)
    names, nlls, scores, complexity, files = [], [], [], [], {}
    for path in paths:
        for suffix, model_name, validation_name in [('nll', 'fit.npz', 'validation.npz'),
                                                     ('kernel', 'kernel_fit.npz', 'kernel_validation.npz')]:
            names.append(f'{path.name}/{suffix}')
            with np.load(path/model_name) as data:
                complexity.append(sum(data[k].size for k in data.files if k.startswith('signal_')))
            with np.load(path/validation_name) as data:
                nlls.append(data['logs'])
                phi = data['characteristic']
            score = []
            for start in range(0, len(valid), 1024):
                phase = np.exp(1j*valid[start:start+1024]@frequencies.T)
                score.append(np.mean(np.abs(phi)**2)-2*np.real(phase@np.conj(phi))/len(phi))
            scores.append(np.concatenate(score))
            for name in [model_name, validation_name, 'result.json', 'config.json']:
                p = path/name
                files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    cluster_ids = np.arange(len(valid)) % 200 if case.startswith('gl') else None
    best_nll, nll_detail = select_by_paired_noisy_nll(np.stack(nlls), complexity, cluster_ids=cluster_ids)
    best_kernel, kernel_detail = select_by_paired_noisy_nll(-np.stack(scores), complexity, cluster_ids=cluster_ids)
    # The helper names its generic loss fields NLL; record that kernel fields
    # refer to the proper kernel score, not log densities.
    record = {'selected_nll': names[best_nll], 'selected_kernel': names[best_kernel], 'names': names,
              'nll_selection': nll_detail, 'kernel_selection': kernel_detail,
              'kernel_selection_field_units': 'kernel score (helper uses generic NLL field names)',
              'parameter_counts': complexity, 'uses_truth': False,
              'rule': 'paired one-SE then fewest actual coefficients; GL SE clusters by MCMC chain',
              'files': files}
    if (folder/'frozen.json').exists():
        assert json.loads((folder/'frozen.json').read_text()) == record
    write_record(folder/'frozen.json', record)
    print(case, record['selected_nll'], record['selected_kernel'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    for case in parser.parse_args().cases:
        freeze(case)

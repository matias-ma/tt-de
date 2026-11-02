"""Summarize frozen-kernel fits and matched optimization-seed checks."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    records, configurations = {}, {}
    for path in sorted(args.root.glob('*/experiment.json')):
        if not (path.parent/'result.json').exists():
            continue
        name = path.parent.name
        config = json.loads(path.read_text())
        record = json.loads((path.parent/'result.json').read_text())
        if 'common_h2' not in record:
            # Signed-density candidates have a separate full-joint clipping audit.
            continue
        configurations[name] = config
        records[name] = {key: record[key] for key in ['best_nll', 'best_step', 'common_h2', 'seconds', 'quadrature_log_error']}
        records[name]['configuration'] = config
    pairs = []
    keys = ['base', 'source_sha256', 'kernel_sha256', 'rank', 'modes', 'order', 'strength', 'steps', 'lr', 'preconditioning']
    names = list(records)
    for index, first in enumerate(names):
        for second in names[index+1:]:
            a, b = configurations[first].copy(), configurations[second].copy()
            a.setdefault('preconditioning', 'l2')
            b.setdefault('preconditioning', 'l2')
            if a['seed'] == b['seed'] or any(a[k] != b[k] for k in keys):
                continue
            loss_delta = np.load(args.root/first/'validation.npy')-np.load(args.root/second/'validation.npy')
            row = {'first': first, 'second': second, 'second_minus_first_nll': float(loss_delta.mean()),
                   'paired_se': float(loss_delta.std(ddof=1)/np.sqrt(len(loss_delta))), 'marginals': {}}
            for label in ['01', '23']:
                aa = np.load(args.root/first/f'marginal_{label}.npz')
                bb = np.load(args.root/second/f'marginal_{label}.npz')
                np.testing.assert_array_equal(aa['xs'], bb['xs'])
                np.testing.assert_array_equal(aa['ys'], bb['ys'])
                area = np.diff(aa['xs'])[0]*np.diff(aa['ys'])[0]
                row['marginals'][label] = {'l1_between_seeds': float(np.abs(aa['after']-bb['after']).sum()*area),
                    'first_l1_change_from_previous': float(np.abs(aa['after']-aa['before']).sum()*area),
                    'second_l1_change_from_previous': float(np.abs(bb['after']-bb['before']).sum()*area)}
            pairs.append(row)
    summary = {'fits': records, 'matched_seed_checks': pairs,
               'stability_scope': 'Different minibatch and rank-perturbation seeds from the same frozen warm start and observations; not independent data or bootstrap replications.'}
    (args.root/'numerical_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'completed_fits': len(records), 'matched_seed_checks': pairs,
                      'largest_quadrature_log_error': max(r['quadrature_log_error'] for r in records.values())}, indent=2))


if __name__ == '__main__':
    main()

"""Post-selection full-dimensional GL diagnostics for the refreshed gallery."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from evaluate_gl_l2 import integrated_terms, relative_l2_from_terms
from refresh_density_fits import read_model

from ftde.ginzburg_landau import GL_BETA, ginzburg_landau_log_normalizer


def evaluate_case(case):
    settings = json.loads((case/'settings.json').read_text())
    if settings['family'] != 'gl':
        return
    fingerprint = hashlib.sha256((case/'selected.npz').read_bytes()).hexdigest()
    if (case/'integrated_l2.json').exists():
        previous = json.loads((case/'integrated_l2.json').read_text())
        if previous.get('checkpoint_sha256') == fingerprint:
            return
    # No fitting or model selection occurs in this evaluator.
    model = read_model(case/'selected.npz')
    dim = settings['dim']
    log_z = ginzburg_landau_log_normalizer(dim)
    p2 = float(np.exp(ginzburg_landau_log_normalizer(dim, beta=2*GL_BETA)-2*log_z))
    order = max(64, 2*max(c.coeffs.shape[-1] for c in model.cores)-1)
    terms, orders = [], []
    for n in (order, 2*order, 4*order, 8*order):
        terms.append(tuple(map(float, jax.jit(
            lambda n=n, model=model, log_z=log_z: integrated_terms(model, n, log_z))())))
        orders.append(n)
        if len(terms) > 1:
            refinement = max(abs(a-b) for a, b in zip(terms[-2], terms[-1], strict=True))/p2
            if refinement < 1e-7:
                break
    if refinement > 1e-7:
        raise FloatingPointError(f'{case}: integrated error quadrature needs refinement: {refinement}')
    q2, pq = terms[-1]
    record = {'checkpoint_sha256': fingerprint, 'relative_l2': relative_l2_from_terms(q2, pq, p2),
              'quadrature_relative_change': refinement, 'orders': orders,
              'integral_estimate_squared': q2, 'integral_truth_times_estimate': pq,
              'integral_truth_squared': p2, 'metric': 'Lebesgue L2 of the latent full-dimensional density'}
    (case/'integrated_l2.json').write_text(json.dumps(record, indent=2)+'\n')
    print(case.name, json.dumps(record), flush=True)
    jax.clear_caches()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', type=Path)
    args = parser.parse_args()
    for case in args.cases:
        evaluate_case(case)


if __name__ == '__main__':
    main()

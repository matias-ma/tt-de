"""Fit polynomial log-density marginal references using noisy samples only.

Finite-dimensional exponential families are a sieve, not an assumption about
the GL formula. Degree and optional sharing across standardized boxes are
selected from noisy marginal likelihood, with paired chain-cluster errors.
The saved reference is only used as a soft prior for the final single TT.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark_noisy_recovery import load_model
from gl_gpu_noisy import _noniso_covariance
from scipy.special import logsumexp, roots_legendre

from ftde.ginzburg_landau import sample_ginzburg_landau
from ftde.marginal_reference import fit_noisy_marginal_reference
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, default=Path('artifacts/gl_rsi_trial/paired_setup.npz'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/noisy_reference'))
    parser.add_argument('--degrees', nargs='+', type=int, default=[2, 4, 6, 8])
    parser.add_argument('--kernel-marginals', action='store_true', help='use the prescribed TT noise law without Gaussian parameters')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = np.load(args.setup)
    domain = saved['domain']
    d = len(domain)
    covariance = _noniso_covariance(d, 1.5)
    std = np.sqrt(np.diag(covariance))
    noise = load_model(saved, 'kernel', saved['kernel_domain']) if args.kernel_marginals else None
    marginals = noise_marginals_from_tt(noise) if noise is not None else None

    def observations(seed, count):
        return np.asarray(sample_ginzburg_landau(seed, count, d))+np.random.default_rng(seed).multivariate_normal(
            np.zeros(d), covariance, count)

    train, valid = observations(0, 100000), observations(2, 2048)
    np.testing.assert_allclose(valid[:256], saved['probe'], atol=1e-14)
    records, coefficients, logs = [], [], []
    for degree in args.degrees:
        for shared in (True, False):
            noise_arguments = {'noise_tt': noise} if noise is not None else {'std': std}
            c, score, record = fit_noisy_marginal_reference(train, valid, domain, degree=degree, shared=shared, **noise_arguments)
            coefficients.append(c)
            records.append(record)
            logs.append(score)
            print(record, flush=True)
            np.savez(args.output/f'degree{degree}_{"shared" if shared else "separate"}.npz',
                     domain=domain, log_coeffs=c, validation=score)
    # For the low-dimensional pilot, prefer fewer parameters among
    # statistically indistinguishable noisy marginal likelihoods.
    chosen, detail = select_by_paired_noisy_nll(np.stack(logs),
        [r['degree']*(1 if r['shared'] else d) for r in records], cluster_ids=np.arange(len(valid)) % 200)
    # Check the selected signal quadrature by recomputing its continuous
    # validation likelihood at twice the order (not a histogram-bias check).
    c = coefficients[chosen]
    t, w = roots_legendre(320)
    refined = []
    for k, (a, b) in enumerate(domain):
        lm = np.polynomial.chebyshev.chebval(t, c[k])+np.log(w)
        if marginals is None:
            nodes = a+(b-a)*(t+1)/2
            lm -= logsumexp(lm)
            lk = -.5*((valid[:, k, None]-nodes)/std[k])**2-np.log(std[k]*np.sqrt(2*np.pi))
            refined.append(logsumexp(lk+lm, axis=1))
        else:
            xx, ww = marginals[k].signal_quadrature(valid[:, k], (a, b), 320)
            log_rho = np.polynomial.chebyshev.chebval(2*(xx-a)/(b-a)-1, c[k])
            weights = ww*marginals[k].pdf(valid[:, k, None]-xx)
            log_weights = np.log(weights, where=weights > 0, out=np.full_like(weights, -np.inf))
            refined.append(logsumexp(log_rho+log_weights, axis=1)-logsumexp(lm)-np.log((b-a)/2))
    error = float(np.max(np.abs(np.mean(refined, axis=0)-logs[chosen])))
    if error > 1e-8:
        raise FloatingPointError(f'Reference quadrature error {error}')
    np.savez(args.output/'selected.npz', domain=domain, log_coeffs=c)
    (args.output/'selection.json').write_text(json.dumps({'selected_index': chosen, 'records': records,
        'selection': detail, 'quadrature_log_error': error,
        'kernel_marginals': args.kernel_marginals,
        'selection_metric': 'mean univariate noisy log density; parameter count as complexity'}, indent=2)+'\n')
    print('SELECTED reference', records[chosen], 'quadrature discrepancy', error, flush=True)


if __name__ == '__main__':
    main()

"""Independent observed-data test scores after gallery selections are frozen.

No model is fitted or selected here. Test NLL and paired uncertainty help
interpret differences between the frozen candidates' latent errors.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from evaluate_noise_diagnosis import paired_se
from plot_dimension_gallery import verify
from run_dimension_gallery import ROOT, digest, read

from ftde.direct_training import load_tt, write_record
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import log_density_tt


def test_observations(protocol):
    dim, n = protocol['dim'], 8192
    if protocol['family'] == 'swirl':
        from ftde.swirl import sample_swirl
        test = sample_swirl(852731, n, dim)
    else:
        from ftde.ginzburg_landau import sample_ginzburg_landau
        test = np.asarray(sample_ginzburg_landau(852731, n, dim))
    if protocol['noise'] != 'free':
        rng = np.random.default_rng(852732)
        weights, cov = np.array(protocol['noise_weights']), np.array(protocol['noise_covariances'])
        labels = rng.choice(len(weights), n, p=weights)
        test = test+np.einsum('nij,nj->ni', np.linalg.cholesky(cov)[labels], rng.normal(size=test.shape))
    return test


def run(case):
    folder = ROOT/case
    frozen = verify(folder)
    protocol = read(folder/'protocol.json')
    if (folder/'test_scores.json').exists():
        previous = read(folder/'test_scores.json')
        if previous['selection_sha256'] != digest(folder/'selection.json'):
            raise RuntimeError('Selection changed since test evaluation')
        return
    test = test_observations(protocol)
    n = len(test)
    if protocol['noise'] != 'free':
        template = load_tt(folder/frozen['selected']/'fit.npz')
        box = np.stack([test.min(0)-.02, test.max(0)+.02], axis=1)
        forward = TTConvolution.from_tt(template, load_tt(protocol['kernel'])).prepare(box, tolerance=1e-14)
        score = jax.jit(lambda model, x: forward.log_density(model, x))
    else:
        score = jax.jit(log_density_tt)
    logs = {}
    for name in frozen['names']:
        model = load_tt(folder/name/'fit.npz')
        values = np.concatenate([np.asarray(score(model, jnp.asarray(test[start:start+8]))) for start in range(0, n, 8)])
        if not np.isfinite(values).all():
            raise FloatingPointError(f'Nonfinite independent test likelihood: {case}/{name}')
        logs[name] = values
    reference = frozen['selected']
    results = {}
    for name, values in logs.items():
        delta = logs[reference]-values
        results[name] = {'test_nll': -float(values.mean()), 'test_nll_minus_selected': float(delta.mean()),
                         'paired_se': paired_se(delta, case)}
    np.savez_compressed(folder/'test_logs.npz', **logs)
    record = {'selection_sha256': digest(folder/'selection.json'), 'selected': reference, 'models': results,
              'test_count': n, 'signal_seed': 852731, 'noise_seed': 852732, 'used_for_selection': False}
    write_record(folder/'test_scores.json', record)
    print('TEST', case, record, flush=True)


def truth_gap(case):
    """Post-selection full observed KL diagnostic, never a selection score."""
    from ftde.swirl import swirl_convolved_log_density
    folder = ROOT/case
    frozen = verify(folder)
    protocol = read(folder/'protocol.json')
    if protocol['family'] != 'swirl' or protocol['noise'] == 'free':
        raise ValueError('This diagnostic requires noisy Swirl')
    scores = read(folder/'test_scores.json')
    if scores['selection_sha256'] != digest(folder/'selection.json'):
        raise RuntimeError('Independent test scores belong to another selection')
    test = test_observations(protocol)
    kwargs = {'covariances': protocol['noise_covariances'], 'weights': protocol['noise_weights'], 'chunk': 32}
    truth = swirl_convolved_log_density(test, order=512, **kwargs)
    refined = swirl_convolved_log_density(test[:64], order=1024, **kwargs)
    difference = float(np.max(np.abs(truth[:64]-refined)))
    if difference > 1e-8 or not np.isfinite(truth).all():
        raise FloatingPointError('Observed-truth path quadrature did not converge')
    estimates = {}
    with np.load(folder/'test_logs.npz') as logs:
        for name in frozen['names']:
            delta = truth-logs[name]
            estimates[name] = {'observed_kl_estimate': float(delta.mean()),
                               'standard_error': float(delta.std(ddof=1)/np.sqrt(len(delta)))}
    np.save(folder/'test_truth_logs.npy', truth)
    record = {'selection_sha256': digest(folder/'selection.json'), 'selected': frozen['selected'],
              'models': estimates, 'quadrature_max_log_difference': difference,
              'true_observed_test_nll': -float(truth.mean()), 'test_count': len(test),
              'used_for_selection': False, 'metric': 'Monte Carlo KL(true observed || fitted observed)'}
    write_record(folder/'observed_truth_gap.json', record)
    print('OBSERVED KL', case, record, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--truth-gap', action='store_true')
    args = parser.parse_args()
    ROOT = args.root
    run(args.case)
    if args.truth_gap:
        truth_gap(args.case)

"""Check a selected RSI fit on all noisy validation points, without clean labels.

The setup NPZ comes from benchmark_gl_convolution.py --rsi-setup-only with
the same example settings. Missing historical pivot rows are reconstructed
from that initialization, never selected anew from the fitted model.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.computations import convolve_tt
from chebttge.cores import ChebyshevCoeffCore
from gl_gpu_noisy import NOISE_STDS, NOISE_WEIGHTS, _noniso_covariance

from ftde.cheb_product import square_tt
from ftde.ginzburg_landau import sample_ginzburg_landau
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import squared_integral_tt


def load_tt(saved, prefix, domain):
    return ChebTT(cores=tuple(
        ChebyshevCoeffCore(jnp.asarray(saved[f'{prefix}_{k}']), tuple(bounds))
        for k, bounds in enumerate(domain)), domain=jnp.asarray(domain))


def check(record_path, setup_path):
    record = json.loads(record_path.read_text())
    if record['convolution_backend'] != 'rsi':
        raise ValueError('Expected a fit trained with RSI convolution')
    with np.load(setup_path) as saved:
        init = load_tt(saved, 'signal', saved['domain'])
        kernel = load_tt(saved, 'kernel', saved['kernel_domain'])
        output_domain = saved['observation_domain']
        initial_probe = saved['probe']
    with np.load(record_path.with_suffix('.npz')) as saved:
        np.testing.assert_allclose(saved['domain'], init.domain, rtol=0, atol=1e-12)
        fitted = load_tt(saved, 'deconvolved', saved['domain'])

    dim, scale = record['dim'], record.get('noise_scale', 1.)
    valid = np.asarray(sample_ginzburg_landau(seed=2, n=2048, d=dim)).copy()
    rng = np.random.default_rng(2)
    if record['noise'] == 'paired':
        valid += rng.multivariate_normal(np.zeros(dim), _noniso_covariance(dim, scale), size=len(valid))
    elif record['noise'] == 'mixture':
        component = rng.choice(2, size=len(valid), p=np.asarray(NOISE_WEIGHTS))
        valid += rng.normal(size=valid.shape) * np.asarray(NOISE_STDS)[component, None] * scale
    else:
        raise ValueError('Expected a prescribed paired or mixture noise kernel')
    np.testing.assert_allclose(initial_probe, valid[:len(initial_probe)], rtol=0, atol=1e-12)

    setup = record['rsi_setup']
    sketch_size = setup.get('sketch_size', record.get('rsi_sketch_size'))
    rows = setup.get('pivot_rows')
    if rows is None:
        print('Reconstructing original interpolation rows...', flush=True)
        initial = convolve_tt(square_tt(init), kernel, output_domain=output_domain,
                              max_rank=record['rsi_max_rank'], sketch_size=sketch_size,
                              validation_samples=64)
        if list(initial.ranks) != list(setup['ranks']):
            raise ValueError('Reconstructed initial ranks differ from the training setup')
        np.testing.assert_allclose(initial.diagnostics['validation_relative_rms_error'],
                                   setup['validation_relative_rms_error'], rtol=1e-6, atol=1e-14)
        rows = [stage['pivot_rows'] for stage in initial.diagnostics['stages']]
        sketch_size = initial.diagnostics['sketch_size']
    print('Checking the frozen training interpolation on 2,048 observations...', flush=True)
    q = convolve_tt(square_tt(fitted), kernel, output_domain=output_domain,
                    max_rank=record['rsi_max_rank'], sketch_size=sketch_size,
                    pivot_rows=rows, validation_samples=0)
    z = squared_integral_tt(fitted)
    direct = TTConvolution.from_tt(fitted, kernel).prepare(output_domain, tolerance=1e-14)
    approximate = jax.jit(lambda y: q(y) / z)
    reference = jax.jit(lambda y: direct.log_density(fitted, y))
    values = np.concatenate([np.asarray(approximate(jnp.asarray(batch)))
                             for batch in np.array_split(valid, 8)])
    exact_log = np.concatenate([np.asarray(reference(jnp.asarray(batch)))
                                for batch in np.array_split(valid, 8)])
    log_values = np.log(np.maximum(values, 1e-300))
    errors = np.abs(log_values - exact_log)
    exact_values = np.exp(exact_log)
    result = {
        'n_observations': len(valid),
        'nonpositive_count': int(np.count_nonzero(values <= 0)),
        'nonfinite_count': int(np.count_nonzero(~np.isfinite(values))),
        'max_abs_log_error': float(np.max(errors)),
        'p99_abs_log_error': float(np.quantile(errors, .99)),
        'relative_density_rms_error': float(np.linalg.norm(values-exact_values) / np.linalg.norm(exact_values)),
        'rsi_nll': float(-log_values.mean()),
        'direct_nll': float(-exact_log.mean()),
    }
    record['checks']['full_validation'] = result
    setup['pivot_rows'], setup['sketch_size'] = rows, sketch_size
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('setup', type=Path)
    args = parser.parse_args()
    check(args.record, args.setup)

"""Compare values, gradients and warm timings on the same arbitrary TT kernel."""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from gl_covariance_common import load_tt

from ftde.tt_convolution import TTConvolution
from ftde.tt_density import rebuild_tt, tt_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/gl_covariance_mixture'))
    args = parser.parse_args()
    data = np.load(args.directory/'data.npz')
    g = load_tt(args.directory/'init_r8_n21.npz')
    u = load_tt(args.directory/'noise.npz')
    base = TTConvolution.from_tt(g, u)
    prepared = base.prepare(data['observation_domain'], tolerance=1e-14)
    params = tt_params(g)
    x = jnp.asarray(data['valid'][:128])
    results, gradients = {}, {}
    for name, op in [('direct', base), ('prepared_dense', replace(prepared, moment_groups=None)), ('prepared', prepared)]:
        evaluate = jax.jit(jax.value_and_grad(lambda p, x, op=op: -op.log_density(rebuild_tt(g, p), x).mean()))
        value, grad = evaluate(params, x)
        jax.block_until_ready(grad)
        timings = []
        for _ in range(10):
            start = time.perf_counter()
            _, gradient = evaluate(params, x)
            jax.block_until_ready(gradient)
            timings.append(time.perf_counter()-start)
        results[name] = {'nll': float(value), 'median_gradient_seconds': float(np.median(timings)), 'timings': timings}
        gradients[name] = grad
        print(name, results[name], flush=True)
    results['relative_gradient_difference'] = float(np.sqrt(
        sum(float(jnp.sum((a-b)**2)) for a, b in zip(gradients['direct'], gradients['prepared'], strict=True)) /
        sum(float(jnp.sum(a*a)) for a in gradients['direct'])))
    results['speed_ratio'] = results['direct']['median_gradient_seconds']/results['prepared']['median_gradient_seconds']
    results['structural_zero_speed_ratio'] = results['prepared_dense']['median_gradient_seconds']/results['prepared']['median_gradient_seconds']
    results['timing_caveat'] = 'Shared GPU allocation; timings include contention from other work.'
    np.savez(args.directory/'operator_benchmark.npz',
             names=np.asarray(['direct', 'prepared_dense', 'prepared']),
             nll=np.asarray([results[name]['nll'] for name in ('direct', 'prepared_dense', 'prepared')]),
             median_gradient_seconds=np.asarray([results[name]['median_gradient_seconds']
                                                 for name in ('direct', 'prepared_dense', 'prepared')]),
             timings=np.asarray([results[name]['timings'] for name in ('direct', 'prepared_dense', 'prepared')]),
             relative_gradient_difference=results['relative_gradient_difference'],
             speed_ratio=results['speed_ratio'],
             structural_zero_speed_ratio=results['structural_zero_speed_ratio'])


if __name__ == '__main__':
    main()

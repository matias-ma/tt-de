"""Compare explicit Tucker squaring/convolution with direct exact contraction."""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebtuckerge.chebtucker import ChebTucker
from chebtuckerge.factors import ChebyshevCoeffFactor

from ftde import TuckerConvolution, init_tucker_constant, log_density_noisy_tucker
from ftde.direct_training import write_record
from ftde.tucker_density import rebuild_tucker, tucker_params


def run(dimension, rank):
    g = init_tucker_constant([[-1., 1.]]*dimension, rank=rank, modes=9, perturbation=.1)
    factors = [ChebyshevCoeffFactor(jnp.array([[1/6, 0., .01]]), (-3., 3.)) for _ in range(dimension)]
    u = ChebTucker(jnp.ones((1,)*dimension), factors, jnp.array([[-3., 3.]]*dimension))
    # Normalize the prescribed polynomial kernel exactly.
    u = ChebTucker(u.core/(.98**dimension), factors, u.domain)
    y = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, (16, dimension)))
    params = tucker_params(g)
    operator = TuckerConvolution.from_tucker(g, u).prepare()
    records, values, gradients = {}, {}, {}
    for name, op in [('explicit', None), ('direct', operator)]:
        loss = lambda p, op=op: jnp.mean(log_density_noisy_tucker(rebuild_tucker(g, p), u, y, forward_operator=op))
        f = jax.jit(jax.value_and_grad(loss))
        start = time.perf_counter()
        value, gradient = f(params)
        jax.block_until_ready(gradient)
        compile_seconds = time.perf_counter()-start
        start = time.perf_counter()
        for _ in range(20):
            value, gradient = f(params)
            jax.block_until_ready(gradient)
        records[name] = {'compile_and_first_seconds': compile_seconds, 'value_and_gradient_seconds': (time.perf_counter()-start)/20}
        values[name], gradients[name] = float(value), jax.tree.leaves(gradient)
    difference = max(float(np.max(np.abs(a-b))) for a, b in zip(gradients['explicit'], gradients['direct'], strict=True))
    if abs(values['explicit']-values['direct']) > 1e-10 or difference > 1e-9:
        raise AssertionError('Convolution value/gradient mismatch')
    return {'dimension': dimension, 'rank': rank, 'amplitude_core_entries': rank**dimension,
            'explicit_square_core_entries': rank**(2*dimension), 'value_error': abs(values['explicit']-values['direct']),
            'max_gradient_error': difference, 'timings': records,
            'speedup': records['explicit']['value_and_gradient_seconds']/records['direct']['value_and_gradient_seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/tucker_gallery/convolution_benchmark.json'))
    args = parser.parse_args()
    rows = []
    for d, r in [(3, 2), (4, 3), (4, 5), (4, 8)]:
        rows.append(run(d, r))
        print(rows[-1], flush=True)
        jax.clear_caches()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_record(args.output, {'results': rows, 'precision': 'float64', 'device': str(jax.devices()[0])})

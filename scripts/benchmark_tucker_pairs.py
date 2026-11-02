"""Compare exact sequential and paired contractions for a real noise kernel."""
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from run_tucker_gallery import read

from ftde import TuckerConvolution, TuckerTrainingProblem, init_tucker_constant
from ftde.direct_training import load_tt, write_record


def main():
    p = read('artifacts/tucker_capacity/r16_noisy/swirl4_mixture/protocol.json')
    model = init_tucker_constant(p['domain'], rank=16, modes=49)
    with np.load(p['data']) as data:
        observed = np.concatenate([data['train'], data['valid']])
        points = data['valid'][:16]
    domain = np.stack([observed.min(axis=0)-.02, observed.max(axis=0)+.02], axis=1)
    forward = TuckerConvolution.from_tucker(model, load_tt(p['kernel'])).prepare(domain)
    timings, values, gradients = {}, {}, {}
    for label, paired in [('sequential', False), ('paired', True)]:
        problem = TuckerTrainingProblem(model, forward=replace(forward, pair_axes=paired), objective='logpen')
        features = problem.features(points)
        function = jax.jit(jax.value_and_grad(problem.loss))
        result = function(problem.params, features)
        jax.block_until_ready(result)
        start = time.monotonic()
        for _ in range(20):
            result = function(problem.params, features)
            jax.block_until_ready(result)
        timings[label] = (time.monotonic()-start)/20
        values[label] = float(result[0])
        gradients[label] = [np.asarray(a) for a in jax.tree.leaves(result[1])]
    difference = max(float(np.max(np.abs(a-b))) for a, b in zip(*gradients.values(), strict=True))
    error = abs(values['paired']-values['sequential'])
    if error > 1e-10 or difference > 1e-9:
        raise AssertionError('Paired contraction changed value or gradient')
    record = {'rank': 16, 'dimension': 4, 'modes': 49, 'batch': 16,
              'seconds': timings, 'speedup': timings['sequential']/timings['paired'],
              'absolute_value_error': error, 'maximum_absolute_gradient_error': difference,
              'device': str(jax.devices()[0]), 'kernel': p['kernel'],
              'timing_note': 'Warm value-and-gradient calls; other fits may share this allocated GPU'}
    write_record(Path('artifacts/tucker_capacity/pair_benchmark.json'), record)
    print(record)


if __name__ == '__main__':
    main()

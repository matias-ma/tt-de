"""Real-kernel value/gradient equivalence and isolated separable-pair timings."""
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
    model = init_tucker_constant(p['domain'], rank=24, modes=49)
    with np.load(p['data']) as data:
        observed = np.concatenate([data['train'], data['valid']])
        points = data['valid'][:256]
    domain = np.stack([observed.min(axis=0)-.02, observed.max(axis=0)+.02], axis=1)
    forward = TuckerConvolution.from_tucker(model, load_tt(p['kernel'])).prepare(domain)
    measurements = []
    for batch in [64, 256]:
        gradients, values = [], []
        row = {'batch': batch}
        for label, enabled in [('dense', False), ('separable', True)]:
            problem = TuckerTrainingProblem(model, forward=replace(forward, separable_pairs=enabled), objective='logpen')
            features = problem.features(points[:batch])
            function = jax.jit(jax.value_and_grad(problem.loss)).lower(problem.params, features).compile()
            result = function(problem.params, features)
            jax.block_until_ready(result)
            start = time.monotonic()
            for _ in range(10):
                result = function(problem.params, features)
                jax.block_until_ready(result)
            row[label] = {'seconds': (time.monotonic()-start)/10,
                          'temporary_bytes': function.memory_analysis().temp_size_in_bytes}
            values.append(float(result[0]))
            gradients.append([np.asarray(a) for a in jax.tree.leaves(result[1])])
        row['value_error'] = abs(values[0]-values[1])
        row['gradient_error'] = max(float(np.max(np.abs(a-b))) for a, b in zip(*gradients, strict=True))
        if row['value_error'] > 1e-10 or row['gradient_error'] > 1e-9:
            raise AssertionError('Separable contraction changed the value or gradient')
        row['speedup'] = row['dense']['seconds']/row['separable']['seconds']
        measurements.append(row)
        print(row, flush=True)
    write_record(Path('artifacts/tucker_4d_tuning/separable_benchmark.json'),
                 {'rank': 24, 'dimension': 4, 'device': str(jax.devices()[0]), 'measurements': measurements})


if __name__ == '__main__':
    main()

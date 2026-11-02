"""Measure 6D clean/noisy Tucker value-gradient costs with real observations."""
import argparse
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from run_tucker_gallery import read

from ftde import TuckerConvolution, TuckerTrainingProblem, init_tucker_constant
from ftde.direct_training import load_tt, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--ungrouped', action='store_true')
    args = parser.parse_args()
    p = read('artifacts/tucker_capacity/r6_noisy/swirl6_mixture/protocol.json')
    model = init_tucker_constant(p['domain'], rank=args.rank, modes=p['modes'])
    with np.load(p['data']) as data:
        observations = np.concatenate([data['train'], data['valid']])
        box = np.asarray(p['domain'])
        inside = np.all((data['valid'] >= box[:, 0]) & (data['valid'] <= box[:, 1]), axis=1)
        points = data['valid'][inside][:args.batch]
    domain = np.stack([observations.min(axis=0)-.02, observations.max(axis=0)+.02], axis=1)
    forward = TuckerConvolution.from_tucker(model, load_tt(p['kernel'])).prepare(domain)
    if args.ungrouped and hasattr(forward, 'pair_groups'):
        forward = replace(forward, pair_groups=None)
    record = {'rank': args.rank, 'dimension': 6, 'batch': args.batch, 'core_entries': int(model.core.size),
              'device': jax.devices()[0].device_kind, 'ungrouped': args.ungrouped, 'measurements': {}}
    for label, op in [('clean', None), ('noisy', forward)]:
        problem = TuckerTrainingProblem(model, forward=op, objective='logpen')
        features = problem.features(points)
        start = time.monotonic()
        function = jax.jit(jax.value_and_grad(problem.loss)).lower(problem.params, features).compile()
        compile_seconds = time.monotonic()-start
        jax.block_until_ready(function(problem.params, features))
        start = time.monotonic()
        for _ in range(10):
            result = function(problem.params, features)
            jax.block_until_ready(result)
        memory = function.memory_analysis()
        row = {'seconds': (time.monotonic()-start)/10, 'compile_seconds': compile_seconds,
               'compiled_temporary_bytes': memory.temp_size_in_bytes,
               'compiled_argument_bytes': memory.argument_size_in_bytes,
               'compiled_output_bytes': memory.output_size_in_bytes, 'loss': float(result[0])}
        record['measurements'][label] = row
        print(label, row, flush=True)
    folder = Path('artifacts/tucker_noisy_rank10/profiles')
    folder.mkdir(parents=True, exist_ok=True)
    write_record(folder/f'r{args.rank}_b{args.batch}_{"ungrouped" if args.ungrouped else "grouped"}.json', record)


if __name__ == '__main__':
    main()

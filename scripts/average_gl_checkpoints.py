"""Data-only late-iterate averaging; no GL density or L2 values are read."""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from gl_covariance_common import load_record, load_tt, save_record, save_tt

from ftde.tt_convolution import TTConvolution
from ftde.tt_density import rebuild_tt, tt_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/gl_covariance_mixture'))
    args = parser.parse_args()
    directory = args.directory
    data = np.load(directory/'data.npz')
    noise = load_tt(directory/'noise.npz')
    records = [load_record(p) for p in sorted(directory.glob('*/result.npz'))]
    for record in records:
        if record['naive'] or record.get('averaged', False):
            continue
        name = record['name']
        output = directory/('avg_'+name)
        if (output/'result.npz').exists():
            continue
        steps = [h['step'] for h in record['history'][-5:]]
        models = [load_tt(directory/name/f'step_{step:06d}.npz') for step in steps]
        template = models[-1]
        # Remove positive scalar core gauges before averaging. This does not
        # change any source density; no bond rotation/truncation is performed.
        parameters = [[np.asarray(c)/np.linalg.norm(np.asarray(c)) for c in tt_params(g)] for g in models]
        averaged = tuple(jnp.asarray(np.mean([p[k] for p in parameters], axis=0)) for k in range(len(template.cores)))
        g = rebuild_tt(template, averaged)
        alignment = min(float(np.sum(a*b)) for p in parameters[:-1] for a, b in zip(p, parameters[-1], strict=True))
        op = TTConvolution.from_tt(template, noise).prepare(data['observation_domain'], tolerance=1e-14)
        evaluate = jax.jit(lambda x, op=op, g=g: op.log_density(g, x))
        start = time.monotonic()
        valid = data['valid']
        values = np.concatenate([np.asarray(evaluate(jnp.asarray(valid[j:j+128]))) for j in range(0, len(valid), 128)])
        nll = -float(values.mean())
        if not np.isfinite(nll):
            raise FloatingPointError('Nonfinite averaged-model likelihood')
        output.mkdir(exist_ok=True)
        save_tt(output/'best.npz', g)
        save_tt(output/f"step_{record['steps']:06d}.npz", g)
        result = {**record, 'name': output.name, 'averaged': True, 'source': name,
                  'source_steps': steps, 'minimum_core_cosine_alignment': alignment,
                  'best_nll': nll, 'best_step': record['steps'],
                  'history': [{'step': record['steps'], 'validation_nll': nll, 'seconds': 0.}],
                  'training_seconds': 0., 'averaging_evaluation_seconds': time.monotonic()-start,
                  'checks': {}}
        save_record(output/'result.npz', result)
        print(output.name, nll, 'minimum core alignment', alignment, flush=True)
        jax.clear_caches()


if __name__ == '__main__':
    main()

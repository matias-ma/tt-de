"""Compare frozen old/new tensors on the already frozen fresh swirl test set."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from refine_cryoem_gallery import expand_tt
from refresh_density_fits import read_model

from ftde.tt_convolution import TTConvolution
from ftde.tt_density import rebuild_tt, tt_params

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    previous, current = read_model(args.base/'selected.npz'), read_model(args.fit)
    rank = max(max(previous.ranks), max(current.ranks))
    modes = max(c.coeffs.shape[-1] for m in (previous, current) for c in m.cores)
    previous, current = [expand_tt(m, modes, rank, jitter=0.) for m in (previous, current)]
    data = np.load(args.test)
    x, truth = data['observations'], data['true_log_density']
    domain = np.stack([x.min(0)-.01, x.max(0)+.01], axis=1)
    kernel = read_model(args.base/'kernel.npz')
    op = TTConvolution.from_tt(previous, kernel).prepare(domain, tolerance=1e-14)
    evaluate = jax.jit(lambda params, xs: op.log_density(rebuild_tt(previous, params), xs))
    logs = []
    for model in (previous, current):
        params = tt_params(model)
        scores = np.concatenate([np.asarray(evaluate(params, jnp.asarray(x[s:s+16]))) for s in range(0, len(x), 16)])
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError('Invalid fresh-test likelihood')
        logs.append(scores)
    delta = logs[1]-logs[0]
    result = {'n': len(x), 'previous_nll': float(-logs[0].mean()), 'new_nll': float(-logs[1].mean()),
              'paired_improvement': float(delta.mean()), 'paired_se': float(delta.std(ddof=1)/np.sqrt(len(x))),
              'previous_gap_to_truth': float(np.mean(truth-logs[0])),
              'new_gap_to_truth': float(np.mean(truth-logs[1])), 'used_for_selection': False}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result, flush=True)

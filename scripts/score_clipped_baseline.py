"""Frozen squared-TT comparator on the same validation and fresh test rows."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from refresh_density_fits import read_model
from swirl_fits import observations

from ftde.tt_convolution import TTConvolution

root = Path('artifacts/clipped_tt')
base = Path('artifacts/swirl/swirl_mixture_dim4')
model, kernel = read_model(base/'selected.npz'), read_model(base/'kernel.npz')
valid = np.load(base/'data.npz')['valid']
test = observations(4, 'mixture', 8192, 73129)
all_points = np.concatenate([valid, test])
box = np.stack([all_points.min(0)-.02, all_points.max(0)+.02], axis=1)
op = TTConvolution.from_tt(model, kernel).prepare(box, tolerance=1e-14)
evaluate = jax.jit(lambda x: op.log_density(model, x))
scores = {}
for name, data in [('validation', valid), ('test', test)]:
    logs = np.concatenate([np.asarray(evaluate(jnp.asarray(data[k:k+64]))) for k in range(0, len(data), 64)])
    np.save(root/f'baseline_{name}.npy', logs)
    scores[name+'_nll'] = float(-logs.mean())
(root/'baseline.json').write_text(json.dumps(scores, indent=2)+'\n')
print(scores, flush=True)

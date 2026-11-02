"""Compare saved models on the same untrimmed held-out observations, no labels."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from ftde.gaussian_convolution import GaussianConvolution
from ftde.kde import make_gaussian_kde


def main():
    directory = Path('.scratch/cryoem')
    cache = np.load('cryoem_data/cache/path_noise1_dim10.npz')
    raw, covariance = cache['points'], cache['cov_mean']
    indices = np.random.default_rng(0).permutation(len(raw))
    valid = raw[indices[:len(raw)//5]][:4096]
    rows = []
    for path in sorted(directory.glob('*.json')):
        result = json.loads(path.read_text())
        if not isinstance(result, dict) or result.get('backend') not in ('gaussian', 'spectral'):
            continue
        if result['dim'] != 10 or result['seed'] != 0:
            continue
        saved = np.load(path.with_suffix('.npz'))
        domain = saved['domain']
        rotation = saved['rotation'] if 'rotation' in saved else np.eye(10)
        cores = tuple(ChebyshevCoeffCore(jnp.asarray(saved[f'core_{k}']), tuple(domain[k]))
                      for k in range(10))
        g = ChebTT(cores=cores, domain=jnp.asarray(domain))
        std = np.sqrt(np.diag(rotation.T @ covariance @ rotation))
        op = GaussianConvolution.from_tt(g, std)
        evaluate = jax.jit(lambda x, op=op, g=g: op.log_density(g, x))
        transformed = valid @ rotation
        scores = np.concatenate([np.asarray(evaluate(jnp.asarray(transformed[i:i+256])))
                                 for i in range(0, len(valid), 256)])
        row = {'run': path.stem, 'validation_nll': float(-scores.mean()),
               'nll_standard_error': float(scores.std(ddof=1)/np.sqrt(len(scores))),
               'n_observations': len(scores)}
        rows.append(row)
        print(row, flush=True)
        jax.clear_caches()
    train = jnp.asarray(raw[indices[len(raw)//5:]])
    kde = make_gaussian_kde(train)
    scores = np.log(np.asarray(kde(jnp.asarray(valid))))
    rows.append({'run': 'kde_all_observations', 'validation_nll': float(-scores.mean()),
                 'nll_standard_error': float(scores.std(ddof=1)/np.sqrt(len(scores))),
                 'n_observations': len(scores)})
    (directory/'common_validation.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()

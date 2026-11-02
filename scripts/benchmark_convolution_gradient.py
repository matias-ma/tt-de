"""Time matched, compiled NLL gradients on a saved cryo-EM TT."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from ftde.gaussian_convolution import GaussianConvolution
from ftde.noise_kernel import build_diagonal_gaussian_noise_kernel_tt
from ftde.noisy_tt_density import log_density_noisy_tt
from ftde.tt_density import rebuild_tt, tt_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('.scratch/cryoem/gaussian_r16_n3.npz'))
    parser.add_argument('--output', type=Path, default=Path('.scratch/cryoem/rank16_gradient_timing.json'))
    args = parser.parse_args()
    saved = np.load(args.checkpoint)
    domain = saved['domain']
    g = ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(saved[f'core_{k}']), tuple(domain[k]))
                          for k in range(len(domain))), domain=jnp.asarray(domain))
    data = np.load(f'cryoem_data/cache/path_noise1_dim{len(domain)}.npz')
    rotation = saved['rotation'] if 'rotation' in saved else np.eye(len(domain))
    points = data['points'] @ rotation
    inside = np.all((points >= domain[:, 0]) & (points <= domain[:, 1]), axis=1)
    # Spectral output is only valid on the signal box; do not extrapolate it.
    y = jnp.asarray(points[inside][:512])
    std = jnp.sqrt(jnp.diag(jnp.asarray(rotation.T @ data['cov_mean'] @ rotation)))
    op = GaussianConvolution.from_tt(g, std)
    u = build_diagonal_gaussian_noise_kernel_tt(std**2, g.domain, n_coeffs=(33, 65, 129))
    params = tt_params(g)
    result = {'device': str(jax.devices()[0]), 'jax_version': jax.__version__,
              'batch_size': len(y), 'ranks': list(g.ranks),
              'kernel_n_coeffs': [c.coeffs.shape[-1] for c in u.cores]}
    gradients = {}
    for name, operator in [('gaussian', op), ('spectral', None)]:
        f = jax.jit(jax.value_and_grad(
            lambda p, operator=operator: -jnp.mean(log_density_noisy_tt(
                rebuild_tt(g, p), u, y, gaussian_convolution=operator))))
        tic = time.monotonic()
        first = f(params)
        jax.block_until_ready(first)
        compile_time = time.monotonic() - tic
        tic = time.monotonic()
        for _ in range(5):
            jax.block_until_ready(f(params))
        result[name] = {'gradient_seconds': (time.monotonic()-tic)/5,
                        'compile_and_first_seconds': compile_time, 'nll': float(first[0])}
        gradients[name] = np.concatenate([np.asarray(v).ravel() for v in first[1]])
        print(name, result[name], flush=True)
    result['gradient_relative_l2_difference'] = float(
        np.linalg.norm(gradients['gaussian']-gradients['spectral']) / np.linalg.norm(gradients['gaussian']))
    result['gradient_speedup'] = result['spectral']['gradient_seconds']/result['gaussian']['gradient_seconds']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Encode an existing squared 4D TT density directly, without a signal mixture.

Physical quadrature projection and randomized TT-SVD provide a controlled
change of representation. The source is a previously data-fitted model.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from refresh_density_fits import read_model, save_model
from scipy.special import roots_legendre

from ftde.tt_density import squared_integral_tt


def convert(source, rank, modes, seed=0):
    if len(source.cores) != 4:
        raise ValueError('Dense reference conversion is four-dimensional only')
    order = max(modes, 2*max(c.coeffs.shape[-1] for c in source.cores)-1)
    t, w = roots_legendre(order)
    values, orthogonal, inverse = [], [], []
    for core in source.cores:
        a, b = core.domain
        weights = w*(b-a)/2
        vander = np.polynomial.chebyshev.chebvander(t, modes-1)
        q, r = np.linalg.qr(vander*np.sqrt(weights[:, None]))
        orthogonal.append(jnp.asarray(q))
        inverse.append(jnp.asarray(np.linalg.inv(r.T)))
        values.append(core(jnp.asarray(a+(b-a)*(t+1)/2)))
    left = jnp.einsum('ia,jab->ijb', values[0][:, 0], values[1]).reshape(order**2, -1)
    right = jnp.einsum('kab,lb->akl', values[2], values[3][:, :, 0]).reshape(-1, order**2)
    tensor = ((left@right)**2/squared_integral_tt(source)).reshape((order,)*4)
    for axis, core in enumerate(source.cores):
        shape = [1]*4
        shape[axis] = order
        tensor *= jnp.sqrt(jnp.asarray(w*(core.domain[1]-core.domain[0])/2)).reshape(shape)
    original_norm2 = float(jnp.sum(tensor**2))
    for axis, q in enumerate(orthogonal):
        tensor = jnp.moveaxis(jnp.tensordot(q.T, tensor, axes=(1, axis)), 0, axis)
    projected_norm2 = float(jnp.sum(tensor**2))
    residual, arrays, key = tensor, [], jax.random.PRNGKey(seed)
    left_rank = 1
    for axis in range(3):
        matrix = residual.reshape(left_rank*modes, -1)
        keep = min(rank, *matrix.shape)
        sketch_rank = min(keep+24, *matrix.shape)
        key, sub = jax.random.split(key)
        omega = jax.random.normal(sub, (matrix.shape[1], sketch_rank), dtype=jnp.float64)
        q, _ = jnp.linalg.qr(matrix@omega, mode='reduced')
        for _ in range(3):
            z, _ = jnp.linalg.qr(matrix.T@q, mode='reduced')
            q, _ = jnp.linalg.qr(matrix@z, mode='reduced')
        u, _, _ = jnp.linalg.svd(q.T@matrix, full_matrices=False)
        u = q@u[:, :keep]
        arrays.append(u.reshape(left_rank, modes, keep).transpose(0, 2, 1))
        residual = u.T@matrix
        left_rank = keep
    arrays.append(residual.reshape(left_rank, 1, modes))
    retained_norm2 = float(jnp.sum(residual**2))
    cores = tuple(ChebyshevCoeffCore(c@inv, old.domain)
                  for c, inv, old in zip(arrays, inverse, source.cores, strict=True))
    result = ChebTT(cores=cores, domain=source.domain)
    return result, {'quadrature_order': order, 'modes': modes, 'rank': rank, 'seed': seed,
                    'projection_relative_l2': float(np.sqrt(max(0., 1-projected_norm2/original_norm2))),
                    'total_relative_l2': float(np.sqrt(max(0., 1-retained_norm2/original_norm2))),
                    'method': 'physical projection and randomized TT-SVD of previous squared density'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--modes', type=int, default=65)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, report = convert(read_model(args.source), args.rank, args.modes)
    save_model(args.output, model)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report, flush=True)

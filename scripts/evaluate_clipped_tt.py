"""Post-fit integration convergence, held-out likelihood, and swirl truth checks."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from prescribed_noise import prescribed_noise
from refresh_density_fits import read_model

from ftde.clipped_tt_density import ClippedTTQuadrature4D


def evaluate(base, directory, orders):
    config = json.loads((base/'settings.json').read_text())
    data = np.load(base/'data.npz')
    f = read_model(directory/'fit.npz')
    covs, mix = prescribed_noise(base)
    results = []
    for order in orders:
        op = ClippedTTQuadrature4D.from_tt(f, covs, mix, order=order)
        weighted, mass, negative = jax.jit(op.weighted_grid)(f)
        convolve = jax.jit(op.convolve_weighted)
        noise = jax.jit(op.noise_factors)
        def logs(x, convolve=convolve, weighted=weighted, noise=noise):
            return np.concatenate([np.log(np.asarray(convolve(weighted, noise(jnp.asarray(x[k:k+256])))))
                                   for k in range(0, len(x), 256)])
        validation = logs(data['valid'])
        np.save(directory/f'validation_order{order}.npy', validation)
        record = {'order': order, 'nll': float(-validation.mean()), 'positive_mass': float(mass),
                  'negative_to_positive_mass': float(negative/mass)}
        if config['family'] == 'swirl':
            from swirl_fits import observations

            from ftde.swirl import TUBE_STD, curve, path_quadrature, swirl_squared_norm
            t, w = path_quadrature(1600)
            truth_op = ClippedTTQuadrature4D.from_tt(f, np.eye(4)[None]*TUBE_STD**2, [1.], order=order)
            truth_noise = jax.jit(truth_op.noise_factors)
            truth_conv = jax.jit(truth_op.convolve_weighted)
            means = curve(t, 4)
            cross = 0.
            for k in range(0, len(t), 256):
                vals = np.asarray(truth_conv(weighted, truth_noise(jnp.asarray(means[k:k+256]))))
                cross += float(w[k:k+256]@vals)
            norm_q = float(jnp.sum(weighted**2/(op.pair_weights[0][:, None]*op.pair_weights[1][None])))
            norm_p = swirl_squared_norm(4, order=1600)
            record.update(relative_l2=float(np.sqrt(max(0., (norm_q-2*cross+norm_p)/norm_p))),
                          norm_q_squared=norm_q, cross_integral=cross, norm_p_squared=norm_p)
            fresh = observations(4, 'mixture', 8192, 73129)
            test_logs = logs(fresh)
            np.save(directory/f'test_order{order}.npy', test_logs)
            record['test_nll'] = float(-test_logs.mean())
        marginals = op.pair_marginals(f)
        np.savez(directory/f'marginals_order{order}.npz', nodes=np.asarray(op.nodes),
                 pair01=np.asarray(marginals[0]), pair23=np.asarray(marginals[1]))
        results.append(record)
        print('AUDIT', directory, record, flush=True)
        del weighted, op, convolve, noise
        jax.clear_caches()
    (directory/'audit.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--directories', type=Path, nargs='+', required=True)
    parser.add_argument('--orders', type=int, nargs='+', default=[65, 97, 129])
    args = parser.parse_args()
    for directory in args.directories:
        evaluate(args.base, directory, args.orders)

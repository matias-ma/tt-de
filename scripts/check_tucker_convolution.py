"""Validate selected real-kernel Tucker fits against unprepared quadrature."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from run_tucker_gallery import current_folder, digest, read

from ftde import TuckerConvolution, load_tucker
from ftde.direct_training import load_tt, write_record

ROOT = Path('artifacts/tucker_gallery')


def check(folder):
    p, s = read(folder/'protocol.json'), read(folder/'selection.json')
    saved = folder/'convolution_check.json'
    if saved.exists() and read(saved)['selection_sha256'] == digest(folder/'selection.json'):
        return
    model = load_tucker(folder/s['selected']/'fit.npz')
    with np.load(p['data']) as data:
        train, valid = data['train'], data['valid']
    observations = np.concatenate([train, valid])
    domain = np.stack([observations.min(axis=0)-.02, observations.max(axis=0)+.02], axis=1)
    points = jnp.asarray(valid[[0, 1, int(np.argmax(np.linalg.norm(valid-np.median(valid, axis=0), axis=1))) ]])
    direct = TuckerConvolution.from_tucker(model, load_tt(p['kernel']))
    prepared = direct.prepare(domain)
    a = np.asarray(jax.jit(lambda y: direct.log_density(model, y))(points))
    b = np.asarray(jax.jit(lambda y: prepared.log_density(model, y))(points))
    sequential = replace(prepared, pair_axes=False)
    c = np.asarray(jax.jit(lambda y: sequential.log_density(model, y))(points))
    error = float(np.max(np.abs(a-b)))
    pair_error = float(np.max(np.abs(b-c)))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(c).all() or max(error, pair_error) > 1e-8:
        raise AssertionError(f'Prepared convolution discrepancy: {error}')
    record = {'max_abs_log_density_error': error, 'quadrature_orders': list(direct.moments.orders),
              'pair_vs_sequential_log_density_error': pair_error,
              'probe_count': len(points), 'includes_tail_observation': True,
              'selection_sha256': digest(folder/'selection.json')}
    write_record(folder/'convolution_check.json', record)
    print(folder.name, record, flush=True)


if __name__ == '__main__':
    for folder in sorted(ROOT.iterdir()):
        folder = current_folder(folder.name)
        if not (folder/'selection.json').exists():
            continue
        if read(folder/'protocol.json').get('kernel'):
            check(folder)
            jax.clear_caches()

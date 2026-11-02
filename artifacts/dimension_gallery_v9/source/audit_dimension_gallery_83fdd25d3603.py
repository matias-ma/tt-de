"""Audit frozen gallery records and the selected noisy forward operators."""
import argparse
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from run_dimension_gallery import ROOT, digest, read

from ftde.direct_training import load_tt, write_record
from ftde.tt_convolution import TTConvolution


def audit(case):
    from plot_dimension_gallery import verify
    folder = ROOT/case
    frozen = verify(folder)
    protocol = read(folder/'protocol.json')
    evaluation = read(folder/'evaluation.json')
    if evaluation['selection_sha256'] != digest(folder/'selection.json'):
        raise RuntimeError('Evaluation selection changed')
    for name in frozen['names']:
        result = read(folder/name/'result.json')
        config = read(folder/name/'config.json')
        budget = config['sweeps'] if config['optimizer'] == 'core_sweeps' else config['steps']
        if result['history'][-1]['step'] != budget:
            raise RuntimeError('Incomplete training budget')
        if result['best_step'] < config['selection_start']:
            raise RuntimeError('Checkpoint selected before eligibility')
        if evaluation['models'][name]['model_sha256'] != digest(folder/name/'fit.npz'):
            raise RuntimeError('Evaluation model changed')
    report = {'frozen_files_verified': len(frozen['files']), 'selection_sha256': digest(folder/'selection.json')}
    if protocol['noise'] != 'free':
        model = load_tt(folder/frozen['selected']/'fit.npz')
        kernel = load_tt(protocol['kernel'])
        with np.load(protocol['data']) as data:
            train, valid = data['train'], data['valid']
        box = np.stack([np.minimum(train.min(0), valid.min(0))-.02,
                        np.maximum(train.max(0), valid.max(0))+.02], axis=1)
        direct = TTConvolution.from_tt(model, kernel)
        prepared = direct.prepare(box, tolerance=1e-14)
        refined = TTConvolution.from_tt(model, kernel, quadrature_order=2*np.asarray(direct.orders))
        points = jnp.asarray(valid[:8])
        a = np.asarray(prepared.log_density(model, points))
        b = np.asarray(refined.log_density(model, points))
        error = float(np.max(np.abs(a-b)))
        if not np.isfinite(error) or error > 1e-6:
            raise FloatingPointError(f'Forward quadrature mismatch: {error}')
        report['forward_max_log_density_difference'] = error
        report['forward_check_points'] = len(points)
    write_record(folder/'audit.json', report)
    print('AUDITED', case, report, flush=True)
    jax.clear_caches()


def snapshot_sources():
    destination = ROOT/'source'
    destination.mkdir(exist_ok=True)
    files = [Path('src/ftde')/name for name in ('benchmark_defaults.py', 'direct_training.py', 'core_sweeps.py',
              'mode_schedule.py', 'marginal_regularization.py', 'tt_convolution.py', 'simple_initialization.py')]
    files += [Path('scripts')/name for name in ('run_dimension_gallery.py', 'plot_dimension_gallery.py', 'audit_dimension_gallery.py')]
    for path in files:
        copy = destination/path.name
        if copy.exists() and digest(copy) != digest(path):
            copy = destination/f'{path.stem}_{digest(path)[:12]}{path.suffix}'
        if not copy.exists():
            shutil.copy2(path, copy)
    manifest = {str(p): digest(p) for p in files}
    manifest_path = destination/'sha256.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        shutil.copy2(manifest_path, destination/f'sha256_{digest(manifest_path)[:12]}.json')
    write_record(manifest_path, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='*')
    args = parser.parse_args()
    for case in args.cases:
        audit(case)
    snapshot_sources()

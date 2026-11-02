"""Audit frozen gallery records and the selected noisy forward operators."""
import argparse
import re
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from run_dimension_gallery import ROOT, digest, read

from ftde.direct_training import load_tt, write_record
from ftde.tt_convolution import TTConvolution


def audit(case, *, analytic_noise=False):
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
        if analytic_noise:
            from noise_convolution_audit import analytic_pair_reference
            # Check central observations and a tail against the prescribed
            # Gaussian formula, independently of the polynomial noise TT.
            indices = np.unique([0, 1, int(np.argmax(np.linalg.norm(valid-valid.mean(0), axis=1)))])
            probes = valid[indices]
            production = np.asarray(prepared.log_density(model, jnp.asarray(probes)))
            values = []
            for order in (128, 256):
                reference = analytic_pair_reference(model, probes, np.asarray(protocol['noise_weights']),
                                                     np.asarray(protocol['noise_covariances']), order)
                values.append(np.asarray(reference(tuple(c.coeffs for c in model.cores))))
            mismatch = float(np.max(np.abs(values[-1]-production)))
            refinement = float(np.max(np.abs(values[0]-values[1])))
            if not np.isfinite(mismatch+refinement) or max(mismatch, refinement) > 1e-6:
                raise FloatingPointError('Analytic Gaussian convolution mismatch')
            report.update(analytic_noise_max_log_difference=mismatch,
                          analytic_noise_quadrature_refinement=refinement,
                          analytic_noise_probe_indices=indices.tolist())
    write_record(folder/'audit.json', report)
    print('AUDITED', case, report, flush=True)
    jax.clear_caches()


def snapshot_sources():
    destination = ROOT/'source'
    destination.mkdir(exist_ok=True)
    files = [Path('src/ftde')/name for name in ('benchmark_defaults.py', 'direct_training.py', 'core_sweeps.py',
              'mode_schedule.py', 'marginal_regularization.py', 'tt_convolution.py', 'simple_initialization.py',
              'noisy_model_selection.py', 'swirl.py', 'ginzburg_landau.py', 'tt_sampling.py')]
    files += [Path('scripts')/name for name in ('run_dimension_gallery.py', 'plot_dimension_gallery.py',
              'audit_dimension_gallery.py', 'diagnose_dimension_results.py', 'run_dimension_followup.py',
              'run_noisy_dimension_capacity.py', 'refine_cryoem_gallery.py', 'swirl_figures.py',
              'evaluate_gl_l2.py', 'plot_objective_study.py', 'noise_convolution_audit.py',
              'diagnose_swirl_joint.py')]
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


def audit_publication():
    """Check the displayed matrix, image provenance, and final diagnostics."""
    from plot_dimension_gallery import verify
    expected = {f'cryoem/{name}.png' for name in ('dim4', 'dim10', 'dim20', 'training')}
    expected |= {'convergence/dimensional_errors.png', 'convergence/joint_tube_recovery.png'}
    rows = read(ROOT/'summary.json')
    if {(r['family'], r['dim']) for r in rows} != {(f, d) for f in ('gl', 'swirl') for d in (4, 10, 20)} or len(rows) != 6:
        raise RuntimeError('Published matrix is incomplete')
    sources = {}
    for row in rows:
        family, dim = row['family'], row['dim']
        subdir = 'swirl' if family == 'swirl' else 'ginzburg_landau'
        for noise, key in (('free', 'noise_free'), ('mixture', 'noisy')):
            case = f'{family}{dim}_{noise}'
            folder = Path(row[key+'_source'])
            frozen = verify(folder)
            protocol, evaluation = read(folder/'protocol.json'), read(folder/'evaluation.json')
            if (protocol['family'], protocol['dim'], protocol['noise']) != (family, dim, noise):
                raise RuntimeError('Published source describes another case')
            sha = digest(folder/'selection.json')
            if evaluation['selection_sha256'] != sha or read(folder/'audit.json')['selection_sha256'] != sha:
                raise RuntimeError('Evaluation/audit is stale')
            if evaluation['selected'] != frozen['selected'] or abs(evaluation['models'][frozen['selected']]['relative_l2']-row[key]) > 1e-12:
                raise RuntimeError('Published error does not match frozen selection')
            if noise != 'free' and read(folder/'test_scores.json')['selection_sha256'] != sha:
                raise RuntimeError('Independent test score is stale')
            prefix = f'noise_free_dim{dim}' if noise == 'free' else f'noisy_{family}{dim}_mixture'
            for suffix in ('marg_01', 'marg_23', 'convergence'):
                name = f'{subdir}/{prefix}_{suffix}.png'
                expected.add(name)
                if digest(Path('figs')/name) != digest(folder.parent/'figs'/name):
                    raise RuntimeError(f'Published image does not match its source: {name}')
            sources[case] = {'folder': str(folder), 'selection_sha256': sha,
                             'selected_model_sha256': digest(folder/frozen['selected']/'fit.npz')}
    geometry = read(ROOT/'joint_geometry.json')
    for noise in ('free', 'mixture'):
        if geometry['models'][noise]['selection_sha256'] != sources[f'swirl20_{noise}']['selection_sha256']:
            raise RuntimeError('Joint diagnostic uses an outdated model')
    name = 'convergence/joint_tube_recovery.png'
    if digest(Path('figs')/name) != digest(ROOT/'figs'/name):
        raise RuntimeError('Joint diagnostic image is stale')
    actual = {str(p.relative_to('figs')) for p in Path('figs').rglob('*') if p.is_file()}
    if actual != expected | {'README.md'}:
        raise RuntimeError(f'Unexpected or missing gallery files: {actual ^ (expected | {"README.md"})}')
    archive = Path('artifacts/gallery_archive/before_dimension_gallery_v9')
    for name in expected:
        if name.startswith('cryoem/') and digest(Path('figs')/name) != digest(archive/name):
            raise RuntimeError('Retained Cryo-EM figure changed')
    for target in re.findall(r'\]\(([^)]+)\)', Path('figs/README.md').read_text()):
        if '://' not in target and not (Path('figs')/target).exists():
            raise RuntimeError(f'Broken gallery link: {target}')
    report = {'published_cases': sources, 'png_count': len(expected),
              'archived_png_count': len(list(archive.rglob('*.png'))),
              'images': {name: digest(Path('figs')/name) for name in sorted(expected)},
              'summary_sha256': digest(ROOT/'summary.json'), 'joint_geometry_sha256': digest(ROOT/'joint_geometry.json')}
    write_record(ROOT/'publication_audit.json', report)
    print('PUBLICATION AUDITED', len(sources), 'cases,', len(expected), 'PNGs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='*')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--analytic-noise', action='store_true')
    parser.add_argument('--publication', action='store_true')
    args = parser.parse_args()
    ROOT = args.root
    for case in args.cases:
        audit(case, analytic_noise=args.analytic_noise)
    if args.publication:
        audit_publication()
    snapshot_sources()

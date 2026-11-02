"""Evaluate frozen dimensional runs and publish a compact, consistent gallery."""
import argparse
import shutil
from pathlib import Path

import jax
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_dimension_gallery import ROOT, digest, read

from ftde.direct_training import load_tt, write_record
from ftde.marginals import squared_pair_grid

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'figure.dpi': 130})
STAGE = ROOT/'figs'
LABELS = {'gl': 'Ginzburg–Landau', 'swirl': 'Swirl'}


def verify(folder):
    selection = read(folder/'selection.json')
    for path, sha in selection['files'].items():
        if digest(path) != sha:
            raise RuntimeError(f'Frozen experiment changed: {path}')
    return selection


def evaluate(case):
    folder = ROOT/case
    selection = verify(folder)
    protocol = read(folder/'protocol.json')
    dim, family = protocol['dim'], protocol['family']
    if (folder/'evaluation.json').exists():
        result = read(folder/'evaluation.json')
        if result['selection_sha256'] != digest(folder/'selection.json'):
            raise RuntimeError('Evaluation belongs to a different selection')
        return result
    results = {}
    for name in selection['names']:
        out = folder/name
        if (out/'evaluation.json').exists():
            results[name] = read(out/'evaluation.json')
            continue
        model = load_tt(out/'fit.npz')
        if family == 'swirl':
            from swirl_figures import integrated_l2
            result = integrated_l2(folder, model, dim, name)
            if result['relative_squared_l2'] < -1e-9 or result['path_quadrature_relative_change'] > 1e-6:
                raise FloatingPointError('Swirl quadrature did not converge')
        else:
            from evaluate_gl_l2 import integrated_terms, relative_l2_from_terms

            from ftde.ginzburg_landau import GL_BETA, ginzburg_landau_log_normalizer
            log_z = ginzburg_landau_log_normalizer(dim)
            p2 = float(np.exp(ginzburg_landau_log_normalizer(dim, beta=2*GL_BETA)-2*log_z))
            order = max(98, 2*max(c.coeffs.shape[-1] for c in model.cores))
            q2, pq = map(float, jax.jit(lambda model=model, order=order, log_z=log_z: integrated_terms(model, order, log_z))())
            q2f, pqf = map(float, jax.jit(lambda model=model, order=order, log_z=log_z: integrated_terms(model, order*2, log_z))())
            result = {'relative_l2': relative_l2_from_terms(q2f, pqf, p2), 'integral_q_squared': q2f,
                'integral_pq': pqf, 'integral_p_squared': p2, 'quadrature_relative_change': max(abs(q2-q2f), abs(pq-pqf))/p2}
            if result['quadrature_relative_change'] > 1e-6:
                raise FloatingPointError('GL quadrature did not converge')
        if not np.isfinite(result['relative_l2']):
            raise FloatingPointError('Nonfinite full-density error')
        result['model_sha256'] = digest(out/'fit.npz')
        write_record(out/'evaluation.json', result)
        print('EVALUATED', case, name, result, flush=True)
        results[name] = result
        jax.clear_caches()
    evaluation = {'models': results, 'selected': selection['selected'], 'minimum_nll': selection['minimum_nll'],
                      'selection_sha256': digest(folder/'selection.json'), 'metric': 'unsquared full-dimensional relative Lebesgue L2'}
    write_record(folder/'evaluation.json', evaluation)
    return evaluation


def save_panels(arrays, titles, groups, grid, pair, heading, footer, path):
    noisy = len(arrays) == 6
    fig, axes = plt.subplots(2 if noisy else 1, 3, figsize=(12.8, 8.6 if noisy else 4.9))
    maxima = {g: max(float(a.max()) for a, group in zip(arrays, groups) if group == g) for g in groups}
    for ax, array, title, group in zip(np.ravel(axes), arrays, titles, groups, strict=True):
        im = ax.imshow(array, origin='lower', extent=(grid[0], grid[-1], grid[0], grid[-1]),
                       cmap='magma', vmin=0, vmax=maxima[group], aspect='equal')
        ax.set(title=title, xlabel=f'x{pair[0]}', ylabel=f'x{pair[1]}')
        fig.colorbar(im, ax=ax, shrink=.8, pad=.025, label='Density')
    fig.suptitle(heading, fontsize=15, y=.975)
    fig.text(.035, .018, footer, fontsize=9, color='#444444', va='bottom')
    fig.tight_layout(rect=(0, .11 if noisy else .16, 1, .92 if noisy else .88))
    fig.savefig(path, dpi=165)
    plt.close(fig)


def plot(case):
    from plot_objective_study import convolve_pair, noise_grid
    folder = ROOT/case
    evaluation = evaluate(case)
    protocol = read(folder/'protocol.json')
    family, dim, noisy = protocol['family'], protocol['dim'], protocol['noise'] != 'free'
    name = evaluation['selected']
    model = load_tt(folder/name/'fit.npz')
    config, _result = read(folder/name/'config.json'), read(folder/name/'result.json')
    with np.load(protocol['data']) as data:
        train = data['train']
    destination = STAGE/('swirl' if family == 'swirl' else 'ginzburg_landau')
    destination.mkdir(parents=True, exist_ok=True)
    prefix = f'noisy_{family}{dim}_mixture' if noisy else f'noise_free_dim{dim}'
    grid = np.linspace(-2.6 if family == 'swirl' else -2.5, 2.6 if family == 'swirl' else 2.5, 181)
    error = evaluation['models'][name]['relative_l2']
    mode = 'mixture noise' if noisy else 'noise-free'
    heading = f'{LABELS[family]} · {dim}D · {mode}\nFull {dim}D latent relative L2 = {error:.3f}'
    algorithm = 'core sweeps' if config['optimizer'] == 'core_sweeps' else 'Adam'
    footer = (f'Full-Hessian regularization · {name.replace("_", " ")} · {algorithm} · rank {config["rank"]} · {config["modes"]} modes\n'
              f'{len(train):,} observations · held-out selection · exact TT marginals · matched color scales · cropped densities retain their mass')
    diagnostics = {}
    for i, j in [(0, 1), (2, 3)]:
        if family == 'swirl':
            from ftde.swirl import swirl_pair_marginal
            truth = swirl_pair_marginal(dim, i, j, grid, grid)
        else:
            from gl_common import _true_marginal_grid

            from ftde.ginzburg_landau import ginzburg_landau_log_normalizer
            truth = _true_marginal_grid(dim, i, j, np.tile([grid[0], grid[-1]], (dim, 1)),
                                        ginzburg_landau_log_normalizer(dim), n_grid=len(grid))
        latent = squared_pair_grid(model, i, j, grid, grid)
        hist, _, _ = np.histogram2d(train[:, i], train[:, j], bins=90, range=[(grid[0], grid[-1])]*2)
        hist = hist.T/(len(train)*((grid[-1]-grid[0])/90)**2)
        row = {'latent_pair_relative_l2': float(np.linalg.norm(latent-truth)/np.linalg.norm(truth))}
        if noisy:
            weights, cov = np.array(protocol['noise_weights']), np.array(protocol['noise_covariances'])
            true_obs, fitted_obs = convolve_pair(model, case, i, j, grid, weights, cov, .0125)
            coarse = convolve_pair(model, case, i, j, grid, weights, cov, .025)
            changes = [float(np.linalg.norm(a-b)/np.linalg.norm(a)) for a, b in zip((true_obs, fitted_obs), coarse)]
            if max(changes) > .01:
                raise FloatingPointError('Displayed convolution requires grid refinement')
            row.update(observed_pair_relative_l2=float(np.linalg.norm(fitted_obs-true_obs)/np.linalg.norm(true_obs)),
                       convolution_grid_refinement=changes)
            arrays = [truth, latent, noise_grid(weights, cov, i, j, grid), true_obs, fitted_obs, hist]
            titles = ['True latent density', 'Recovered latent density', 'Known measurement noise',
                      'True observed density', 'Reconvolved estimator', 'Observed data']
            groups = [0, 0, 1, 2, 2, 2]
        else:
            arrays, titles, groups = [truth, latent, hist], ['True density', 'Estimated density', 'Observed data'], [0, 0, 0]
        save_panels(arrays, titles, groups, grid, (i, j), heading, footer,
                    destination/f'{prefix}_marg_{i}{j}.png')
        diagnostics[f'pair{i}{j}'] = row
    write_record(folder/'plot_checks.json', diagnostics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    selection = read(folder/'selection.json')
    for candidate in selection['names']:
        history = read(folder/candidate/'result.json')['history']
        selected = candidate == name
        label = candidate.replace('_', ' ')+(' (selected)' if selected else '')
        axes[0].plot([h['step'] for h in history], [h['validation_nll'] for h in history], label=label, lw=2 if selected else 1)
        axes[1].semilogy([h['step'] for h in history], [max(h['roughness_hessian'], 1e-14) for h in history], lw=2 if selected else 1)
    axes[0].set(xlabel='Updates within each stage' if noisy else ('Core sweeps' if algorithm == 'core sweeps' else 'Adam updates'), ylabel='Validation NLL')
    axes[1].set(xlabel=axes[0].get_xlabel(), ylabel='Amplitude Hessian energy')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f'{LABELS[family]} · {dim}D · {mode} — training')
    fig.tight_layout()
    fig.savefig(destination/f'{prefix}_convergence.png', dpi=165)
    plt.close(fig)


def publish():
    expected = [f'{f}{d}_{n}' for f in ('gl', 'swirl') for d in (4, 10, 20) for n in ('free', 'mixture')]
    for case in expected:
        verify(ROOT/case)
        if not (ROOT/case/'plot_checks.json').exists():
            raise RuntimeError(f'Cannot publish incomplete gallery: {case}')
    archive = Path('artifacts/gallery_archive/before_dimension_gallery_v9')
    if not archive.exists():
        shutil.copytree('figs', archive)
        old_index = archive/'README.md'
        old_index.write_text(old_index.read_text().replace('../docs/', '../../../docs/').replace('../artifacts/', '../../../artifacts/'))
    # The complete old gallery is preserved before removing superseded PNGs.
    for family in ('swirl', 'ginzburg_landau', 'convergence'):
        destination = Path('figs')/family
        destination.mkdir(exist_ok=True)
        for old in destination.glob('*.png'):
            if not (archive/family/old.name).exists():
                raise RuntimeError(f'Unarchived figure: {old}')
            old.unlink()
        staged = STAGE/family
        if staged.exists():
            for image in staged.glob('*.png'):
                shutil.copy2(image, destination/image.name)
    # Retain real-data 4D/10D/20D and training; archive the extra path variant.
    path_variant = Path('figs/cryoem/path_dim4.png')
    if path_variant.exists():
        if not (archive/'cryoem/path_dim4.png').exists():
            raise RuntimeError('Cryo-EM path figure not archived')
        path_variant.unlink()
    # Historical reports should continue to point to their historical figures.
    for doc in Path('docs').glob('*.md'):
        if doc.name != 'dimension_gallery.md':
            original = doc.read_text()
            updated = original.replace('../figs/', '../artifacts/gallery_archive/before_dimension_gallery_v9/')
            if original != updated:
                doc.write_text(updated)
    rows, table = [], []
    for family in ('gl', 'swirl'):
        for dim in (4, 10, 20):
            errors = [read(ROOT/f'{family}{dim}_{noise}'/'evaluation.json') for noise in ('free', 'mixture')]
            values = [r['models'][r['selected']]['relative_l2'] for r in errors]
            rows.append({'family': family, 'dim': dim, 'noise_free': values[0], 'noisy': values[1]})
            table.append(f'| {LABELS[family]} | {dim} | {values[0]:.4f} | {values[1]:.4f} |')
    write_record(ROOT/'summary.json', rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, family in zip(axes, ('gl', 'swirl')):
        selected_rows = [r for r in rows if r['family'] == family]
        for key, label in [('noise_free', 'Noise-free'), ('noisy', 'Mixture noise')]:
            ax.plot([r['dim'] for r in selected_rows], [r[key] for r in selected_rows], 'o-', label=label)
        ax.axhspan(.2, .3, color='green', alpha=.1, label='0.2–0.3 target')
        ax.set(title=LABELS[family], xlabel='Dimensions', ylabel='Full-density relative L2', xticks=[4, 10, 20], ylim=(0, None))
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig('figs/convergence/dimensional_errors.png', dpi=165)
    plt.close(fig)
    index = '# Density estimation: 4D, 10D, 20D\n\n'
    index += 'Fresh single squared-Chebyshev-TT fits from near-uniform starts. Full-Hessian regularization is the default. '
    index += 'Noise kernels retain the earlier benchmark scales, including the correlated Swirl mixture.\n\n'
    index += '| Benchmark | Dimension | Noise-free relative L2 | Noisy relative L2 |\n|---|---:|---:|---:|\n'+'\n'.join(table)+'\n\n'
    index += 'These are unsquared **full-dimensional latent** errors, evaluated after observation-only choices were frozen. '
    index += 'Each benchmark folder contains two pair marginals and a convergence figure per condition. '
    index += 'True and estimated latent densities share their color scale; reconvolved fits and observed densities share another.\n\n'
    index += 'Cryo-EM retains its previous 4D/10D/20D noisy fits and training figure; no clean-density error is available. '
    index += 'Its fits were not rerun in this synthetic matrix.\n\n'
    index += 'See [protocol and interpretation](../docs/dimension_gallery.md), '
    index += '[error comparison](convergence/dimensional_errors.png), and '
    index += '[archived previous gallery](../artifacts/gallery_archive/before_dimension_gallery_v9/). '
    index += 'Records, models and logs live in `artifacts/dimension_gallery_v9`.\n'
    Path('figs/README.md').write_text(index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='*')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    for case in args.cases:
        plot(case)
    if args.publish:
        publish()

"""Publish density/noise/data/reconvolution panels from frozen objective fits."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_objective_study import model_path
from objective_study import ROOT, setup
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

from ftde.direct_training import load_tt, write_record
from ftde.marginals import squared_pair_coefficients

CASE_LABELS = {'gl6_free': 'Noise-free Ginzburg–Landau, 6D', 'swirl4_free': 'Noise-free swirl, 4D',
               'gl6_paired': 'Ginzburg–Landau, 6D · correlated Gaussian noise',
               'gl6_mixture': 'Ginzburg–Landau, 6D · broad mixture noise',
               'swirl4_mixture': 'Swirl, 4D · correlated mixture noise',
               'swirl10_mixture': 'Swirl, 10D · correlated mixture noise'}


def label(folder, name):
    run, checkpoint = name.rsplit('/', 1)
    c = json.loads((folder/run/'config.json').read_text())
    r = json.loads((folder/run/'result.json').read_text())
    objective = {'nll': 'NLL', 'mmd': f'{c["kernel_weight"]:g} MMD²', 'hybrid': f'NLL + {c["kernel_weight"]:g} MMD²'}[c['objective']]
    step = r['best_step'] if checkpoint == 'nll' else r['kernel_step']
    return f'{objective}; r={c["rank"]}, modes={c["modes"]}, λ={c["strength"]:g}', f'Adam checkpoint {step}/{c["steps"]}'


def noise_grid(weights, covariances, i, j, grid):
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack([xx, yy], axis=-1)
    result = np.zeros_like(xx)
    for weight, c in zip(weights, covariances, strict=True):
        c = c[np.ix_([i, j], [i, j])]
        result += weight*np.exp(-.5*np.einsum('...i,ij,...j->...', points, np.linalg.inv(c), points))/(2*np.pi*np.sqrt(np.linalg.det(c)))
    return result


def convolve_pair(model, case, i, j, grid, weights, covariance, spacing):
    coefficient, box = squared_pair_coefficients(model, i, j)
    radius = np.ceil(max(3., np.max(np.abs(box)))/spacing)*spacing
    source = np.arange(-round(radius/spacing), round(radius/spacing)+1)*spacing
    noise_radius = int(np.ceil((radius+max(abs(grid[0]), abs(grid[-1])))/spacing))
    offsets = np.arange(-noise_radius, noise_radius+1)*spacing
    bases = [np.polynomial.chebyshev.chebvander(np.clip(2*(source-a)/(b-a)-1, -1, 1), n-1)
             for (a, b), n in zip(box, coefficient.shape[::-1], strict=True)]
    estimated = bases[1]@coefficient@bases[0].T
    estimated[:, (source < box[0, 0]) | (source > box[0, 1])] = 0
    estimated[(source < box[1, 0]) | (source > box[1, 1]), :] = 0
    dim = len(model.domain)
    if case.startswith('gl'):
        from gl_common import _true_marginal_grid

        from ftde.ginzburg_landau import ginzburg_landau_log_normalizer
        truth = _true_marginal_grid(dim, i, j, np.tile([-radius, radius], (dim, 1)),
                                    ginzburg_landau_log_normalizer(dim), n_grid=len(source))
    else:
        from ftde.swirl import swirl_pair_marginal
        truth = swirl_pair_marginal(dim, i, j, source, source)
    kernel = noise_grid(weights, covariance, i, j, offsets)
    coordinate = source[0]+offsets[0]+np.arange(len(source)+len(offsets)-1)*spacing
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack([yy.ravel(), xx.ravel()], axis=-1)
    return tuple(RegularGridInterpolator((coordinate, coordinate), fftconvolve(p, kernel, mode='full')*spacing**2)(points).reshape(xx.shape)
                 for p in (truth, estimated))


def panels(arrays, titles, grid, i, j, heading, footer, path, groups=None, coordinate_pairs=None):
    columns = len(arrays)//2
    fig, axes = plt.subplots(2, columns, figsize=(4.85*columns, 9.8), layout='constrained')
    groups = list(range(len(arrays))) if groups is None else groups
    maxima = {group: max(float(a.max()) for a, g in zip(arrays, groups, strict=True) if g == group) for group in groups}
    coordinate_pairs = [(i, j)]*len(arrays) if coordinate_pairs is None else coordinate_pairs
    for ax, array, title, group, (xi, yi) in zip(axes.flat, arrays, titles, groups, coordinate_pairs, strict=True):
        image = ax.imshow(array, origin='lower', extent=(grid[0], grid[-1], grid[0], grid[-1]),
                          cmap='magma', vmin=0, vmax=maxima[group], aspect='equal', interpolation='nearest')
        ax.set(title=title, xlabel=f'x{xi}', ylabel=f'x{yi}')
        fig.colorbar(image, ax=ax, shrink=.78, label='Density')
    fig.suptitle(heading, fontsize=15, y=.985).set_in_layout(False)
    fig.get_layout_engine().set(rect=(0, .12, 1, .79))
    fig.text(.025, .015, footer, fontsize=9, va='bottom')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *[ax.title for ax in axes.flat]]:
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height:
            raise ValueError(f'Figure text outside canvas: {text.get_text()}')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot(case):
    folder, _, train, valid, _, weights, covariance = setup(case)
    result = json.loads((folder/'evaluation.json').read_text())
    frozen = json.loads((folder/'frozen.json').read_text())
    selected, kernel = frozen['selected_nll'], frozen['selected_kernel']
    truth = np.load(folder/'truth.npz')
    grid = truth['grid']
    dim = len(json.loads((folder/'protocol.json').read_text())['domain'])
    destination = Path('figs/swirl' if case.startswith('swirl') else 'figs/ginzburg_landau')
    def marginal(name, pair):
        run, suffix = name.rsplit('/', 1)
        return np.load(folder/run/f'marginals_{suffix}.npz')[pair]
    description, step = label(folder, selected)
    noise_description = {'gl6_paired': 'Correlated Gaussian: marginal σ=0.830, adjacent-pair ρ=0.5',
                         'gl6_mixture': 'Gaussian-mixture noise: σ=0.45/1.35, weights 0.7/0.3',
                         'swirl4_mixture': 'Noise weights 0.5/0.3/0.2: σ=0.22; diagonal σ=0.45/0.25; σ=0.5 with adjacent-pair ρ=±0.6',
                         'swirl10_mixture': 'Noise weights 0.5/0.3/0.2: σ=0.22; diagonal σ=0.45/0.25; σ=0.5 with adjacent-pair ρ=±0.6'}
    footer = (f'Squared, normalized Chebyshev functional TT · {description} · {step}\n'
              f'Ntrain={len(train):,}, Nvalidation={len(valid):,} · Adam lr=0.001, batch=1024 · near-uniform start, random scale 0.01, seed 0\n'
              'Fit selected using held-out observations only; known density used afterward for evaluation. Cropped panels are not renormalized.\n')
    diagnostics = {}
    for i, j in [(0, 1), (2, 3)]:
        pair = f'pair{i}{j}'
        hist, _, _ = np.histogram2d(train[:, i], train[:, j], bins=len(grid), range=[(grid[0], grid[-1])]*2)
        hist = hist.T/(len(train)*((grid[-1]-grid[0])/len(grid))**2)
        latent = marginal(selected, pair)
        heading = f'{CASE_LABELS[case]}\nFull {dim}D latent relative L2 = {result["models"][selected]["relative_l2"]:.3f}'
        if not case.endswith('free'):
            model = load_tt(model_path(folder, selected))
            truth_noisy, fit_noisy = convolve_pair(model, case, i, j, grid, weights, covariance, .0125)
            coarse = convolve_pair(model, case, i, j, grid, weights, covariance, .025)
            errors = [float(np.linalg.norm(a-b)/np.linalg.norm(a)) for a, b in zip((truth_noisy, fit_noisy), coarse, strict=True)]
            if max(errors) > .01:
                raise FloatingPointError(f'Convolution figure requires refinement: {errors}')
            diagnostics[pair] = {'convolution_refinement_errors': errors,
                                  'latent_pair_relative_l2': float(np.linalg.norm(latent-truth[pair])/np.linalg.norm(truth[pair])),
                                  'observed_pair_relative_l2': float(np.linalg.norm(fit_noisy-truth_noisy)/np.linalg.norm(truth_noisy))}
            kernel_model = load_tt(model_path(folder, kernel))
            _, kernel_noisy = convolve_pair(kernel_model, case, i, j, grid, weights, covariance, .0125)
            _, kernel_coarse = convolve_pair(kernel_model, case, i, j, grid, weights, covariance, .025)
            kernel_error = float(np.linalg.norm(kernel_noisy-kernel_coarse)/np.linalg.norm(kernel_noisy))
            if kernel_error > .01:
                raise FloatingPointError('Kernel-selected figure requires finer convolution grid')
            kernel_latent = marginal(kernel, pair)
            diagnostics[pair].update(kernel_convolution_refinement_error=kernel_error,
                kernel_latent_pair_relative_l2=float(np.linalg.norm(kernel_latent-truth[pair])/np.linalg.norm(truth[pair])),
                kernel_observed_pair_relative_l2=float(np.linalg.norm(kernel_noisy-truth_noisy)/np.linalg.norm(truth_noisy)),
                marginal_error_window=[float(grid[0]), float(grid[-1])])
            arrays = [truth[pair], noise_grid(weights, covariance, i, j, grid), truth_noisy, hist,
                      fit_noisy, latent, kernel_noisy, kernel_latent]
            titles = ['True latent density', 'Prescribed noise kernel', 'True noisy density', 'Histogram of noisy observations',
                      'NLL selection: reconvolved fit', 'NLL selection: latent fit',
                      'MMD selection: reconvolved fit', 'MMD selection: latent fit']
            kernel_description, kernel_step = label(folder, kernel)
            noise_footer = (f'NLL selection: {description} · {step}\n'
                            f'MMD selection: {kernel_description} · {kernel_step}\n'
                            f'Squared, normalized Chebyshev TTs · Ntrain={len(train):,}, Nvalidation={len(valid):,} · Adam lr=0.001, batch=256 · full-Hessian scales=5% box width\n'
                            'Near-uniform starts (random scale 0.01, seed 0). MMD is a multiscale kernel score. Choices use held-out observations; truth is evaluated afterward.\n'
                            +noise_description[case]+' · Cropped panels are not renormalized.')
            heading = (CASE_LABELS[case]+f'\nFull {dim}D latent relative L2: NLL selection {result["models"][selected]["relative_l2"]:.3f}'
                       f' · MMD selection {result["models"][kernel]["relative_l2"]:.3f}')
            panels(arrays, titles, grid, i, j, heading, noise_footer,
                   destination/f'noisy_{case}_marg_{i}{j}.png', groups=[0, 1, 2, 2, 2, 0, 2, 0])
        else:
            capacities = json.loads((folder/'capacity_selection.json').read_text())['names']
            def best_rank(rank, capacities=capacities):
                eligible = [n for n in capacities if json.loads((folder/n/'config.json').read_text())['rank'] == rank]
                return min(eligible, key=lambda n: json.loads((folder/n/'result.json').read_text())['best_nll'])+'/nll'
            controls = [best_rank(32), best_rank(10), selected, kernel]
            arrays = [truth[pair], hist]+[marginal(n, pair) for n in controls]
            titles = ['True latent density', 'Histogram of observations']
            for name, prefix in zip(controls, ['Rank-32 control', 'Rank-10 control', 'NLL-selected fit', 'MMD-selected fit'], strict=True):
                titles.append(prefix+'\n'+label(folder, name)[0]+f'\nFull {dim}D L2={result["models"][name]["relative_l2"]:.3f}')
            panels(arrays, titles, grid, i, j, heading, footer+'Full-Hessian penalty; scales=5% box width. MMD is a multiscale kernel score. All controls use the elementary start.',
                   destination/f'noise_free_dim{dim}_marg_{i}{j}.png', groups=[0]*6)
    # Show both observation-based choices and the pure alternative, even when
    # the kernel score prefers a likelihood-trained TT.
    pure = next(n for n in result['models'] if n.startswith('mmd_') and n.endswith('/kernel'))
    shown = [selected, kernel, pure]
    arrays, titles = [], []
    for i, j in [(0, 1), (2, 3)]:
        for name, prefix in zip(shown, ['NLL selection', 'MMD selection', 'Pure MMD objective'], strict=True):
            arrays.append(marginal(name, f'pair{i}{j}'))
            titles.append(prefix+'\n'+label(folder, name)[0]+f'\nFull {dim}D L2={result["models"][name]["relative_l2"]:.3f}; pair ({i},{j})')
    panels(arrays, titles, grid, 0, 1, CASE_LABELS[case]+'\nObjective comparison after freezing both choices',
           'One Chebyshev TT per panel. MMD is a multiscale kernel score. Each selection uses its own held-out score; pure MMD uses its MMD checkpoint.\n'
           'Top row: coordinates (0,1). Bottom row: coordinates (2,3). L2 evaluates the full latent density, not just these images.',
           destination/f'objectives_{case}.png', groups=[0]*3+[1]*3, coordinate_pairs=[(0, 1)]*3+[(2, 3)]*3)
    write_record(folder/'plot_checks.json', diagnostics)
    if case.endswith('free'):
        summary([case], destination/f'noise_free_dim{dim}_training.png')


def summary(cases, path=Path('figs/convergence/objectives.png')):
    fig, axes = plt.subplots(len(cases), 3, figsize=(16, 3.8*len(cases)+.4), layout='constrained', squeeze=False)
    for row, case in enumerate(cases):
        folder = ROOT/case
        evaluated = json.loads((folder/'evaluation.json').read_text())
        frozen = json.loads((folder/'frozen.json').read_text())
        names = list(dict.fromkeys([frozen['selected_nll'], frozen['selected_kernel'],
                                   next(n for n in evaluated['models'] if n.startswith('mmd_') and n.endswith('/kernel'))]))
        for name in names:
            record = json.loads((folder/name.rsplit('/', 1)[0]/'result.json').read_text())
            history = record['history']
            text = label(folder, name)[0]
            for col, metric in enumerate(['validation_nll', 'validation_mmd']):
                axes[row, col].plot([r['step'] for r in history], [r[metric] for r in history], '.-', label=text, ms=3)
            axes[row, 0].set(title=CASE_LABELS[case]+'\nHeld-out NLL', ylabel='NLL ↓', xlabel='Coefficient updates')
            axes[row, 1].set(title='Independent-frequency kernel discrepancy', ylabel='MMD² ↓', xlabel='Coefficient updates', yscale='log')
        axes[row, 0].legend(fontsize=7)
        values = [evaluated['models'][n]['relative_l2'] for n in names]
        axes[row, 2].bar(range(len(names)), values)
        axes[row, 2].set_xticks(range(len(names)), [label(folder, n)[0] for n in names], rotation=15, ha='right', fontsize=7)
        axes[row, 2].set(title='Latent error after selection', ylabel='Full-dimensional relative L2 ↓')
        for ax in axes[row]:
            ax.grid(alpha=.2)
    fig.suptitle('Objectives, capacity and strong-noise recovery · all fits start near uniform', fontsize=16)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    for case in args.cases:
        plot(case)
    if args.summary:
        summary(args.cases)

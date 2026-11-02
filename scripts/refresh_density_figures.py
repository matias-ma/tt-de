"""Render the flat gallery from frozen fits; truth is used only here.

All density panels are unconditionally normalized, including histograms:
cropping a view never renormalizes the probability inside that view.
Convolution panels use zero-padded linear FFT quadrature of exact pair
marginals, independently checked at two grid spacings.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gl_common import _true_marginal_grid, dim_pairs_for
from refresh_density_fits import read_model
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve

from ftde.cheb_gram import chebyshev_moment_vector
from ftde.cheb_product import square_tt
from ftde.ginzburg_landau import ginzburg_landau_log_normalizer
from ftde.marginals import squared_pair_grid
from ftde.tt_density import density_tt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.titlesize': 13, 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white'})


def pair_coefficients(tt, i, j):
    """Integrate all other coordinates; retain a small 2D coefficient matrix."""
    state = np.ones(1)
    for k, core in enumerate(tt.cores):
        c = np.asarray(core.coeffs)
        if k in (i, j):
            state = np.einsum('...a,abn->...nb', state, c)
        else:
            moments = np.asarray(chebyshev_moment_vector(c.shape[-1], tuple(core.domain)))
            state = state @ np.einsum('abn,n->ab', c, moments)
    coeff = state[..., 0]
    mi = np.asarray(chebyshev_moment_vector(coeff.shape[0], tuple(tt.domain[i])))
    mj = np.asarray(chebyshev_moment_vector(coeff.shape[1], tuple(tt.domain[j])))
    mass = mi@coeff@mj
    if not np.isfinite(mass) or mass <= 0:
        raise FloatingPointError('Invalid marginal normalization')
    return coeff/mass, np.asarray(tt.domain)[[i, j]]


def evaluate_pair(coeff, box, xs, ys, *, negative_tolerance=1e-7, diagnostics=None):
    """Rows are y, columns x; density is zero outside its declared support."""
    tx = 2*(np.asarray(xs)-box[0, 0])/np.ptp(box[0])-1
    ty = 2*(np.asarray(ys)-box[1, 0])/np.ptp(box[1])-1
    vx = np.polynomial.chebyshev.chebvander(np.clip(tx, -1, 1), coeff.shape[0]-1)
    vy = np.polynomial.chebyshev.chebvander(np.clip(ty, -1, 1), coeff.shape[1]-1)
    z = (vx@coeff@vy.T).T
    z[:, np.abs(tx) > 1] = 0
    z[np.abs(ty) > 1, :] = 0
    if diagnostics is not None:
        diagnostics['minimum_before_clipping'] = float(z.min())
        diagnostics['clipped_negative_mass'] = float(-np.minimum(z, 0.).sum()
            * (xs[1]-xs[0]) * (ys[1]-ys[0]))
    if z.min() < -negative_tolerance*max(1., z.max()):
        raise FloatingPointError(f'Negative pair marginal: {z.min()}')
    return np.maximum(z, 0.)


def kde_grid(train, bandwidth, i, j, xs, ys):
    z = np.zeros((len(ys), len(xs)))
    for start in range(0, len(train), 10000):
        batch = train[start:start+10000]
        x = np.exp(-.5*((xs[:, None]-batch[:, i])/bandwidth[i])**2)/bandwidth[i]
        y = np.exp(-.5*((ys[:, None]-batch[:, j])/bandwidth[j])**2)/bandwidth[j]
        z += y@x.T
    return z/(len(train)*2*np.pi)


def histogram(train, i, j, extent, bins=55):
    h, xedges, yedges = np.histogram2d(train[:, i], train[:, j], bins=bins,
                                      range=[extent[:2], extent[2:]])
    return h.T/(len(train)*np.diff(xedges)[0]*np.diff(yedges)[0])


def convolved_grids(dim, i, j, p_coeff, p_box, u_coeff, u_box, xs, log_z, spacing, *, kernel_check=None):
    radius = np.ceil(max(3., np.max(np.abs(p_box)))/spacing)*spacing
    source = np.arange(-round(radius/spacing), round(radius/spacing)+1)*spacing
    kernel_radius = int(np.ceil((radius+2.5)/spacing))
    offsets = np.arange(-kernel_radius, kernel_radius+1)*spacing
    display_domain = np.tile([-radius, radius], (dim, 1))
    truth = _true_marginal_grid(dim, i, j, display_domain, log_z, n_grid=len(source))
    fit = evaluate_pair(p_coeff, p_box, source, source)
    check = {} if kernel_check is None else kernel_check
    noise = evaluate_pair(u_coeff, u_box, offsets, offsets, negative_tolerance=1e-5, diagnostics=check)
    # Cross-approximated kernels can have tiny negative tails. Bound the
    # actual probability removed rather than silently accepting arbitrary
    # negativity. Convolution changes by at most this mass times max(p).
    if check['clipped_negative_mass'] > 1e-5:
        raise FloatingPointError(f'Noise approximation has too much negative mass: {check}')
    coordinate = source[0]+offsets[0]+np.arange(len(source)+len(offsets)-1)*spacing
    xx, yy = np.meshgrid(xs, xs)
    points = np.stack([yy.ravel(), xx.ravel()], axis=1)
    result = []
    for density in (truth, fit):
        convolved = fftconvolve(density, noise, mode='full')*spacing**2
        result.append(RegularGridInterpolator((coordinate, coordinate), convolved)(points).reshape(xx.shape))
    return result


def physical_noise_grid(kernel, i, j, xs, ys, scale):
    """Noise density on physical displacement axes; no observation-mean shift."""
    coeff, box = pair_coefficients(kernel, i, j)
    return evaluate_pair(coeff, box, np.asarray(xs)/scale[i], np.asarray(ys)/scale[j])/(scale[i]*scale[j])


def draw_panels(panels, titles, extent, i, j, heading, footer, path, groups=None):
    n = len(panels)
    rows, cols = (2, 3) if n == 6 else (1, n)
    fig, axes = plt.subplots(rows, cols, figsize=(14.5, 10.5) if rows == 2 else (4.75*cols, 6.3),
                             layout='constrained', squeeze=False)
    groups = groups or [0]*n
    vmax = {g: max(float(np.max(p)) for p, gg in zip(panels, groups, strict=True) if gg == g)
            for g in set(groups)}
    for ax, z, title, group in zip(axes.flat, panels, titles, groups, strict=True):
        im = ax.imshow(z, extent=extent, origin='lower', aspect='auto', cmap='magma',
                       vmin=0, vmax=vmax[group], interpolation='nearest')
        ax.set_box_aspect(1)
        ax.set(title=title, xlabel=f'$x_{{{i}}}$', ylabel=f'$x_{{{j}}}$',
               xlim=extent[:2], ylim=extent[2:])
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 3), useMathText=True)
        ax.spines[['top', 'right']].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=.042, pad=.025, label='Density')
    fig.suptitle(heading, fontsize=17, fontweight='bold')
    fig.supxlabel(footer, fontsize=8, color='#52525b')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print('Saved', path, flush=True)


def duration_label(seconds):
    seconds = round(seconds)
    if seconds >= 3600:
        return f'{seconds//3600}h {(seconds%3600)//60:02d}m'
    if seconds >= 60:
        return f'{seconds//60}m {seconds%60:02d}s'
    return f'{seconds}s'


def candidate_fit_seconds(case):
    """Sum recorded candidate runs, deduplicating inherited/symlinked fits."""
    records, visited = set(), set()

    def collect(folder):
        folder = Path(folder).resolve()
        if folder in visited:
            return
        visited.add(folder)
        selection = json.loads((folder/'selection.json').read_text())
        for name in {*selection.get('names', [selection['selected']]), 'warmup'}:
            path = folder/name/'result.json'
            if path.exists():
                records.add(path.resolve())
        if (folder/'refinement.json').exists():
            collect(json.loads((folder/'refinement.json').read_text())['base'])

    collect(case)
    return sum(json.loads(path.read_text())['seconds'] for path in records)


def settings_caption(case, fit, train, bandwidth):
    settings = json.loads((case/'settings.json').read_text())
    selection = json.loads((case/'selection.json').read_text())
    name = selection['selected']
    record = json.loads((case/name/'result.json').read_text())
    data = np.load(case/'data.npz')
    ranks = fit.ranks
    modes = [c.coeffs.shape[-1] for c in fit.cores]
    count = str(modes[0]) if len(set(modes)) == 1 else f'{min(modes)}–{max(modes)}'
    refinement = json.loads((case/'refinement.json').read_text()) if (case/'refinement.json').exists() else None
    initialization = 'expanded previous fit, normalized TT gauge' if refinement else '64 starts, 10 search steps'
    if name.startswith('polish_'):
        initialization = 'higher-rank fit, continued smoothing'
    lines = [(f'N train = {len(train):,} · N validation = {len(data["valid"]):,} · max TT rank = {max(ranks)} · '
              f'Chebyshev amplitude modes/axis = {count} (degree {max(modes)-1}) · init: {initialization}')]
    lines.append(f'Adam · {record["steps"]:,} steps per candidate · selected step {record["best_step"]:,} · '
                 f'batch {record["batch_size"]} · learning rate {record["lr"]:g} → {record["lr"]*.1:g} (cosine) · gradient clip 1 · '
                 f'{"Sobolev" if record.get("preconditioning") == "sobolev" else "Gram"} preconditioning')
    noisy = settings['family'] == 'cryoem' or settings['noise'] != 'free'
    if refinement and refinement.get('continuity'):
        controls = refinement['continuity']
        lines.append(f'Fixed-kernel continuity controls: ranks {controls["ranks"]}, modes {controls["modes"]} · '
                     f'selected amplitude H{controls["selected_order"]} strength {record["strength"]:g} · regularized checkpoint')
        lines.append('Selection: paired noisy one-SE, then minimum common amplitude H² · '
                     f'noisy validation NLL {refinement["baseline_nll"]:.4f} → {refinement["selected_nll"]:.4f}')
        lines.append(f'Minibatch seed {record.get("seed", 0)} · same observations, noise kernel and latent support as previous fit')
        kernel = read_model(case/'kernel.npz')
        lines.append(f'Kernel max rank {max(kernel.ranks)}, max modes {max(c.coeffs.shape[-1] for c in kernel.cores)} · '
                     'prepared TT convolution · latent box: 0.5–99.5% data quantiles + 5% padding')
    elif refinement:
        config = refinement['configuration']
        strengths = sorted(set(config['strengths']+(refinement.get('continuation') or {}).get('strengths', [])))
        lines.append(f'Refinement: amplitude H² strengths {strengths} · selected {record["strength"]:g} · '
                     'regularized checkpoint · paired noisy one-SE selection, then minimum curvature')
        lines.append(f'Prior fit: 8,000-step warm-up + Fisher search · refinement {config["steps"]:,} steps/candidate, max rank {config["rank"]}, modes {config["modes"]} · '
                     f'noisy validation NLL {refinement["baseline_nll"]:.4f} → {refinement["selected_nll"]:.4f}')
        if name.startswith('polish_'):
            continuation = refinement['continuation']
            lines.append(f'Continuation: {continuation["steps"]:,} further steps · H² strengths {continuation["strengths"]} · source {continuation["source"]}')
        kernel = read_model(case/'kernel.npz')
        lines.append(f'Kernel max rank {max(kernel.ranks)}, max modes {max(c.coeffs.shape[-1] for c in kernel.cores)} · '
                     'prepared TT convolution · latent box: 0.5–99.5% data quantiles + 5% padding')
    elif noisy:
        pilot = json.loads((case/'reference_selection.json').read_text())
        chosen = pilot['records'][pilot['selected']]
        kernel = read_model(case/'kernel.npz')
        selected_label = name.replace('_', ' ')
        if name.startswith('fisher_') and record['best_step'] == 0:
            selected_label += ' (warm-up retained)'
        lines.append(f'Warm-up: H² penalty 3e−5, 25 modes, {settings["steps"]:,} steps · Fisher strengths '
                     f'{settings["strengths"]}, floor 0.01 · selected {selected_label} · '
                     f'pilot degree {chosen["degree"]}, {"shared" if chosen["shared"] else "separate"}')
        lines.append(f'Kernel max rank {max(kernel.ranks)}, max modes {max(c.coeffs.shape[-1] for c in kernel.cores)} · '
                     'prepared TT convolution · latent box: 0.5–99.5% data quantiles + 5% padding · all noisy observations retained')
    else:
        lines.append('Unregularized NLL · training min/max box + 10% padding · checkpoint selected by validation NLL')
    seeds = 'train/validation seeds 0/2, 200 MCMC chains' if settings['family'] == 'gl' else 'split seed 0; training-based coordinate standardization'
    lines.append(f'KDE: LSCV, bandwidth range {float(min(bandwidth)):.3g}–{float(max(bandwidth)):.3g} · {seeds}')
    device = f' on {record["device"]}' if record.get('device') else ''
    lines.append(f'TT fit wall time: chosen run {duration_label(record["seconds"])}{device} · recorded candidate search {duration_label(candidate_fit_seconds(case))} '
                 '· includes compilation/validation; excludes initialization, KDE/kernel/pilot construction and plotting')
    return '\n'.join(lines)


def plot_gl(case, destination):
    from evaluate_gallery_fits import evaluate_case
    evaluate_case(case)
    error = json.loads((case/'integrated_l2.json').read_text())['relative_l2']
    settings = json.loads((case/'settings.json').read_text())
    dim, noise = settings['dim'], settings['noise']
    data = np.load(case/'data.npz')
    train = data['train']
    bandwidth = np.load(case/'bandwidth.npy')
    fit = read_model(case/'selected.npz')
    ptt = square_tt(fit)
    kernel = read_model(case/'kernel.npz') if noise != 'free' else None
    log_z = ginzburg_landau_log_normalizer(dim)
    domain = np.tile([-2.5, 2.5], (dim, 1))
    xs, extent = np.linspace(-2.5, 2.5, 180), (-2.5, 2.5, -2.5, 2.5)
    metrics = {}
    caption = settings_caption(case, fit, train, bandwidth)
    for i, j in dim_pairs_for(dim):
        coeff, box = pair_coefficients(ptt, i, j)
        true = _true_marginal_grid(dim, i, j, domain, log_z, n_grid=len(xs))
        kde = kde_grid(train, bandwidth, i, j, xs, xs)
        hist = histogram(train, i, j, extent)
        footer = caption + f'\nCoordinates {i}, {j} (zero based) · all panels on [−2.5, 2.5]² · error: unsquared, full-dimensional Lebesgue L2'
        if noise == 'free':
            pred = evaluate_pair(coeff, box, xs, xs)
            panels, titles = [true, hist, kde, pred], ['True GL marginal', 'Histogram of data', 'KDE', 'ft-de']
            heading = f'Noise-free Ginzburg–Landau · {dim} dimensions\nft-de relative L2 = {error:.4f}'
            filename = f'noise_free_dim{dim}_marg_{i}{j}.png'
            groups = None
        else:
            uc, ub = pair_coefficients(kernel, i, j)
            uq = evaluate_pair(uc, ub, xs, xs, negative_tolerance=1e-5)
            kernel_check = {}
            true_q, fit_q = convolved_grids(dim, i, j, coeff, box, uc, ub, xs, log_z, .0125,
                                           kernel_check=kernel_check)
            coarse = convolved_grids(dim, i, j, coeff, box, uc, ub, xs, log_z, .025)
            errors = [float(np.linalg.norm(a-b)/np.linalg.norm(a)) for a, b in zip((true_q, fit_q), coarse, strict=True)]
            if max(errors) > .01:
                raise FloatingPointError(f'Plot convolution grid needs refinement: {errors}')
            metrics[f'{i}{j}'] = {'fft_relative_refinement_change': errors,
                'observed_marginal_relative_l2': float(np.linalg.norm(fit_q-true_q)/np.linalg.norm(true_q)),
                'kernel_approximation': kernel_check}
            latent = evaluate_pair(coeff, box, xs, xs)
            metrics[f'{i}{j}']['latent_marginal_relative_l2'] = float(np.linalg.norm(latent-true)/np.linalg.norm(true))
            panels = [true, uq, true_q, hist, kde, latent]
            titles = ['True GL marginal', 'Noise kernel', 'Convolved marginal',
                      'Histogram of noisy data', 'KDE of noisy data', 'Deconvolved ft-de']
            groups = [0, 1, 2, 2, 2, 0]
            heading = (f'Noisy Ginzburg–Landau · {dim} dimensions · {"paired Gaussian" if noise == "paired" else "Gaussian mixture"} · scale 1.5'
                       f'\nDeconvolved ft-de relative L2 = {error:.4f}')
            filename = f'noisy_{noise}_scale1p5_dim{dim}_marg_{i}{j}.png'
        draw_panels(panels, titles, extent, i, j, heading, footer, destination/filename, groups)
    (case/'plot_checks.json').write_text(json.dumps(metrics, indent=2)+'\n')


def plot_cryoem(case, destination):
    settings = json.loads((case/'settings.json').read_text())
    dim = settings['dim']
    data = np.load(case/'data.npz')
    train, scale, shift = data['train'], data['scale'], data['shift']
    bandwidth = np.load(case/'bandwidth.npy')
    fit, naive = read_model(case/'selected.npz'), read_model(case/'naive/fit.npz')
    caption = settings_caption(case, fit, train, bandwidth)
    baseline = json.loads((case/'naive/result.json').read_text())
    caption += (f'\nNaive baseline: max rank {max(naive.ranks)}, modes {max(c.coeffs.shape[-1] for c in naive.cores)}, '
                f'{baseline["steps"]:,} steps (selected {baseline["best_step"]:,}), batch {baseline["batch_size"]}, '
                f'learning rate {baseline["lr"]:g} → {baseline["lr"]*.1:g} · fit wall time {duration_label(baseline["seconds"])}')
    if (case/'naive_initialization.json').exists():
        initialization = json.loads((case/'naive_initialization.json').read_text())
        if initialization['method'] == 'broadened KDE':
            widths = initialization['bandwidth']
            caption += f'\nNaive initialization: broadened KDE, bandwidth range {min(widths):.3g}–{max(widths):.3g}; unregularized NLL training'
    destination.mkdir(parents=True, exist_ok=True)
    selection = json.loads((case/'selection.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    for name in ['warmup', *[n for n in selection['names'] if n != 'warmup']]:
        record = json.loads((case/name/'result.json').read_text())
        ax = axes[0 if name == 'warmup' else 1]
        ax.plot([r['step'] for r in record['history']], [r['nll'] for r in record['history']],
                label=name.replace('_', ' '), linewidth=2)
        ax.scatter([record['best_step']], [record['best_nll']], s=25)
    refined = (case/'refinement.json').exists()
    corrected_projection = refined and json.loads((case/'refinement.json').read_text())['configuration'].get('correct_projection', False)
    warmup_title = 'Original warm-up (historical kernel)' if refined else 'Smoothness warm-up'
    for ax, title in zip(axes, [warmup_title, 'Refinement candidates' if refined else 'Fisher regularization candidates'], strict=True):
        ax.set(title=title, xlabel='Gradient steps', ylabel='Noisy validation NLL (standardized coordinates)')
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
    fig.suptitle(f'Cryo-EM · {dim} dimensions · selected {selection["selected"].replace("_", " ")}')
    fig.supxlabel(caption, fontsize=8, color='#52525b')
    fig.savefig(destination/f'cryoem_{"path_" if settings["path"] else ""}training_dim{dim}.png', dpi=180)
    plt.close(fig)
    if not settings['path']:
        kernel = read_model(case/'kernel.npz')
        pairs = [(0, 1), (2, 3)] + ([(8, 9)] if dim >= 10 else []) + ([(0, 19)] if dim == 20 else [])
        for i, j in pairs:
            box = np.asarray(fit.domain)[[i, j]]
            extent_z = tuple(box[0])+tuple(box[1])
            xs, ys = np.linspace(*box[0], 180), np.linspace(*box[1], 180)
            jacobian = scale[i]*scale[j]
            panels = [histogram(train, i, j, extent_z), kde_grid(train, bandwidth, i, j, xs, ys),
                      squared_pair_grid(naive, i, j, xs, ys), squared_pair_grid(fit, i, j, xs, ys)]
            panels = [z/jacobian for z in panels]
            noise = physical_noise_grid(kernel, i, j, xs*scale[i]+shift[i], ys*scale[j]+shift[j], scale)
            panels.insert(0, noise)
            extent = tuple(box[0]*scale[i]+shift[i])+tuple(box[1]*scale[j]+shift[j])
            noise_description = (json.loads((case/'refinement.json').read_text())['kernel_description']
                                 if refined else 'Prescribed diagonal Gaussian noise approximation')
            draw_panels(panels, ['Noise kernel', 'Histogram of data', 'KDE', 'ft-de of observations', 'Deconvolved ft-de'],
                        extent, i, j, f'Cryo-EM · {dim} dimensions',
                        caption+'\n'+noise_description+'; no clean-density ground truth',
                        destination/f'cryoem_dim{dim}_marg_{i}{j}.png', [0, 1, 1, 1, 2])
    if dim == 10 or settings['path']:
        import jax.numpy as jnp
        from cryoem_path_density import path_anchors, true_path_probability

        from ftde.kde import make_gaussian_kde
        cache = np.load(f'cryoem_data/cache/path_noise1_dim{dim}.npz')
        labels = cache['labels']
        prob, count = true_path_probability(labels)
        anchors = (path_anchors(cache['points'], labels, count)-shift)/scale
        # Simulation labels are first read here, after all fitting/selection.
        curves = {'State frequencies': prob}
        curves['KDE'] = np.asarray(make_gaussian_kde(jnp.asarray(train), jnp.asarray(bandwidth))(jnp.asarray(anchors)))
        for name, model in [('ft-de of observations', naive), ('Deconvolved ft-de', fit)]:
            inside = np.all((anchors >= np.asarray(model.domain)[:, 0]) & (anchors <= np.asarray(model.domain)[:, 1]), axis=1)
            curves[name] = np.where(inside, np.asarray(density_tt(model, jnp.asarray(anchors))), 0.)
        fig, ax = plt.subplots(figsize=(15, 6.3), layout='constrained')
        for name, values in curves.items():
            ax.plot(np.arange(count), values/values.sum(), label=name, linewidth=2,
                    linestyle='--' if name == 'State frequencies' else '-')
        ax.set(title=f'Cryo-EM path diagnostic · {dim} dimensions', xlabel='Conformational state',
               ylabel='Normalized anchor density / state frequency')
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
        noise_note = ''
        if refined:
            noise_note = '\n'+json.loads((case/'refinement.json').read_text())['kernel_description']
            if corrected_projection:
                noise_note += ' · covariance projected after full precision inversion'
        fig.supxlabel(caption+noise_note+'\nLabels used only after model selection. Anchor densities are a diagnostic, not state probability estimates.', fontsize=8)
        fig.savefig(destination/f'cryoem_path_dim{dim}.png', dpi=180)
        plt.close(fig)
        np.savez(case/'path_diagnostic.npz', **{k.replace(' ', '_'): v/v.sum() for k, v in curves.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', type=Path)
    parser.add_argument('--figures', type=Path, default=Path('figs'))
    args = parser.parse_args()
    for case in args.cases:
        settings = json.loads((case/'settings.json').read_text())
        if settings['family'] == 'gl':
            plot_gl(case, args.figures/'ginzburg_landau')
        else:
            plot_cryoem(case, args.figures/'cryoem')
        import jax
        jax.clear_caches()


if __name__ == '__main__':
    main()

"""Render swirl-pretzel figures from frozen fits; the clean truth is used only here.

Panels mirror the GL gallery: noise-free cases show truth, histogram, KDE and
ft-de; noisy cases add the noise kernel and the true convolved marginal and
show the deconvolved (latent) ft-de. The true convolved marginal is exact:
the tube density convolved with a Gaussian mixture is again a t-integral of
Gaussians. Every candidate from the fit search is also drawn, with ridge
profiles along the true path, to separate representation artifacts from
deconvolution artifacts. Candidate L2 errors are post-selection diagnostics.

Relative L2 is the unsquared, full-dimensional Lebesgue error
sqrt(int (q - p)^2 / int p^2): int q^2 is exact Gauss-Legendre quadrature of
the degree-4 amplitude polynomial, int p^2 is a double t-quadrature, and
int p q is a path quadrature of exact Gauss-Hermite axis integrals.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from gl_common import dim_pairs_for
from refresh_density_figures import draw_panels, duration_label, histogram, kde_grid
from refresh_density_fits import read_model
from scipy.special import roots_legendre
from swirl_fits import NOISE_WEIGHTS, noise_covariances

from ftde.marginals import squared_pair_grid
from ftde.swirl import (
    TUBE_STD,
    curve,
    path_quadrature,
    swirl_convolved_log_density,
    swirl_log_density,
    swirl_pair_marginal,
    swirl_squared_norm,
)

FIGS = Path('figs/swirl')
WINDOW = 2.5


def fourth_power_integral(g):
    """log int g^4 dx, exact: 4*(modes-1)-degree polynomial per axis, r^4 state."""
    log_total = 0.
    state = jnp.ones((1, 1, 1, 1))
    for core in g.cores:
        c = jnp.asarray(core.coeffs)/jnp.max(jnp.abs(core.coeffs))
        log_total += 4*np.log(float(jnp.max(jnp.abs(core.coeffs))))
        a, b = core.domain
        order = 2*c.shape[-1]
        t, w = roots_legendre(order)
        x = a+(b-a)*(t+1)/2
        basis = np.polynomial.chebyshev.chebvander(t, c.shape[-1]-1)
        values = jnp.einsum('xn,abn->xab', jnp.asarray(basis), c)
        weights = jnp.asarray(w*(b-a)/2)
        del x

        @jax.jit
        def advance(state, values=values, weights=weights):
            def body(acc, item):
                v, wt = item
                s = jnp.einsum('abcd,ai->ibcd', state, v)
                s = jnp.einsum('ibcd,bj->ijcd', s, v)
                s = jnp.einsum('ijcd,ck->ijkd', s, v)
                return acc+wt*jnp.einsum('ijkd,dl->ijkl', s, v), None
            r = values.shape[-1]
            out, _ = jax.lax.scan(body, jnp.zeros((r, r, r, r)), (values, weights))
            return out
        state = advance(state)
        scale = float(jnp.max(jnp.abs(state)))
        state = state/scale
        log_total += np.log(scale)
    return log_total+np.log(float(state.reshape(-1)[0]))


def cross_integral(scaled, dim, t_order, hermite_order=None):
    """int q_unnormalized(x) p(x) dx = int w(t) int g(x)^2 prod_k N(x_k; gamma_k(t), s^2) dx dt.

    For each path node, every axis integral is a Gaussian-weighted polynomial
    of degree 2*(modes-1), exact under Gauss-Hermite quadrature. q vanishes
    outside its box, so nodes there are dropped (their weights are < 1e-30
    whenever the path is at least a few tube widths inside the box).
    """
    modes = max(c.coeffs.shape[-1] for c in scaled.cores)
    hermite_order = hermite_order or modes+1
    h, hw = np.polynomial.hermite.hermgauss(hermite_order)
    t, w = path_quadrature(t_order)
    centers = curve(t, dim)
    total, log_scale = np.zeros(len(t)), np.zeros(len(t))
    for s in range(0, len(t), 256):
        state = jnp.ones((min(256, len(t)-s), 1, 1))
        logs = jnp.zeros(state.shape[0])
        for k, core in enumerate(scaled.cores):
            a, b = core.domain
            x = centers[s:s+256, k, None]+np.sqrt(2)*TUBE_STD*h[None]
            inside = (x >= a) & (x <= b)
            basis = np.polynomial.chebyshev.chebvander(np.clip(2*(x-a)/(b-a)-1, -1, 1), core.coeffs.shape[-1]-1)
            values = jnp.einsum('tzn,abn->tzab', jnp.asarray(basis), core.coeffs)
            weights = jnp.asarray(np.where(inside, hw[None]/np.sqrt(np.pi), 0.))
            state = jnp.einsum('tac,tz,tzab,tzcd->tbd', state, weights, values, values)
            scale = jnp.max(jnp.abs(state), axis=(1, 2))
            state, logs = state/scale[:, None, None], logs+jnp.log(scale)
        total[s:s+256] = np.asarray(state[:, 0, 0])
        log_scale[s:s+256] = np.asarray(logs)
    peak = log_scale.max()
    return float(np.log(np.sum(w*total*np.exp(log_scale-peak)))+peak)


def integrated_l2(case, fit, dim, name='selected'):
    """Full-dimensional relative L2 of q = g^2/int g^2 against the swirl truth."""
    from ftde.tt_density import squared_integral_tt
    cores = [c.coeffs/jnp.max(jnp.abs(c.coeffs)) for c in fit.cores]
    scaled = fit.__class__(cores=tuple(type(c)(k, c.domain) for c, k in zip(fit.cores, cores, strict=True)),
                           domain=fit.domain)
    log_z = float(jnp.log(squared_integral_tt(scaled)))
    q2 = float(np.exp(fourth_power_integral(scaled)-2*log_z))
    p2 = swirl_squared_norm(dim)
    pq, pq_fine = (float(np.exp(cross_integral(scaled, dim, n)-log_z)) for n in (4000, 8000))
    squared = (q2-2*pq_fine+p2)/p2
    return {'relative_l2': float(np.sqrt(max(squared, 0.))), 'relative_squared_l2': float(squared),
            'integral_q_squared': q2, 'integral_pq': pq_fine, 'integral_p_squared': p2,
            'path_quadrature_relative_change': abs(pq-pq_fine)/p2, 'model': name,
            'metric': 'Lebesgue L2 of the latent full-dimensional density'}


def cached_l2(case, name, fit, dim):
    path = case/'integrated_l2.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    fingerprint = hashlib.sha256(b''.join(np.asarray(c.coeffs).tobytes() for c in fit.cores)).hexdigest()
    if records.get(name, {}).get('sha256') != fingerprint:
        records[name] = integrated_l2(case, fit, dim, name=name) | {'sha256': fingerprint}
        path.write_text(json.dumps(records, indent=2)+'\n')
        print(case.name, name, 'relative L2', records[name]['relative_l2'], flush=True)
    return records[name]


def oracle_nll(case, settings):
    """Validation NLL of the true observation density (reference only)."""
    valid = np.load(case/'data.npz')['valid']
    if settings['noise'] == 'free':
        return float(-swirl_log_density(valid).mean())
    return float(-swirl_convolved_log_density(valid, noise_covariances(settings['dim']), NOISE_WEIGHTS).mean())


def mixture_pair(covs, weights, i, j, xs, ys, center=None):
    xx, yy = np.meshgrid(xs, ys)
    out = np.zeros_like(xx)
    for w, c in zip(weights, covs, strict=True):
        c2 = c[np.ix_([i, j], [i, j])]
        inv = np.linalg.inv(c2)
        q = inv[0, 0]*xx**2+2*inv[0, 1]*xx*yy+inv[1, 1]*yy**2
        out += w*np.exp(-.5*q)/(2*np.pi*np.sqrt(np.linalg.det(c2)))
    return out


def convolved_truth(dim, i, j, xs, ys, order=4000):
    """Exact pair marginal of (swirl tube) * (Gaussian mixture noise)."""
    t, w = path_quadrature(order)
    g = curve(t, dim)
    out = np.zeros((len(ys), len(xs)))
    for pi, c in zip(NOISE_WEIGHTS, noise_covariances(dim), strict=True):
        cov = c[np.ix_([i, j], [i, j])]+TUBE_STD**2*np.eye(2)
        inv = np.linalg.inv(cov)
        norm = pi/(2*np.pi*np.sqrt(np.linalg.det(cov)))
        dx = np.asarray(xs)[None, :, None]-g[None, None, :, i]
        dy = np.asarray(ys)[:, None, None]-g[None, None, :, j]
        for s in range(0, len(ys), 20):
            q = inv[0, 0]*dx**2+2*inv[0, 1]*dx*dy[s:s+20]+inv[1, 1]*dy[s:s+20]**2
            out[s:s+20] += norm*np.exp(-.5*q)@w
    return out


def ridge_profile(grid, xs, dim, i, j, t):
    """Bilinear lookup of a (y, x) grid along the true projected path."""
    from scipy.interpolate import RegularGridInterpolator
    path = curve(t, dim)[:, [j, i]]
    return RegularGridInterpolator((xs, xs), grid, bounds_error=False, fill_value=0.)(path)


def caption(case, settings, fit, record, train, bandwidth, selection, l2):
    noisy = settings['noise'] != 'free'
    modes = max(c.coeffs.shape[-1] for c in fit.cores)
    init = f'KDE, rank {settings["init_rank"]}, 25 modes, 64 starts, 10 search steps'
    if (case/'initialization.json').exists():
        factor = json.loads((case/'initialization.json').read_text())['attempts'][-1]['bandwidth_factor']
        if factor != '1.0':
            init += f' (bandwidth broadened ×{factor} after pivot-search collapse)'
    lines = [(f'N train = {len(train):,} · N validation = {settings["n_valid"]:,} · max TT rank = {max(fit.ranks)} · '
              f'Chebyshev amplitude modes/axis = {modes} (degree {modes-1}) · init: {init}')]
    chosen_nll = selection['validation_nll'][selection['names'].index(selection['selected'])]
    lines.append(f'Validation NLL {chosen_nll:.4f} · true {"observation " if noisy else ""}density on the same validation set '
                 f'{oracle_nll(case, settings):.4f} (reference only; never used for fitting or selection)')
    lines.append(f'Selected {selection["selected"].replace("_", " ")} · Adam · {record["steps"]:,} steps · selected step {record["best_step"]:,} · '
                 f'batch {record["batch_size"]} · learning rate {record["lr"]:g} → {record["lr"]*.1:g} (cosine) · gradient clip 1 · '
                 f'{"Sobolev" if record.get("preconditioning") == "sobolev" else "Gram"} preconditioning')
    if noisy:
        kernel = read_model(case/'kernel.npz')
        lines.append(f'Base: amplitude H² 3e−5, 25 modes, {settings["base_steps"]:,} steps · continuations: ranks {settings["ranks"]}, '
                     f'{settings["modes"]} modes, H² strengths {settings["strengths"]}, {settings["steps"]:,} steps, regularized checkpoints')
        lines.append(f'Selection: {selection["rule"]} · kernel TT ranks ≤ {max(kernel.ranks)}, {kernel.cores[0].coeffs.shape[-1]} modes · '
                     'prepared TT convolution · latent box: 0.5–99.5% data quantiles + 5% padding · all noisy observations retained')
        lines.append('Noise: 0.5·N(0, 0.22² I) + 0.3·N(0, diag(0.45², 0.25², …)) + 0.2·N(0, 0.5² [[1, ±0.6], [±0.6, 1]] per pair) · '
                     f'clean tube σ {TUBE_STD}')
    else:
        lines.append(f'Base: unregularized NLL, {settings["base_steps"]:,} steps, lr 0.001, batch 2048 · continuations: ranks {settings["ranks"]}, '
                     f'{settings["modes"]} modes, {settings["steps"]:,} steps · {selection["rule"]} · training min/max box + 10% padding')
    lines.append(f'KDE: LSCV, bandwidth range {float(min(bandwidth)):.3g}–{float(max(bandwidth)):.3g} · exact i.i.d. samples, train/validation seeds 0/2')
    device = f' on {record["device"]}' if record.get('device') else ''
    total = sum(json.loads((case/n/'result.json').read_text())['seconds'] for n in selection['names'])
    lines.append(f'TT fit wall time: chosen run {duration_label(record["seconds"])}{device} · recorded candidate search {duration_label(total)} '
                 '· includes compilation/validation; excludes initialization, KDE/kernel construction and plotting')
    lines.append('Error: unsquared, full-dimensional Lebesgue L2 · ∫q² Gauss–Legendre, ∫pq path × Gauss–Hermite quadrature, '
                 f'∫p² double path quadrature (path-quadrature change {l2["path_quadrature_relative_change"]:.1e})')
    return '\n'.join(lines)


def plot_case(case, figures):
    settings = json.loads((case/'settings.json').read_text())
    dim, noise = settings['dim'], settings['noise']
    noisy = noise != 'free'
    data = np.load(case/'data.npz')
    train = data['train']
    bandwidth = np.load(case/'bandwidth.npy')
    selection = json.loads((case/'selection.json').read_text())
    fit = read_model(case/'selected.npz')
    record = json.loads((case/selection['selected']/'result.json').read_text())
    l2 = cached_l2(case, selection['selected'], fit, dim)
    xs = np.linspace(-WINDOW, WINDOW, 180)
    extent = (-WINDOW, WINDOW, -WINDOW, WINDOW)
    text = caption(case, settings, fit, record, train, bandwidth, selection, l2)
    label = 'noise_free' if not noisy else 'noisy_mixture'
    for i, j in dim_pairs_for(dim):
        truth = swirl_pair_marginal(dim, i, j, xs, xs)
        estimate = squared_pair_grid(fit, i, j, xs, xs)
        hist = histogram(train, i, j, extent)
        kde = kde_grid(train, bandwidth, i, j, xs, xs)
        footer = text+f'\nCoordinates {i}, {j} (zero based) · all panels on [−{WINDOW}, {WINDOW}]²'
        if noisy:
            kernel = mixture_pair(noise_covariances(dim), NOISE_WEIGHTS, i, j, xs, xs)
            panels = [truth, kernel, convolved_truth(dim, i, j, xs, xs), hist, kde, estimate]
            titles = ['True swirl-pretzel marginal', 'Noise kernel', 'Convolved marginal',
                      'Histogram of noisy data', 'KDE of noisy data', 'Deconvolved ft-de']
            groups = [0, 1, 2, 2, 2, 0]
            heading = (f'Noisy swirl-pretzel · {dim} dimensions · Gaussian mixture'
                       f'\nDeconvolved ft-de relative L2 = {l2["relative_l2"]:.4f}')
        else:
            panels, titles, groups = [truth, hist, kde, estimate], ['True swirl-pretzel marginal', 'Histogram of data', 'KDE', 'ft-de'], None
            heading = f'Noise-free swirl-pretzel · {dim} dimensions\nft-de relative L2 = {l2["relative_l2"]:.4f}'
        draw_panels(panels, titles, extent, i, j, heading, footer, figures/f'{label}_dim{dim}_marg_{i}{j}.png', groups)
    plot_candidates(case, figures/f'{label}_candidates_dim{dim}.png', settings, selection)


def plot_candidates(case, path, settings, selection):
    """Every searched candidate on the (0,1) and (2,3) pairs, plus ridge profiles."""
    dim = settings['dim']
    names = selection['names']
    xs = np.linspace(-WINDOW, WINDOW, 180)
    extent = (-WINDOW, WINDOW, -WINDOW, WINDOW)
    t = np.linspace(0, 1, 1500)
    pairs = [(0, 1), (2, 3)]
    fig, axes = plt.subplots(len(pairs)*2, len(names)+1, figsize=(3.3*(len(names)+1), 13.5),
                             layout='constrained', squeeze=False,
                             gridspec_kw={'height_ratios': [1, .55]*len(pairs)})
    fits = {n: read_model(case/n/'fit.npz') for n in names}
    errors = {n: cached_l2(case, n, fits[n], dim)['relative_l2'] for n in names}
    for row, (i, j) in enumerate(pairs):
        truth = swirl_pair_marginal(dim, i, j, xs, xs)
        grids = [truth]+[squared_pair_grid(fits[n], i, j, xs, xs) for n in names]
        vmax = max(g.max() for g in grids)
        true_profile = ridge_profile(truth, xs, dim, i, j, t)
        for col, (grid, name) in enumerate(zip(grids, ['truth']+names, strict=True)):
            ax = axes[2*row, col]
            ax.imshow(grid, extent=extent, origin='lower', cmap='magma', vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            if name == 'truth':
                ax.set_title(f'True marginal ({i},{j})')
            else:
                nll = selection['validation_nll'][names.index(name)]
                star = ' ★' if name == selection['selected'] else ''
                ax.set_title(f'{name.replace("_", " ")}{star}\nNLL {nll:.4f} · L2 {errors[name]:.3f}', fontsize=10)
            prof = axes[2*row+1, col]
            prof.plot(t, true_profile, color='#52525b', lw=1.2, label='truth')
            if name != 'truth':
                prof.plot(t, ridge_profile(grid, xs, dim, i, j, t), color='#dd5b00', lw=1.2, label='ft-de')
            prof.set_ylim(0, 1.05*vmax)
            prof.set_xlabel('path parameter t', fontsize=9)
            prof.tick_params(labelsize=8)
            prof.spines[['top', 'right']].set_visible(False)
            if col == 0:
                prof.set_ylabel('density on true ridge', fontsize=9)
                prof.legend(frameon=False, fontsize=8)
    kind = 'noise-free' if settings['noise'] == 'free' else 'noisy (deconvolved latent density)'
    fig.suptitle(f'Swirl-pretzel · {dim} dimensions · {kind} · all searched candidates\n'
                 f'true {"observation " if settings["noise"] != "free" else ""}density validation NLL {oracle_nll(case, settings):.4f}',
                 fontsize=15, fontweight='bold')
    fig.supxlabel('Profiles evaluate each pair marginal along the projected true path. A continuous ridge gives a smooth '
                  'profile; beads appear as oscillations.\nNLL = validation NLL (selection criterion) · L2 = full-dimensional '
                  'relative L2 against the clean truth (post-selection diagnostic only) · ★ selected', fontsize=9, color='#52525b')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('Saved', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+', type=Path)
    parser.add_argument('--figures', type=Path, default=FIGS)
    args = parser.parse_args()
    for case in args.cases:
        plot_case(case, args.figures)
        jax.clear_caches()


if __name__ == '__main__':
    main()

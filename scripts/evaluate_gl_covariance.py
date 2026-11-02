"""Final evaluation and figures, after all noisy-NLL checkpoint selections."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from evaluate_gl_l2 import integrated_terms, relative_l2_from_terms
from gl_covariance_common import load_record, load_tt, noise_density
from scipy.special import roots_legendre
from select_gl_likelihood import select_with_paired_uncertainty

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.gaussian_convolution import _transfer
from ftde.ginzburg_landau import GL_BETA, GL_LAM, ginzburg_landau_log_normalizer
from ftde.tt_density import rebuild_tt, squared_integral_tt, tt_params


def marginal(g, points, axes):
    state = jnp.ones((len(points), 1, 1))
    for k, core in enumerate(g.cores):
        if k in axes:
            a = core(points[:, axes.index(k)])
            state = jnp.einsum('bpq,bpi,bqj->bij', state, a, a)
        else:
            gram = chebyshev_gram_matrix(core.coeffs.shape[-1], core.domain)
            state = _transfer(state, core.coeffs, gram[None])
    return state[:, 0, 0]/squared_integral_tt(g)


def gl_marginal(d, x, y=None, axis=0):
    """1D or adjacent-pair marginal from independent GL transfer integration."""
    t, w = roots_legendre(160)
    nodes, weights = 3*t, 3*w
    scale = GL_BETA*GL_LAM*(d+1)**2/2
    site = lambda z: np.exp(-GL_BETA*(1-z**2)**2/(4*GL_LAM))
    edge = lambda a, b: np.exp(-scale*(a-b)**2)
    matrix = edge(nodes[:, None], nodes[None, :])
    def message(length, query):
        if length == 0:
            return edge(0, query)
        state = edge(0, nodes)*site(nodes)*weights
        for _ in range(length-1):
            state = (state@matrix)*site(nodes)*weights
        return state@edge(nodes[:, None], np.asarray(query).reshape(1, -1))
    z = np.exp(ginzburg_landau_log_normalizer(d))
    if y is None:
        return message(axis, x)*site(x)*message(d-axis-1, x)/z
    return (message(axis, x)*site(x)*edge(x, y)*site(y)*message(d-axis-2, y))/z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/gl_covariance_mixture'))
    parser.add_argument('--selection-rule', choices=['minimum', 'paired_one_se'], default='minimum')
    parser.add_argument('--keep-workfiles', action='store_true',
                        help='retain datasets, checkpoints and run metadata after plotting')
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/gl_covariance_mixture'))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    directory = args.directory
    records = [load_record(p) for p in sorted(directory.glob('*/result.npz'))]
    if not records:
        raise ValueError('No finished runs')
    # Freeze selection before evaluating any true GL density errors.
    eligible = [r for r in records if not r['naive']]
    selected = min(eligible, key=lambda r: r['best_nll'])['name']
    data = np.load(directory/'data.npz')
    if args.selection_rule == 'paired_one_se':
        selected, _ = select_with_paired_uncertainty(directory, records, data)
    d = data['train'].shape[1]
    log_z = ginzburg_landau_log_normalizer(d)
    p2 = float(np.exp(ginzburg_landau_log_normalizer(d, beta=2*GL_BETA)-2*log_z))
    metrics = {}
    for record in records:
        name = record['name']
        best = load_tt(directory/name/'best.npz')
        evaluate = jax.jit(lambda p, best=best: integrated_terms(rebuild_tt(best, p), 80, log_z))
        q2, pq = map(float, evaluate(tt_params(best)))
        fine = jax.jit(lambda p, best=best: integrated_terms(rebuild_tt(best, p), 160, log_z))
        q2fine, pqfine = map(float, fine(tt_params(best)))
        relative = relative_l2_from_terms(q2fine, pqfine, p2)
        history = []
        for entry in record['history']:
            g = load_tt(directory/name/f"step_{entry['step']:06d}.npz")
            a, b = map(float, evaluate(tt_params(g)))
            history.append({**entry, 'relative_l2': relative_l2_from_terms(a, b, p2)})
        metrics[name] = {'relative_l2': float(relative), 'validation_nll': record['best_nll'],
                         'best_step': record['best_step'], 'history': history,
                         'quadrature_change_relative_to_p2': max(abs(q2-q2fine), abs(pq-pqfine))/p2}
        print(name, relative, flush=True)
        jax.clear_caches()
    figures = args.figure_dir
    g = load_tt(directory/selected/'best.npz')
    selected_record = next(r for r in records if r['name'] == selected)
    naive_name = next((r['name'] for r in records if r['naive']), None)
    naive_record = next((r for r in records if r['name'] == naive_name), None)
    naive = load_tt(directory/naive_name/'best.npz') if naive_name else None
    error = metrics[selected]['relative_l2']
    naive_error = metrics[naive_name]['relative_l2'] if naive_name else float('nan')
    degree = selected_record['n_coeffs']-1
    noise_scale = selected_record.get('noise_scale', float(data['noise_scale']) if 'noise_scale' in data else 1.)
    caption = (f'{d}D noisy GL | noise std scale={noise_scale:.2f} | n={len(data["train"]):,} | '
               f'rank={selected_record["rank"]}, '
               f'{selected_record["n_coeffs"]} coeffs (degree {degree}), batch={selected_record["batch_size"]}\n'
               f'deconvolution steps={selected_record["steps"]:,} (selected {selected_record["best_step"]:,}) | '
               f'naive steps={naive_record["steps"]:,} (selected {naive_record["best_step"]:,}) | '
               f'full L2={error:.4f}, naive full L2={naive_error:.4f}')

    def save(fig, name):
        fig.savefig(figures/f'{name}.png', dpi=170)
        plt.close(fig)

    # The two compact error summaries replace loose metric/report files.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for name, result in metrics.items():
        if name.startswith('avg_'):
            continue
        steps = [h['step'] for h in result['history']]
        axs[0].plot(steps, [h['validation_nll'] for h in result['history']], label=f"{name}: L2={result['relative_l2']:.3f}")
        axs[1].plot(steps, [h['relative_l2'] for h in result['history']], label=name)
    axs[0].set(xlabel='Step', ylabel='Validation NLL', title='Naive NLL uses in-box validation only')
    axs[1].set(xlabel='Step', ylabel='Integrated relative L2', title='Computed only after checkpoint selection')
    if selected.startswith('avg_'):
        point = metrics[selected]
        axs[0].scatter(point['best_step'], point['validation_nll'], marker='*', s=100, color='black', label='Selected late average')
        axs[1].scatter(point['best_step'], point['relative_l2'], marker='*', s=100, color='black')
    axs[0].legend(fontsize=7)
    axs[1].set_ylim(bottom=0)
    fig.suptitle(caption)
    save(fig, 'training_errors')

    fig, ax = plt.subplots(figsize=(14, max(4.5, .38*len(metrics)+1)), constrained_layout=True)
    bars = ax.barh(list(metrics), [m['relative_l2'] for m in metrics.values()],
                   color=['tab:green' if name == selected else 'tab:blue' for name in metrics])
    ax.bar_label(bars, fmt='%.4f')
    ax.invert_yaxis()
    ax.set(xlabel='Integrated relative L2', title=caption)
    ax.set_xlim(right=1.13*max(m['relative_l2'] for m in metrics.values()))
    save(fig, 'full_density_errors')

    # Previous-GL-style composite: truth, observed data and noise on the top;
    # naive fit, deconvolution and its signed error on the bottom.
    covs, weights = data['covs'], data['weights']
    x = np.linspace(-1.85, 1.85, 121)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    for pair in sorted({0, 2*(d//4), d-2}):
        truth = gl_marginal(d, points[:, 0], points[:, 1], pair).reshape(xx.shape)
        estimate = np.asarray(jax.jit(lambda p, pair=pair: marginal(g, p, (pair, pair+1)))(jnp.asarray(points))).reshape(xx.shape)
        naive_grid = np.asarray(jax.jit(lambda p, pair=pair: marginal(naive, p, (pair, pair+1)))(jnp.asarray(points))).reshape(xx.shape) if naive else None
        marginal_error = np.linalg.norm(estimate-truth)/np.linalg.norm(truth)
        naive_marginal_error = np.linalg.norm(naive_grid-truth)/np.linalg.norm(truth)
        h, xedges, yedges = np.histogram2d(data['train'][:, pair], data['train'][:, pair+1],
                                           bins=120, range=[[-1.85, 1.85], [-1.85, 1.85]], density=True)
        noise = noise_density(points, weights, covs[:, pair:pair+2, pair:pair+2]).reshape(xx.shape)
        vmax = max(truth.max(), estimate.max(), naive_grid.max(), h.max())
        fig, axs = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
        ordinary = ((axs[0, 0], truth, 'True GL marginal'),
                    (axs[1, 0], naive_grid, f'Naive TT; marginal L2={naive_marginal_error:.3f}'),
                    (axs[1, 1], estimate, f'Deconvolved TT; marginal L2={marginal_error:.3f}'))
        for ax, z, title in ordinary:
            ax.imshow(z.T, extent=(-1.85, 1.85, -1.85, 1.85), origin='lower', aspect='equal',
                      cmap='viridis', vmin=0, vmax=vmax)
            ax.set(title=title, xlabel=f'x{pair+1}', ylabel=f'x{pair+2}')
        axs[0, 1].imshow(h.T, extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), origin='lower',
                         aspect='equal', cmap='viridis', vmin=0, vmax=vmax)
        axs[0, 1].set(title=f'Observed noisy data ({len(data["train"]):,} points)',
                      xlabel=f'y{pair+1}', ylabel=f'y{pair+2}')
        noise_image = axs[0, 2].imshow(noise.T, extent=(-1.85, 1.85, -1.85, 1.85), origin='lower',
                                       aspect='equal', cmap='magma')
        axs[0, 2].set(title='Prescribed 3-component noise mixture',
                      xlabel=f'noise {pair+1}', ylabel=f'noise {pair+2}')
        fig.colorbar(noise_image, ax=axs[0, 2], shrink=.8)
        signed = estimate-truth
        signed_scale = np.max(np.abs(signed))
        signed_image = axs[1, 2].imshow(signed.T, extent=(-1.85, 1.85, -1.85, 1.85), origin='lower',
                                        aspect='equal', cmap='coolwarm', vmin=-signed_scale, vmax=signed_scale)
        axs[1, 2].set(title='Deconvolved signed marginal error',
                      xlabel=f'x{pair+1}', ylabel=f'x{pair+2}')
        fig.colorbar(signed_image, ax=axs[1, 2], shrink=.8)
        fig.suptitle(caption)
        save(fig, f'marginal_{pair+1}_{pair+2}')
    print('selected', selected, 'figures', figures, flush=True)
    data.close()


if __name__ == '__main__':
    main()

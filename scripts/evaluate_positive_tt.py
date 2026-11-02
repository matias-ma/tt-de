"""Freeze observation-only selection, then evaluate/plot positive Chebyshev TTs."""
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
from refresh_density_fits import read_model
from scipy.special import roots_legendre

from ftde.marginals import squared_pair_grid
from ftde.positive_chebyshev import PositiveChebyshevTT
from ftde.positive_tt_fit import PositiveTTForward
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import squared_integral_tt


def cross_integral(model, path_order, axis_order):
    from ftde.swirl import TUBE_STD, curve, path_quadrature
    t, weights = path_quadrature(path_order)
    centers = curve(t, len(model.tt.cores))
    state = np.ones((len(t), 1))
    nodes, w = roots_legendre(axis_order)
    for k, core in enumerate(model.tt.cores):
        a, b = core.domain
        x = a+(b-a)*(nodes+1)/2
        kernel = np.exp(-.5*((centers[:, k, None]-x)/TUBE_STD)**2)/(np.sqrt(2*np.pi)*TUBE_STD)
        basis = np.polynomial.chebyshev.chebvander(nodes, core.coeffs.shape[-1]-1)
        moments = (kernel*(w*(b-a)/2))@basis
        values = np.einsum('tm,abm->tab', moments, np.asarray(core.coeffs))
        state = np.einsum('ta,tab->tb', state, values)
    return float(weights@state[:, 0])


def l2_error(model):
    from ftde.swirl import swirl_squared_norm
    q2 = float(squared_integral_tt(model.tt))
    p2 = swirl_squared_norm(len(model.tt.cores), order=2400)
    pq = cross_integral(model, 1600, 160)
    pq_fine = cross_integral(model, 2400, 240)
    squared = (q2-2*pq_fine+p2)/p2
    if squared < -1e-8:
        raise FloatingPointError('Negative squared error')
    return {'relative_l2': float(np.sqrt(max(0., squared))), 'integral_q_squared': q2,
            'integral_p_squared': p2, 'integral_pq': pq_fine,
            'refined_cross_integral_change_relative_to_p2': abs(pq-pq_fine)/p2}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--directories', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = json.loads((args.base/'settings.json').read_text())
    dim = settings['dim']
    results = [json.loads((d/'result.json').read_text()) for d in args.directories]
    chosen = int(np.argmin([r['best_nll'] for r in results]))
    directory = args.directories[chosen]
    selection = {'selected': str(directory), 'rule': 'minimum noisy validation NLL; frozen before truth evaluation',
                 'candidates': [str(d) for d in args.directories], 'validation_nll': [r['best_nll'] for r in results],
                 'selected_sha256': hashlib.sha256((directory/'fit.npz').read_bytes()).hexdigest()}
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    model = PositiveChebyshevTT.load(directory/'fit.npz')
    model.save(args.output/'selected.npz')
    old = read_model(args.base/'selected.npz')
    kernel = read_model(args.base/'kernel.npz')
    data = np.load(args.base/'data.npz')
    audit = model.audit()
    is_swirl = settings['family'] == 'swirl'
    fresh = None
    if is_swirl:
        from swirl_fits import observations
        fresh = observations(dim, 'mixture', 8192, 73129)
    all_obs = np.concatenate([data['train'], data['valid']]+([] if fresh is None else [fresh]))
    obs_domain = np.stack([all_obs.min(axis=0)-.02, all_obs.max(axis=0)+.02], axis=1)
    forward = PositiveTTForward(model, obs_domain, kernel)
    params = forward.parameters(model)
    evaluate = jax.jit(lambda x: forward.log_density(params, x))
    old_op = TTConvolution.from_tt(old, kernel).prepare(obs_domain, tolerance=1e-14)
    evaluate_old = jax.jit(lambda x: old_op.log_density(old, x))
    for label, x in [('validation', data['valid'])]+([] if fresh is None else [('test', fresh)]):
        logs = np.concatenate([np.asarray(evaluate(jnp.asarray(x[k:k+256]))) for k in range(0, len(x), 256)])
        old_logs = np.concatenate([np.asarray(evaluate_old(jnp.asarray(x[k:k+128]))) for k in range(0, len(x), 128)])
        np.save(args.output/f'{label}.npy', logs)
        np.save(args.output/f'old_{label}.npy', old_logs)
        delta = logs-old_logs
        audit[label] = {'nll': float(-logs.mean()), 'old_nll': float(-old_logs.mean()),
                        'gain': float(delta.mean()), 'paired_se': float(delta.std(ddof=1)/np.sqrt(len(delta)))}
    np.testing.assert_allclose(np.load(directory/'validation.npy'), np.load(args.output/'validation.npy'), rtol=0, atol=2e-9)
    # Compare prepared moments against independently doubled integration.
    from ftde.linear_tt_convolution import LinearTTConvolution
    refined = LinearTTConvolution.from_tt(model.tt, kernel, obs_domain,
                 quadrature_order=2*np.asarray(forward.operator.prepared.orders))
    check = np.asarray(refined.log_density(model.tt, jnp.asarray(data['valid'][:16])))
    audit['doubled_convolution_quadrature_max_log_change'] = float(np.max(np.abs(check-np.load(args.output/'validation.npy')[:16])))
    if is_swirl:
        audit.update(l2_error(model))
        original_metrics = json.loads((args.base/'integrated_l2.json').read_text())
        old_hash = hashlib.sha256(b''.join(np.asarray(c.coeffs).tobytes() for c in old.cores)).hexdigest()
        matching = [r for r in original_metrics.values() if r.get('sha256') == old_hash]
        if not matching:
            raise ValueError('No matching original full-dimensional L2 audit')
        audit['old_relative_l2'] = matching[0]['relative_l2']
    (args.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('AUDIT', args.output, audit, flush=True)

    # Figures retain existing folders; numerical records stay in artifacts.
    figs = Path('figs/swirl' if is_swirl else 'figs/cryoem')
    figs.mkdir(exist_ok=True)
    pairs = [(0, 1), (2, 3)]
    if dim >= 10:
        pairs.append((8, 9))
    xs = np.linspace(-2.5, 2.5, 181) if is_swirl else None
    fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 4*len(pairs)), layout='constrained')
    for row, (i, j) in enumerate(pairs):
        xv = xs if xs is not None else np.linspace(*np.asarray(model.tt.domain)[i], 181)
        yv = xs if xs is not None else np.linspace(*np.asarray(model.tt.domain)[j], 181)
        grids = [squared_pair_grid(old, i, j, xv, yv), model.pair_grid(i, j, xv, yv)]
        titles = ['Original squared TT', 'Positive-core Chebyshev TT']
        if is_swirl:
            from ftde.swirl import swirl_pair_marginal
            grids.insert(0, swirl_pair_marginal(dim, i, j, xv, yv))
            titles.insert(0, 'True latent density')
        else:
            from scipy.stats import gaussian_kde
            xx, yy = np.meshgrid(xv, yv)
            observed = gaussian_kde(data['train'][:20000, [i, j]].T)
            grids.insert(0, observed(np.stack([xx.ravel(), yy.ravel()])).reshape(xx.shape))
            titles.insert(0, 'Observed KDE (contains noise)')
        peak = max(float(g.max()) for g in grids)
        for ax, grid, title in zip(axes[row], grids, titles, strict=True):
            im = ax.pcolormesh(xv, yv, grid, shading='nearest', cmap='magma', vmin=0, vmax=peak)
            ax.set(title=title, xlabel=f'x{i}', ylabel=f'x{j}', aspect='equal')
            fig.colorbar(im, ax=ax, shrink=.8)
    caption = f'Rank {max(model.tt.ranks)}, {model.tt.cores[0].coeffs.shape[-1]} Chebyshev modes. Direct density TT; certified positive polynomial cores; no final square or clipping.'
    if is_swirl:
        caption += f'\nFull {dim}D relative L2: original {audit["old_relative_l2"]:.3f}; positive cores {audit["relative_l2"]:.3f}. Truth evaluated only after noisy-likelihood selection.'
    else:
        caption += '\nSelection uses noisy validation likelihood. The latent cryo-EM density is unknown; likelihood gains do not establish latent recovery.'
    fig.suptitle(f'{dim}D {"noisy swirl" if is_swirl else "cryo-EM"}: positive Chebyshev cores', fontweight='bold')
    fig.supxlabel(caption, fontsize=9)
    name = ('noisy_mixture_positive' if is_swirl else 'positive_chebyshev')+f'_dim{dim}'
    if settings.get('path'):
        name += '_path'
    fig.savefig(figs/f'{name}.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), layout='constrained')
    for d, result in zip(args.directories, results, strict=True):
        history = result['history']
        axes[0].plot([r['sweep'] for r in history], [r['nll'] for r in history], label=d.name)
        axes[1].plot([r['sweep'] for r in history], [r['nll'] for r in history])
    axes[0].axhline(audit['validation']['old_nll'], color='black', ls='--', label='Original squared TT')
    axes[0].set(xlabel='Alternating sweep', ylabel='Noisy validation NLL', title='Checkpoint selection sees observations only')
    axes[0].legend(fontsize=7)
    axes[1].axhline(audit['validation']['old_nll'], color='black', ls='--')
    low = min(audit['validation']['old_nll'], min(r['best_nll'] for r in results))-.003
    high = min(max(r['best_nll'] for r in results)+.008, low+.15)
    axes[1].set(xlabel='Alternating sweep', ylabel='Noisy validation NLL', title='Detail near the best likelihoods', ylim=(low, high))
    labels = ['validation']+([] if fresh is None else ['test'])
    axes[2].errorbar(labels, [audit[k]['gain'] for k in labels], yerr=[2*audit[k]['paired_se'] for k in labels], fmt='o', capsize=5)
    axes[2].axhline(0, color='black', ls='--')
    axes[2].set(ylabel='Log-likelihood gain over original TT', title='Selected positive-core TT (bars: ±2 paired SE)')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f'{dim}D: training and likelihood comparison')
    fig.savefig(figs/f'{name}_training.png', dpi=180)
    plt.close(fig)

    if is_swirl:
        from scipy.interpolate import RegularGridInterpolator

        from ftde.swirl import curve, swirl_pair_marginal
        t = np.linspace(0, 1, 1200)
        path = curve(t, dim)
        fig, axes = plt.subplots(1, len(pairs), figsize=(6*len(pairs), 4.5), layout='constrained')
        for ax, (i, j) in zip(axes, pairs, strict=True):
            grids = [swirl_pair_marginal(dim, i, j, xs, xs),
                     squared_pair_grid(old, i, j, xs, xs), model.pair_grid(i, j, xs, xs)]
            for grid, label, color in zip(grids, ['Truth', 'Original squared TT', 'Positive-core Chebyshev TT'],
                                           ['black', '#6b7280', '#1976b9'], strict=True):
                interpolation = RegularGridInterpolator((xs, xs), grid, bounds_error=False, fill_value=0.)
                ax.plot(t, interpolation(path[:, [j, i]]), label=label, color=color)
            ax.set(title=f'Marginal ({i}, {j}) along the true curve', xlabel='Path parameter (post-fit diagnosis)', ylabel='Marginal density')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(f'{dim}D swirl: recovered ridge profiles after noisy-likelihood selection')
        fig.savefig(figs/f'{name}_ridges.png', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()

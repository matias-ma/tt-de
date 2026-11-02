"""Same marginal/convergence panels as TT, published alongside existing figures."""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_tucker_gallery import digest, read

from ftde.direct_training import write_record
from ftde.marginals import squared_pair_grid
from ftde.tucker_operations import load_tucker, tucker_to_tt


def evaluate(folder, model, protocol, native=None):
    from swirl_figures import fourth_power_integral

    from ftde.tt_density import squared_integral_tt
    result_path = folder/'evaluation.json'
    if result_path.exists():
        result = read(result_path)
        if result['selection_sha256'] != digest(folder/'selection.json'):
            raise RuntimeError('Evaluation belongs to a different selection')
        return result
    dim = protocol['dim']
    if protocol['family'] == 'swirl' and native is not None and (dim == 4 or protocol['rank'] > 4):
        from ftde.swirl import TUBE_STD, curve, path_quadrature, swirl_squared_norm
        from ftde.tucker_density import squared_integral_tucker
        from ftde.tucker_diagnostics import (
            density_squared_integral_qmc,
            fourth_power_integral_4d,
            product_measure_integral,
        )
        z = float(squared_integral_tucker(native))
        qmc = None
        if dim == 4:
            q2 = fourth_power_integral_4d(native)/z**2
        else:
            with np.load(protocol['data']) as data:
                qmc = density_squared_integral_qmc(native, data['train'])
            q2 = qmc['value']
        h, hw = np.polynomial.hermite.hermgauss(protocol['modes']+1)
        def cross(order):
            t, w = path_quadrature(order)
            nodes = curve(t, dim)[:, :, None]+np.sqrt(2)*TUBE_STD*h
            weights = np.broadcast_to(hw/np.sqrt(np.pi), nodes.shape)
            return float(w@product_measure_integral(native, nodes, weights)/z)
        pq, fine = cross(4000), cross(8000)
        p2 = swirl_squared_norm(dim)
        squared = (q2-2*fine+p2)/p2
        result = {'relative_l2': float(np.sqrt(max(0., squared))), 'relative_squared_l2': squared,
                  'integral_q_squared': q2, 'integral_pq': fine, 'integral_p_squared': p2,
                  'path_quadrature_relative_change': abs(pq-fine)/p2,
                  'integration': 'exact blocked Tucker quartic quadrature; native product-measure cross integral'}
        if qmc is not None:
            result.update(integration='importance-QMC quartic integral; native product-measure cross integral', qmc=qmc,
                          relative_l2_standard_error=qmc['standard_error']/(2*p2*result['relative_l2']))
    elif protocol['family'] == 'swirl':
        from swirl_figures import integrated_l2
        result = integrated_l2(folder, model, dim)
    else:
        # Sequential quartic contraction avoids order*r^4 batched storage.
        from scipy.special import roots_legendre

        from ftde.ginzburg_landau import GL_BETA, GL_LAM, ginzburg_landau_log_normalizer
        z = float(squared_integral_tt(model))
        q2 = float(np.exp(fourth_power_integral(model)))/z**2
        log_z = ginzburg_landau_log_normalizer(dim)
        p2 = float(np.exp(ginzburg_landau_log_normalizer(dim, beta=2*GL_BETA)-2*log_z))
        def cross(order):
            t, w = roots_legendre(order)
            state, previous = None, None
            edge = GL_BETA*GL_LAM*(dim+1)**2/2
            for c in model.cores:
                a, b = c.domain
                nodes = jnp.asarray(a+(b-a)*(t+1)/2)
                weights = jnp.asarray(w*(b-a)/2)*jnp.exp(-GL_BETA*(1-nodes**2)**2/(4*GL_LAM))
                incoming = jnp.exp(-edge*nodes**2)[:, None, None] if state is None else jnp.einsum('st,sab->tab', jnp.exp(-edge*(previous[:, None]-nodes[None, :])**2), state)
                v = c(nodes)
                state = jnp.einsum('tab,tai,tbj->tij', incoming, v, v, optimize='optimal')*weights[:, None, None]
                previous = nodes
            return float(jnp.sum(state[:, 0, 0]*jnp.exp(-edge*previous**2))*jnp.exp(-log_z)/z)
        pq, fine = cross(98), cross(196)
        from evaluate_gl_l2 import relative_l2_from_terms
        result = {'relative_l2': relative_l2_from_terms(q2, fine, p2), 'integral_q_squared': q2,
                  'integral_pq': fine, 'integral_p_squared': p2, 'quadrature_relative_change': abs(pq-fine)/p2}
    change = result.get('quadrature_relative_change', result.get('path_quadrature_relative_change'))
    if not np.isfinite(result['relative_l2']) or change > 1e-6 or result.get('relative_squared_l2', 0.) < -1e-9:
        raise FloatingPointError('Full-dimensional L2 integration did not converge')
    result['selection_sha256'] = digest(folder/'selection.json')
    result['metric'] = 'unsquared full-dimensional latent relative Lebesgue L2'
    write_record(result_path, result)
    return result


def plot(folder):
    import fcntl
    with (folder/'.plot.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _plot(folder)


def _plot(folder):
    from plot_dimension_gallery import save_panels
    from plot_objective_study import convolve_pair, noise_grid
    p, selection = read(folder/'protocol.json'), read(folder/'selection.json')
    for path, sha in selection['files'].items():
        if digest(path) != sha:
            raise RuntimeError(f'Frozen input changed: {path}')
    name = selection['selected']
    native = load_tucker(folder/name/'fit.npz')
    # Exact, untruncated core SVD solely to reuse validated TT diagnostic code.
    model = native if p['family'] == 'swirl' and (p['dim'] == 4 or p['rank'] > 4) else tucker_to_tt(native)
    evaluation = evaluate(folder, model, p, native=native)
    family, dim, noisy = p['family'], p['dim'], p['noise'] != 'free'
    figure_root = Path('figs') if folder.parent.resolve() == Path('artifacts/tucker_gallery').resolve() else folder.parent/'figs'
    destination = figure_root/('ginzburg_landau' if family == 'gl' else 'swirl')
    destination.mkdir(parents=True, exist_ok=True)
    prefix = f'tucker_{"noisy" if noisy else "noise_free"}_dim{dim}'
    grid = np.linspace(-2.6 if family == 'swirl' else -2.5, 2.6 if family == 'swirl' else 2.5, 181)
    with np.load(p['data']) as data:
        train = data['train']
    heading = f'{"Ginzburg–Landau" if family == "gl" else "Swirl"} · Tucker · {dim}D · {"mixture noise" if noisy else "noise-free"}\nFull {dim}D latent relative L2 = {evaluation["relative_l2"]:.3f}'
    if 'relative_l2_standard_error' in evaluation:
        heading += f' ± {evaluation["relative_l2_standard_error"]:.3f} (QMC SE)'
    config = read(folder/name/'config.json')
    penalty = f'Full-Hessian λ={config["strength"]:g}'
    if config.get('marginal_strength', 0.):
        penalty += f' · marginal λ={config["marginal_strength"]:g}'
    footer = f'{penalty} · Adam · mode rank {p["rank"]} · {p["modes"]} modes · {name}\n{len(train):,} observations · held-out selection · exact marginals · matched color scales'
    checks = {}
    for i, j in [(0, 1), (2, 3)]:
        if family == 'swirl':
            from ftde.swirl import swirl_pair_marginal
            truth = swirl_pair_marginal(dim, i, j, grid, grid)
        else:
            from gl_common import _true_marginal_grid

            from ftde.ginzburg_landau import ginzburg_landau_log_normalizer
            truth = _true_marginal_grid(dim, i, j, np.tile([grid[0], grid[-1]], (dim, 1)), ginzburg_landau_log_normalizer(dim), n_grid=len(grid))
        latent = squared_pair_grid(model, i, j, grid, grid)
        hist = np.histogram2d(train[:, i], train[:, j], bins=90, range=[(grid[0], grid[-1])]*2)[0].T/(len(train)*((grid[-1]-grid[0])/90)**2)
        row = {'latent_pair_relative_l2': float(np.linalg.norm(latent-truth)/np.linalg.norm(truth))}
        if noisy:
            weights, cov = np.array(p['noise_weights']), np.array(p['noise_covariances'])
            obs, recon = convolve_pair(model, folder.name, i, j, grid, weights, cov, .0125)
            coarse = convolve_pair(model, folder.name, i, j, grid, weights, cov, .025)
            changes = [float(np.linalg.norm(a-b)/np.linalg.norm(a)) for a, b in zip((obs, recon), coarse, strict=True)]
            if max(changes) > .01:
                raise FloatingPointError('Displayed convolution requires refinement')
            row.update(observed_pair_relative_l2=float(np.linalg.norm(obs-recon)/np.linalg.norm(obs)), convolution_grid_refinement=changes)
            arrays, titles, groups = [truth, latent, noise_grid(weights, cov, i, j, grid), obs, recon, hist], ['True latent density', 'Recovered Tucker density', 'Known measurement noise', 'True observed density', 'Reconvolved Tucker', 'Observed data'], [0, 0, 1, 2, 2, 2]
        else:
            arrays, titles, groups = [truth, latent, hist], ['True density', 'Estimated Tucker density', 'Observed data'], [0, 0, 0]
        save_panels(arrays, titles, groups, grid, (i, j), heading, footer, destination/f'{prefix}_marg_{i}{j}.png')
        checks[f'pair{i}{j}'] = row
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for recipe in p['recipes']:
        r = read(folder/recipe/'result.json')
        x = np.array([h['step'] for h in r['history']])
        y = np.array([h['validation_nll'] for h in r['history']])
        h = np.array([h['roughness_hessian'] for h in r['history']])
        label = recipe+(' (selected)' if recipe == name else '')
        axes[0].plot(x, y, label=label)
        axes[0].scatter([r['best_step']], [r['best_nll']], s=25)
        tail = x >= x[-1]/2
        axes[1].plot(x[tail], y[tail])
        axes[2].semilogy(x[1:], h[1:])
    for ax, title in zip(axes, ['Held-out NLL', 'Final-half NLL', 'Full-Hessian energy'], strict=True):
        ax.set(title=title, xlabel='Adam updates')
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(heading)
    fig.tight_layout()
    fig.savefig(destination/f'{prefix}_convergence.png', dpi=165)
    plt.close(fig)
    write_record(folder/'plot_checks.json', checks)
    print(folder.name, evaluation, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    for folder in parser.parse_args().folders:
        plot(folder)
        jax.clear_caches()

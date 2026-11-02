"""Compare latent marginals and noisy scores; average normalized densities.

Grid roughness and conditional-bootstrap spread are descriptive diagnostics,
not clean-density errors or calibrated uncertainty bounds.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from refresh_density_fits import read_model
from scipy.special import logsumexp

from ftde.marginals import (
    relative_pair_hessian_energy,
    squared_pair_coefficients,
    squared_pair_grid,
)
from ftde.noise_marginals import noise_marginals_from_tt


def integrate_grid(values, xs, ys):
    return np.trapezoid(np.trapezoid(values, xs, axis=1), ys)


def grid_diagnostics(grid, xs, ys, scales):
    dy, dx = np.gradient(grid, ys, xs, edge_order=2)
    dyy = np.gradient(dy, ys, axis=0, edge_order=2)
    dxx = np.gradient(dx, xs, axis=1, edge_order=2)
    dxy = np.gradient(dx, ys, axis=0, edge_order=2)
    sx, sy = scales
    curvature = integrate_grid(sx**4*dxx**2+2*sx**2*sy**2*dxy**2+sy**4*dyy**2, xs, ys)
    return {'mass_on_grid': float(integrate_grid(grid, xs, ys)),
            'finite_difference_relative_density_hessian_energy': float(curvature/integrate_grid(grid**2, xs, ys)),
            'differential_entropy': float(-integrate_grid(grid*np.log(np.maximum(grid, 1e-300)), xs, ys))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidates', nargs='+', type=Path, required=True)
    parser.add_argument('--ensemble', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    args.candidates = [p.resolve() for p in args.candidates]
    args.ensemble = [p.resolve() for p in args.ensemble]
    if len({p.name for p in args.candidates}) != len(args.candidates):
        raise ValueError('Candidate folder names must be unique')
    if any(p not in args.candidates for p in args.ensemble):
        raise ValueError('Ensemble members must be candidate folders')
    if len(args.ensemble) == 1:
        raise ValueError('Use at least two ensemble members to measure variation')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    source_name = json.loads((args.base/'selection.json').read_text())['selected']
    previous = read_model(args.base/'selected.npz')
    models = {'previous': previous, **{p.name: read_model(p/'fit.npz') for p in args.candidates}}
    if any(not np.array_equal(g.domain, previous.domain) for g in models.values()):
        raise ValueError('This comparison requires the same latent support for every model')
    base_logs = np.load(args.base/source_name/'validation.npy')
    configs = {p.name: json.loads((p/'experiment.json').read_text()) for p in args.candidates}
    for p in args.candidates:
        if Path(configs[p.name]['base']).resolve() != args.base.resolve():
            raise ValueError('Candidates must share the same frozen base')
        for key, filename in [('kernel_sha256', 'kernel.npz'), ('data_sha256', 'data.npz')]:
            if key in configs[p.name] and configs[p.name][key] != hashlib.sha256((args.base/filename).read_bytes()).hexdigest():
                raise ValueError('The frozen kernel or observations changed')
    records = {p.name: json.loads((p/'result.json').read_text()) for p in args.candidates}
    for p in args.ensemble:
        indices = np.load(p/'bootstrap_indices.npy')
        records[p.name]['bootstrap_draws'] = len(indices)
        records[p.name]['bootstrap_unique_training_rows'] = len(np.unique(indices))
    logs = {'previous': base_logs, **{p.name: np.load(p/'validation.npy') for p in args.candidates}}
    if args.ensemble:
        logs['density_average'] = logsumexp(np.stack([logs[p.name] for p in args.ensemble]), axis=0)-np.log(len(args.ensemble))
        # This manifest defines the actual joint mixture, not just an image.
        manifest = {'representation': 'convex mixture of normalized squared ChebTT densities',
                    'members': [str((p/'fit.npz').resolve()) for p in args.ensemble],
                    'weights': [1/len(args.ensemble)]*len(args.ensemble),
                    'kernel': str((args.base/'kernel.npz').resolve()),
                    'base': str(args.base.resolve())}
        (out/'ensemble.json').write_text(json.dumps(manifest, indent=2)+'\n')
        np.save(out/'ensemble_validation.npy', logs['density_average'])
    scores = {}
    for name, values in logs.items():
        delta = base_logs-values
        scores[name] = {'noisy_validation_nll': float(-values.mean()),
                        'nll_change_from_previous': float(delta.mean()),
                        'paired_se': float(delta.std(ddof=1)/np.sqrt(len(delta)))}
    best_single = min(records, key=lambda n: scores[n]['noisy_validation_nll'])
    for name, values in logs.items():
        delta = logs[best_single]-values
        scores[name]['nll_change_from_best_single'] = float(delta.mean())
        scores[name]['paired_se_to_best_single'] = float(delta.std(ddof=1)/np.sqrt(len(delta)))
    data = np.load(args.base/'data.npz')
    scales = [m.moments()[1] for m in noise_marginals_from_tt(read_model(args.base/'kernel.npz'))]
    summary = {'scores': scores, 'records': records, 'best_single': best_single,
               'ensemble_members': [p.name for p in args.ensemble], 'marginals': {},
               'stability_scope': 'Conditional resampled continuations from a fit trained on all original training observations. Not independent-data replication or calibrated confidence intervals.'}
    for i, j in [(0, 1), (2, 3), (8, 9)]:
        if j >= len(previous.cores):
            continue
        box = np.asarray(previous.domain)[[i, j]]
        xs, ys = np.linspace(*box[0], 240), np.linspace(*box[1], 240)
        grids = {'previous': squared_pair_grid(previous, i, j, xs, ys)}
        coefficients = {name: squared_pair_coefficients(model, i, j)[0] for name, model in models.items()}
        for p in args.candidates:
            path = p/f'marginal_{i}{j}.npz'
            if path.exists():
                saved = np.load(path)
                np.testing.assert_allclose(saved['xs'], xs)
                np.testing.assert_allclose(saved['ys'], ys)
                grids[p.name] = saved['after']
            else:
                grids[p.name] = squared_pair_grid(models[p.name], i, j, xs, ys)
        members = np.stack([grids[p.name] for p in args.ensemble]) if args.ensemble else None
        if members is not None:
            grids['density_average'] = members.mean(0)
            sizes = np.max([coefficients[p.name].shape for p in args.ensemble], axis=0)
            coefficients['density_average'] = np.mean([
                np.pad(coefficients[p.name], tuple((0, int(n-m)) for n, m in zip(sizes, coefficients[p.name].shape, strict=True)))
                for p in args.ensemble], axis=0)
        diagnostics = {}
        for name, grid in grids.items():
            diagnostics[name] = grid_diagnostics(grid, xs, ys, [scales[i], scales[j]])
            diagnostics[name]['relative_density_hessian_energy'] = float(relative_pair_hessian_energy(coefficients[name], box, [scales[i], scales[j]]))
            diagnostics[name]['l1_change_from_previous'] = float(integrate_grid(abs(grid-grids['previous']), xs, ys))
        if members is not None:
            diagnostics['conditional_spread'] = {'integrated_pointwise_sd': float(integrate_grid(members.std(0, ddof=1), xs, ys)),
                'mean_member_l1_from_average': float(np.mean([integrate_grid(abs(g-members.mean(0)), xs, ys) for g in members]))}
        summary['marginals'][f'{i}{j}'] = diagnostics
        names = ['previous', best_single]+(['density_average'] if args.ensemble else [])
        n = len(names)+(1 if args.ensemble else 0)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 6.4), layout='constrained')
        peak = max(grids[name].max() for name in names)
        extent = tuple(box[0]*data['scale'][i]+data['shift'][i])+tuple(box[1]*data['scale'][j]+data['shift'][j])
        jacobian = data['scale'][i]*data['scale'][j]
        for ax, name in zip(axes, names):
            im = ax.imshow(grids[name]/jacobian, origin='lower', extent=extent, aspect='auto',
                           cmap='magma', vmin=0, vmax=peak/jacobian, interpolation='nearest')
            label = {'previous': 'Previous fit', 'density_average': 'Average of resampled densities'}.get(name, 'Best single continuation')
            ax.set(title=f'{label}\nNoisy NLL {scores[name]["noisy_validation_nll"]:.5f}', xlabel=f'x{i}', ylabel=f'x{j}')
            ax.set_box_aspect(1)
            fig.colorbar(im, ax=ax, fraction=.045, label='Density')
        if args.ensemble:
            im = axes[-1].imshow(members.std(0, ddof=1)/jacobian, origin='lower', extent=extent,
                                 aspect='auto', cmap='viridis', vmin=0, interpolation='nearest')
            axes[-1].set(title='Variation between resampled fits\nPointwise SD; not a confidence interval', xlabel=f'x{i}', ylabel=f'x{j}')
            axes[-1].set_box_aspect(1)
            fig.colorbar(im, ax=axes[-1], fraction=.045, label='Density SD')
        best = records[best_single]
        config = configs[best_single]
        fig.suptitle(f'Cryo-EM stability · {len(previous.cores)} dimensions · same observations and noise kernel')
        footer = (f'N train {len(data["train"]):,}; validation {len(data["valid"]):,}; 49 amplitude modes/axis; best single rank {config["rank"]}; '
                  f'H² strength {best["strength"]:g}; Adam {best["steps"]} steps; batch 256; LR 2e-4 → 2e-5; Sobolev preconditioning\n'
                  f'Best continuation fit {best["seconds"]:.1f}s; displayed candidate fits {sum(r["seconds"] for r in records.values()):.1f}s total; includes compilation/validation; excludes initialization, plotting and ancestral fits\n'
                  'Density averages and marginals are computed before plotting; no image smoothing. Resampled continuations share a warm start; no clean-density ground truth.')
        if args.ensemble:
            footer += ('\nResampled members: '+', '.join(
                f'seed {configs[p.name]["seed"]}, rank {configs[p.name]["rank"]}, H² {records[p.name]["strength"]:g}, {records[p.name]["steps"]} steps'
                for p in args.ensemble))
        fig.supxlabel(footer, fontsize=8)
        fig.savefig(out/f'comparison_{i}{j}.png', dpi=160)
        plt.close(fig)
        np.savez(out/f'marginal_{i}{j}.npz', xs=xs, ys=ys, **grids,
                 **({'conditional_sd': members.std(0, ddof=1)} if members is not None else {}))
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__':
    main()

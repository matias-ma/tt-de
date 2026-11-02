"""Freeze selection from noisy validation, then evaluate saved GL recovery fits."""

import argparse
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_noisy_recovery import load_model
from evaluate_gl_l2 import integrated_terms, relative_l2_from_terms
from gl_common import _marginal_grid, _true_marginal_grid

from ftde.ginzburg_landau import (
    GL_BETA,
    ginzburg_landau_density,
    ginzburg_landau_log_normalizer,
    sample_ginzburg_landau,
)
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.reference_score import ReferenceScorePenalty
from ftde.tt_density import density_tt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--baseline', type=Path, default=Path('artifacts/gl_rsi_trial/paired_rsi_accurate.npz'))
    parser.add_argument('--baseline-name', default='original_rsi')
    parser.add_argument('--baseline-label', default='Original RSI fit')
    parser.add_argument('--selection-reference', type=Path,
                        help='use this common relative-score measure when comparing different penalties')
    parser.add_argument('--selection-only', action='store_true', help='save noisy selection and checkpoint without evaluating truth')
    parser.add_argument('--no-plots', action='store_true', help='evaluate metrics without regenerating figures')
    parser.add_argument('--output', type=Path, default=Path('artifacts/noisy_recovery_evaluation'))
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/ginzburg_landau/noisy_recovery'))
    args = parser.parse_args()
    paths = []
    for directory in args.directories:
        if not (directory/'selection.json').exists():
            raise ValueError(f'Experiment is incomplete: {directory}')
        paths.extend(sorted(directory.glob('*/result.json')))
    records = [json.loads(p.read_text()) for p in paths]
    # Different polynomial bases are allowed; validation observations and the
    # definition of roughness must match exactly across the comparison.
    for record in records[1:]:
        common = ('setup', 'n_train', 'train_seed', 'valid_seed', 'noise_scale', 'warm_start')
        penalty_settings = () if args.selection_reference else ('order', 'reference', 'fisher_floor', 'kernel_marginals')
        for key in common+penalty_settings:
            default = False if key == 'kernel_marginals' else None
            if record['settings'].get(key, default) != records[0]['settings'].get(key, default):
                raise ValueError(f'Incompatible experiment setting: {key}')
    def load(path):
        saved = np.load(path)
        return load_model(saved, 'deconvolved', saved['domain'])

    roughness = [r['roughness'] for r in records]
    if args.selection_reference:
        reference = np.load(args.selection_reference)
        setup = np.load(records[0]['settings']['setup'])
        noise_tt = load_model(setup, 'kernel', setup['kernel_domain'])
        std = np.array([m.moments()[1] for m in noise_marginals_from_tt(noise_tt)])
        roughness = []
        for path in paths:
            g = load(path.parent/'fit.npz')
            np.testing.assert_array_equal(np.asarray(g.domain), reference['domain'])
            penalty = ReferenceScorePenalty.from_tt(g, reference['log_coeffs'], length_scales=std)
            roughness.append(float(jax.jit(penalty)(g)))
        # This measure and all selection inputs use noisy data only. In
        # particular, integrated_terms and the GL truth are not used here.
    scores = np.stack([np.load(p.parent/'validation.npy') for p in paths])
    selected, detail = select_by_paired_noisy_nll(scores, roughness,
                                                cluster_ids=np.arange(scores.shape[1]) % 200)
    names = [f'{p.parent.parent.name}/{p.parent.name}' for p in paths]
    detail['selected'] = names[selected]
    detail['names'] = names
    detail['selection_reference'] = str(args.selection_reference) if args.selection_reference else None
    detail['checkpoint'] = str(paths[selected].parent/'fit.npz')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'selection.json').write_text(json.dumps(detail, indent=2)+'\n')
    shutil.copyfile(paths[selected].parent/'fit.npz', args.output/'selected_fit.npz')
    print('FROZEN SELECTION', names[selected], flush=True)
    if args.selection_only:
        return

    models = {args.baseline_name: load(args.baseline)}
    models.update({name: load(path.parent/'fit.npz') for name, path in zip(names, paths, strict=True)})
    chosen = models[names[selected]]
    d = len(chosen.cores)
    log_z = ginzburg_landau_log_normalizer(d)
    p2 = float(np.exp(ginzburg_landau_log_normalizer(d, beta=2*GL_BETA)-2*log_z))
    clean = sample_ginzburg_landau(seed=1, n=4000, d=d)
    truth = np.asarray(ginzburg_landau_density(clean, log_z=log_z))
    metrics = {}
    for name, g in models.items():
        pred = np.asarray(jax.jit(density_tt)(g, clean))
        inside = np.all((np.asarray(clean) >= np.asarray(g.domain)[:, 0])
                        & (np.asarray(clean) <= np.asarray(g.domain)[:, 1]), axis=1)
        pred = np.where(inside, pred, 0.)
        sample_error = float(np.linalg.norm(pred-truth)/np.linalg.norm(truth))
        order = max(64, 2*max(c.coeffs.shape[-1] for c in g.cores)-1)
        terms = [tuple(map(float, jax.jit(lambda q=q, g=g: integrated_terms(g, q, log_z))()))
                 for q in (order, 2*order)]
        q2, pq = terms[-1]
        metrics[name] = {'relative_sample_l2': sample_error, 'relative_squared_sample_error': sample_error**2,
                         'integrated_relative_l2': relative_l2_from_terms(q2, pq, p2),
                         'quadrature_relative_change': max(abs(a-b) for a, b in zip(*terms, strict=True))/p2,
                         'integral_estimate_squared': q2, 'integral_product': pq, 'integral_truth_squared': p2}
        print(name, json.dumps(metrics[name]), flush=True)
        (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        jax.clear_caches()

    if args.no_plots:
        return
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(0, 1), (2, 3), (0, d-1)]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(11, 10), constrained_layout=True)
    for row, (i, j) in enumerate(pairs):
        # Same physical window and color scale within each comparison.
        display = jnp.array([[-2., 2.]]*d)
        true = _true_marginal_grid(d, i, j, display, log_z)
        extent, old = _marginal_grid(models[args.baseline_name], i, j, display)
        _, new = _marginal_grid(chosen, i, j, display)
        vmax = max(true.max(), old.max(), new.max())
        for col, values in enumerate((true, old, new)):
            im = axes[row, col].imshow(values, extent=extent, origin='lower', vmin=0, vmax=vmax, cmap='viridis')
            axes[row, col].set(xlabel=f'x{i}', ylabel=f'x{j}')
        fig.colorbar(im, ax=axes[row, :], label='Marginal density', shrink=.85)
    axes[0, 0].set_title('True GL density')
    axes[0, 1].set_title(f'{args.baseline_label}\nIntegrated rel. L2 {metrics[args.baseline_name]["integrated_relative_l2"]:.3f}')
    chosen_settings = records[selected]['settings']
    chosen_label = ('Fisher-filter TT fit' if chosen_settings.get('fisher_floor') is not None
                    else 'Reference-score TT fit' if chosen_settings.get('reference')
                    else 'Regularized TT fit')
    axes[0, 2].set_title(f'{chosen_label}\nIntegrated rel. L2 {metrics[names[selected]]["integrated_relative_l2"]:.3f}')
    settings = records[0]['settings']
    fig.suptitle(f'{d}D GL · paired noise ×{settings["noise_scale"]} · '
                 f'{settings["n_train"]:,} observations\nSelection uses noisy validation only')
    fig.savefig(args.figure_dir/'paired_comparison.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    def display_label(record):
        settings = record['settings']
        method = ('Legacy' if record['name'] == 'legacy' else
                  'Fisher' if settings.get('fisher_floor') is not None else
                  'Score' if settings.get('reference') else f'H{settings["order"]}')
        stopping = ' · 1SE stop' if settings.get('checkpoint_rule') == 'paired-one-se' else ''
        multiple_sizes = len({r['settings'].get('n_coeffs') for r in records}) > 1
        basis = f' · n={settings.get("n_coeffs") or "base"}' if multiple_sizes else ''
        return f'{method} λ={record["strength"]:g}{basis}{stopping}'

    for name, record in zip(names, records, strict=True):
        ax.plot([r['step'] for r in record['history']], [r['nll'] for r in record['history']],
                label=display_label(record))
    all_nll = [h['nll'] for r in records for h in r['history']]
    padding = max(.0005, .05*(max(all_nll)-min(all_nll)))
    ax.set(xlabel='Adam step', ylabel='Held-out noisy NLL', ylim=(min(all_nll)-padding, max(all_nll)+padding))
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(args.figure_dir/'validation_curves.png', dpi=180)
    plt.close(fig)

    labels = list(metrics)
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.bar(positions-.18, [metrics[n]['relative_sample_l2'] for n in labels], width=.36,
           label='Relative error on clean sample points')
    ax.bar(positions+.18, [metrics[n]['integrated_relative_l2'] for n in labels], width=.36,
           label='Integrated relative L2')
    plot_labels = [args.baseline_label]+[display_label(r)+(' *' if n == names[selected] else '')
                                        for n, r in zip(names, records, strict=True)]
    ax.set_xticks(positions, plot_labels, rotation=30, ha='right')
    ax.axhline(.2, color='black', linestyle='--', linewidth=1, label='Requested integrated-L2 target')
    ax.set(ylabel='Relative error (square root)', title='* Selected using noisy validation; clean truth used only for this evaluation')
    ax.legend()
    fig.savefig(args.figure_dir/'recovery_errors.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

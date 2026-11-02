"""Plots for the GL-convolution scaling sweep (rank / n_coeffs / n_train) and
the explicit naive-steps=0 example. Reads artifacts/gl_convolution_scaling/
(run_gl_convolution_experiments.py's output); does not delete anything.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {'kde': 'tab:blue', 'naive': 'tab:orange', 'deconvolved': 'tab:green'}
LABELS = {'kde': 'KDE', 'naive': 'Naive TT', 'deconvolved': 'Deconvolved TT'}
SWEEP_LABELS = {'rank': 'TT rank', 'n_coeffs': 'Chebyshev coefficients (basis modes)',
                'n_train': 'training points n'}


def load(directory):
    summary = json.loads((directory/'run_summary.json').read_text())
    records = {}
    for config in summary:
        path = directory/f"{config['name']}.json"
        if config['ok'] and path.exists():
            records[config['name']] = {**config, **json.loads(path.read_text())}
    return records


def plot_scaling(records, directory):
    sweeps = ['rank', 'n_coeffs', 'n_train']
    fig, axes = plt.subplots(1, len(sweeps), figsize=(6*len(sweeps), 4.5), constrained_layout=True)
    for ax, sweep in zip(axes, sweeps, strict=True):
        points = sorted((r for r in records.values() if r.get('sweep') == sweep), key=lambda r: r['value'])
        if not points:
            ax.axis('off')
            continue
        xs = [p['value'] for p in points]
        for key in ('kde', 'naive', 'deconvolved'):
            ys = [p['metrics'][key]['relative_sample_l2'] for p in points]
            ax.plot(xs, ys, marker='o', color=COLORS[key], label=LABELS[key])
        ax.set(xlabel=SWEEP_LABELS[sweep], ylabel='Relative error on clean test samples',
               title=f'Error vs. {SWEEP_LABELS[sweep]}', ylim=(0, 1.05))
        if sweep == 'n_train':
            ax.set_xscale('log')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Noisy GL (8D, mixture noise): naive vs. deconvolved TT scaling')
    fig.savefig(directory/'scaling_summary.png', dpi=170)
    plt.close(fig)


def plot_training_curves(records, directory):
    baseline = records.get('scaling_rank_8')
    if baseline is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, color in (('naive', 'tab:orange'), ('deconvolved', 'tab:green')):
        history = baseline['histories'][label]['history']
        steps = [h['step'] for h in history]
        nll = [h['validation_nll'] for h in history]
        best_step = baseline['histories'][label]['best_step']
        ax.plot(steps, nll, color=color, label=f'{LABELS[label]} (selected step {best_step:,})')
        ax.axvline(best_step, color=color, linestyle=':', alpha=.5)
    zero = records.get('scaling_naive_steps_0')
    if zero is not None:
        ax.axhline(zero['histories']['naive']['history'][0]['validation_nll'], color='tab:orange',
                   linestyle='--', alpha=.6, label='Naive TT at step 0 (KDE cross-approx init, untrained)')
    ax.set(xlabel='training step', ylabel='validation NLL',
           title=f"Training curves, {baseline['dim']}D GL {baseline['noise']} noise "
                 f"(rank={baseline['rank']}, {baseline['n_coeffs']} coeffs, n={baseline['n_train']:,})")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(directory/'training_curves.png', dpi=170)
    plt.close(fig)


def plot_naive_steps_zero(records, directory):
    baseline, zero = records.get('scaling_rank_8'), records.get('scaling_naive_steps_0')
    if baseline is None or zero is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    bars = [('KDE', baseline['metrics']['kde']['relative_sample_l2'], 'tab:blue'),
            ('Naive TT\n(0 steps: raw KDE\ncross-approx init)', zero['metrics']['naive']['relative_sample_l2'], 'tab:red'),
            (f"Naive TT\n({baseline['naive_steps']:,} steps)", baseline['metrics']['naive']['relative_sample_l2'], 'tab:orange'),
            (f"Deconvolved TT\n({baseline['deconv_steps']:,} steps)", baseline['metrics']['deconvolved']['relative_sample_l2'], 'tab:green')]
    x = np.arange(len(bars))
    rects = ax.bar(x, [b[1] for b in bars], color=[b[2] for b in bars])
    ax.bar_label(rects, fmt='%.3f')
    ax.set_xticks(x, [b[0] for b in bars])
    ax.set(ylabel='Relative error on clean test samples', ylim=(0, 1.05),
           title=f"Effect of training and deconvolution, {baseline['dim']}D GL {baseline['noise']} noise")
    ax.grid(axis='y', alpha=.2)
    fig.savefig(directory/'naive_steps_zero_comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/gl_convolution_scaling'))
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/gl_convolution_scaling'))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    records = load(args.directory)
    plot_scaling(records, args.figure_dir)
    plot_training_curves(records, args.figure_dir)
    plot_naive_steps_zero(records, args.figure_dir)
    print(f'saved plots to {args.figure_dir}')


if __name__ == '__main__':
    main()

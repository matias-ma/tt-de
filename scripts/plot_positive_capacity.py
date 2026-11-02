"""Post-selection diagnosis of matched-budget Chebyshev degree continuations."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_positive_tt import l2_error

from ftde.positive_chebyshev import PositiveChebyshevTT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    args = parser.parse_args()
    # Require selection to have been frozen before evaluating any clean truth.
    selection = json.loads((args.comparison/'selection.json').read_text())
    audit = json.loads((args.comparison/'audit.json').read_text())
    root = Path('artifacts/positive_tt')
    stages = [[f'swirl{args.dim}_r32_m49_control', f'swirl{args.dim}_r32_m65_v3'],
              [f'swirl{args.dim}_r32_m65_control', f'swirl{args.dim}_r32_m97_v3']]
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    for row, names in enumerate(stages):
        configs = [json.loads((root/name/'experiment.json').read_text()) for name in names]
        for key in ['initial', 'sweeps', 'inner_steps', 'batch_size', 'seed', 'barrier', 'barrier_final', 'strength']:
            if configs[0][key] != configs[1][key]:
                raise ValueError(f'The supposed matched-budget controls differ in {key}')
        modes = [c['modes'] for c in configs]
        logs = [np.load(root/name/'validation.npy') for name in names]
        initial = [PositiveChebyshevTT.load(root/name/'init.npz') for name in names]
        initial_difference = max(float(np.max(np.abs(
            np.pad(np.asarray(a.coeffs), ((0, 0), (0, 0), (0, b.coeffs.shape[-1]-a.coeffs.shape[-1])))
            -np.asarray(b.coeffs)))) for a, b in zip(initial[0].tt.cores, initial[1].tt.cores, strict=True))
        delta = logs[1]-logs[0]
        gain = float(delta.mean())
        se = float(delta.std(ddof=1)/np.sqrt(len(delta)))
        errors = [l2_error(PositiveChebyshevTT.load(root/name/'fit.npz')) for name in names]
        records.append({'stage': row+1, 'names': names, 'higher_degree_validation_gain': gain,
                        'paired_se': se, 'errors': errors,
                        'maximum_initial_core_coefficient_difference': initial_difference})
        axes[row, 0].errorbar(modes, [0., gain], yerr=[0., 2*se], fmt='o-', capsize=5, color='#1976b9')
        axes[row, 0].axhline(0, color='black', ls='--', lw=1)
        axes[row, 0].set(xticks=modes, xlabel='Direct-density Chebyshev modes', ylabel='Noisy validation gain over lower degree',
                         title=f'Stage {row+1}: same parent, data and update budget')
        axes[row, 1].bar([str(m) for m in modes], [e['relative_l2'] for e in errors], color=['#6b7280', '#1976b9'])
        axes[row, 1].axhline(audit['old_relative_l2'], color='black', ls='--', label='Original squared TT')
        axes[row, 1].set(xlabel='Direct-density Chebyshev modes', ylabel=f'Full {args.dim}D relative L2',
                         title='Latent recovery, evaluated only afterward')
        axes[row, 1].legend(fontsize=8)
        for ax in axes[row]:
            ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'{args.dim}D swirl: polynomial resolution versus extra optimization', fontweight='bold')
    fig.supxlabel('Each pair uses the same parent checkpoint, batch sequence and optimization budget; each checkpoint is selected by noisy validation likelihood.\n'
                  '97 direct modes match the degree of a squared 49-mode amplitude. The positive-core rank is 32; the original amplitude rank is 64.', fontsize=9)
    fig.savefig(f'figs/swirl/noisy_mixture_positive_capacity_dim{args.dim}.png', dpi=180)
    plt.close(fig)
    (args.comparison/'capacity.json').write_text(json.dumps({'frozen_selection': selection['selected'], 'stages': records}, indent=2)+'\n')
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()

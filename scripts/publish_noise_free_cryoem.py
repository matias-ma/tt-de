"""Freeze and plot the direct-optimization cryo-EM transfer; no clean oracle."""
import hashlib
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from transfer_direct_cryoem import BASES

from ftde.direct_training import DirectTrainingProblem, load_tt, write_record
from ftde.marginals import squared_pair_grid
from ftde.noisy_model_selection import select_by_paired_noisy_nll

ROOT = Path('artifacts/noise_free_v2')


def publish(case):
    folder = ROOT/f'cryoem_{case}'
    protocol = json.loads((folder/'protocol.json').read_text())
    names = [*protocol['recipes'], 'lbfgs']
    records = [json.loads((folder/n/'result.json').read_text()) for n in names]
    logs = np.stack([np.load(folder/n/'validation.npy') for n in names])
    models = [load_tt(folder/n/'fit.npz') for n in names]
    roughness = []
    for model in models:
        problem = DirectTrainingProblem(model)
        energies = np.asarray(problem.energies(problem.params))
        roughness.append(float(energies[3]/energies[0]))
    selected, details = select_by_paired_noisy_nll(logs, roughness)
    previous = Path('artifacts/direct_optimization')/f'cryoem_{case}'
    old_selection = json.loads((previous/'selection.json').read_text())
    oldname = old_selection['new_selected']
    oldfit = previous/oldname/'fit.npz'
    oldnll = old_selection['validation_nll'][old_selection['names'].index(oldname)]
    oldlogs = np.load(previous/oldname/'validation.npy')
    assert logs.shape[1:] == oldlogs.shape
    difference = oldlogs[None]-logs
    paired = [{'delta_nll': float(x.mean()), 'paired_se': float(x.std(ddof=1)/np.sqrt(len(x)))} for x in difference]
    result = {'selected': names[selected], 'selection': details, 'previous_direct_name': oldname,
              'previous_direct_nll': oldnll, 'paired_new_minus_previous': paired, 'new_nll': (-logs.mean(axis=1)).tolist(),
              'uses_signal_pilot': False, 'uses_clean_truth': False,
              'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [oldfit, *[folder/n/'fit.npz' for n in names]]}}
    write_record(folder/'frozen.json', result)
    domain = np.asarray(models[0].domain)
    plotmodels = [load_tt(oldfit), *models]
    labels = ['Previous direct TT', *[n.replace('hessian_', 'Full Hessian λ=').replace('h1_', 'Slope penalty λ=').replace('_nll', '').replace('_logpen', ' (log)').replace('lbfgs', 'Selected Adam → L-BFGS') for n in names]]
    if records[-1].get('restart_from_step') is not None:
        labels[-1] += f"\nRestarted at step {records[-1]['restart_from_step']}"
    nlls = [oldnll, *(-logs.mean(axis=1)).tolist()]
    with np.load(Path(BASES[case])/'data.npz') as saved:
        train = saved['train']
    fig, axes = plt.subplots(2, len(plotmodels)+1, figsize=(4*(len(plotmodels)+1), 9), layout='constrained')
    for row, (i, j) in enumerate([(0, 1), (2, 3)]):
        xs, ys = np.linspace(*domain[i], 180), np.linspace(*domain[j], 180)
        marginals = [squared_pair_grid(model, i, j, xs, ys) for model in plotmodels]
        hist, xe, ye = np.histogram2d(train[:, i], train[:, j], bins=100, range=[domain[i], domain[j]], density=True)
        vmax = max(float(m.max()) for m in marginals)
        axes[row, 0].pcolormesh(xe, ye, hist.T, cmap='magma', vmin=0, vmax=vmax, shading='auto')
        if row == 0:
            axes[row, 0].set_title('Noisy observations\nHistogram', fontsize=11)
        for k, (marginal, label, nll) in enumerate(zip(marginals, labels, nlls, strict=True)):
            image = axes[row, k+1].pcolormesh(xs, ys, marginal, cmap='magma', vmin=0, vmax=vmax, shading='auto')
            if row == 0:
                title = label+f'\nNoisy validation NLL = {nll:.4f}'
                if k == selected+1:
                    title += '\nSelected new fit'
                axes[row, k+1].set_title(title, fontsize=11)
        for ax in axes[row]:
            ax.set(xlabel=f'x{i}', ylabel=f'x{j}', xlim=domain[i], ylim=domain[j], aspect='equal')
        fig.colorbar(image, ax=axes[row], shrink=.7, label='Density; fitted panels are latent marginals')
    fig.suptitle(f'Cryo-EM {case.replace("_", " ")}: direct Chebyshev TT optimization', fontsize=16)
    fig.get_layout_engine().set(rect=(0, .09, 1, .89))
    fig.text(.015, .017, 'New fits start near uniform and optimize convolved likelihood plus an amplitude smoothness penalty; 49 modes, rank 32, 4,000 Adam updates, then full-data L-BFGS.\n'
             'Regularizer recipes transfer from observation-selected noise-free swirl fits; no learned signal initializer. Prescribed measurement noise is unchanged.\n'
             'Selection compares noisy validation likelihoods, not unknown latent truth. The earlier direct TT is a reference, outside the new-recipe selection.', fontsize=9)
    fig.savefig(Path('figs/cryoem')/f'{case}.png', dpi=160)
    plt.close(fig)
    return names, records, result


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for case, ax in zip(BASES, axes.flat, strict=True):
        names, records, selected = publish(case)
        for name, record in zip(names, records, strict=True):
            h = record['history']
            ax.plot([v['step']+(4000 if name == 'lbfgs' else 0) for v in h], [v['validation_nll'] for v in h], '.-', label=name)
            best = next(v for v in h if v['step'] == record['best_step'])
            ax.scatter(best['step']+(4000 if name == 'lbfgs' else 0), best['validation_nll'], marker='*', s=100)
        ax.axhline(selected['previous_direct_nll'], color='gray', ls='--', label='Previous direct TT')
        title = case.replace('_', ' ')
        if records[-1].get('restart_from_step') is not None:
            title += f"\nL-BFGS restarted at step {records[-1]['restart_from_step']}"
        ax.set(title=title, xlabel='Adam updates, then L-BFGS updates (different costs)', ylabel='Noisy held-out NLL ↓')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        print(case, selected, flush=True)
        jax.clear_caches()
    fig.suptitle('Cryo-EM transfer: all signal structure is learned by TT optimization', fontsize=15)
    fig.savefig('figs/cryoem/training.png', dpi=160)
    plt.close(fig)

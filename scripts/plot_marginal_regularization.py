"""Compare frozen noisy fits, including actual one-dimensional marginals."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_marginal_regularization import CASES_RUN, ROOT, read

STEM = 'marginal_regularization'


def model_label(name):
    if name == 'original':
        return {'marginal_regularization': 'Previous control',
                'smooth_initialization': 'Default init, seed 0',
                'calibrated_marginals': 'Selected weak-penalty source'}[STEM]
    if STEM == 'calibrated_marginals':
        return {'control': 'Matched continuation, no marginal penalty',
                'marginal_light': 'Marginal penalty, initial budget 0.001',
                'marginal_moderate': 'Marginal penalty, initial budget 0.003'}[name]
    return name.replace('_', ' ')


def chosen_models(case):
    info = read(ROOT/'frozen.json')['cases'][case]
    names = info['models']
    marginal = min([n for n in names if n.startswith('marginal') or n == 'fresh_marginal'],
                   key=lambda n: names[n]['validation_nll'])
    smooth = read(ROOT/'smooth_selection.json')['cases'][case]['selected']
    return list(dict.fromkeys(['original', info['selected'], smooth, marginal]))


def plot():
    frozen = read(ROOT/'frozen.json')
    for case in CASES_RUN:
        info = frozen['cases'][case]
        evaluation = read(ROOT/case/'evaluation.json')
        curves = np.load(ROOT/case/'univariate.npz')
        chosen = chosen_models(case)
        dim = curves['truth'].shape[0]
        fig, axes = plt.subplots(2, (dim+1)//2, figsize=(13, 7), layout='constrained')
        for k, ax in enumerate(axes.flat):
            ax.plot(curves['grid'], curves['truth'][k], color='black', lw=2, label='Truth (evaluation only)')
            for c, name in enumerate(chosen):
                smooth = read(ROOT/'smooth_selection.json')['cases'][case]['selected']
                label = model_label(name)+(' [min NLL]' if name == info['selected'] else '')
                if name == smooth:
                    label += ' [smooth choice]'
                ax.plot(curves['grid'], curves[name][k], color=f'C{c}', lw=1.5, label=label)
            ax.set(title=f'Coordinate {k}', xlabel='x', ylabel='Marginal density', xlim=(-2.6, 2.6))
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'{case}: latent 1D marginals, selected using noisy validation likelihood')
        dest = Path('figs/swirl' if case.startswith('swirl') else 'figs/ginzburg_landau')
        fig.savefig(dest/f'{STEM}_1d.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, len(chosen)+1, figsize=(4*(len(chosen)+1), 7), layout='constrained')
        truth = np.load(ROOT/case/'truth.npz')
        for row, pair in enumerate(['pair01', 'pair23']):
            panels = [truth[pair]]+[np.load(ROOT/case/f'marginals_{n}.npz')[pair] for n in chosen]
            vmax = max(float(v.max()) for v in panels)
            for col, (label, values) in enumerate(zip(['Truth']+chosen, panels, strict=True)):
                ax = axes[row, col]
                mesh = ax.imshow(values, origin='lower', extent=[truth['grid'][0], truth['grid'][-1]]*2,
                                 vmin=0, vmax=vmax, cmap='magma')
                ax.set_title('Truth' if label == 'Truth' else model_label(label), fontsize=10)
                ax.set_xlabel(f'x{pair[-2]}')
                ax.set_ylabel(f'x{pair[-1]}')
            fig.colorbar(mesh, ax=axes[row].tolist(), shrink=.8)
        fig.suptitle(f'{case}: latent pair marginals')
        fig.savefig(dest/f'{STEM}_2d.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
    for row, case in enumerate(CASES_RUN):
        info = frozen['cases'][case]
        evaluation = read(ROOT/case/'evaluation.json')
        names = list(info['models'])
        colors = ['C1' if n == info['selected'] else 'C0' for n in names]
        axes[row, 0].bar(range(len(names)), [evaluation['models'][n]['relative_l2'] for n in names], color=colors)
        axes[row, 0].set(title=case, ylabel='Full-dimensional latent relative L2')
        axes[row, 1].bar(range(len(names)), [evaluation['models'][n]['marginal_curvature'] for n in names], color=colors)
        axes[row, 1].set(ylabel='Integrated 1D density curvature', yscale='log')
        compare = evaluation['test_comparisons']
        axes[row, 2].errorbar(range(len(names)), [compare[n]['test_nll_minus_reference'] for n in names],
                            yerr=[1.96*compare[n]['paired_se'] for n in names], fmt='o', capsize=3)
        axes[row, 2].axhline(0, color='gray', ls='--')
        axes[row, 2].set(ylabel='Independent noisy test NLL minus original\n±1.96 paired SE (GL: chain clustered)')
        for ax in axes[row]:
            ax.set_xticks(range(len(names)), [n.replace('_', '\n') for n in names], rotation=35, ha='right', fontsize=8)
    fig.suptitle(STEM.replace('_', ' ')+' — orange: minimum observed validation NLL; truth evaluated afterward')
    fig.savefig(f'figs/convergence/{STEM}.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    plot()


def plot_initialization_training():
    """Show whether smooth initial amplitudes remain smooth during training."""
    frozen = read(ROOT/'frozen.json')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for row, case in enumerate(CASES_RUN):
        info = frozen['cases'][case]
        for name, model in info['models'].items():
            history = model['result']['history']
            steps = np.array([x['step'] for x in history])
            nll = np.array([x['validation_nll'] for x in history])
            rough = np.array([x['roughness_hessian'] for x in history])
            label = 'default seed0' if name == 'original' else name.replace('_', ' ')
            axes[row, 0].plot(steps[1:], nll[1:], label=label)
            axes[row, 1].semilogy(steps, rough, label=label)
        best = min(m['validation_nll'] for m in info['models'].values())
        axes[row, 0].set(title=case+' — validation NLL (zoom)', xlabel='Adam update', ylabel='Noisy NLL',
                         ylim=(best-.003, best+.04))
        axes[row, 1].set(title='Amplitude roughness grows during learning', xlabel='Adam update',
                         ylabel='Integrated amplitude Hessian energy')
        for ax in axes[row]:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.suptitle('Matched objectives, capacities and batches; only random initialization differs')
    fig.savefig('figs/convergence/smooth_initialization_training.png', dpi=160)
    plt.close(fig)

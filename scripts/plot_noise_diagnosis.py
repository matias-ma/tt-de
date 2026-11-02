"""Publish only figures into the established gallery directories."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from noise_diagnosis import ROOT, SIGMAS
from objective_study import CASES, noise_parameters
from plot_objective_study import convolve_pair, noise_grid, panels

from ftde.direct_training import load_tt, write_record

LABELS = {'old_adam': 'Matched Adam control', 'seed0': 'Fresh start 0', 'seed1': 'Fresh start 1',
          'seed2': 'Fresh start 2', 'longer_adam': '+4000 Adam', 'lbfgs': 'L-BFGS selected', 'lbfgs_last': 'L-BFGS final'}


def read(path):
    return json.loads(Path(path).read_text())


def export(fig, path, footer=None):
    if footer:
        fig.text(.02, .015, footer, va='bottom', fontsize=9)
        fig.get_layout_engine().set(rect=(0, .12, 1, .85))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def optimization():
    frozen = read(ROOT/'frozen.json')
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout='constrained')
    for row, case in enumerate(['swirl4_mixture', 'gl6_mixture']):
        info, results = frozen['cases'][case], read(ROOT/case/'evaluation.json')
        names = ['old_adam']+[n for n in info['models'] if n != 'old_adam']
        for k, name in enumerate(names):
            entry = info['models'][name]
            history = entry['result']['history']
            x = np.array([r['step'] for r in history])
            y = np.array([r['validation_nll'] for r in history])
            # Starting uniform losses are much higher; show the resolved tail.
            if name.startswith('lbfgs'):
                continue
            x = x+(4000 if name == 'longer_adam' else 0)
            mask = x >= 400
            axes[row, 0].plot(x[mask], y[mask], '.-', label=LABELS[name], color=f'C{k}')
        title = 'Swirl 4D, r=32, 49 modes' if row == 0 else 'GL 6D, r=10, 25 modes'
        axes[row, 0].set(title=title+'\nObserved validation NLL', xlabel='Adam updates', ylabel='NLL ↓')
        axes[row, 0].legend(fontsize=8)
        delta = results['test_comparisons']
        axes[row, 1].errorbar(range(len(names)), [delta[n]['test_nll_minus_reference'] for n in names],
                             yerr=[1.96*delta[n]['paired_se'] for n in names], fmt='o', capsize=4)
        axes[row, 1].axhline(0, color='gray', ls='--', lw=1)
        axes[row, 1].set(title='Independent noisy test data', ylabel='NLL difference from matched Adam control\n(points ± 1.96 paired standard errors)')
        axes[row, 2].bar(range(len(names)), [results['models'][n]['relative_l2'] for n in names],
                        color=['C1' if n == info['selected'] else 'C0' for n in names])
        axes[row, 2].set(title='Latent error evaluated after selection', ylabel='Full-dimensional relative L2 ↓')
        for ax in axes[row, 1:]:
            ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=25, ha='right', fontsize=8)
        for ax in axes[row]:
            ax.grid(alpha=.2)
    fig.suptitle('NLL optimization with prescribed mixture noise · one squared Chebyshev TT', fontsize=16)
    export(fig, 'figs/convergence/noisy_optimization.png',
        '100,000 training observations; validation: 4096 swirl / 2048 GL; independent test: 4096. Same data, capacity and Hessian penalty λ=0.001 within each case.\n'
        'Fresh starts: constant + 0.01 random coefficients, 4000 Adam updates. Continuations start from the previous noisy NLL fit. Orange: lowest validation NLL.\n'
        'Adam batch 256; lr=0.001 (fresh), 0.0003 (continuation). L-BFGS uses all 100,000 observations. Paired SE: IID for swirl; 200-chain clustered for GL.')


def accuracy():
    reports = [read(p) for p in sorted((ROOT/'convolution').glob('*.json'))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout='constrained')
    for ax, key, title in zip(axes, ['analytic_gaussian_q256', 'relative_gradient_error'],
        ['Forward value: largest absolute log-density difference', 'NLL gradient: relative Euclidean difference'], strict=True):
        ax.bar(range(len(reports)), [r['errors'][key] for r in reports])
        ax.set_xticks(range(len(reports)), [r['case'].replace('_', '\n') for r in reports])
        ax.set(yscale='log', title=title, ylabel='Error versus analytic Gaussian block quadrature')
        ax.grid(alpha=.2)
    fig.suptitle('Independent convolution check: represented noise versus analytic noise', fontsize=14)
    export(fig, 'figs/convergence/convolution_accuracy.png',
        '12 held-out observations per case, including four observed tails; reference Gauss orders 128 and 256, quadrature-block SVD tolerance 1e-14.\n'
        'Also checked unprepared integration, doubled polynomial quadrature, untruncated moments and a finite-difference coefficient derivative.\n'
        'This is a probe-based numerical audit, not a uniform error bound over every possible TT and observation.')


def noisy_panels(case):
    frozen = read(ROOT/'frozen.json')['cases'][case]
    evaluation = read(ROOT/case/'evaluation.json')
    selected = frozen['selected']
    truth = np.load(ROOT/case/'truth.npz')
    grid = truth['grid']
    model = load_tt(frozen['models'][selected]['path'])
    baseline = load_tt(frozen['models']['old_adam']['path'])
    dim = len(model.cores)
    weights, covariance = noise_parameters(case, dim)
    train = np.load(Path(CASES[case])/'data.npz')['train']
    destination = Path('figs/swirl' if case.startswith('swirl') else 'figs/ginzburg_landau')
    checks = {}
    for i, j in [(0, 1), (2, 3)]:
        pair = f'pair{i}{j}'
        latent = np.load(ROOT/case/f'marginals_{selected}.npz')[pair]
        old = np.load(ROOT/case/'marginals_old_adam.npz')[pair]
        true_noisy, fitted_noisy = convolve_pair(model, case, i, j, grid, weights, covariance, .0125)
        coarse = convolve_pair(model, case, i, j, grid, weights, covariance, .025)
        refinement = max(np.linalg.norm(a-b)/np.linalg.norm(a) for a, b in zip((true_noisy, fitted_noisy), coarse, strict=True))
        if refinement > .01:
            raise FloatingPointError('Plot convolution refinement failed')
        _, old_noisy = convolve_pair(baseline, case, i, j, grid, weights, covariance, .0125)
        hist, _, _ = np.histogram2d(train[:, i], train[:, j], bins=len(grid), range=[(grid[0], grid[-1])]*2)
        hist = hist.T/(len(train)*((grid[-1]-grid[0])/len(grid))**2)
        checks[pair] = {'convolution_refinement': float(refinement),
            'selected_latent_pair_relative_l2': float(np.linalg.norm(latent-truth[pair])/np.linalg.norm(truth[pair])),
            'selected_observed_pair_relative_l2': float(np.linalg.norm(fitted_noisy-true_noisy)/np.linalg.norm(true_noisy)),
            'old_latent_pair_relative_l2': float(np.linalg.norm(old-truth[pair])/np.linalg.norm(truth[pair])),
            'old_observed_pair_relative_l2': float(np.linalg.norm(old_noisy-true_noisy)/np.linalg.norm(true_noisy))}
        arrays = [truth[pair], noise_grid(weights, covariance, i, j, grid), true_noisy, hist,
                  fitted_noisy, latent, old_noisy, old]
        titles = ['True latent density', 'Prescribed noise kernel', 'True noisy density', 'Noisy observation histogram',
                  'Selected NLL: reconvolved fit', 'Selected NLL: latent fit', 'Matched Adam: reconvolved fit', 'Matched Adam: latent fit']
        config = frozen['models'][selected]['config']
        checkpoint = frozen['models'][selected]['result']['best_step']
        noise_description = ('Noise weights 0.5/0.3/0.2: σ=0.22; diagonal σ=0.45/0.25; σ=0.5 with adjacent-pair ρ=±0.6.'
                             if case.startswith('swirl') else 'Noise: isotropic Gaussian components σ=0.45/1.35, weights 0.7/0.3.')
        heading = (f'{case.replace("_", " ")} · NLL optimizer comparison, fixed λ=0.001\nFull {dim}D latent relative L2: selected {evaluation["models"][selected]["relative_l2"]:.3f}'
                   f' · matched Adam control {evaluation["models"]["old_adam"]["relative_l2"]:.3f}')
        footer = (f'Observed-NLL selection: {LABELS[selected]}, checkpoint {checkpoint}/{config["steps"]}; rank={config["rank"]}, modes={config["modes"]}, Hessian λ=0.001, scales=5% box width.\n'
                  f'Ntrain=100,000; Nvalidation={4096 if dim == 4 else 2048}. Constant + 0.01 random start; continuations use the earlier noisy NLL fit. No signal pilot.\n'
                  'One normalized squared Chebyshev TT. Known noise mixtures describe measurement noise. Truth used only after all choices were frozen.\n'
                  +noise_description+'\n'
                  f'Pair-window L2: latent {checks[pair]["selected_latent_pair_relative_l2"]:.3f}, after convolution {checks[pair]["selected_observed_pair_relative_l2"]:.3f}. Cropped panels are not renormalized.')
        panels(arrays, titles, grid, i, j, heading, footer, destination/f'noisy_{case}_marg_{i}{j}.png', groups=[0, 1, 2, 2, 2, 0, 2, 0])
    write_record(ROOT/case/'plot_checks.json', checks)


def strength():
    frozen = read(ROOT/'frozen.json')
    records, latent, convolved, truths, grid = [], [], [], [], None
    for sigma in SIGMAS:
        case = f'gaussian_{sigma:g}'
        metric = read(ROOT/case/'evaluation.json')['models']['seed0']
        model = load_tt(frozen['cases'][case]['models']['seed0']['path'])
        truth = np.load(ROOT/case/'truth.npz')
        grid = truth['grid']
        fit = np.load(ROOT/case/'marginals_seed0.npz')['pair01']
        if sigma:
            p, q = convolve_pair(model, 'swirl4_gaussian', 0, 1, grid, np.ones(1), np.eye(4)[None]*sigma**2, .0125)
            coarse = convolve_pair(model, 'swirl4_gaussian', 0, 1, grid, np.ones(1), np.eye(4)[None]*sigma**2, .025)
            refinement = max(np.linalg.norm(a-b)/np.linalg.norm(a) for a, b in zip((p, q), coarse, strict=True))
            if refinement > .01:
                raise FloatingPointError('Noise sweep plot quadrature failed')
        else:
            p, q = truth['pair01'], fit
            refinement = 0.
        metric.update(sigma=sigma, observed_pair01_relative_l2=float(np.linalg.norm(p-q)/np.linalg.norm(p)),
                      plot_convolution_refinement=float(refinement))
        records.append(metric)
        latent.append(fit)
        convolved.append(q)
        truths.append(p)
    write_record(ROOT/'noise_strength_evaluation.json', {'records': records})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout='constrained')
    axes[0].plot(SIGMAS, [r['relative_l2'] for r in records], 'o-')
    axes[0].set(title='Latent full 4D density', ylabel='Relative L2 error ↓')
    axes[1].plot(SIGMAS, [r['pair01_relative_l2'] for r in records], 'o-', label='Latent pair')
    axes[1].plot(SIGMAS, [r['observed_pair01_relative_l2'] for r in records], 'o-', label='After convolution')
    axes[1].set(title='Same coordinate pair (0,1)', ylabel='Pair-window relative L2 error ↓')
    axes[1].legend()
    axes[2].errorbar(SIGMAS, [r['estimated_observed_kl'] for r in records], yerr=[1.96*r['observed_kl_se'] for r in records], fmt='o-', capsize=4)
    axes[2].axhline(0, color='gray', ls='--')
    axes[2].set(title='Observed likelihood versus true noisy density', ylabel='Estimated KL (nats) ± 1.96 standard errors')
    for ax in axes:
        ax.set(xlabel='Independent Gaussian measurement-noise σ')
        ax.grid(alpha=.2)
    fig.suptitle('Matched noise experiment: convolution can hide a worsening latent fit', fontsize=15)
    footer = ('One squared Chebyshev TT: rank 32, 49 modes/axis, same box [−3,3]⁴ from prior noisy-only bounds, same near-uniform start and batch sequence.\n'
              '100,000 training / 4096 validation observations; 4000 Adam steps, batch 256, lr=0.001; Hessian λ=0.001, length scales 0.3.\n'
              'Identical latent draws and standard noise draws across levels. Each fit sees only its own observations. Checkpoints selected by observed NLL before truth evaluation.')
    export(fig, 'figs/convergence/noise_strength.png', footer)
    panels(truths+latent, [f'True observed density · σ={s:g}' for s in SIGMAS]+
           [f'Latent fit · σ={s:g}\nFull 4D L2={r["relative_l2"]:.3f}' for s, r in zip(SIGMAS, records, strict=True)], grid, 0, 1,
           'Swirl 4D · fixed capacity, box, regularizer, initialization and update budget', footer,
           Path('figs/swirl/noise_strength_marg_01.png'), groups=[1, 2, 3, 4]+[0]*4)


def box_control():
    cases = ['gl6_free', 'gl6_wide_free']
    arrays, titles, pairs = [], [], []
    truth = np.load(ROOT/cases[0]/'truth.npz')
    for i, j in [(0, 1), (2, 3)]:
        pair = f'pair{i}{j}'
        arrays.append(truth[pair])
        titles.append(f'True GL marginal · ({i},{j})')
        pairs.append((i, j))
        for case, name, label in [('gl6_free', 'seed0', 'Original box · 25 modes'),
                                 ('gl6_wide_free', 'seed0', 'Wider box · 25 modes'),
                                 ('gl6_wide_free', 'modes49', 'Wider box · 49 modes')]:
            result = read(ROOT/case/'evaluation.json')['models'][name]
            arrays.append(np.load(ROOT/case/f'marginals_{name}.npz')[pair])
            titles.append(label+f'\nFull 6D L2={result["relative_l2"]:.3f}')
            pairs.append((i, j))
    panels(arrays, titles, truth['grid'], 0, 1, 'GL box-width control · all fits use noise-free observations',
        'Rank 10; modes as labeled; Ntrain=100,000, Nvalidation=2048; Adam 4000 updates, batch 256, lr=0.001; constant + 0.01 random start.\n'
        'Original box widths ≈4.3 versus noisy-data box widths ≈7.5; Hessian λ=0.001, scales=5% of box width. More modes test physical resolution.\n'
        'The box change affects representation and regularization geometry together. No clean fit initializes or selects a noisy fit.',
        Path('figs/ginzburg_landau/noise_box_control.png'), groups=[0]*4+[1]*4, coordinate_pairs=pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['accuracy', 'optimization', 'strength', 'box', 'swirl4_mixture', 'gl6_mixture'])
    part = parser.parse_args().part
    if part in ['swirl4_mixture', 'gl6_mixture']:
        noisy_panels(part)
    else:
        {'accuracy': accuracy, 'optimization': optimization, 'strength': strength, 'box': box_control}[part]()

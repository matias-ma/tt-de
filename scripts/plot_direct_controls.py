"""Plot frozen noise-free controls separating capacity and regularization."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('artifacts/direct_optimization')
case = root/'swirl_free4'
evaluation = json.loads((case/'evaluation_free.json').read_text())['models']
training = json.loads((root/'frozen_training_objectives.json').read_text())
strengths = ['0', '1e-05', '0.0001', '0.001']
names = [f'squared_n49_r32_l{s}_b1024_s0' for s in strengths]
colors = ['tab:red', 'goldenrod', 'tab:green', 'tab:blue']
fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
truth = np.load(case/'truth_evaluation.npz')
axes[0, 0].plot(truth['t'], truth['density'], color='black', lw=2, label='Truth (evaluation only)')
for name, strength, color in zip(names, strengths, colors, strict=True):
    density = np.load(case/name/'profile_free.npz')['density']
    axes[0, 0].plot(truth['t'], density, color=color, lw=.9, label=f'λ = {strength}')
axes[0, 0].set(title='Noise-free: oscillations remain without sign crossings',
               xlabel='True curve parameter (evaluation only)', ylabel='Joint density on curve')
axes[0, 0].legend(fontsize=8)
energies = [training['swirl_free4/'+name]['roughness'] for name in names]
axes[0, 1].bar(strengths, energies, color=colors)
axes[0, 1].set(yscale='log', title='Curvature penalty suppresses rapid oscillations',
               xlabel='Curvature strength λ', ylabel='Integrated, scaled amplitude curvature H²')
for i, energy in enumerate(energies):
    axes[0, 1].text(i, energy*1.08, f'{energy:.1f}', ha='center', fontsize=9)
axes[0, 1].set_ylim(1, max(energies)*2)
refined = 'squared_n49_r32_l0.0001_b1024_s0_lbfgs'
compare = [*names, refined, 'previous_squared']
axes[1, 0].bar(np.arange(len(compare)), [evaluation[n]['relative_l2'] for n in compare],
               color=[*colors, 'tab:purple', 'gray'])
axes[1, 0].set_xticks(np.arange(len(compare)), ['λ = '+s for s in strengths]+['λ = 0.0001\n+ L-BFGS', 'Historical'], fontsize=8)
axes[1, 0].set(title='Independent density recovery', ylabel='Full 4D relative L2 ↓', ylim=(0, .12))
for i, name in enumerate(compare):
    error = evaluation[name]['relative_l2']
    axes[1, 0].text(i, error+.001, f'{error:.3f}', ha='center', fontsize=9)
capacity = ['squared_n25_r16_l0.0001_b1024_s0', names[2],
            'squared_n65_r32_l0.0001_b1024_s0', 'squared_n49_r64_l0.0001_b1024_s0']
positions = np.arange(len(capacity))
axes[1, 1].plot(positions, [training['swirl_free4/'+n]['training_nll'] for n in capacity],
                'o-', label='All 100,000 training observations')
axes[1, 1].plot(positions, [evaluation[n]['nll'] for n in capacity], 'o-', label='4,096 validation observations')
axes[1, 1].set_xticks(positions, ['25 modes\nrank 16', '49 modes\nrank 32', '65 modes\nrank 32', '49 modes\nrank 64'])
axes[1, 1].set(title='Larger fits improve training more than validation', ylabel='Negative log likelihood ↓')
axes[1, 1].legend(fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle('Noise-free 4D swirl: regularization, capacity, and optimization', fontsize=16)
fig.get_layout_engine().set(rect=(0, .065, 1, .89))
fig.text(.015, .018, 'New fits: Chebyshev TT, near-uniform start, 4,000 Adam updates, batch 1,024; same seed. Historical fit has a different recipe.\n'
         'Regularization controls use 49 modes/rank 32. All four amplitudes have zero sampled sign crossings on the true curve.\n'
         'All models and validation choices were frozen before truth evaluation. Curvature is measured over the whole box, not just the curve.', fontsize=9)
fig.savefig('figs/swirl/direct_optimization_free_dim4_controls.png', dpi=160)
plt.close(fig)

selected = json.loads((case/'evaluation_free.json').read_text())['displayed']
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
for key, ax in zip(('signed', 'penalized'), axes, strict=True):
    name = selected[key]
    result = evaluation[name]
    profile = np.load(case/name/'profile_free.npz')
    np.testing.assert_allclose(profile['density'], np.maximum(profile['raw_tt'], 0)/result['positive_mass'], atol=1e-12)
    ax.plot(truth['t'], truth['density'], color='black', lw=2, label='True density (evaluation only)')
    ax.plot(truth['t'], profile['raw_tt'], color='tab:blue', ls='--', label='Optimized signed TT: raw values')
    ax.plot(truth['t'], profile['density'], color='tab:orange', label='Final truncated, normalized density')
    description = 'Signed TT' if key == 'signed' else 'Signed TT + negative-part penalty'
    reduction = 100*(1-1/result['positive_mass'])
    ax.set(title=f'{description}\nRemoved mass {result["negative_mass"]:.3f} → positive values fall {reduction:.1f}%',
           xlabel='True curve parameter (evaluation only)', ylabel='Joint function value on the curve')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
fig.suptitle('Clipping at the end changes even the already-positive parts of the fit', fontsize=15)
fig.get_layout_engine().set(rect=(0, .13, 1, .82))
fig.text(.015, .02, 'Both trained signed TTs integrate to one. Their positive parts integrate to 1 + removed negative mass, so final normalization rescales the entire curve.\n'
         'The raw signed TT is not a probability density: it remains negative elsewhere. These are joint values on a post-fit diagnostic curve, not integrated errors.\n'
         'The final densities assign zero probability to 107/8,192 and 87/8,192 independent test samples, respectively; their test NLL is infinite.', fontsize=9)
fig.savefig('figs/swirl/direct_optimization_free_dim4_clipping.png', dpi=160)
plt.close(fig)

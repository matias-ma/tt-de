"""Frozen spatial-smoothing tradeoffs, with the data-selected amount marked."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('artifacts/direct_optimization/smoothing_free4')
selection = json.loads((root/'selection.json').read_text())
evaluation = json.loads((root/'evaluation.json').read_text())
names = sorted(selection['models'], key=lambda n: selection['models'][n]['fraction'])
scales = [selection['models'][n]['fraction'] for n in names]
fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
risk = [selection['models'][n]['validation_risk'] for n in names]
axes[0, 0].plot(scales, risk, 'o-', color='tab:blue')
chosen = names.index(selection['selected'])
axes[0, 0].scatter(scales[chosen], risk[chosen], marker='*', s=170, color='black', zorder=5, label='Validation-selected amount')
axes[0, 0].set(title='Held-out error worsens with substantial smoothing', ylabel='Validation least-squares risk ↓')
axes[0, 0].legend(fontsize=9)
axes[0, 1].plot(scales, [selection['models'][n]['negative_mass'] for n in names], 'o-', color='tab:orange')
axes[0, 1].set(title='Spatial smoothing does reduce negative mass', ylabel='Negative mass removed at final truncation')
axes[1, 0].plot(scales, [evaluation['models'][n]['relative_l2'] for n in names], 'o-', color='tab:purple')
axes[1, 0].set(title='Independent evaluation after the choice is frozen', ylabel='Full 4D relative L2 ↓')
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set(xscale='symlog', xlabel='Smoothing standard deviation / coordinate box width')
    ax.set_xscale('symlog', linthresh=.0025)
    ax.set_xticks([0, .0025, .01, .04, .16], ['0', '.0025', '.01', '.04', '.16'])
truth = np.load(root/'truth.npz')
axes[1, 1].plot(truth['t'], truth['density'], color='black', lw=2, label='Truth (evaluation only)')
for name, label in [('heat_0', 'Final truncation only'), ('heat_0.0025', 'Very light smoothing'),
                    ('heat_0.01', 'Moderate smoothing'), ('heat_0.04', 'Strong smoothing')]:
    profile = np.load(root/name/'profile.npz')['density']
    axes[1, 1].plot(truth['t'], profile, lw=1, label=label)
axes[1, 1].set(title='Reducing negative mass also broadens the curved tube',
               xlabel='True curve parameter (evaluation only)', ylabel='Joint density on curve')
axes[1, 1].legend(fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle('Noise-free signed TT: spatial smoothing before final truncation', fontsize=16)
fig.get_layout_engine().set(rect=(0, .085, 1, .87))
fig.text(.015, .018, 'Fixed TT learned by optimization; no new density estimator is fitted. Heat diffusion preserves Chebyshev modes, TT ranks and signed mass.\n'
         'Remaining negative values are truncated only after smoothing. Natural no-flux boundaries keep mass inside the original box.\n'
         'The strict validation minimum is no smoothing; the smallest positive amount is nearly tied. Truth is used only to assess the frozen grid, never to reselect it.', fontsize=9)
fig.savefig('figs/swirl/direct_optimization_free_dim4_smoothing.png', dpi=160)
plt.close(fig)

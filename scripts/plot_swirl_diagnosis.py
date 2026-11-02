"""Display post-fit amplitude zeros, mode tails and physical rank spectra."""
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from refresh_density_fits import read_model

from ftde.swirl import curve

root = Path('artifacts/swirl_improvement')
old_path = 'artifacts/swirl/swirl_mixture_dim10/selected.npz'
new_path = str(root/'tt_tangent_smallbatch_dim10_r64/fit.npz')
old = json.loads((root/'diagnostics10.json').read_text())[old_path]
new = json.loads((root/'diagnostics_tangent10.json').read_text())[new_path]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
t = np.linspace(0, 1, 4001)
for label, path, record, color in [('Previous TT', old_path, old, '#6b7280'),
                                   ('Positive-pilot TT', new_path, new, '#d95f02')]:
    amplitude = np.asarray(read_model(path)(jnp.asarray(curve(t, 10))))
    amplitude = amplitude*np.sign(amplitude[0])/np.max(np.abs(amplitude))
    axes[0].plot(t, amplitude, label=f'{label}: {record["amplitude_sign_changes_on_true_curve"]} sign changes', color=color)
    modes = record['top_eight_mode_energy']
    axes[1].semilogy(range(10), [modes[str(k)] for k in range(10)], 'o-', label=label, color=color)
    energy = np.asarray(record['schmidt_spectra']['5']['energy'])
    tail = np.maximum(0, 1-np.cumsum(energy))
    axes[2].semilogy(np.arange(1, len(tail)+1), np.maximum(tail, 1e-16), label=label, color=color)
axes[0].axhline(0, color='black', lw=.8, ls=':')
axes[0].set(title='Artificial zeros along the curve', xlabel='True path parameter (diagnosis only)', ylabel='Signed amplitude / maximum magnitude')
axes[1].set(title='Highest eight polynomial modes', xlabel='Coordinate', ylabel='Fraction of squared amplitude norm')
axes[2].set(title='Central physical Schmidt spectrum', xlabel='Retained rank', ylabel='Fraction of squared amplitude norm discarded')
for rank in (16, 32, 64):
    axes[2].axvline(rank, color='black', ls=':', lw=.6)
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
fig.suptitle('10D noisy swirl: same rank 64 and 49 modes, different data-derived initialization', fontweight='bold')
fig.supxlabel('The density is the normalized square of the amplitude: each sign crossing forces a zero. Global amplitude sign is arbitrary.\n'
              'Basis energies and singular values use the physical Lebesgue L2 metric. The true curve is used only after fitting; it never enters initialization or selection.', fontsize=8)
fig.savefig('figs/swirl/noisy_mixture_diagnosis_dim10.png', dpi=180)
plt.close(fig)

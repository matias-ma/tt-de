"""Native 4D Tucker fit on the same cryo-EM observations and prescribed kernel."""
import json
from pathlib import Path

import jax
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_tucker_gallery import digest, freeze
from transfer_direct_cryoem import BASES

from ftde import (
    TuckerConvolution,
    TuckerTrainingProblem,
    init_tucker_constant,
    load_tucker,
)
from ftde.direct_training import load_tt, write_record
from ftde.marginals import squared_pair_grid
from ftde.tucker_operations import tucker_to_tt

ROOT = Path('artifacts/tucker_gallery/cryoem4')


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    base = Path(BASES['dim4'])
    with np.load(base/'data.npz') as data:
        train, valid = data['train'], data['valid']
    with np.load(base/'selected.npz') as data:
        domain = data['domain']
    protocol = {'family': 'cryoem', 'dim': 4, 'decomposition': 'tucker', 'data': str(base/'data.npz'),
                'data_sha256': digest(base/'data.npz'), 'kernel': str(base/'kernel.npz'), 'domain': domain.tolist(),
                'recipes': ['strong', 'weak'], 'rank': 8, 'modes': 49, 'steps': 4000,
                'selection': 'minimum held-out noisy NLL; no latent oracle', 'uses_signal_pilot': False}
    if (ROOT/'protocol.json').exists():
        assert json.loads((ROOT/'protocol.json').read_text()) == protocol
    else:
        write_record(ROOT/'protocol.json', protocol)
    for name, strength in [('strong', .001), ('weak', 1e-5)]:
        out = ROOT/name
        if (out/'result.json').exists():
            continue
        initial = init_tucker_constant(domain, rank=8, modes=49)
        obs = np.concatenate([train, valid])
        observation_domain = np.stack([obs.min(axis=0)-.02, obs.max(axis=0)+.02], axis=1)
        forward = TuckerConvolution.from_tucker(initial, load_tt(base/'kernel.npz')).prepare(observation_domain)
        problem = TuckerTrainingProblem(initial, forward=forward, strength=strength)
        out.mkdir(exist_ok=True)
        write_record(out/'config.json', protocol | {'strength': strength, 'batch_size': 256, 'microbatches': 4})
        problem.fit_adam(train, valid, out, steps=4000, batch_size=256, microbatches=4, validation_batch_size=16)
        jax.clear_caches()
    freeze(ROOT)
    selection = json.loads((ROOT/'selection.json').read_text())
    model = tucker_to_tt(load_tucker(ROOT/selection['selected']/'fit.npz'))
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout='constrained')
    for row, (i, j) in enumerate([(0, 1), (2, 3)]):
        xs, ys = np.linspace(*domain[i], 181), np.linspace(*domain[j], 181)
        marginal = squared_pair_grid(model, i, j, xs, ys)
        hist, xe, ye = np.histogram2d(train[:, i], train[:, j], bins=90, range=[domain[i], domain[j]])
        hist = hist.T/(len(train)*(xe[1]-xe[0])*(ye[1]-ye[0]))
        vmax = max(marginal.max(), hist.max())
        axes[row, 0].pcolormesh(xe, ye, hist, cmap='magma', vmin=0, vmax=vmax)
        im = axes[row, 1].pcolormesh(xs, ys, marginal, cmap='magma', vmin=0, vmax=vmax)
        axes[row, 0].set(title='Observed data', xlabel=f'x{i}', ylabel=f'x{j}')
        axes[row, 1].set(title='Tucker latent marginal', xlabel=f'x{i}', ylabel=f'x{j}')
        fig.colorbar(im, ax=axes[row], shrink=.8)
    fig.suptitle(f'Cryo-EM · 4D Tucker · rank 8 · 49 modes\nSelected {selection["selected"]}; noisy validation NLL {min(selection["scores"].values()):.4f}')
    fig.savefig('figs/cryoem/tucker_dim4.png', dpi=165)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for name in protocol['recipes']:
        r = json.loads((ROOT/name/'result.json').read_text())
        ax.plot([h['step'] for h in r['history']], [h['validation_nll'] for h in r['history']], label=name)
    ax.set(title='Cryo-EM · 4D Tucker', xlabel='Adam updates', ylabel='Noisy held-out NLL')
    ax.legend()
    fig.savefig('figs/cryoem/tucker_training.png', dpi=165)
    plt.close(fig)


if __name__ == '__main__':
    run()

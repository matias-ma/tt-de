"""Frozen-kernel capacity and smoothness controls for cryo-EM beads."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from refine_cryoem_gallery import expand_tt
from refresh_density_fits import fit_candidate, read_model

from ftde.marginals import squared_pair_grid
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.sobolev import AmplitudeSobolevPenalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--modes', type=int, default=49)
    parser.add_argument('--order', type=int, choices=[1, 2], default=2)
    parser.add_argument('--strength', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--preconditioning', choices=['l2', 'sobolev'], default='l2')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['source_sha256'] = hashlib.sha256((args.base/'selected.npz').read_bytes()).hexdigest()
    config['kernel_sha256'] = hashlib.sha256((args.base/'kernel.npz').read_bytes()).hexdigest()
    path = args.output/'experiment.json'
    if path.exists():
        previous = json.loads(path.read_text())
        previous.setdefault('preconditioning', 'l2')
        if previous != config:
            raise ValueError('Different experiment; use a new output directory')
    path.write_text(json.dumps(config, indent=2)+'\n')
    original, kernel = read_model(args.base/'selected.npz'), read_model(args.base/'kernel.npz')
    data = np.load(args.base/'data.npz')
    init = expand_tt(original, args.modes, args.rank, seed=args.seed)
    scales = [m.moments()[1] for m in noise_marginals_from_tt(kernel)]
    penalty = AmplitudeSobolevPenalty.from_tt(init, order=args.order, length_scales=scales)
    fit = fit_candidate(init, data['train'], data['valid'], args.output, kernel=kernel,
        penalty=penalty, strength=args.strength, steps=args.steps, lr=args.lr,
        batch_size=256, checkpoint_rule='regularized', seed=args.seed,
        preconditioning=args.preconditioning)
    record = json.loads((args.output/'result.json').read_text())
    common = AmplitudeSobolevPenalty.from_tt(fit, order=2, length_scales=scales)
    record['common_h2'] = float(common(fit))
    (args.output/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    for i, j in [(0, 1), (2, 3)]:
        box = np.asarray(fit.domain)[[i, j]]
        xs, ys = np.linspace(*box[0], 240), np.linspace(*box[1], 240)
        grids = [squared_pair_grid(g, i, j, xs, ys) for g in (original, fit)]
        np.savez(args.output/f'marginal_{i}{j}.npz', xs=xs, ys=ys, before=grids[0], after=grids[1])
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
        for ax, grid, title in zip(axes, grids, ['Previous fit', args.output.name], strict=True):
            im = ax.imshow(grid, origin='lower', extent=tuple(box[0])+tuple(box[1]),
                aspect='auto', vmin=0, vmax=max(z.max() for z in grids), cmap='magma')
            ax.set(title=title, xlabel=f'x{i} (standardized)', ylabel=f'x{j} (standardized)')
            fig.colorbar(im, ax=ax)
        fig.suptitle(f'Fixed noise kernel; noisy validation NLL {record["best_nll"]:.6f}')
        fig.savefig(args.output/f'comparison_{i}{j}.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()

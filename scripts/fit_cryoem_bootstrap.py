"""Conditional bootstrap continuations with a fixed noise kernel and support.

The source fit used all original training rows. These resampled continuations
measure local stability; they are not independent bootstrap fits from scratch.
Only training observations are resampled; validation rows remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

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
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--rank', type=int, default=48)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--strength', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resample', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    initial = args.initial or args.base/'selected.npz'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['source_sha256'] = hashlib.sha256(initial.read_bytes()).hexdigest()
    config['kernel_sha256'] = hashlib.sha256((args.base/'kernel.npz').read_bytes()).hexdigest()
    config['data_sha256'] = hashlib.sha256((args.base/'data.npz').read_bytes()).hexdigest()
    config['initial'] = str(initial)
    config.update(modes=49, order=2, preconditioning='sobolev', lr=2e-4, batch_size=256)
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text()) != config:
        raise ValueError('Settings changed; use a new output directory')
    (out/'experiment.json').write_text(json.dumps(config, indent=2)+'\n')
    data = np.load(args.base/'data.npz')
    train = data['train']
    if args.resample:
        indices = np.random.default_rng(args.seed).integers(0, len(train), len(train))
        np.save(out/'bootstrap_indices.npy', indices)
        train = train[indices]
    original = read_model(initial)
    if args.initial is not None and not np.array_equal(original.domain, read_model(args.base/'selected.npz').domain):
        raise ValueError('Alternative initialization must preserve the frozen latent support')
    kernel = read_model(args.base/'kernel.npz')
    init = expand_tt(original, 49, args.rank, seed=args.seed)
    scales = [m.moments()[1] for m in noise_marginals_from_tt(kernel)]
    penalty = AmplitudeSobolevPenalty.from_tt(init, order=2, length_scales=scales)
    model = fit_candidate(init, train, data['valid'], out, kernel=kernel, penalty=penalty,
                          strength=args.strength, steps=args.steps, lr=2e-4,
                          batch_size=256, checkpoint_rule='regularized', seed=args.seed,
                          preconditioning='sobolev')
    record = json.loads((out/'result.json').read_text())
    record['common_h2'] = float(penalty(model))
    (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    for i, j in [(0, 1), (2, 3), (8, 9)]:
        if j >= len(model.cores):
            continue
        box = np.asarray(model.domain)[[i, j]]
        xs, ys = np.linspace(*box[0], 240), np.linspace(*box[1], 240)
        np.savez(out/f'marginal_{i}{j}.npz', xs=xs, ys=ys,
                 after=squared_pair_grid(model, i, j, xs, ys))


if __name__ == '__main__':
    main()

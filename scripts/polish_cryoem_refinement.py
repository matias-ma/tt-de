"""Continue smoothing after the higher-rank representation has been learned."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from refresh_density_fits import fit_candidate, read_model, save_model

from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.sobolev import AmplitudeSobolevPenalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--strengths', type=float, nargs='+', default=[1e-4, 3e-4])
    args = parser.parse_args()
    out = args.experiment
    config = json.loads((out/'experiment.json').read_text())
    base = Path(config['base'])
    previous = out/'selection_before_continuation.json'
    if not previous.exists():
        shutil.copy2(out/'selection.json', previous)
        shutil.copy2(out/'selected.npz', out/'continuation_init.npz')
    selection = json.loads(previous.read_text())
    source = selection['selected']
    description = {'source': source, 'steps': args.steps, 'strengths': args.strengths,
                   'lr': .0001, 'batch_size': 256}
    settings = out/'continuation.json'
    if settings.exists() and json.loads(settings.read_text()) != description:
        raise ValueError('Continuation configuration differs')
    settings.write_text(json.dumps(description, indent=2)+'\n')
    gaussian_path = out/'corrected_kernel.npz' if config.get('correct_projection') else base/'kernel.npz'
    gaussian = read_model(gaussian_path)
    kernel = read_model(out/'mixture_kernel.npz') if source.startswith('mixture') else gaussian
    init = read_model(out/'continuation_init.npz')
    scales = [m.moments()[1] for m in noise_marginals_from_tt(gaussian)]
    penalty = AmplitudeSobolevPenalty.from_tt(init, order=2, length_scales=scales)
    data = np.load(base/'data.npz')
    names, curvature = list(selection['names']), list(selection['curvature'])
    logs = [np.load(out/'baseline_validation.npy' if n == 'baseline' else out/n/'validation.npy') for n in names]
    for strength in args.strengths:
        name = f'polish_h2_{strength:g}'
        fitted = fit_candidate(init, data['train'], data['valid'], out/name, kernel=kernel,
            penalty=penalty, strength=strength, steps=args.steps, lr=.0001, batch_size=256,
            checkpoint_rule='regularized')
        names.append(name)
        curvature.append(float(penalty(fitted)))
        logs.append(np.load(out/name/'validation.npy'))
    chosen, detail = select_by_paired_noisy_nll(np.stack(logs), curvature)
    selected = names[chosen]
    model = read_model(base/'selected.npz' if selected == 'baseline' else out/selected/'fit.npz')
    save_model(out/'selected.npz', model)
    selection.update(selected=selected, names=names, curvature=curvature, selection=detail)
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    print('SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()

"""Select frozen-kernel controls by noisy validation and common curvature."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from refresh_density_fits import read_model

from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.sobolev import AmplitudeSobolevPenalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('candidates', nargs='+', type=Path)
    args = parser.parse_args()
    base, out = args.base.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prior_name = json.loads((base/'selection.json').read_text())['selected']
    folders = [base/prior_name, *[p.resolve() for p in args.candidates]]
    names = ['previous', *[p.name for p in args.candidates]]
    if len(set(names)) != len(names):
        raise ValueError('Candidate names must be unique')
    kernel = read_model(base/'kernel.npz')
    scales = [m.moments()[1] for m in noise_marginals_from_tt(kernel)]
    original = read_model(base/'selected.npz')
    roughness = [float(AmplitudeSobolevPenalty.from_tt(original, order=2, length_scales=scales)(original))]
    records = [json.loads((folder/'result.json').read_text()) for folder in folders]
    roughness.extend(r['common_h2'] for r in records[1:])
    chosen, details = select_by_paired_noisy_nll(
        np.stack([np.load(p/'validation.npy') for p in folders]), roughness)
    configs = [json.loads((p/'experiment.json').read_text()) for p in folders[1:]]
    for config in configs:
        if Path(config['base']).resolve() != base:
            raise ValueError('Candidates must use the same frozen base')
        for key, filename in [('source_sha256', 'selected.npz'), ('kernel_sha256', 'kernel.npz')]:
            if config[key] != hashlib.sha256((base/filename).read_bytes()).hexdigest():
                raise ValueError('The frozen source or noise kernel changed')

    def link(name, target):
        path = out/name
        if path.is_symlink():
            path.unlink()
        if path.exists():
            raise FileExistsError(path)
        path.symlink_to(target.resolve())

    for name in ['data.npz', 'split.npz', 'bandwidth.npy', 'warmup', 'naive', 'naive_initialization.json']:
        if (base/name).exists():
            link(name, base/name)
    for name, folder in zip(names, folders, strict=True):
        link(name, folder)
    shutil.copy2(base/'selected.npz' if chosen == 0 else folders[chosen]/'fit.npz', out/'selected.npz')
    shutil.copy2(base/'kernel.npz', out/'kernel.npz')
    settings = json.loads((base/'settings.json').read_text())
    settings['output'] = str(out)
    (out/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    selection = {'selected': names[chosen], 'names': names, 'roughness': roughness, 'selection': details,
                 'rule': 'paired noisy one-SE, then minimum common physical amplitude H2'}
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    baseline_metadata = json.loads((base/'refinement.json').read_text())
    metadata = {'base': str(base), 'configuration': {'correct_projection': baseline_metadata['configuration'].get('correct_projection', False)},
        'kernel_description': baseline_metadata['kernel_description'], 'baseline_nll': records[0]['best_nll'],
        'selected_nll': records[chosen]['best_nll'], 'continuity': {
            'ranks': sorted({c['rank'] for c in configs}), 'modes': sorted({c['modes'] for c in configs}),
            'selected_order': 2 if chosen == 0 else configs[chosen-1]['order'], 'candidates': configs}}
    (out/'refinement.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Materialize a selected refinement as a reproducible gallery case."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment, output = args.experiment.resolve(), args.output.resolve()
    if (experiment/'INVALID_MIXTURE.json').exists():
        raise ValueError('This pilot contains invalid mixture scores and cannot be published')
    config = json.loads((experiment/'experiment.json').read_text())
    base = Path(config['base']).resolve()
    selection = json.loads((experiment/'selection.json').read_text())
    previous = json.loads((base/'selection.json').read_text())['selected']
    selected = selection['selected']
    continuation = json.loads((experiment/'continuation.json').read_text()) if (experiment/'continuation.json').exists() else None
    output.mkdir(parents=True, exist_ok=True)

    def link(name, source):
        target = output/name
        if target.is_symlink():
            target.unlink()
        if target.exists():
            raise FileExistsError(target)
        target.symlink_to(source)

    for name in ['data.npz', 'split.npz', 'bandwidth.npy', 'warmup', 'naive', 'naive_initialization.json']:
        if (base/name).exists():
            link(name, base/name)
    link('previous', experiment/'baseline_rechecked' if config.get('correct_projection') else base/previous)
    for name in selection['names']:
        if name != 'baseline':
            link(name, experiment/name)
    uses_mixture = selected.startswith('mixture') or (selected.startswith('polish') and continuation['source'].startswith('mixture'))
    if uses_mixture:
        kernel = experiment/'mixture_kernel.npz'
        description = '4-component stationary Gaussian noise mixture; diagonal within components; supplied noise metadata only'
    else:
        kernel = experiment/'corrected_kernel.npz' if config.get('correct_projection') else base/'kernel.npz'
        description = 'Prescribed diagonal Gaussian noise approximation'
    shutil.copy2(kernel, output/'kernel.npz')
    shutil.copy2(experiment/'selected.npz', output/'selected.npz')
    settings = json.loads((base/'settings.json').read_text())
    settings['output'] = str(output)
    (output/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    selection['names'] = ['previous' if n == 'baseline' else n for n in selection['names']]
    selection['selected'] = 'previous' if selected == 'baseline' else selected
    (output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    rows = selection['selection']['candidates']
    chosen = selection['selection']['selected_index']
    metadata = {'base': str(base), 'experiment': str(experiment), 'configuration': config,
        'continuation': continuation,
        'kernel_description': description,
        'baseline_nll': rows[0]['validation_nll'], 'selected_nll': rows[chosen]['validation_nll'],
        'baseline_excess_nll': rows[0]['excess_nll'], 'baseline_paired_se': rows[0]['paired_cluster_se'],
        'selection_note': 'Held-out noisy likelihood with paired one-SE rule and common amplitude curvature; no clean data or labels'}
    (output/'refinement.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

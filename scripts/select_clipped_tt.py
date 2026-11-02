"""Freeze clipped candidates using only noisy validation likelihood."""
import argparse
import json
from pathlib import Path

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/clipped_tt'))
    args = parser.parse_args()
    candidates = sorted(p.parent for p in args.root.glob('swirl4_*/result.json'))
    names, nlls, records = [], [], []
    for directory in candidates:
        r = json.loads((directory/'result.json').read_text())
        logs = np.load(directory/'validation.npy')
        names.append(directory.name)
        nlls.append(float(-logs.mean()))
        records.append({'name': directory.name, 'nll': nlls[-1], 'best_step': r['best_step']})
    selected = names[int(np.argmin(nlls))]
    warm = [r for r in records if '_warm_' in r['name']]
    warm_selected = min(warm, key=lambda r: r['nll'])['name'] if warm else None
    result = {'selected': selected, 'warm_selected': warm_selected, 'candidates': records,
              'rule': 'minimum noisy validation NLL; no target evaluation'}
    (args.root/'selection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result)

"""Freeze direct-TT choices using held-out observations only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ftde.direct_training import write_record
from ftde.noisy_model_selection import select_by_paired_noisy_nll

ROOT = Path('artifacts/noise_free_v2')


def freeze(case):
    folder = ROOT/case
    paths = sorted(p for p in folder.iterdir() if (p/'result.json').exists())
    records = [json.loads((p/'result.json').read_text()) for p in paths]
    logs = np.stack([np.load(p/'validation.npy') for p in paths])
    roughness = [next(h['roughness_hessian'] for h in r['history'] if h['step'] == r['best_step']) for r in records]
    index, details = select_by_paired_noisy_nll(logs, roughness)
    names = [p.name for p in paths]
    selected = names[index]
    result = {'selected': selected, 'names': names, 'selection': details,
              'rule': 'paired one-SE joint held-out NLL, then lowest full-Hessian amplitude energy',
              'uses_clean_truth': False, 'uses_signal_pilot': False,
              'files': {str(p/'fit.npz'): hashlib.sha256((p/'fit.npz').read_bytes()).hexdigest() for p in paths},
              'result_files': {str(p/'result.json'): hashlib.sha256((p/'result.json').read_bytes()).hexdigest() for p in paths}}
    if (folder/'frozen.json').exists():
        assert json.loads((folder/'frozen.json').read_text()) == result, 'Frozen selection cannot change after evaluation'
    write_record(folder/'frozen.json', result)
    print(case, selected, details['candidates'][index], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    for case in parser.parse_args().cases:
        freeze(case)

"""Recover an interrupted cryo-EM refinement from its saved best checkpoint.

The restart preserves the observed-data selection and resets L-BFGS history.
It is recorded explicitly; it is not equivalent to an uninterrupted solve.
"""
import argparse
import json
import shutil

import numpy as np
from run_noise_free_cryoem import BASES, ROOT

from ftde.direct_lbfgs import fit_lbfgs
from ftde.direct_training import NoisyDirectTrainingProblem, load_tt, write_record
from ftde.tt_convolution import TTConvolution


def resume(case):
    folder = ROOT/f'cryoem_{case}'
    out = folder/'lbfgs'
    if (out/'result.json').exists():
        return
    if (out/'interrupted_progress.json').exists():
        raise RuntimeError('A previous recovery exists; inspect it before another restart')
    protocol = json.loads((folder/'protocol.json').read_text())
    old = json.loads((out/'progress.json').read_text())
    offset = old['best_step']
    remaining = protocol['refinement_iterations']-offset
    if remaining <= 0:
        raise RuntimeError('No remaining refinement iterations')
    prefix = [row for row in old['history'] if row['step'] <= offset]
    write_record(out/'interrupted_progress.json', old)
    shutil.copy2(out/'fit.npz', out/'restart.npz')
    initial = load_tt(out/'restart.npz')
    parent = json.loads((out/'config.json').read_text())['parent']
    with np.load(f'{BASES[case]}/data.npz') as data:
        train, valid = data['train'], data['valid']
    kernel = load_tt(f'{BASES[case]}/kernel.npz')
    forward = TTConvolution.from_tt(initial, kernel).prepare(tolerance=1e-14)
    problem = NoisyDirectTrainingProblem(initial, forward, **protocol['recipes'][parent])
    _, record = fit_lbfgs(problem, train, valid, out, iterations=remaining,
                         batch_size=128, validation_batch_size=128)
    elapsed = prefix[-1]['seconds']
    record['history'] = prefix+[
        row | {'step': row['step']+offset, 'seconds': row['seconds']+elapsed}
        for row in record['history'][1:]]
    record['best_step'] += offset
    record['iterations'] += offset
    record['seconds'] += elapsed
    record.update(restart_from_step=offset, restart_resets_lbfgs_history=True,
                  interruption='Allocation terminated; restored last saved best checkpoint',
                  evaluation_trace_scope='restart segment only',
                  timing_scope='completed prefix checkpoints plus restart; excludes lost work')
    write_record(out/'result.json', record)
    write_record(out/'progress.json', {k: record[k] for k in ('history', 'best_step', 'best_nll')})
    print('RECOVERED', case, record['best_step'], record['best_nll'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=BASES)
    resume(parser.parse_args().case)

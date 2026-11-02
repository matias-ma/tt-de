"""Reproducible Hessian recipes transferred from the completed repo studies.

These are benchmark starting points, not universal optimal hyperparameters.
Every noisy recipe uses only observations and a prescribed noise kernel.
"""

DIMENSIONS = (4, 10, 20)


def recipes(family, dim, noisy):
    if family not in ('swirl', 'gl') or dim not in DIMENSIONS:
        raise ValueError('Supported benchmarks: swirl/gl in 4, 10, 20 dimensions')
    common = {'regularizer': 'hessian', 'strength': .001, 'rank': 32 if family == 'swirl' else 10,
                  'modes': 49 if family == 'swirl' else 25, 'seed': 0, 'steps': 4000,
                  'learning_rate': .001, 'batch_size': 256 if noisy else 1024,
                  'marginal_strength': 0., 'parent': None, 'mode_schedule': None, 'selection_start': 0,
                  'optimizer': 'adam'}
    if noisy and dim == 20:
        common['microbatches'] = 2
    if not noisy:
        if family == 'swirl':
            return {'hessian': common | {'optimizer': 'core_sweeps', 'sweeps': 20,
                                             'rank': 64 if dim == 20 else 32,
                                             'strength': {4: .01, 10: .003, 20: .001}[dim]}}
        return {'hessian': common | {'rank': 5}}
    schedule = [(0, 13), (1000, 25), (2000, 49)] if family == 'swirl' else [(0, 9), (1000, 17), (2000, 25)]
    result = {'curriculum': common | {'strength': 1e-5, 'mode_schedule': schedule, 'selection_start': 2000},
              'strong': common}
    if family == 'swirl':
        result['weak_continuation'] = common | {'parent': 'strong', 'strength': 1e-5,
                                                    'steps': 2000, 'learning_rate': .0003}
        result['marginal_continuation'] = common | {'parent': 'weak_continuation', 'strength': 1e-5,
            'steps': 2000, 'learning_rate': .0001, 'marginal_budget': .003}
    return result

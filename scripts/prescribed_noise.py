"""Read fixed noise metadata shared by the clipped-TT experiments."""

import json
from pathlib import Path

import numpy as np


def prescribed_noise(base):
    settings = json.loads((base/'settings.json').read_text())
    d = settings['dim']
    if settings.get('noise') == 'free':
        return np.zeros((1, d, d)), np.ones(1)
    if settings['family'] == 'swirl':
        from swirl_fits import NOISE_WEIGHTS, noise_covariances
        return noise_covariances(d), NOISE_WEIGHTS
    path = bool(settings.get('path'))
    name = 'path_dim4' if path else f'dim{d}'
    candidates = list(Path('artifacts/cryoem_refinement').glob(f'{name}_rank20/mixture_noise.json'))
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous noise metadata: {candidates}')
    info = json.loads(candidates[0].read_text())
    # Verify these parameters reproduce the frozen prescribed TT kernel.
    from refine_cryoem_gallery import diagonal_mixture_kernel
    from refresh_density_fits import read_model
    kernel = read_model(base/'kernel.npz')
    signal = read_model(base/'selected.npz')
    rebuilt = diagonal_mixture_kernel(np.array(info['variances_standardized']), np.array(info['weights']), signal.domain)
    for a, b in zip(kernel.cores, rebuilt.cores, strict=True):
        np.testing.assert_allclose(a.coeffs, b.coeffs, atol=1e-12, rtol=1e-10)
    return np.array([np.diag(v) for v in info['variances_standardized']]), np.array(info['weights'])

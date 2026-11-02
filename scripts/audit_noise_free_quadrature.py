"""Independent bounded-box Gaussian quadrature for frozen swirl cross integrals."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ftde.direct_training import load_tt, write_record
from ftde.gaussian_convolution import GaussianConvolution
from ftde.swirl import TUBE_STD, curve, path_quadrature

ROOT = Path('artifacts/noise_free_v2')


if __name__ == '__main__':
    checks = {}
    for case in ('swirl4', 'swirl10'):
        frozen = json.loads((ROOT/case/'frozen.json').read_text())
        name = frozen['selected']
        model = load_tt(ROOT/case/name/'fit.npz')
        dim = len(model.cores)
        result = json.loads((ROOT/case/name/'evaluation.json').read_text())
        forward = GaussianConvolution.from_tt(model, [TUBE_STD]*dim, quadrature_order=193)
        evaluate = jax.jit(lambda x, forward=forward, model=model: jnp.exp(forward.log_density(model, x)))
        t, weights = path_quadrature(1600)
        points = curve(t, dim)
        values = np.concatenate([np.asarray(evaluate(jnp.asarray(points[i:i+64]))) for i in range(0, len(t), 64)])
        cross = float(weights@values)
        discrepancy = abs(cross-result['integral_pq'])/result['integral_p_squared']
        if not np.isfinite(discrepancy) or discrepancy > 1e-7:
            raise FloatingPointError(f'{case}: independent quadratures disagree by {discrepancy}')
        checks[case] = {'fit': name, 'legendre_order': 193, 'path_order': 1600,
                        'bounded_box_cross_integral': cross, 'relative_to_p2_difference': discrepancy}
        print(case, checks[case], flush=True)
        jax.clear_caches()
    write_record(ROOT/'quadrature_audit.json', checks)

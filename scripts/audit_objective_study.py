"""Check publication completeness, frozen hashes and figure-only gallery."""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from objective_study import CASES, noise_parameters
from PIL import Image
from scipy.special import roots_legendre

from ftde.kernel_score import gaussian_mixture_characteristic


def check_noise_kernels():
    checks = {}
    for case, base in CASES.items():
        if case.endswith('free'):
            continue
        path = Path(base)/'kernel.npz'
        with np.load(path) as kernel:
            dim = len(kernel['domain'])
            frequencies = np.random.default_rng(12).normal(size=(64, dim))
            frequencies[0] = 0.
            state = np.ones((len(frequencies), 1), dtype=complex)
            for k, (a, b) in enumerate(kernel['domain']):
                c = kernel[f'signal_{k}']
                t, w = roots_legendre(2*c.shape[-1])
                x = (a+b)/2+(b-a)*t/2
                basis = np.polynomial.chebyshev.chebvander(t, c.shape[-1]-1)
                moments = (np.exp(1j*frequencies[:, k, None]*x)*w*(b-a)/2)@basis
                state = np.einsum('fi,ijn,fn->fj', state, c, moments)
        weights, covariance = noise_parameters(case, dim)
        exact = gaussian_mixture_characteristic(frequencies, weights, covariance)
        error = float(np.max(np.abs(state[:, 0]-exact)))
        assert error < 1e-5
        checks[case] = {'maximum_absolute_error_64_probes': error,
                        'mass': float(state[0, 0].real), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return checks


def check_swirl_cross_integrals(root, cases):
    """Independently check the bounded-support reference integral used by L2."""
    from evaluate_objective_study import model_path

    from ftde.direct_training import load_tt
    from ftde.gaussian_convolution import GaussianConvolution
    from ftde.swirl import TUBE_STD, curve, path_quadrature

    checks = {}
    for case in cases:
        if not case.startswith('swirl'):
            continue
        folder = root/case
        frozen = json.loads((folder/'frozen.json').read_text())
        evaluation = json.loads((folder/'evaluation.json').read_text())['models']
        for name in dict.fromkeys([frozen['selected_nll'], frozen['selected_kernel']]):
            model = load_tt(model_path(folder, name))
            forward = GaussianConvolution.from_tt(model, [TUBE_STD]*len(model.cores), quadrature_order=193)
            evaluate = jax.jit(lambda x, forward=forward, model=model: jnp.exp(forward.log_density(model, x)))
            t, weights = path_quadrature(1600)
            points = curve(t, len(model.cores))
            values = np.concatenate([np.asarray(evaluate(jnp.asarray(points[k:k+64]))) for k in range(0, len(t), 64)])
            cross = float(weights@values)
            result = evaluation[name]
            discrepancy = abs(cross-result['integral_pq'])/result['integral_p_squared']
            if not np.isfinite(discrepancy) or discrepancy >= 1e-7:
                raise FloatingPointError(f'{case}/{name}: independent cross-integral difference {discrepancy}')
            checks[f'{case}/{name}'] = {'bounded_box_cross_integral': cross,
                'relative_to_p2_difference': discrepancy, 'legendre_order': 193, 'path_order': 1600}
            print('INDEPENDENT CROSS INTEGRAL', case, name, discrepancy, flush=True)
            del model, forward, evaluate
            jax.clear_caches()
    return checks


def main():
    root = Path('artifacts/objective_v3')
    publication = json.loads((root/'publication.json').read_text())
    assert publication['fits'] == 61 and publication['all_figures_published']
    fits = list(root.glob('*/*/result.json'))
    assert len(fits) == 61
    zero_nll, zero_kernel = [], []
    for path in fits:
        result = json.loads(path.read_text())
        assert result['steps'] == 4000 and not result['uses_signal_pilot']
        assert result['history'][-1]['step'] == 4000
        assert all(np.isfinite(r['validation_nll']) and np.isfinite(r['validation_mmd']) for r in result['history'])
        if result['best_step'] == 0:
            zero_nll.append(str(path.parent))
        if result['kernel_step'] == 0:
            zero_kernel.append(str(path.parent))
    checked = 0
    evaluated = 0
    for case in publication['cases']:
        folder = root/case
        frozen = json.loads((folder/'frozen.json').read_text())
        assert not frozen['uses_truth']
        for path, digest in frozen['files'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            checked += 1
        protocol = json.loads((folder/'protocol.json').read_text())
        source = Path(protocol['source'])/'data.npz'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol['data_sha256']
        assert not protocol['uses_signal_pilot'] and not protocol['uses_clean_truth']
        evaluation = json.loads((folder/'evaluation.json').read_text())
        assert evaluation['selection_frozen_before_truth']
        assert evaluation['selected_nll'] == frozen['selected_nll']
        assert evaluation['selected_kernel'] == frozen['selected_kernel']
        if case.endswith('free'):
            generalization = json.loads((folder/'generalization.json').read_text())
            assert not generalization['uses_truth']
            for name, row in generalization['models'].items():
                assert np.isfinite(row['training_nll'])
                saved = json.loads((folder/name/'result.json').read_text())
                assert abs(row['validation_nll']-saved['best_nll']) < 1e-9
        for result in evaluation['models'].values():
            assert np.isfinite(result['relative_l2']) and result['relative_l2'] >= 0
            assert result['fourier_refinement_max_error'] < 1e-9
            assert result.get('quadrature_relative_change', result.get('path_quadrature_relative_change')) < 1e-6
            evaluated += 1
        for result in json.loads((folder/'plot_checks.json').read_text()).values():
            assert max(result['convolution_refinement_errors']) < .01
            assert result['kernel_convolution_refinement_error'] < .01
        family = 'swirl' if case.startswith('swirl') else 'ginzburg_landau'
        assert (Path('figs')/family/f'objectives_{case}.png').exists()
        if not case.endswith('free'):
            for pair in ['01', '23']:
                assert (Path('figs')/family/f'noisy_{case}_marg_{pair}.png').exists()
    gallery = [p for p in Path('figs').rglob('*') if p.is_file()]
    assert all(p.suffix == '.png' or p == Path('figs/README.md') for p in gallery)
    for path in gallery:
        if path.suffix == '.png':
            with Image.open(path) as image:
                image.verify()
    assert Path('figs/convergence/capacity.png').exists()
    report = {'fits': len(fits), 'frozen_hashes_checked': checked, 'evaluated_checkpoints': evaluated,
              'zero_update_nll_choices': zero_nll, 'zero_update_kernel_choices': zero_kernel,
              'noise_kernel_checks': check_noise_kernels(),
              'swirl_cross_integral_checks': check_swirl_cross_integrals(root, publication['cases']),
              'gallery_figures': sum(p.suffix == '.png' for p in gallery),
              'artifact_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}
    (root/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

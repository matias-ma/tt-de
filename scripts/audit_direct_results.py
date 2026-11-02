"""Final integrity and provenance audit of the direct TT optimization study."""

import datetime
import hashlib
import importlib.metadata
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

root = Path('artifacts/direct_optimization')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


previous = json.loads((root/'swirl4/previous_figures.json').read_text())
for path, expected in previous.items():
    assert digest(path) == expected, f'Previous figure changed: {path}'
assert not [p for p in Path('figs').rglob('*') if p.is_file() and p.suffix not in ('.png', '.md')]
frozen_count, trajectories = 0, {}
for case in ('swirl4', 'swirl10', 'swirl_free4', 'swirl_free10'):
    folder = root/case
    problem = json.loads((folder/'problem.json').read_text())
    base = Path(problem['base'])
    assert digest(base/'data.npz') == problem['data_sha256']
    if 'kernel_sha256' in problem:
        assert digest(base/'kernel.npz') == problem['kernel_sha256']
    frozen = json.loads((folder/'frozen_evaluation.json').read_text())
    assert frozen['truth_used_for_selection'] is False
    for path, expected in frozen['files'].items():
        assert digest(path) == expected, f'Frozen file changed: {path}'
        frozen_count += 1
    for path in folder.glob('*/result.json'):
        result = json.loads(path.read_text())
        summary = {'selected_update': result['best_step'], 'recorded_checkpoints': len(result['history'])}
        if path.parent.name.startswith('signed'):
            h = result['history']
            maximum_increase = max(b['objective']-a['objective'] for a, b in pairwise(h))
            maximum_mass_error = max(abs(v['mass']-1) for v in h)
            assert maximum_increase <= 1e-7 and maximum_mass_error <= 1e-7
            summary.update(maximum_objective_increase=maximum_increase, maximum_mass_error=maximum_mass_error,
                           maximum_cg_residual=max(v['cg_relative_residual'] for v in h))
        trajectories[str(path.parent)] = summary
    filename = {'swirl4': 'evaluation_q97.json', 'swirl10': 'evaluation.json',
                'swirl_free4': 'evaluation_free.json', 'swirl_free10': 'evaluation.json'}[case]
    evaluation = json.loads((folder/filename).read_text())
    for value in evaluation['models'].values():
        assert digest(value['fit_path']) == value['fit_sha256']
        assert value['relative_l2'] >= 0
        if case in ('swirl10', 'swirl_free10'):
            assert value['max_sample_mean_z'] <= 6 and value['quadrature_log_discrepancy'] <= 1e-7

cryo_checks = {}
for case in ('dim4', 'path_dim4', 'dim10', 'dim20'):
    folder = root/f'cryoem_{case}'
    config = json.loads((folder/'config.json').read_text())
    base = Path(config['base'])
    for filename, key in [('data.npz', 'data_sha256'), ('kernel.npz', 'kernel_sha256'),
                          ('selected.npz', 'historical_sha256')]:
        assert digest(base/filename) == config[key]
    noise_path = Path('artifacts/swirl_improvement')/f'cryoem_{case}'/'noise.npz'
    assert digest(noise_path) == config['noise_sha256']
    assert config['uses_signal_pilot_or_oracle_or_labels'] is False
    frozen = json.loads((folder/'frozen.json').read_text())
    for path, expected in frozen['files'].items():
        assert digest(path) == expected
        frozen_count += 1
    noise = json.loads((folder/'noise_check.json').read_text())
    assert noise['max_relative_error'] < 1e-6
    selection = json.loads((folder/'selection.json').read_text())
    assert selection['uses_clean_truth'] is False
    for name, nll in zip(selection['names'][1:], selection['validation_nll'][1:], strict=True):
        result = json.loads((folder/name/'result.json').read_text())
        logs = np.load(folder/name/'validation.npy')
        assert logs.shape == (config['validation_count'],) and np.isfinite(logs).all()
        np.testing.assert_allclose(-logs.mean(), nll, rtol=1e-10, atol=1e-10)
        assert np.isfinite(selection['roughness']).all()
        if name.startswith('constant'):
            assert result['best_step'] > 0
            assert result['best_nll'] < result['history'][0]['nll']
        else:
            source = json.loads((folder/name/'config.json').read_text())
            assert digest(folder/source['source']/'fit.npz') == source['source_sha256']
            assert digest(folder/name/'init.npz') == source['source_sha256']
        trajectories[str(folder/name)] = {'selected_update': result['best_step'],
                                         'recorded_checkpoints': len(result['history'])}
    cryo_checks[case] = {'selection': selection, 'noise_check': noise,
                         'file_sha256': {str(p): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}}

training = json.loads((root/'frozen_training_objectives.json').read_text())
for name, result in training.items():
    assert digest(root/name/'fit.npz') == result['fit_sha256']

smooth = root/'smoothing_free4'
protocol = json.loads((smooth/'protocol.json').read_text())
assert digest(protocol['source']) == protocol['source_sha256']
frozen = json.loads((smooth/'frozen.json').read_text())
assert frozen['truth_used_for_selection'] is False
for path, expected in frozen['files'].items():
    assert digest(path) == expected
    frozen_count += 1
selection = json.loads((smooth/'selection.json').read_text())
evaluation = json.loads((smooth/'evaluation.json').read_text())
assert evaluation['selected'] == selection['selected'] == frozen['selected']
baseline = np.load(smooth/'heat_0/validation.npy')
smoothing_checks = {}
for name, r in evaluation['models'].items():
    assert digest(smooth/name/'fit.npz') == r['fit_sha256']
    assert abs(r['positive_mass']-r['negative_mass']-1) < 1e-7
    original = selection['models'][name]
    values = np.load(smooth/name/'validation.npy')
    risk = r['norm_q_squared']-2*float(values.mean())*original['positive_mass']/r['positive_mass']
    assert abs(risk-original['validation_risk']) < 1e-4
    smoothing_checks[name] = {'high_order_validation_risk': risk,
                              'paired_se_against_zero': 2*float(np.std(values-baseline, ddof=1))/np.sqrt(len(baseline))}
assert min(smoothing_checks, key=lambda n: smoothing_checks[n]['high_order_validation_risk']) == selection['selected']

figures = {str(p): digest(p) for p in Path('figs').rglob('direct_optimization_*.png')}
assert len(figures) >= 19, 'Missing expected figures'
tests = (root/'logs/tests_complete.log').read_text()
assert '18 passed' in tests
assert '3 passed' in (root/'logs/tests_smoothing.log').read_text()
assert '1 failed' in (root/'logs/padding_before.log').read_text()
assert '1 passed' in (root/'logs/padding_after.log').read_text()
style = (root/'logs/style.log').read_text()
assert 'All checks passed!' in style
source = list(Path('scripts').glob('*direct*.py'))+[
    Path('src/ftde/simple_initialization.py'), Path('src/ftde/signed_tt_least_squares.py'),
    Path('src/ftde/full_batch_optimization.py'), Path('src/ftde/__init__.py'),
    Path('src/ftde/linear_tt_convolution.py'), Path('src/ftde/tt_convolution.py'),
    Path('src/ftde/clipped_tt_density.py'), Path('src/ftde/sobolev.py'),
    Path('src/ftde/tt_sampling.py'), Path('tests/test_direct_optimization.py'),
    Path('src/ftde/tt_smoothing.py'), Path('tests/test_tt_smoothing.py'),
    Path('README.md'), Path('docs/direct_optimization.md'), Path('figs/README.md')]
result = {'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'allocations': [14507300, 14518032, 14523863, 14566334], 'previous_png_count': len(previous), 'new_figure_sha256': figures,
          'frozen_file_checks': frozen_count, 'training_trajectories': trajectories,
          'source_sha256': {str(p): digest(p) for p in sorted(set(source))},
          'tests_sha256': digest(root/'logs/tests_complete.log'), 'style_sha256': digest(root/'logs/style.log'),
          'protocol_sha256': digest(root/'protocol.json'),
          'cryo_transfer_protocol_sha256': digest(root/'transfer_protocol.json'),
          'cryo_checks': cryo_checks,
          'frozen_training_objectives': training,
          'smoothing_checks': smoothing_checks,
          'smoothing_tests_sha256': digest(root/'logs/tests_smoothing.log'),
          'shifted_domain_padding_regression': {
              'before_sha256': digest(root/'logs/padding_before.log'),
              'after_sha256': digest(root/'logs/padding_after.log'),
              'result': 'Same shifted-domain regression fails before the fix and passes after it'},
          'numerical_checks': json.loads((root/'numerical_checks.json').read_text()),
          'clipping_quadrature': json.loads((root/'swirl4/clipping_quadrature.json').read_text()),
          'noise_free_clipping_quadrature': json.loads((root/'swirl_free4/clipping_quadrature.json').read_text()),
          'versions': {name: importlib.metadata.version(name) for name in ('jax', 'numpy', 'scipy', 'matplotlib', 'chebttge')}}
(root/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'previous_figures_preserved': len(previous), 'new_figures': len(figures),
                  'frozen_file_checks': frozen_count, 'successful_runs': len(trajectories)}, indent=2))

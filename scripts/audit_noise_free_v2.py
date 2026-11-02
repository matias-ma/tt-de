"""Audit the frozen current study and publish a figures-only gallery."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path('artifacts/noise_free_v2')
FIGS = Path('figs')


def main(archive_old=False):
    expected = [f'{kind}/noise_free_dim{dim}_{suffix}.png' for kind, dims in
                [('swirl', [4, 10]), ('ginzburg_landau', [6])] for dim in dims
                for suffix in ['marg_01', 'marg_23', 'training']]
    expected += [f'cryoem/{case}.png' for case in ['dim4', 'path_dim4', 'dim10', 'dim20', 'training']]
    expected += ['convergence/noise_free_optimization.png', 'convergence/initialization.png', 'convergence/objectives.png']
    for name in expected:
        path = FIGS/name
        assert path.exists() and path.stat().st_size > 10000, path
        with Image.open(path) as image:
            image.verify()
    count, hashes = 0, {}
    for case in ['swirl4', 'swirl10', 'gl6', 'cryoem_dim4', 'cryoem_path_dim4', 'cryoem_dim10', 'cryoem_dim20']:
        frozen = json.loads((ROOT/case/'frozen.json').read_text())
        for path, sha in (frozen['files'] | frozen.get('result_files', {})).items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
            count += 1
        for path in (ROOT/case).glob('*/result.json'):
            result = json.loads(path.read_text())
            assert result['uses_signal_pilot'] is False
            assert result['best_step'] in [h['step'] for h in result['history']]
        if not case.startswith('cryoem'):
            report = json.loads((ROOT/case/'evaluation.json').read_text())
            assert report['selected'] == frozen['selected']
            assert report['selection_frozen_before_truth']
            assert all(r['relative_l2'] >= 0 for r in report['models'].values())
    quadrature = json.loads((ROOT/'quadrature_audit.json').read_text())
    assert all(c['relative_to_p2_difference'] < 1e-7 for c in quadrature.values())
    for path in (ROOT/'initialization_check').glob('*/protocol.json'):
        protocol = json.loads(path.read_text())
        assert not protocol['uses_clean_truth'] and not protocol['uses_signal_pilot'] and not protocol['reselects_main_fit']
    if archive_old:
        archive = Path('artifacts/gallery_archive/before_noise_free_v2')
        for path in FIGS.rglob('*.png'):
            relative = path.relative_to(FIGS)
            if str(relative) not in expected:
                target = archive/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                assert not target.exists(), target
                path.rename(target)
    assert sorted(str(p.relative_to(FIGS)) for p in FIGS.rglob('*.png')) == sorted(expected)
    assert all(p.suffix.lower() in ('.png', '.md') for p in FIGS.rglob('*') if p.is_file())
    for name in expected:
        path = FIGS/name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    stored = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    report = {'figures': hashes, 'frozen_file_checks': count, 'study_storage_bytes': stored,
              'fits': len(list(ROOT.glob('*/*/result.json'))), 'figs_contains_no_numerical_sidecars': True}
    (ROOT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-old', action='store_true')
    main(parser.parse_args().archive_old)

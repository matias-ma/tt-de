"""Extract supplied noise metadata, aligned with the existing gallery splits.

The path cache predates the covariance projection correction. Retain its
observation mask for a controlled comparison, but invert the full precision
before marginalizing the noise. Never read simulation labels.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from cryoem_cache import marginal_noise_covariances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/cryoem_refinement'))
    root = parser.parse_args().output
    root.mkdir(parents=True, exist_ok=True)
    embedding = joblib.load('cryoem_data/cryoem_test.joblib')
    records = {}
    for name, dim, load_dim in [('dim4', 4, 4), ('dim10', 10, 10),
                                 ('dim20', 20, 20), ('path_dim4', 4, 10)]:
        precision = np.asarray(embedding['cov_zs'][f'{load_dim}_noreg'])
        covariance = marginal_noise_covariances(precision, dim)
        old_covariance = np.linalg.inv(precision[:, :dim, :dim])
        filename = f'path_noise1_dim{dim}.npz' if name.startswith('path') else f'dim{dim}.npz'
        cached = np.load(Path('cryoem_data/cache')/filename)
        source_points = np.asarray(embedding['zs'][f'{load_dim}_noreg'])[:, :dim]
        for mask_name, mask_covariance in [('historical', old_covariance), ('corrected', covariance)]:
            trace = np.trace(mask_covariance, axis1=1, axis2=2)
            indices = np.flatnonzero(trace < np.percentile(trace, 99))
            if np.array_equal(cached['points'], source_points[indices]):
                break
        else:
            raise ValueError(f'{filename}: cached observations do not match either known source mask')
        covariance = covariance[indices]
        variance = np.diagonal(covariance, axis1=1, axis2=2)
        correlation = covariance/np.sqrt(variance[:, :, None]*variance[:, None, :])
        off = ~np.eye(dim, dtype=bool)
        records[name] = {'count': len(indices),
            'variance_cv': (variance.std(0)/variance.mean(0)).tolist(),
            'abs_correlation_quantiles': np.quantile(np.abs(correlation[:, off]), [.5, .9, .99]).tolist(),
            'correct_to_cached_variance_ratio': (variance.mean(0)/np.diag(cached['cov_mean'])).tolist(),
            'mask': f'Retained {mask_name} cached rows; corrected noise marginalization'}
        extra = {'covariances': covariance} if dim == 4 else {}
        np.savez(root/f'noise_{name}.npz', variances=variance, indices=indices,
                 points=source_points[indices], **extra)
        print(name, records[name], flush=True)
    (root/'noise_audit.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()

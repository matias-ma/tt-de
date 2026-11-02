"""Post-fit fidelity of noise approximations against held-out noise metadata."""

import argparse
import json
from pathlib import Path

import numpy as np
from refresh_density_fits import read_model

from ftde.cheb_gram import chebyshev_moment_vector
from ftde.noise_marginals import noise_marginals_from_tt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', nargs='+', type=Path)
    args = parser.parse_args()
    for experiment in args.experiments:
        config = json.loads((experiment/'experiment.json').read_text())
        base = Path(config['base'])
        data, split = np.load(base/'data.npz'), np.load(base/'split.npz')
        supplied = np.load(config['noise_metadata'])['variances'][split['valid_idx']]/data['scale']**2
        mixture = json.loads((experiment/'mixture_noise.json').read_text())
        represented_mixture = read_model(experiment/'mixture_kernel.npz')
        mass = np.ones(1)
        for core in represented_mixture.cores:
            moments = np.asarray(chebyshev_moment_vector(core.coeffs.shape[-1], core.domain))
            mass = mass@np.einsum('abn,n->ab', np.asarray(core.coeffs), moments)
        if abs(float(mass[0])-1.) > 1e-10:
            raise FloatingPointError('Mixture kernel is not normalized')
        component = np.asarray(mixture['variances_standardized'])
        weights = np.asarray(mixture['weights'])
        kernel = read_model(experiment/'corrected_kernel.npz' if config.get('correct_projection') else base/'kernel.npz')
        marginal_std = np.array([m.moments()[1] for m in noise_marginals_from_tt(kernel)])
        frequencies = np.linspace(0., 8., 129)
        gaussian_errors, mixture_errors = [], []
        for k, std in enumerate(marginal_std):
            omega = frequencies/std
            target = np.exp(-.5*supplied[:, k, None]*omega**2).mean(0)
            gaussian = np.exp(-.5*frequencies**2)
            mixed = weights@np.exp(-.5*component[:, k, None]*omega**2)
            gaussian_errors.append(float(np.linalg.norm(gaussian-target)))
            mixture_errors.append(float(np.linalg.norm(mixed-target)))
        record = {'method': 'L2 errors of marginal characteristic functions on 129 frequencies, 0 to 8 noise-standardized units',
                  'rows': 'held-out noise metadata only; no signal values or labels',
                  'gaussian_errors': gaussian_errors, 'mixture_errors': mixture_errors,
                  'mixture_mass': float(mass[0]),
                  'aggregate_error_ratio': float(np.linalg.norm(mixture_errors)/np.linalg.norm(gaussian_errors))}
        (experiment/'noise_approximation_check.json').write_text(json.dumps(record, indent=2)+'\n')
        print(experiment.name, 'mixture/Gaussian noise approximation error', record['aggregate_error_ratio'])


if __name__ == '__main__':
    main()

"""Plot the noisy-data marginal reference and its information spectrum."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import roots_legendre

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=Path('artifacts/noisy_reference'))
    parser.add_argument('--spectra', type=Path, default=Path('artifacts/noisy_fisher_fits/fisher_spectra.npz'))
    parser.add_argument('--output', type=Path, default=Path('figs/ginzburg_landau/noisy_information/reference_and_information.png'))
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for filename, label, style in [('degree2_shared.npz', 'Gaussian-shaped pilot', '--'),
                                   ('selected.npz', 'Selected noisy-data pilot', '-')]:
        saved = np.load(args.reference_dir/filename)
        a, b = saved['domain'][0]
        t = np.linspace(-1., 1., 800)
        nodes, weights = roots_legendre(320)
        c = saved['log_coeffs'][0]
        log_values = np.polynomial.chebyshev.chebval(t, c)
        shift = log_values.max()
        z = np.sum(weights*np.exp(np.polynomial.chebyshev.chebval(nodes, c)-shift))*(b-a)/2
        axes[0].plot(a+(t+1)*(b-a)/2, np.exp(log_values-shift)/z, style, label=label)
    axes[0].set(xlim=(-2., 2.), xlabel='Signal coordinate', ylabel='Reference marginal density',
                title='Sharp structure learned through the noise')
    axes[0].legend(fontsize=8)
    spectra = np.load(args.spectra)
    for k, name in enumerate(spectra.files):
        values = np.sort(spectra[name])[::-1]
        axes[1].semilogy(np.arange(1, len(values)+1), np.maximum(values, 1e-16),
                         color='C0', alpha=.4, label='Marginal information modes' if k == 0 else None)
    axes[1].axhline(.01, color='C1', linestyle='--', label='Tested visibility floor')
    axes[1].set(xlim=(1, 20), ylim=(1e-12, 1.), xlabel='Information mode (not Chebyshev degree)',
                ylabel='Retained fraction of noise-free information', title='Convolution hides amplitude directions')
    axes[1].legend(fontsize=8)
    fig.suptitle('10D GL, paired noise ×1.5 · reference and spectrum use noisy data only')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

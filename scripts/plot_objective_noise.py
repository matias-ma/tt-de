"""Known-noise Fourier attenuation; no fitted or true signal density enters."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from objective_study import noise_parameters

from ftde.kernel_score import gaussian_mixture_characteristic


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    frequency = np.linspace(0, 15, 401)
    cases = [('gl6_paired', 6, 'GL: correlated Gaussian'), ('gl6_mixture', 6, 'GL: narrow/broad mixture'),
             ('swirl4_mixture', 4, 'Swirl: correlated mixture')]
    for ax, (case, dim, title) in zip(axes, cases, strict=True):
        weights, cov = noise_parameters(case, dim)
        directions = np.zeros((3, dim))
        directions[0, 0] = 1.
        directions[1, :2] = 1/np.sqrt(2)
        directions[2, :2] = [1/np.sqrt(2), -1/np.sqrt(2)]
        for direction, name, linestyle in zip(directions, ['Coordinate x0', '(x0+x1)/√2', '(x0−x1)/√2'], ['-', '--', ':'], strict=True):
            value = gaussian_mixture_characteristic(frequency[:, None]*direction, weights, cov)
            ax.semilogy(frequency, value, label=name, linestyle=linestyle)
        sigma = np.sqrt(np.einsum('c,c->', weights, cov[:, 0, 0]))
        ax.axhline(1/np.sqrt(100000), color='gray', linestyle='-.', label='iid reference N⁻¹ᐟ²')
        ax.set(title=f'{title}\nMarginal noise σ={sigma:.3f}', xlabel='Angular frequency in physical coordinates',
               ylabel='Known-noise characteristic function |φu|', ylim=(1e-9, 1.05))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Equal noise variance can hide very different amounts of detail', y=.985, fontsize=15).set_in_layout(False)
    fig.get_layout_engine().set(rect=(0, .15, 1, .73))
    fig.text(.02, .015, 'Convolution multiplies signal Fourier coefficients by φu. In the population kernel discrepancy, squared signal errors are weighted by |φu|².\n'
             'The dashed reference is only an iid sampling scale for N=100,000, not a detectability cutoff; GL samples also have within-chain dependence.\n'
             'The narrow component of the broad-tailed GL mixture preserves more high-frequency information than the matched-variance Gaussian.', fontsize=9)
    path = Path('figs/convergence/noise_information.png')
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()

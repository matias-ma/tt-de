"""Compare fitted densities before/after convolution, without any latent truth."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from noise_diagnosis import ROOT
from objective_study import noise_parameters
from scipy.special import roots_legendre

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.direct_training import load_tt, write_record
from ftde.marginals import squared_pair_coefficients


def frequency_geometry(box, sizes, weights, covariances, pair, *, step=.2, order=256):
    """Parseval quadrature for the squared norm of an analytically blurred pair."""
    frequency = np.arange(-round(40/step), round(40/step)+1)*step
    t, w = roots_legendre(order)
    transforms = []
    for interval, size in zip(box, sizes[::-1], strict=True):
        x = interval[0]+np.ptp(interval)*(t+1)/2
        v = np.polynomial.chebyshev.chebvander(t, size-1)
        transforms.append((np.exp(1j*frequency[:, None]*x)*(w*np.ptp(interval)/2)[None])@v)
    sx, sy = np.meshgrid(frequency, frequency)
    multiplier = np.zeros_like(sx)
    for weight, covariance in zip(weights, covariances, strict=True):
        c = covariance[np.ix_(pair, pair)]
        multiplier += weight*np.exp(-.5*(c[0, 0]*sx**2+2*c[0, 1]*sx*sy+c[1, 1]*sy**2))
    quadrature = np.full(len(frequency), step)
    quadrature[[0, -1]] *= .5
    measure = np.outer(quadrature, quadrature)*multiplier**2/(2*np.pi)**2
    return transforms, measure


def compare(case):
    capacity = 'r32_n49' if case.startswith('swirl') else 'r10_n25'
    original = Path('artifacts/objective_v3')/case/f'nll_{capacity}_l0.001_w100'
    baseline = load_tt(original/'fit.npz')
    logs = np.load(original/'validation.npz')['logs']
    names = [p.parent.name for p in sorted((ROOT/case).glob('*/result.json'))]
    weights, covariance = noise_parameters(case, len(baseline.cores))
    records = {}
    for pair in [(0, 1), (2, 3)]:
        reference, box = squared_pair_coefficients(baseline, *pair)
        reference, box = np.asarray(reference), np.asarray(box)
        grams = [np.asarray(chebyshev_gram_matrix(n, b)) for n, b in zip(reference.shape[::-1], box, strict=True)]
        def norm(c, grams=grams):
            return float(np.sum(c*(grams[1]@c@grams[0])))
        geometry = [frequency_geometry(box, reference.shape, weights, covariance, pair, step=s, order=q)
                    for s, q in [(.2, 256), (.1, 512)]]
        def observed_norm(c, geometry):
            transform, measure = geometry
            characteristic = transform[1]@c@transform[0].T
            return float(np.sum(np.abs(characteristic)**2*measure))
        denominator = norm(reference)
        observed_denominator = [observed_norm(reference, g) for g in geometry]
        answer = {}
        for name in names:
            model = load_tt(ROOT/case/name/'fit.npz')
            coefficient, candidate_box = squared_pair_coefficients(model, *pair)
            np.testing.assert_array_equal(box, candidate_box)
            delta = np.asarray(coefficient)-reference
            convolved = [observed_norm(delta, g) for g in geometry]
            change = max(abs(convolved[0]-convolved[1]), abs(observed_denominator[0]-observed_denominator[1]))/observed_denominator[1]
            if change > 1e-9:
                raise FloatingPointError('Parseval quadrature needs refinement')
            difference = logs-np.load(ROOT/case/name/'validation.npy')
            answer[name] = {'latent_pair_relative_change': float(np.sqrt(max(norm(delta), 0.)/denominator)),
                'observed_pair_relative_change': float(np.sqrt(convolved[1]/observed_denominator[1])),
                'parseval_refinement': float(change), 'validation_nll_minus_previous': float(difference.mean())}
        records[str(pair)] = answer
    write_record(ROOT/case/'stability.json', {'pairs': records, 'uses_truth': False,
        'latent_metric': 'exact full-plane pair L2 relative to previous fitted pair',
        'observed_metric': 'full-plane pair L2 via Parseval under analytic known noise; refined in frequency spacing and polynomial quadrature'})
    return records


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for row, case in enumerate(['swirl4_mixture', 'gl6_mixture']):
        records = compare(case)
        for col, (pair, results) in enumerate(records.items()):
            names = list(results)
            x = np.arange(len(names))
            axes[row, col].bar(x-.18, [results[n]['latent_pair_relative_change'] for n in names], width=.36, label='Latent fitted density')
            axes[row, col].bar(x+.18, [results[n]['observed_pair_relative_change'] for n in names], width=.36, label='After convolution')
            axes[row, col].set_xticks(x, [n.replace('_', ' ') for n in names], rotation=20, ha='right')
            axes[row, col].set(title=f'{case.replace("_", " ")} · pair {pair}', ylabel='Relative L2 difference from previous fit')
            axes[row, col].set_ylim(0, max(max(r['latent_pair_relative_change'], r['observed_pair_relative_change']) for r in results.values())*1.4)
            axes[row, col].grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Variation between fitted densities before and after convolution\nThese comparisons use no true latent density.', fontsize=14)
    fig.text(.02, .015, 'Same pairs and full-plane L2 metric before/after the known noise. Latent integrals are exact polynomial contractions;\n'
             'observed integrals use independently refined Parseval quadrature. Each candidate is its observed-NLL-selected checkpoint.\n'
             'Swirl: r=32, modes=49; GL: r=10, modes=25. Ntrain=100,000, Hessian λ=0.001. One squared Chebyshev TT per fit.\n'
             'Zero bars mean validation retained the earlier noisy fit. A small blurred change need not imply a small latent change.', fontsize=9, va='bottom')
    fig.get_layout_engine().set(rect=(0, .12, 1, .84))
    Path('figs/convergence').mkdir(parents=True, exist_ok=True)
    fig.savefig('figs/convergence/noisy_stability.png', dpi=160)
    plt.close(fig)

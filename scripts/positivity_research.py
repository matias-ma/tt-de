"""Algebra checks and an explanatory figure, not a fitted-density benchmark."""
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit, roots_legendre


def check_algebra():
    rng = np.random.default_rng(7391)
    edges = np.linspace(-2, 2, 6)
    degree = 3
    knots = np.r_[np.repeat(edges[0], degree), edges, np.repeat(edges[-1], degree)]
    modes = len(knots)-degree-1
    basis = BSpline(knots, np.eye(modes), degree, extrapolate=False)
    masses = basis.integrate(edges[0], edges[-1])
    t, w = roots_legendre(24)
    nodes = np.concatenate([a+(b-a)*(t+1)/2 for a, b in pairwise(edges)])
    weights = np.concatenate([w*(b-a)/2 for a, b in pairwise(edges)])
    values = basis(nodes)/masses
    np.testing.assert_allclose(weights@values, 1., atol=2e-14)
    ranks = [1, 3, 3, 1]
    cores = []
    for left, right in pairwise(ranks):
        c = rng.uniform(.1, 1., (left, right, modes))
        cores.append(c/c.sum(axis=(1, 2), keepdims=True))
    normalizer = np.ones(1)
    for c in cores:
        normalizer = normalizer@c.sum(axis=2)
    np.testing.assert_allclose(normalizer, 1., atol=2e-14)
    points = rng.uniform(-2, 2, (512, 3))
    state = np.ones((len(points), 1))
    for k, c in enumerate(cores):
        state = np.einsum('na,nab->nb', state, np.einsum('abm,nm->nab', c, basis(points[:, k])/masses))
    assert np.min(state) > 0

    # Arbitrary observation locations, not samples from a known target.
    # Integrating the known noise is part of the algebra check.
    observations = rng.uniform(-2.5, 2.5, (40, 3))
    features = []
    for k, sigma in enumerate([.3, .45, .25]):
        kernel = np.exp(-.5*((observations[:, k, None]-nodes[None])/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        features.append((kernel*weights)@values)
    factors = [np.einsum('abm,nm->nab', c, f) for c, f in zip(cores, features, strict=True)]
    convolved = np.einsum('na,nab,nb->n', factors[0][:, 0], factors[1], factors[2][:, :, 0])
    design = np.einsum('na,nb,nm->nabm', factors[0][:, 0], factors[2][:, :, 0], features[1]).reshape(len(observations), -1)
    c = cores[1].ravel()
    np.testing.assert_allclose(design@c, convolved, atol=2e-15)
    mu = .001
    hessian = (design.T/(design@c)**2)@design/len(observations)+np.diag(mu/c**2)
    eigenvalue = float(np.linalg.eigvalsh(hessian)[0])
    assert eigenvalue > 0
    other = rng.uniform(.1, 1., cores[1].shape)
    other = (other/other.sum(axis=(1, 2), keepdims=True)).ravel()
    def loss(v):
        return -np.log(design@v).mean()-mu*np.log(v).sum()
    gaps = [loss(a*c+(1-a)*other)-(a*loss(c)+(1-a)*loss(other)) for a in np.linspace(.1, .9, 9)]
    assert max(gaps) < 1e-12
    return {'integral': float(normalizer[0]), 'minimum_density_on_512_probes': float(state.min()),
            'convolution_single_core_linearity_error': float(np.max(np.abs(design@c-convolved))),
            'minimum_core_hessian_eigenvalue': eigenvalue, 'maximum_jensen_gap': float(max(gaps)),
            'scope': 'algebra validation only; no fitted density or benchmark improvement'}


def plot():
    x = np.linspace(-1.5, 1.5, 501)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    for values, color, label in [(x*x, '#6b7280', 'Square'), (np.maximum(x, 0), '#d95f02', 'Hard clipping'),
                                 (np.logaddexp(0, x), '#1976b9', 'Softplus')]:
        axes[0].plot(x, values, color=color, label=label, lw=2)
    axes[0].set(title='Ways to map a real value to a positive one', xlabel='Input t', ylabel='Unnormalized output')
    axes[0].legend()
    axes[1].plot(x, 2*x, color='#6b7280', label='Square: 2t', lw=2)
    axes[1].plot(x[x < 0], np.zeros_like(x[x < 0]), color='#d95f02', lw=2)
    axes[1].plot(x[x > 0], np.ones_like(x[x > 0]), color='#d95f02', label='Clipping: 0 or 1', lw=2)
    axes[1].plot(x, expit(x), color='#1976b9', label='Softplus: sigmoid(t)', lw=2)
    axes[1].set(title='Derivative of the map (not the full loss)', xlabel='Input t', ylabel='Map derivative')
    axes[1].legend()
    c = np.geomspace(.001, 1., 300)
    axes[2].loglog(c, 1/c, color='#16856c', lw=2)
    axes[2].set(title='Barrier on an already positive coefficient', xlabel='Coefficient c > 0', ylabel='Magnitude of barrier derivative: |d(-log c)/dc|')
    axes[2].text(.03, .94, 'Constrained updates stay above zero.\nThe barrier strengthens near the boundary.', transform=axes[2].transAxes,
                 va='top', fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Positivity alternatives: explanatory mathematics, not density-fit results', fontweight='bold')
    fig.supxlabel('Global softplus still breaks TT integration. Positive univariate cores preserve separability; normalized spline cores also allow convex single-core updates.\n'
                  'Softplus can saturate for very negative inputs. Positive-core methods change the model class and may require higher ranks.', fontsize=9)
    fig.savefig('figs/swirl/positivity_options.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    report = check_algebra()
    destination = Path('artifacts/positivity_research')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'algebra_checks.txt').write_text('\n'.join(f'{k}: {v}' for k, v in report.items())+'\n')
    plot()
    print(report)

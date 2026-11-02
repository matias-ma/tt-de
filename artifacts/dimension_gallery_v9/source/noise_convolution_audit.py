"""Independent analytic-noise quadrature audit of the production TT forward map.

The reference integrates the prescribed Gaussian blocks directly. Its SVD
separates quadrature weights for contraction; it is not a density estimator
or an initializer. All probes are held-out noisy observations.
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from noise_diagnosis import ROOT
from objective_study import CASES, noise_parameters
from scipy.special import roots_legendre

from ftde.direct_training import load_tt, write_record
from ftde.gaussian_convolution import _rescale, _transfer
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import rebuild_tt


def analytic_pair_reference(model, observations, weights, covariances, order):
    """Return a differentiable fixed-probe reference, with analytic noise."""
    t, w = roots_legendre(order)
    nodes, basis, grams = [], [], []
    for c, box in zip(model.cores, np.asarray(model.domain), strict=True):
        x = box[0]+np.ptp(box)*(t+1)/2
        v = np.polynomial.chebyshev.chebvander(t, c.coeffs.shape[-1]-1)
        nodes.append(x)
        basis.append(v)
        grams.append(jnp.asarray(v.T@(w[:, None]*v)*np.ptp(box)/2))
    components = []
    for cov in covariances:
        pairs = []
        for k in range(0, len(nodes), 2):
            sub = cov[k:k+2, k:k+2]
            inv = np.linalg.inv(sub)
            parts = []
            for y in observations:
                dx, dy = y[k]-nodes[k][:, None], y[k+1]-nodes[k+1][None, :]
                kernel = np.exp(-.5*(inv[0, 0]*dx**2+2*inv[0, 1]*dx*dy+inv[1, 1]*dy**2))
                kernel *= w[:, None]*w[None, :]*np.prod(np.ptp(np.asarray(model.domain)[k:k+2], axis=1))/4
                kernel /= 2*np.pi*np.sqrt(np.linalg.det(sub))
                u, s, vh = np.linalg.svd(kernel, full_matrices=False)
                rank = max(1, int(np.sum(s > s[0]*1e-14)))
                left = np.einsum('tz,ti,tj->zij', u[:, :rank]*np.sqrt(s[:rank]), basis[k], basis[k])
                right = np.einsum('tz,ti,tj->zij', vh[:rank].T*np.sqrt(s[:rank]), basis[k+1], basis[k+1])
                parts.append((left, right))
            rank = max(len(a) for a, b in parts)
            left = jnp.asarray(np.stack([np.pad(a, ((0, rank-len(a)), (0, 0), (0, 0))) for a, b in parts]))
            right = jnp.asarray(np.stack([np.pad(b, ((0, rank-len(b)), (0, 0), (0, 0))) for a, b in parts]))
            pairs.append((left, right))
        components.append(pairs)

    @jax.jit
    def logs(cores):
        norm = jnp.ones((1, 1, 1))
        for c, gram in zip(cores, grams, strict=True):
            norm = _transfer(norm, c, gram[None])
        answers = []
        for pairs in components:
            state = jnp.ones((len(observations), 1, 1))
            scale = jnp.zeros(len(observations))
            for k, (leftgram, rightgram) in enumerate(pairs):
                c, d = cores[2*k:2*k+2]
                left = jnp.einsum('bpq,pki->bqki', state, c)
                left = jnp.einsum('bqki,bzij->bqkzj', left, leftgram)
                mid = jnp.einsum('bqkzj,qlj->bklz', left, c)
                left = jnp.einsum('bpqz,pki->bqkzi', mid, d)
                left = jnp.einsum('bqkzi,bzij->bqkj', left, rightgram)
                state = jnp.einsum('bqkj,qlj->bkl', left, d)
                state, inc = _rescale(state)
                scale += inc
            answers.append(scale+jnp.log(state[:, 0, 0])-jnp.log(norm[0, 0, 0]))
        return jax.scipy.special.logsumexp(jnp.stack(answers)+jnp.log(jnp.asarray(weights))[:, None], axis=0)
    return logs


def audit(case):
    base = Path(CASES[case])
    capacity = 'r32_n49' if case.startswith('swirl') else 'r10_n25'
    path = Path('artifacts/objective_v3')/case/f'nll_{capacity}_l0.001_w100/fit.npz'
    model, kernel = load_tt(path), load_tt(base/'kernel.npz')
    data = np.load(base/'data.npz')
    train, valid = data['train'], data['valid']
    obs_box = np.stack([np.minimum(train.min(0), valid.min(0))-.02, np.maximum(train.max(0), valid.max(0))+.02], axis=1)
    # Include observed tails as well as randomly chosen observations.
    indices = np.unique(np.r_[np.random.default_rng(917).choice(len(valid), 8, replace=False),
                              np.argsort(np.linalg.norm(valid-valid.mean(0), axis=1))[-4:]])
    probes = valid[indices]
    weights, covariance = noise_parameters(case, probes.shape[1])
    raw = TTConvolution.from_tt(model, kernel)
    prepared = raw.prepare(obs_box, tolerance=1e-14)
    full = raw.prepare(obs_box, tolerance=0.)
    refined = TTConvolution.from_tt(model, kernel, quadrature_order=2*np.array(raw.orders))
    cores = tuple(c.coeffs for c in model.cores)
    def forward(op):
        return jax.jit(lambda p: op.log_density(rebuild_tt(model, p), jnp.asarray(probes)))
    prod = forward(prepared)
    values = np.asarray(prod(cores))
    errors = {}
    for name, op in [('no_moment_truncation', full), ('unprepared', raw), ('double_polynomial_quadrature', refined)]:
        logs = np.asarray(forward(op)(cores))
        errors[name] = float(np.max(np.abs(logs-values)))
        print(case, name, errors[name], flush=True)
    previous = None
    for order in [128, 256]:
        reference = analytic_pair_reference(model, probes, weights, covariance, order)
        answer = np.asarray(reference(cores))
        errors[f'analytic_gaussian_q{order}'] = float(np.max(np.abs(answer-values)))
        if previous is not None:
            errors['analytic_quadrature_refinement'] = float(np.max(np.abs(previous-answer)))
        previous = answer
        print(case, order, errors, flush=True)
    g1 = jax.grad(lambda p: jnp.mean(prod(p)))(cores)
    g2 = jax.grad(lambda p: jnp.mean(reference(p)))(cores)
    errors['relative_gradient_error'] = float(np.sqrt(sum(np.sum((np.asarray(a)-np.asarray(b))**2) for a, b in zip(g1, g2, strict=True))
                                                    /sum(np.sum(np.asarray(a)**2) for a in g2)))
    rng = np.random.default_rng(123)
    direction = tuple(jnp.asarray(rng.normal(size=c.shape)) for c in cores)
    length = jnp.sqrt(sum(jnp.sum(v*v) for v in direction))
    direction = tuple(v/length for v in direction)
    exact = float(sum(jnp.sum(g*v) for g, v in zip(g1, direction, strict=True)))
    eps = 1e-5
    finite = float((jnp.mean(prod(tuple(c+eps*v for c, v in zip(cores, direction, strict=True))))
                  -jnp.mean(prod(tuple(c-eps*v for c, v in zip(cores, direction, strict=True)))))/(2*eps))
    errors['finite_difference_absolute_error'] = abs(exact-finite)
    errors['finite_difference_derivative'] = finite
    out = ROOT/'convolution'
    out.mkdir(parents=True, exist_ok=True)
    write_record(out/f'{case}.json', {'case': case, 'model': str(path), 'probe_indices': indices.tolist(),
        'errors': errors, 'production_log_density_range': [float(values.min()), float(values.max())],
        'uses_truth': False, 'reference': 'analytic Gaussian block quadrature, SVD tolerance 1e-14'})
    print(case, errors, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cases', nargs='+', choices=[c for c in CASES if not c.endswith('free')])
    for case in p.parse_args().cases:
        audit(case)
        jax.clear_caches()

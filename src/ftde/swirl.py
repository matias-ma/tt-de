"""A continuous, curved "swirl-pretzel" tube density, with known truth.

A latent path parameter ``t in [0, 1]`` has smooth density ``w(t)``, and

    x = gamma(t) + TUBE_STD * standard normal noise in R^d.

Hence ``p(x) = int_0^1 w(t) prod_k N(x_k; gamma_k(t), TUBE_STD^2) dt``.
This mirrors the structure of the cryo-EM latent embedding (a single
conformational path embedded in d dimensions), except that the path is
continuous: there are no discrete states, so every pair marginal is a
continuous ridge with smoothly varying intensity. Coordinates (0, 1) trace
an Archimedean spiral, (2, 3) a trefoil ("pretzel") with three crossings,
and every further coordinate is a slow swoop of ``t``, so higher pairs are
Lissajous-like arcs.

Because the curve is one-dimensional, the density, its exact pair marginals
and its squared norm are 1D (or 2D) quadratures in ``t``. Samples are exact
and i.i.d.; no MCMC is involved. The truth is used only for figures and
post-selection diagnostics, never by fitting or model selection.
"""

from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre

TUBE_STD = 0.14
WEIGHT_AMPLITUDE = 0.6


def path_weight(t):
    """Smooth path density on [0, 1] with two brighter stretches; integrates to one."""
    t = np.asarray(t, dtype=float)
    return 1.0 + WEIGHT_AMPLITUDE * np.sin(4 * np.pi * t + 0.4)


def _path_cdf(t):
    return t - WEIGHT_AMPLITUDE / (4 * np.pi) * (np.cos(4 * np.pi * t + 0.4) - np.cos(0.4))


def curve(t, dim):
    """Path gamma(t), shape (..., dim). Requires dim >= 4."""
    if dim < 4:
        raise ValueError('the swirl-pretzel density needs at least four dimensions')
    t = np.asarray(t, dtype=float)
    theta = 2 * np.pi * 1.6 * t + 0.5
    radius = 0.3 + 1.6 * t
    u = 2 * np.pi * t
    coords = [radius * np.cos(theta), radius * np.sin(theta),
              0.62 * (np.sin(u) + 2 * np.sin(2 * u)), 0.62 * (np.cos(u) - 2 * np.cos(2 * u)) + 0.3]
    for k in range(4, dim):
        frequency = 0.5 + 0.25 * ((k - 4) % 4)
        coords.append(1.3 * np.sin(2 * np.pi * frequency * t + 1.3 * k) + 0.8 * (-1) ** k * (t - 0.5))
    return np.stack(coords, axis=-1)


def sample_swirl(seed: int, n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, 200001)
    t = np.interp(rng.uniform(size=n), _path_cdf(grid), grid)
    return curve(t, dim) + TUBE_STD * rng.normal(size=(n, dim))


def path_quadrature(order: int = 4000):
    nodes, weights = roots_legendre(order)
    t = (nodes + 1) / 2
    return t, weights / 2 * path_weight(t)


def swirl_log_density(x, *, order: int = 4000, chunk: int = 2048):
    """Exact log density (up to t-quadrature error) at points x, shape (n, d)."""
    x = np.asarray(x, dtype=float)
    t, w = path_quadrature(order)
    centers = curve(t, x.shape[1])
    out = np.empty(len(x))
    const = -0.5 * x.shape[1] * np.log(2 * np.pi * TUBE_STD**2)
    for s in range(0, len(x), chunk):
        d2 = ((x[s:s + chunk, None, :] - centers[None]) ** 2).sum(-1)
        logs = np.log(w)[None] - d2 / (2 * TUBE_STD**2)
        peak = logs.max(1)
        out[s:s + chunk] = const + peak + np.log(np.exp(logs - peak[:, None]).sum(1))
    return out


def swirl_pair_marginal(dim, i, j, xs, ys, *, order: int = 4000):
    """Exact pair marginal on a grid; rows are y, columns are x."""
    t, w = path_quadrature(order)
    g = curve(t, dim)
    gx = np.exp(-(np.asarray(xs)[:, None] - g[None, :, i]) ** 2 / (2 * TUBE_STD**2))
    gy = np.exp(-(np.asarray(ys)[:, None] - g[None, :, j]) ** 2 / (2 * TUBE_STD**2))
    return (gy * w[None]) @ gx.T / (2 * np.pi * TUBE_STD**2)


def swirl_squared_norm(dim, *, order: int = 4000, chunk: int = 500) -> float:
    """int p(x)^2 dx = int int w(t) w(s) N(gamma(t) - gamma(s); 0, 2 s^2 I) dt ds."""
    t, w = path_quadrature(order)
    g = curve(t, dim)
    total = 0.0
    for s in range(0, len(t), chunk):
        d2 = ((g[s:s + chunk, None, :] - g[None]) ** 2).sum(-1)
        total += float(w[s:s + chunk] @ np.exp(-d2 / (4 * TUBE_STD**2)) @ w)
    return total / (4 * np.pi * TUBE_STD**2) ** (dim / 2)


def swirl_convolved_log_density(x, covariances, weights, *, order: int = 4000, chunk: int = 1024):
    """Exact log density of observations x = clean + noise, noise ~ sum_c w_c N(0, C_c)."""
    x = np.asarray(x, dtype=float)
    dim = x.shape[1]
    t, w = path_quadrature(order)
    centers = curve(t, dim)
    per_component = []
    for weight, cov in zip(np.asarray(weights)/np.sum(weights), np.asarray(covariances), strict=True):
        factor = np.linalg.cholesky(cov+TUBE_STD**2*np.eye(dim))
        inverse = np.linalg.inv(factor)
        const = np.log(weight)-0.5*dim*np.log(2*np.pi)-np.log(np.diag(factor)).sum()
        white_centers = centers@inverse.T
        out = np.empty(len(x))
        for s in range(0, len(x), chunk):
            d2 = ((x[s:s + chunk]@inverse.T)[:, None, :]-white_centers[None]) ** 2
            logs = np.log(w)[None]-0.5*d2.sum(-1)
            peak = logs.max(1)
            out[s:s + chunk] = const+peak+np.log(np.exp(logs-peak[:, None]).sum(1))
        per_component.append(out)
    stacked = np.stack(per_component)
    peak = stacked.max(0)
    return peak+np.log(np.exp(stacked-peak).sum(0))

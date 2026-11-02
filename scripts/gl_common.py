"""Shared plumbing for the Ginzburg-Landau GPU scripts
(gl_gpu_noise_free.py, gl_gpu_noisy.py): target parameters, domain/error
helpers, and marginal-density figure saving (figs/ginzburg_landau/).

Reuses cryoem_common.marginal_density_fn for the exact closed-form 2D
marginal of a fitted squared-TT model (square_tt + chebttge.marginal, same
machinery used for the real cryo-EM marginals).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from chebttge import marginal as marginal_tt_fn
from cryoem_common import marginal_density_fn
from scipy.special import roots_legendre

from ftde.ginzburg_landau import GL_BETA, GL_LAM
from ftde.kde import make_gaussian_kde

# Ginzburg-Landau target parameters live as defaults on the model functions
# themselves now (ftde.ginzburg_landau.GL_BETA/GL_LAM, h=1/(d+1)), matching
# Peng, Khoo & Wang (arXiv:2401.11646) Simulation IV exactly -- just call
# sample_ginzburg_landau/ginzburg_landau_density/_log_normalizer with no
# beta/lam/h kwargs to get that.
FIGS_DIR = Path("figs/ginzburg_landau")


def int_arg(s: str) -> int:
    """argparse ``type=`` for integer CLI flags that also accepts scientific
    notation (``--n-pts 1e5``) -- plain ``int("1e5")`` raises ValueError, so
    a bare ``type=int`` silently rejects the natural way to write a large
    round number on the command line.
    """
    return int(float(s))


def relative_l2_error(y_est: np.ndarray, y_true: np.ndarray) -> float:
    """Return ||estimate - truth||² / ||truth||² on evaluation points."""
    return float(np.sum((y_est - y_true) ** 2) / np.sum(y_true**2))


def data_domain(points: jnp.ndarray, margin_frac: float = 0.1) -> jnp.ndarray:
    """Domain from the RAW min/max of `points`, padded by `margin_frac` of
    the range. Kept for the noise-free case, where there's no heavy-tailed
    noise to distort it -- for noisy data, use `robust_domain` instead (see
    its docstring for why raw min/max is a real problem there).
    """
    lo = jnp.min(points, axis=0)
    hi = jnp.max(points, axis=0)
    margin = margin_frac * (hi - lo)
    return jnp.stack([lo - margin, hi + margin], axis=1)


def robust_domain(points: jnp.ndarray, trim_pct: float = 0.5, margin_frac: float = 0.05) -> jnp.ndarray:
    """Domain from the [trim_pct, 100-trim_pct] percentile range of `points`
    per dimension (instead of raw min/max), padded by `margin_frac` of that
    range.

    Why this matters for noisy data specifically: the domain a squared-TT
    signal is fit on directly controls how much of its fixed Chebyshev
    degree gets "spent" resolving each unit of x -- a domain padded out to
    cover a handful of extreme noise outliers (here, a 2-component mixture
    with a std=0.9 tail component) can end up 1.5-2x wider than where the
    density actually has any mass, for no benefit, at the cost of
    proportionally worse resolution everywhere it does. Trimming the most
    extreme ~1% of points (both tails combined) typically shrinks the
    domain substantially. For a convolved likelihood this is the latent
    signal box: retain noisy observations outside it and ensure that the
    kernel covers all observation-minus-signal differences. Only an
    unconvolved density evaluation requires observations inside its box.
    """
    lo = jnp.percentile(points, trim_pct, axis=0)
    hi = jnp.percentile(points, 100.0 - trim_pct, axis=0)
    margin = margin_frac * (hi - lo)
    return jnp.stack([lo - margin, hi + margin], axis=1)


def filter_to_domain(points: jnp.ndarray, domain: jnp.ndarray) -> jnp.ndarray:
    """Drop rows of `points` that fall outside `domain` in any dimension.
    Use only when deliberately fitting an unconvolved density on a subset.
    Do not apply this to noisy observations for a convolved likelihood.
    """
    inside = jnp.all((points >= domain[:, 0]) & (points <= domain[:, 1]), axis=1)
    kept = points[inside]
    n_dropped = int(points.shape[0] - kept.shape[0])
    if n_dropped:
        print(f"  (dropped {n_dropped}/{points.shape[0]} points outside the trimmed domain)")
    return kept


def dim_pairs_for(dim: int) -> list[tuple[int, int]]:
    """A handful of representative (adjacent + farther-apart) coordinate
    pairs to plot -- adjacent pairs show the strongest coupling (nearest-
    neighbor interaction), a farther pair shows how quickly correlation
    decays along the chain.
    """
    pairs = [(0, 1)]
    if dim >= 4:
        pairs.append((2, 3))
    if dim >= 8:
        pairs.append((0, dim - 1))
    return pairs


def _marginal_grid(fit, i: int, j: int, domain: jnp.ndarray, n_grid: int = 150):
    xmin, xmax = float(domain[i, 0]), float(domain[i, 1])
    ymin, ymax = float(domain[j, 0]), float(domain[j, 1])
    xs = jnp.linspace(xmin, xmax, n_grid)
    ys = jnp.linspace(ymin, ymax, n_grid)
    XX, YY = jnp.meshgrid(xs, ys, indexing="xy")
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    m = marginal_density_fn(fit, "tt", (i, j))
    z = np.concatenate([
        np.asarray(m(grid[start:start + 512]))
        for start in range(0, grid.shape[0], 512)
    ]).reshape(n_grid, n_grid)
    return (xmin, xmax, ymin, ymax), z


def _kde_marginal_grid(train: jnp.ndarray, bandwidth: jnp.ndarray, i: int, j: int, xmin, xmax, ymin, ymax, n_grid: int = 150):
    """Exact 2D marginal of the diagonal-bandwidth Gaussian KDE onto (i, j).

    No approximation needed: a diagonal-covariance Gaussian mixture's
    marginal onto a coordinate subset is exactly the same mixture built
    from only those coordinates (dropping a dimension of an axis-aligned
    Gaussian just drops that dimension's factor) -- so this is just
    ``make_gaussian_kde`` on the 2-column slice of the data with the
    matching 2 entries of the bandwidth vector.
    """
    xs = jnp.linspace(xmin, xmax, n_grid)
    ys = jnp.linspace(ymin, ymax, n_grid)
    XX, YY = jnp.meshgrid(xs, ys, indexing="xy")
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    idx = jnp.array([i, j])
    kde_2d = make_gaussian_kde(train[:, idx], bandwidth=bandwidth[idx])
    return np.asarray(kde_2d(grid)).reshape(n_grid, n_grid)


def _true_marginal_grid(dim: int, i: int, j: int, domain: jnp.ndarray, log_z: float, n_grid: int = 150):
    """GL pair marginal by integrating the intervening chain with 1D quadrature."""
    if not 0 <= i < j < dim:
        raise ValueError("expected ordered, distinct coordinate indices")
    xs = np.linspace(float(domain[i, 0]), float(domain[i, 1]), n_grid)
    ys = np.linspace(float(domain[j, 0]), float(domain[j, 1]), n_grid)
    nodes, weights = roots_legendre(160)
    nodes, weights = 3.0 * nodes, 3.0 * weights
    scale = GL_BETA * GL_LAM * (dim + 1) ** 2 / 2.0

    def edge(a, b):
        return np.exp(-scale * (a - b) ** 2)

    def site(a):
        return np.exp(-GL_BETA * (1.0 - a**2) ** 2 / (4.0 * GL_LAM))

    weighted_site = site(nodes) * weights
    transition = edge(nodes[:, None], nodes[None, :]) * weighted_site[None, :]

    def boundary_message(n_before, values):
        if n_before == 0:
            return edge(0.0, values)
        state = edge(0.0, nodes) * weighted_site
        for _ in range(n_before - 1):
            state = state @ transition
        return state @ edge(nodes[:, None], values[None, :])

    gap = j - i - 1
    bridge = edge(xs[:, None], ys[None, :])
    if gap:
        middle = edge(xs[:, None], nodes[None, :]) * weighted_site[None, :]
        for _ in range(gap - 1):
            middle = middle @ transition
        bridge = middle @ edge(nodes[:, None], ys[None, :])
    left = boundary_message(i, xs) * site(xs)
    right = boundary_message(dim - j - 1, ys) * site(ys)
    return (left[:, None] * bridge * right[None, :] / np.exp(log_z)).T


def save_noise_free_marginal_figure(
    dim: int, fit, clean_test: jnp.ndarray, train: jnp.ndarray, bandwidth: jnp.ndarray,
    domain: jnp.ndarray, i: int, j: int, *, log_z: float, n_steps: int,
    kde_error: float, fit_error: float, rank: int, degree: int, fit_seconds: float,
):
    """Four fixed-window panels, with the histogram of the training data."""
    from refresh_density_figures import (
        draw_panels,
        evaluate_pair,
        histogram,
        kde_grid,
        pair_coefficients,
    )

    from ftde.cheb_product import square_tt

    window = np.tile([-2.5, 2.5], (dim, 1))
    xs = np.linspace(-2.5, 2.5, 180)
    extent = (-2.5, 2.5, -2.5, 2.5)
    coeff, box = pair_coefficients(square_tt(fit), i, j)
    panels = [_true_marginal_grid(dim, i, j, window, log_z, n_grid=len(xs)),
              histogram(np.asarray(train), i, j, extent),
              kde_grid(np.asarray(train), np.asarray(bandwidth), i, j, xs, xs),
              evaluate_pair(coeff, box, xs, xs)]
    draw_panels(panels, ['True GL marginal', 'Histogram of data', 'KDE', 'ft-de'],
                extent, i, j, f'Noise-free Ginzburg–Landau · {dim} dimensions\nft-de sample relative L2 = {np.sqrt(fit_error):.4f}',
                f'{len(train):,} observations · TT rank {rank} · amplitude modes {degree+1} (degree {degree}) · '
                f'{n_steps:,} steps · fit time {fit_seconds:.1f} s\nError: unsquared relative error on held-out full-density values',
                FIGS_DIR/f'noise_free_dim{dim}_marg_{i}{j}.png')


def _kernel_marginal_grid(u, i: int, j: int, extent: tuple[float, float, float, float], n_grid: int = 150):
    """U is already a proper (normalized) density ChebTT -- unlike the
    signal, no squaring/Z-normalizing is needed, just chebttge's marginal()
    directly. Evaluate it on the same window as every other panel so their
    spatial scales are directly comparable. U's declared domain is wider
    than this signal window to satisfy convolution's fixed-window condition.
    """
    xmin, xmax, ymin, ymax = extent
    xs = jnp.linspace(xmin, xmax, n_grid)
    ys = jnp.linspace(ymin, ymax, n_grid)
    XX, YY = jnp.meshgrid(xs, ys, indexing="xy")
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)
    m = marginal_tt_fn(u, keep_modes=(i, j))
    z = np.asarray(m(grid)).reshape(n_grid, n_grid)
    return extent, z


def save_noisy_marginal_figure(
    dim: int, naive_fit, deconv_fit, u, clean_test: jnp.ndarray, noisy_train: jnp.ndarray,
    kde_bandwidth: jnp.ndarray, domain: jnp.ndarray, i: int, j: int, *,
    log_z: float, kde_error: float, naive_error: float, deconv_error: float,
    naive_steps: int, deconv_steps: int, n_density_points: int, noise_type: str,
    naive_rank: int, deconv_rank: int, degree: int, sigma: float, fit_seconds: float,
    selected_step: int | None = None,
    figure_dir: Path | None = None,
    deconv_label: str = "Deconvolved TT",
):
    """Six fixed-window panels displaying the recovered latent density.

    Historical clean-test and error arguments remain accepted for callers;
    the displayed histogram uses noisy training observations only.
    """
    from refresh_density_figures import (
        convolved_grids,
        draw_panels,
        evaluate_pair,
        histogram,
        kde_grid,
        pair_coefficients,
    )

    from ftde.cheb_product import square_tt

    window = np.tile([-2.5, 2.5], (dim, 1))
    xs = np.linspace(-2.5, 2.5, 180)
    extent = (-2.5, 2.5, -2.5, 2.5)
    coeff, box = pair_coefficients(square_tt(deconv_fit), i, j)
    uc, ub = pair_coefficients(u, i, j)
    true_q, _ = convolved_grids(dim, i, j, coeff, box, uc, ub, xs, log_z, .0125)
    panels = [_true_marginal_grid(dim, i, j, window, log_z, n_grid=len(xs)),
              evaluate_pair(uc, ub, xs, xs, negative_tolerance=1e-5), true_q,
              histogram(np.asarray(noisy_train), i, j, extent),
              kde_grid(np.asarray(noisy_train), np.asarray(kde_bandwidth), i, j, xs, xs),
              evaluate_pair(coeff, box, xs, xs)]
    destination = FIGS_DIR if figure_dir is None else figure_dir
    scale_label = f'{sigma:g}'.replace('.', 'p')
    draw_panels(panels, ['True GL marginal', 'Noise kernel', 'Convolved marginal',
                        'Histogram of noisy data', 'KDE of noisy data', 'Deconvolved ft-de'],
                extent, i, j, (f'Noisy Ginzburg–Landau · {dim} dimensions · {noise_type} · scale {sigma:g}'
                              f'\nDeconvolved ft-de sample relative L2 = {np.sqrt(deconv_error):.4f}'),
                f'{len(noisy_train):,} fitted observations · TT rank {deconv_rank} · amplitude modes {degree+1} (degree {degree}) · '
                f'{deconv_steps:,} steps · selected {selected_step} · fit time {fit_seconds:.1f} s\n'
                'Error: unsquared relative error on held-out full-density values',
                destination/f'noisy_{noise_type}_scale{scale_label}_dim{dim}_marg_{i}{j}.png',
                [0, 1, 2, 2, 2, 0])

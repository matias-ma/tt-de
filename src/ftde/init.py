"""Initialize a squared-TT/Tucker density from an elementary KDE.

We fit the *square root* of the KDE via functional cross-approximation
(chebttge/chebtuckerge), so that the resulting squared-TT/Tucker model
starts close to the KDE density before gradient-based NLL refinement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chebttge import ChebTTOptions, tt_qtensge
from chebttge.chebtt import ChebTT
from chebtuckerge import ChebTuckerOptions, tucker_qtensge
from chebtuckerge.chebtucker import ChebTucker

from .kde import bounding_domain, make_gaussian_kde


def _estimate_kde_scale(kde, reference_points: jnp.ndarray) -> float:
    """A rough O(1)-ish estimate of the KDE's own typical/peak magnitude.

    Density values scale roughly like 1/volume, so for data spread over a
    wide domain (or in higher dimension) they can legitimately be many
    orders of magnitude smaller than "small-looking" absolute constants like
    1e-12 -- e.g. ~1e-18 on a real 4D dataset here, and ~1e-42 at 10D. Both
    chebttge/chebtuckerge's cross-approximation convergence tolerances
    (``tol``/``cheb_tol``, default 1e-12) and any hard-coded numerical floor
    are ABSOLUTE, so without rescaling, a target this small looks like "already
    zero" to them regardless of its real relative structure -- this caused a
    degenerate rank-1 fit in both cases (once via an explicit floor set too
    high, once via cross-approximation's own default tolerance). Dividing the
    KDE by its own typical scale before fitting sqrt(kde) keeps those default,
    O(1)-tuned tolerances meaningful at any absolute density scale. This
    rescaling is always safe here: the final density p = g^2 / Z is exactly
    invariant to multiplying g by any positive constant.
    """
    return float(jnp.max(kde(reference_points)))


def _sqrt_kde_target(kde, scale: float = 1.0, floor: float = 1e-300):
    # The KDE is mathematically always >= 0 (a sum of positive Gaussian
    # kernel evaluations), so this floor exists only to guard against
    # sqrt() of a value that's gone slightly negative from floating-point
    # noise -- it should never clip real signal (see _estimate_kde_scale's
    # docstring for why an absolute floor much above ~1e-300 is not safe).
    inv_scale = 1.0 / max(scale, 1e-300)

    def f(x):
        return jnp.sqrt(jnp.maximum(kde(x) * inv_scale, floor))

    return f


def _subsample_for_kde(data: jnp.ndarray, max_kde_points: int | None, seed: int) -> jnp.ndarray:
    """Cap the number of points the KDE itself is built from.

    ``make_gaussian_kde`` costs O(n_query * n_data) per evaluation, and
    cross-approximation's pivot search evaluates the target function on
    candidate batches as large as a few thousand points by default -- with a
    large dataset (this matters starting in the tens of thousands of points)
    that product can reach billions of floating-point pairs in a single
    call, which is slow enough to look hung and can exhaust memory outright
    (this actually happened on a 99k-point real dataset). A KDE used only to
    seed the cross-approximation doesn't need the full dataset to be a
    reasonable starting point, so subsample it; the subsequent NLL training
    (``fit_tt``/``fit_noisy_tt``, etc.) still sees the full data, via
    minibatches if you pass ``batch_size``.
    """
    if max_kde_points is None or data.shape[0] <= max_kde_points:
        return data
    idx = jax.random.choice(jax.random.PRNGKey(seed), data.shape[0], (max_kde_points,), replace=False)
    return data[idx]


def init_tt_from_kde(
    data: jnp.ndarray,
    *,
    max_rank: int = 8,
    domain: jnp.ndarray | None = None,
    margin_sigmas: float = 6.0,
    n_coeffs: tuple[int, ...] = (17, 33, 65),
    bandwidth: jnp.ndarray | None = None,
    n_starts: int = 256,
    gd_steps: int = 50,
    random_seed: int = 0,
    max_kde_points: int | None = 5000,
) -> ChebTT:
    data = jnp.asarray(data, dtype=jnp.float64)
    d = data.shape[1]
    if domain is None:
        domain = bounding_domain(data, margin_sigmas=margin_sigmas)
    kde_data = _subsample_for_kde(data, max_kde_points, random_seed)
    kde = make_gaussian_kde(kde_data, bandwidth=bandwidth)
    target = _sqrt_kde_target(kde, scale=_estimate_kde_scale(kde, kde_data[:200]))

    opts = ChebTTOptions(
        domain=domain,
        max_rank=max_rank,
        core_representation="coeff_tensor",
        coeff_core_n_values=n_coeffs,
        n_starts=n_starts,
        gd_steps=gd_steps,
        random_seed=random_seed,
    )
    return tt_qtensge(target, opts, d=d)


def init_tucker_from_kde(
    data: jnp.ndarray,
    *,
    max_rank: int = 6,
    domain: jnp.ndarray | None = None,
    margin_sigmas: float = 6.0,
    n_coeffs: tuple[int, ...] = (17, 33, 65),
    bandwidth: jnp.ndarray | None = None,
    n_starts: int = 256,
    gd_steps: int = 50,
    random_seed: int = 0,
    max_kde_points: int | None = 5000,
) -> ChebTucker:
    data = jnp.asarray(data, dtype=jnp.float64)
    d = data.shape[1]
    if domain is None:
        domain = bounding_domain(data, margin_sigmas=margin_sigmas)
    kde_data = _subsample_for_kde(data, max_kde_points, random_seed)
    kde = make_gaussian_kde(kde_data, bandwidth=bandwidth)
    target = _sqrt_kde_target(kde, scale=_estimate_kde_scale(kde, kde_data[:200]))

    opts = ChebTuckerOptions(
        domain=domain,
        max_rank=max_rank,
        core_representation="coeff_tensor",
        coeff_core_n_values=n_coeffs,
        n_starts=n_starts,
        gd_steps=gd_steps,
        random_seed=random_seed,
    )
    return tucker_qtensge(target, opts, d=d)

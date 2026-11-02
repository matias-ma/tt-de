"""Legacy frequency-weighted penalties on individual TT/Tucker cores.

Retained to reproduce earlier experiments. These are parameter penalties,
not norms of the normalized density: they depend on tensor gauge, and a
common amplitude rescaling can reduce them without changing the density.
The Chebyshev-degree/Fourier-frequency correspondence is only a heuristic;
Chebyshev polynomials are not eigenfunctions of Gaussian convolution on a
finite interval. Scaling exponential weights by their maximum can leave
all but the last few degrees effectively unregularized.

For a gauge- and scale-invariant TT alternative, use
``AmplitudeSobolevPenalty`` in ``ftde.sobolev``. Select its strength from
noisy validation data; it does not make the inverse problem well-conditioned.
"""

from __future__ import annotations

import jax.numpy as jnp
from chebttge.chebtt import ChebTT
from chebtuckerge.chebtucker import ChebTucker


def chebyshev_tikhonov_weights(n_coeffs: int, domain: tuple[float, float], noise_std: float) -> jnp.ndarray:
    """Per-degree penalty weight ``expm1((noise_std * k * pi / L)^2)``, for
    ``k = 0..n_coeffs-1`` on a domain of width ``L = domain[1] - domain[0]``.

    Zero at ``k=0`` (an unshifted/constant mode passes through any noise
    kernel unchanged) and grows extremely quickly in ``k`` once
    ``noise_std * k * pi / L`` exceeds ~1 -- i.e. once a mode's implied
    physical wavelength drops below roughly the noise's own scale.
    """
    a, b = domain
    width = b - a
    k = jnp.arange(n_coeffs, dtype=jnp.float64)
    omega = k * jnp.pi / width
    return jnp.expm1((noise_std * omega) ** 2)


def _axis_weight(n_coeffs: int, domain: tuple[float, float], std: float, per_axis_target_budget: float | None) -> jnp.ndarray:
    weight = chebyshev_tikhonov_weights(n_coeffs, domain, std)
    if per_axis_target_budget is not None:
        # Rescale THIS axis's own weight array so its own max equals
        # per_axis_target_budget, independent of every other axis -- see
        # this module's own comment (near calibrated_reg_strength) for why
        # a single, externally-computed reg_strength cannot safely
        # substitute for this whenever axes have genuinely different
        # domain widths/noise_std (confirmed empirically on real cryo-EM
        # dim=10 data, see todo.txt: one axis's raw weight can be so much
        # larger than the others' that any single global reg_strength
        # small enough to keep that one axis's penalty finite underflows to
        # ~0 for every other axis, SILENTLY disabling regularization there
        # even though the caller's reg_lambda was nonzero).
        weight = weight * (per_axis_target_budget / jnp.maximum(jnp.max(weight), 1e-300))
    return weight


def tikhonov_penalty_tt(
    g_sq: ChebTT, noise_std: jnp.ndarray, *, per_axis_target_budget: float | None = None,
) -> jnp.ndarray:
    """Sum over modes/cores of ``sum_k weight(k) * coeffs[..., k]^2`` for
    ``g_sq = square_tt(g)`` -- a plain weighted-L2 (ridge) penalty on the
    squared signal's own Chebyshev coefficients, entrywise across the TT
    bond-rank indices.

    ``per_axis_target_budget``, if given, rescales EACH axis's own weight
    array to its own max of this value before summing -- the robust
    alternative to a single externally-computed ``reg_strength`` (e.g. via
    ``calibrated_reg_strength``) whenever axes have heterogeneous domain
    widths/noise_std (see ``_axis_weight``'s comment). Leaving this
    ``None`` (default) reproduces the original, unnormalized-per-axis
    behavior -- safe whenever domains/noise_std are roughly homogeneous
    across axes (e.g. this project's own Ginzburg-Landau benchmark, where a
    single externally-calibrated ``reg_strength`` is validated and this
    flag is intentionally left off to keep that behavior reproducible).
    """
    penalty = jnp.asarray(0.0, dtype=jnp.float64)
    for core, std in zip(g_sq.cores, noise_std, strict=True):
        n_coeffs = core.coeffs.shape[-1]
        weight = _axis_weight(n_coeffs, core.domain, std, per_axis_target_budget)
        penalty = penalty + jnp.sum(core.coeffs**2 * weight)
    return penalty


def tikhonov_penalty_tucker(
    g_sq: ChebTucker, noise_std: jnp.ndarray, *, per_axis_target_budget: float | None = None,
) -> jnp.ndarray:
    """Tucker analogue of ``tikhonov_penalty_tt``: sums the per-mode penalty
    over each factor's own Chebyshev coefficients (the core mixing tensor
    itself carries no per-mode frequency content, so it is not penalized).
    See ``tikhonov_penalty_tt`` for ``per_axis_target_budget``.
    """
    penalty = jnp.asarray(0.0, dtype=jnp.float64)
    for factor, std in zip(g_sq.factors, noise_std, strict=True):
        coeffs = factor.core.coeffs[0]  # (rank, n_coeffs)
        n_coeffs = coeffs.shape[-1]
        weight = _axis_weight(n_coeffs, factor.domain, std, per_axis_target_budget)
        penalty = penalty + jnp.sum(coeffs**2 * weight)
    return penalty


# The Tikhonov penalty's own per-coefficient weight, expm1((noise_std * k *
# pi / L)^2), grows EXPONENTIALLY in noise_std/domain-width/degree -- a
# FIXED reg_strength constant that is safe at one (noise_std, degree,
# domain) is not safe at another (it can be many orders of magnitude off,
# silently blowing up training NLL from the first step or, conversely,
# regularizing so weakly it does nothing). ``calibrated_reg_strength``
# fixes the overall regularization *budget* (a single, external
# reg_strength times the single largest per-coefficient weight across ALL
# modes/axes) and solves for the reg_strength that hits it -- portable
# across degree/noise_std/domain as long as those are roughly HOMOGENEOUS
# across axes (validated this way on the Ginzburg-Landau benchmark, which
# has one common domain width/noise scale per mode).
#
# It is NOT safe when axes have genuinely heterogeneous domain
# widths/noise_std (confirmed empirically on real cryo-EM dim=10 data --
# see todo.txt): the single global reg_strength that keeps the WORST
# axis's penalty sane can be many orders of magnitude too small for every
# other axis, silently underflowing to an effective no-op regularizer on
# most of them even though reg_lambda was nonzero. For that case, prefer
# passing ``per_axis_target_budget`` directly to
# ``tikhonov_penalty_tt``/``_tucker`` (and ``reg_strength=reg_lambda``,
# dimensionless, to ``fit_noisy_tt``/``_tucker``) instead of this function
# -- it normalizes each axis's own weight independently, which reduces to
# the exact same result as this function in the homogeneous case but stays
# correct in the heterogeneous one.
DEFAULT_REG_TARGET_BUDGET = 0.5


def calibrated_reg_strength(
    domain: jnp.ndarray, noise_std: jnp.ndarray, degree: int, reg_lambda: float,
    target_budget: float = DEFAULT_REG_TARGET_BUDGET,
) -> float:
    """Solve for a single, external ``reg_strength`` (see
    ``tikhonov_penalty_tt``/``_tucker``) that gives an effective
    regularization budget of ``reg_lambda * target_budget``, calibrated
    against this run's own actual per-axis domain/noise_std/degree (see
    module comment above for why a fixed constant isn't portable across
    those -- and why this single-scalar approach itself is only safe when
    axes are roughly homogeneous; use ``per_axis_target_budget`` instead
    otherwise). ``degree`` is the signal g's own Chebyshev degree; the
    penalty applies to ``g^2``, whose degree is ``2*degree - 1``.
    """
    g_sq_degree = 2 * degree - 1
    max_weight = 0.0
    for i in range(domain.shape[0]):
        w = chebyshev_tikhonov_weights(g_sq_degree, (float(domain[i, 0]), float(domain[i, 1])), float(noise_std[i]))
        max_weight = max(max_weight, float(jnp.max(w)))
    return reg_lambda * target_budget / max(max_weight, 1e-300)

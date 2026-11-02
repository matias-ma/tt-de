"""A minimal fully-JAX (autodiff-friendly) diagonal Gaussian KDE.

We need the initial density estimate itself to be a JAX function so that
``tt_qtensge``/``tucker_qtensge`` can gradient-refine cross-approximation
pivots against it (a black-box scipy KDE would break that autodiff path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scott_bandwidth(data: jnp.ndarray) -> jnp.ndarray:
    """Per-dimension Scott's-rule bandwidth for a diagonal Gaussian KDE."""
    n, d = data.shape
    std = jnp.std(data, axis=0)
    return std * n ** (-1.0 / (d + 4))


def make_gaussian_kde(
    data: jnp.ndarray, bandwidth: jnp.ndarray | None = None, *, max_pairwise_elements: int = 2_000_000,
):
    """Return ``kde(x)`` evaluating a diagonal-bandwidth Gaussian KDE at ``x``.

    ``data`` has shape (n, d); ``kde`` accepts ``x`` of shape (batch, d) or
    (d,) and returns density values of matching batch shape.

    HARD, ALWAYS-ON memory safeguard: naively evaluating this needs one
    ``(n_query, n, d)`` array, which at real dataset sizes (tens of
    thousands of points -- the scale this project's own GPU scripts use)
    and moderate-to-high dimension can reach tens of gigabytes and exhaust
    memory outright (confirmed: this happened more than once on this
    project's own development machine, including from code that *thought*
    it had picked a safe size). Rather than relying on any caller picking a
    safe batch size, this chunks over BOTH the query axis and the
    reference-data axis so that the largest transient array is always
    capped at ``max_pairwise_elements`` entries (default 2,000,000 -- 16MB
    at float64 -- regardless of ``d``, ``n``, or how many points are
    queried at once). This is exact streaming logsumexp accumulation, not
    a subsample or other approximation -- results are bit-identical (up to
    ordinary floating-point summation-order noise) to the unbatched
    computation regardless of chunk size.
    """
    data = jnp.asarray(data, dtype=jnp.float64)
    n, d = data.shape
    if bandwidth is None:
        bandwidth = scott_bandwidth(data)
    bandwidth = jnp.asarray(bandwidth, dtype=jnp.float64)
    log_norm_const = (
        -jnp.log(n)
        - 0.5 * d * jnp.log(2.0 * jnp.pi)
        - jnp.sum(jnp.log(bandwidth))
    )
    data_batch_size = max(1, max_pairwise_elements // max(d, 1))

    def _log_kernel_sum(xs: jnp.ndarray) -> jnp.ndarray:
        if n <= data_batch_size:
            diff = (xs[:, None, :] - data[None, :, :]) / bandwidth
            return jax_logsumexp(-0.5 * jnp.sum(diff**2, axis=-1))
        running_max = jnp.full((xs.shape[0],), -jnp.inf, dtype=jnp.float64)
        running_sum = jnp.zeros((xs.shape[0],), dtype=jnp.float64)
        for start in range(0, n, data_batch_size):
            chunk = data[start : start + data_batch_size]
            diff = (xs[:, None, :] - chunk[None, :, :]) / bandwidth
            log_kernel = -0.5 * jnp.sum(diff**2, axis=-1)  # (n_query, chunk)
            chunk_max = jnp.max(log_kernel, axis=-1)
            new_max = jnp.maximum(running_max, chunk_max)
            running_sum = running_sum * jnp.exp(running_max - new_max) + jnp.sum(
                jnp.exp(log_kernel - new_max[:, None]), axis=-1
            )
            running_max = new_max
        return running_max + jnp.log(running_sum)

    def kde(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64)
        scalar_input = x.ndim == 1
        xs = x[None, :] if scalar_input else x
        query_batch_size = max(1, max_pairwise_elements // (min(n, data_batch_size) * max(d, 1)))
        if xs.shape[0] <= query_batch_size:
            out = jnp.exp(log_norm_const + _log_kernel_sum(xs))
        else:
            parts = [
                jnp.exp(log_norm_const + _log_kernel_sum(xs[start : start + query_batch_size]))
                for start in range(0, xs.shape[0], query_batch_size)
            ]
            out = jnp.concatenate(parts, axis=0)
        return out[0] if scalar_input else out

    return kde


def jax_logsumexp(a: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    m = jnp.max(a, axis=axis, keepdims=True)
    out = m + jnp.log(jnp.sum(jnp.exp(a - m), axis=axis, keepdims=True))
    return jnp.squeeze(out, axis=axis)


def _lscv_score(data: jnp.ndarray, bandwidth: jnp.ndarray) -> float:
    """Least-squares cross-validation score (up to an additive constant
    independent of bandwidth): an estimate of ``integral (kde - p_true)^2
    dx`` built entirely from the data itself (no knowledge of ``p_true``
    needed) -- the standard, textbook data-only score to *minimize* for
    bandwidth selection when the thing you actually care about is squared
    error (Bowman 1984; Silverman 1986 sec 3.4.3), rather than likelihood.

    ``integral kde^2 dx`` has an exact closed form for a Gaussian kernel:
    convolving two N(0, b^2) Gaussians and evaluating at 0 gives N(0, 2b^2)
    evaluated at the points' own separation, so ``integral kde^2 dx`` is a
    plain pairwise sum with bandwidth ``sqrt(2)*b`` (no numerical
    integration needed). The cross term is the standard leave-one-out
    log-density sum.
    """
    n, d = data.shape
    diff_sq = jnp.sum(((data[:, None, :] - data[None, :, :]) / bandwidth) ** 2, axis=-1)  # (n, n)
    log_norm_b = -0.5 * d * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(bandwidth))
    log_norm_sqrt2b = log_norm_b - 0.5 * d * jnp.log(2.0)
    term1 = jnp.sum(jnp.exp(log_norm_sqrt2b - 0.25 * diff_sq)) / n**2
    off_diag = jnp.exp(log_norm_b - 0.5 * diff_sq) * (1.0 - jnp.eye(n))
    term2 = jnp.sum(off_diag) * 2.0 / (n * (n - 1))
    return float(term1 - term2)


def select_gaussian_kde_bandwidth(
    data: jnp.ndarray,
    *,
    max_points: int = 1200,
    n_scales: int = 24,
    scale_range: tuple[float, float] = (0.05, 2.0),
    seed: int = 0,
) -> jnp.ndarray:
    """Per-dimension bandwidth via least-squares cross-validation (LSCV, see
    ``_lscv_score``), as a single multiplicative correction to Scott's-rule
    bandwidth.

    Scott's rule (``scott_bandwidth``) is derived to minimize integrated MSE
    *assuming the true density is roughly Gaussian* -- for a genuinely
    multimodal target (e.g. the bistable Ginzburg-Landau field), it can
    oversmooth badly: two nearby modes get blurred into one broad hump, so
    the KDE's value at each mode's actual peak location is far below the
    true peak, which a relative-L2 error metric penalizes severely
    (confirmed empirically: Scott's rule alone gave relative L2 error
    ~0.997 -- essentially uninformative -- on an 8D Ginzburg-Landau target).
    Naively shrinking the bandwidth is not safe either -- too small and the
    KDE becomes a near-sum-of-delta-functions with wild variance between
    query points (confirmed: shrinking far enough blows the error up past
    1000). LSCV finds a good scale automatically and generically, without
    hardcoding anything about this specific target.

    (An earlier version of this used likelihood cross-validation instead --
    also standard, but confirmed empirically to be noticeably WORSE here: a
    single held-out point landing in a near-zero-density region creates a
    huge log-likelihood penalty, biasing the selection toward oversmoothing
    to avoid that risk, even when the resulting bandwidth is far from
    squared-error-optimal. LSCV targets squared error directly, which is
    what we actually care about here, and empirically lands much closer to
    the true L2-optimal bandwidth on this target.)

    Deliberately memory-bounded (`max_points`, default 800): this is an
    O(n^2) computation (`_lscv_score`'s pairwise distance array), and
    building large O(n_query * n_ref) arrays at full dataset size (tens of
    thousands of points) is what exhausted memory on this machine earlier --
    bandwidth selection only needs a modest, fixed-size subsample to get a
    good scale, not the full training set.
    """
    data = jnp.asarray(data, dtype=jnp.float64)
    n = data.shape[0]
    if n > max_points:
        idx = jax.random.choice(jax.random.PRNGKey(seed), n, (max_points,), replace=False)
        data = data[idx]

    base_bw = scott_bandwidth(data)
    scales = jnp.geomspace(scale_range[0], scale_range[1], n_scales)
    best_scale, best_score = 1.0, jnp.inf
    for scale in scales:
        score = _lscv_score(data, base_bw * scale)
        if score < best_score:
            best_score, best_scale = score, float(scale)
    return base_bw * best_scale


def bounding_domain(data: jnp.ndarray, margin_sigmas: float = 6.0) -> jnp.ndarray:
    """A (d, 2) domain covering the data with a Gaussian-tail margin."""
    data = jnp.asarray(data, dtype=jnp.float64)
    std = jnp.std(data, axis=0)
    lo = jnp.min(data, axis=0) - margin_sigmas * std
    hi = jnp.max(data, axis=0) + margin_sigmas * std
    return jnp.stack([lo, hi], axis=1)

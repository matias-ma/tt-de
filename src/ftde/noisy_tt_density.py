"""Noisy density estimation: fit the squared-TT signal g so that the known
noise kernel U convolved with p_tilde = g^2/Z matches noisy observations.

U is fixed (prescribed, not trained); only g's Chebyshev coefficients are
optimized. This directly reuses:
  - square_tt (g -> a real ChebTT representing g(x)^2)
  - chebttge.convolve_tt (exact or RSI-compressed convolution)
  - squared_integral_tt (the same normalizer Z as the noise-free model,
    since integrating the convolution over y recovers integral g^2 dx again)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from chebttge import convolve_tt, integrate
from chebttge.chebtt import ChebTT

from .cheb_product import square_tt
from .gaussian_convolution import GaussianConvolution
from .noisy_regularization import tikhonov_penalty_tt
from .tt_convolution import ConvolutionOperator
from .tt_density import TTParams, rebuild_tt, squared_integral_tt, tt_params
from .tt_rounding import round_tt


def log_density_noisy_tt(
    g: ChebTT,
    u: ChebTT | None,
    y: jnp.ndarray,
    eps: float = 1e-300,
    *,
    output_domain: jnp.ndarray | None = None,
    pre_conv_max_rank: int | None = None,
    convolution_max_rank: int | None = None,
    convolution_tol: float = 0.0,
    convolution_method: str = "naive",
    rsi_max_rank: int | None = None,
    rsi_sketch_size: int | None = None,
    rsi_pivot_rows=None,
    gaussian_convolution: GaussianConvolution | None = None,
    forward_operator: ConvolutionOperator | None = None,
) -> jnp.ndarray:
    """log q(y) for the noisy density q = (U * g^2) / Z, y of shape (batch, d) or (d,).

    ``forward_operator`` accepts a direct convolution operator, including
    ``TTConvolution`` for arbitrary prescribed Chebyshev TT noise kernels.
    The operator controls the valid observation domain; no explicit square
    or rank truncation is used. ``u`` may be None in that case.

    When ``gaussian_convolution`` is supplied, use the direct analytic-Gaussian
    quadrature operator instead of ``u``; no explicit square or SVD is needed.
    Observations may lie outside the signal box.

    ``output_domain`` lets ``q`` be evaluated on a domain wider than ``g``'s
    own (e.g. ``g`` fit tightly to where the clean signal actually lives,
    while the noisy observations ``y`` spread further out) -- passed
    straight through to chebdCUR's ``convolve_tt``, which already supports
    this; see ``noise_kernel.gaussian_noise_domain``'s docstring for how the
    kernel ``u`` must be sized to match. Defaults to ``g``'s own domain
    (the original, backward-compatible behavior).

    ``convolution_method="naive"`` uses exact spectral convolution. The RSI
    path uses a compressed TT; coefficient gradients through JIT require
    preselected ``rsi_pivot_rows``. ``make_noisy_train_step`` selects these
    rows from the initial model before compiling the optimizer step.

    ``pre_conv_max_rank`` rounds ``g^2`` (SVD-based, via ``round_tt``) BEFORE
    convolving -- this is the lever that actually controls the expensive
    step's own cost. ``square_tt(g)`` gives ``g^2`` an EXACT rank of
    ``rank(g)^2`` (a worst-case combinatorial bound from the Chebyshev
    linearization identity, not necessarily the rank ``g^2`` actually
    needs), and convolve_tt's cost scales like
    ``O(kernel_degree^2 * rank(g^2)^2 * rank(u)^2)`` -- so at rank(g)=8 this
    convolves an effective rank-64 object even though the true density's
    own shape is often well captured at a much smaller rank (empirically,
    for signals like this project's Ginzburg-Landau target). Rounding
    ``g^2`` back down to a modest rank before convolving directly cuts that
    cost, unlike ``convolution_max_rank`` below (which only rounds the
    ALREADY-computed, already-expensive convolution output -- it speeds up
    nothing upstream of it). This IS an approximation (some of g^2's true
    shape is discarded), so it trades a bit of representational accuracy
    for a real, structural speedup; compare against unrounded results
    before trusting it blindly at very low rounded ranks.
    """
    if forward_operator is not None and gaussian_convolution is not None:
        raise ValueError("provide only one forward operator")
    operator = forward_operator if forward_operator is not None else gaussian_convolution
    if operator is not None:
        if pre_conv_max_rank is not None or convolution_max_rank is not None or convolution_tol:
            raise ValueError("direct convolution does not use TT rank truncation")
        return operator.log_density(g, y)
    if u is None:
        raise ValueError("provide a noise kernel or a direct forward operator")
    g_sq = square_tt(g)
    if pre_conv_max_rank is not None:
        g_sq = round_tt(g_sq, max_rank=pre_conv_max_rank)
    # chebdCUR's convolve_tt resolves output_domain to plain Python floats
    # internally (it is a static, per-call domain spec, not a differentiated
    # array), which fails under jax.jit if it's still a jnp array at that
    # point -- iterating a jnp array inside a trace yields tracers even for
    # a value that is, in fact, a closed-over compile-time constant here.
    # Converting to numpy first sidesteps that (a concrete jnp/np array
    # closed over by a jitted function is never itself a tracer, so this
    # numpy conversion is safe to do at trace time).
    static_output_domain = None if output_domain is None else np.asarray(output_domain, dtype=np.float64)
    q_unnorm = convolve_tt(
        g_sq, u, output_domain=static_output_domain, method=convolution_method,
        max_rank=rsi_max_rank, sketch_size=rsi_sketch_size, pivot_rows=rsi_pivot_rows,
        validation_samples=0 if rsi_pivot_rows is not None else 256,
    )
    z = squared_integral_tt(g)
    if pre_conv_max_rank is not None or convolution_max_rank is not None or convolution_tol:
        if convolution_max_rank is not None or convolution_tol:
            q_unnorm = round_tt(
                q_unnorm, max_rank=convolution_max_rank, tol=convolution_tol
            )
        # Rank truncation (either step) perturbs the convolution integral.
        # Normalize the compressed forward model itself so the opt-in
        # approximation remains a density; the exact path retains the
        # cheaper shared Z.
        z = integrate(q_unnorm)
    return jnp.log(jnp.maximum(q_unnorm(y), eps)) - jnp.log(z)


def density_noisy_tt(g: ChebTT, u: ChebTT | None, y: jnp.ndarray, *, output_domain: jnp.ndarray | None = None,
                     gaussian_convolution: GaussianConvolution | None = None,
                     forward_operator: ConvolutionOperator | None = None) -> jnp.ndarray:
    return jnp.exp(log_density_noisy_tt(g, u, y, output_domain=output_domain,
                                        gaussian_convolution=gaussian_convolution, forward_operator=forward_operator))


def nll_noisy_tt(
    params: TTParams,
    template: ChebTT,
    u: ChebTT | None,
    y: jnp.ndarray,
    *,
    noise_std: jnp.ndarray | None = None,
    reg_strength: float = 0.0,
    regularizer=None,
    per_axis_target_budget: float | None = None,
    pre_conv_max_rank: int | None = None,
    convolution_max_rank: int | None = None,
    convolution_tol: float = 0.0,
    convolution_method: str = "naive",
    rsi_max_rank: int | None = None,
    rsi_sketch_size: int | None = None,
    rsi_pivot_rows=None,
    gaussian_convolution: GaussianConvolution | None = None,
    forward_operator: ConvolutionOperator | None = None,
) -> jnp.ndarray:
    """NLL of the noisy density, optionally plus a regularization penalty.
    A supplied ``regularizer(g)`` is multiplied by ``reg_strength`` and
    replaces the legacy core penalty described below.
    For the legacy penalty, ``noise_std`` (per-mode standard deviation,
    required when ``reg_strength`` is nonzero) must match the kernel ``u``.
    ``per_axis_target_budget`` is passed straight through to
    ``tikhonov_penalty_tt`` -- use it (with ``reg_strength=reg_lambda``
    directly) instead of a pre-computed ``calibrated_reg_strength`` whenever
    modes have heterogeneous domain widths/noise_std; see that function's
    docstring. ``pre_conv_max_rank`` (see ``log_density_noisy_tt``) is the
    lever that actually reduces the expensive convolution step's own cost,
    unlike ``convolution_max_rank`` which only rounds its already-computed
    output.

    With ``gaussian_convolution``, ``u`` may be None and observations may
    lie outside the signal domain. The following limitation applies only
    to the legacy spectral path.

    This training loss uses the template's output domain. Observations must
    lie in that domain on the spectral path.
    """
    g = rebuild_tt(template, params)
    nll = -jnp.mean(
        log_density_noisy_tt(
            g, u, y, pre_conv_max_rank=pre_conv_max_rank,
            convolution_max_rank=convolution_max_rank, convolution_tol=convolution_tol,
            convolution_method=convolution_method, rsi_max_rank=rsi_max_rank,
            rsi_sketch_size=rsi_sketch_size,
            rsi_pivot_rows=rsi_pivot_rows,
            gaussian_convolution=gaussian_convolution, forward_operator=forward_operator,
        )
    )
    if reg_strength:
        penalty = (regularizer(g) if regularizer is not None else tikhonov_penalty_tt(
            square_tt(g), noise_std, per_axis_target_budget=per_axis_target_budget))
        nll = nll + reg_strength * penalty
    return nll


def make_noisy_train_step(
    template: ChebTT,
    u: ChebTT | None,
    optimizer: optax.GradientTransformation,
    *,
    noise_std: jnp.ndarray | None = None,
    reg_strength: float = 0.0,
    regularizer=None,
    per_axis_target_budget: float | None = None,
    pre_conv_max_rank: int | None = None,
    convolution_max_rank: int | None = None,
    convolution_tol: float = 0.0,
    convolution_method: str = "rsi",
    rsi_max_rank: int | None = None,
    rsi_sketch_size: int | None = None,
    rsi_pivot_rows=None,
    gaussian_convolution: GaussianConvolution | None = None,
    forward_operator: ConvolutionOperator | None = None,
):
    if convolution_method == "rsi" and rsi_pivot_rows is None and forward_operator is None and gaussian_convolution is None:
        if u is None:
            raise ValueError("RSI convolution requires a noise kernel")
        initial_square = square_tt(template)
        if pre_conv_max_rank is not None:
            initial_square = round_tt(initial_square, max_rank=pre_conv_max_rank)
        initial_result = convolve_tt(initial_square, u, method="rsi", max_rank=rsi_max_rank,
                                     sketch_size=rsi_sketch_size, validation_samples=0)
        rsi_pivot_rows = tuple(stage["pivot_rows"] for stage in initial_result.diagnostics["stages"])

    def loss_fn(params: TTParams, y: jnp.ndarray) -> jnp.ndarray:
        return nll_noisy_tt(
            params, template, u, y, noise_std=noise_std, pre_conv_max_rank=pre_conv_max_rank,
            reg_strength=reg_strength, regularizer=regularizer, per_axis_target_budget=per_axis_target_budget,
            convolution_max_rank=convolution_max_rank,
            convolution_tol=convolution_tol,
            convolution_method=convolution_method, rsi_max_rank=rsi_max_rank,
            rsi_sketch_size=rsi_sketch_size,
            rsi_pivot_rows=rsi_pivot_rows,
            gaussian_convolution=gaussian_convolution, forward_operator=forward_operator,
        )

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(params: TTParams, opt_state, y: jnp.ndarray):
        loss, grads = loss_and_grad(params, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return step


def fit_noisy_tt(
    init_g: ChebTT,
    u: ChebTT | None,
    noisy_y: jnp.ndarray,
    *,
    n_steps: int = 2000,
    lr: float = 3e-3,
    batch_size: int | None = None,
    seed: int = 0,
    log_every: int = 200,
    max_grad_norm: float = 1.0,
    noise_std: jnp.ndarray | None = None,
    reg_strength: float = 0.0,
    regularizer=None,
    per_axis_target_budget: float | None = None,
    pre_conv_max_rank: int | None = None,
    convolution_max_rank: int | None = None,
    convolution_tol: float = 0.0,
    convolution_method: str = "rsi",
    rsi_max_rank: int | None = None,
    rsi_sketch_size: int | None = None,
    gaussian_convolution: GaussianConvolution | None = None,
    forward_operator: ConvolutionOperator | None = None,
) -> tuple[ChebTT, list[float]]:
    """Minimize the NLL of the noisy density on noisy observations `noisy_y` (n, d).

    ``regularizer`` may be a callable of the amplitude TT, for example
    ``AmplitudeSobolevPenalty.from_tt(init_g, order=2)``. Its contribution
    is multiplied by ``reg_strength``. This measures the normalized
    represented function and requires no ``noise_std``. If omitted, the
    legacy core coefficient penalty is retained for reproducibility; that
    penalty is not invariant to TT gauge or amplitude rescaling.

    ``forward_operator=TTConvolution.from_tt(init_g, u)`` uses the direct
    contraction for any prescribed TT kernel, without a Gaussian assumption.
    It integrates the represented polynomials to floating-point accuracy.

    ``gaussian_convolution`` selects the direct Gaussian forward operator,
    with ``u=None`` allowed. Construct it once using
    ``GaussianConvolution.from_tt(init_g, noise_std)``. It retains the same
    NLL and exact normalizer, uses controlled quadrature, and accepts noisy
    observations outside the signal box. Rank truncation is incompatible.

    With a spectral kernel, training uses RSI convolution by default. The
    initial model selects interpolation rows before JAX compilation and keeps
    them fixed during optimization. ``rsi_max_rank`` caps the output TT rank.
    ``rsi_sketch_size`` controls the number of random suffix projections.
    Use the same sketch size when selecting and reusing interpolation rows.
    Set ``convolution_method="naive"`` for exact spectral convolution.

    Without a custom regularizer, ``reg_strength`` > 0 adds the legacy
    coefficient penalty from ``noisy_regularization.py`` (requires
    ``noise_std``). ``per_axis_target_budget``, if given, normalizes each
    mode's own penalty weight independently (pass ``reg_strength`` as the
    dimensionless ``reg_lambda`` in this case) -- see
    ``noisy_regularization.calibrated_reg_strength``'s docstring for when
    this matters (heterogeneous per-axis domain/noise_std) versus a single
    pre-calibrated ``reg_strength`` (homogeneous axes). Every point of
    ``noisy_y`` must lie inside ``init_g``'s own domain on the legacy
    spectral path (there is no
    separate ``output_domain`` here -- see ``nll_noisy_tt``'s docstring for
    why).

    ``pre_conv_max_rank`` rounds ``g^2`` down to this rank before convolving
    with ``u`` on EVERY step -- this is what actually cuts the per-step
    cost (square_tt gives g^2 an exact but often unnecessarily large rank
    of ``rank(g)^2``; convolve_tt's cost scales quadratically in that rank).
    ``convolution_max_rank`` only rounds the already-computed convolution
    output, so it does not speed up training at all -- see
    ``log_density_noisy_tt``'s docstring for the full explanation of why
    these are different levers.
    """
    if convolution_tol:
        raise ValueError(
            "convolution_tol is intended for one-off round_tt/log-density evaluation; "
            "use convolution_max_rank for JAX-jitted training"
        )
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
    params = tt_params(init_g)
    opt_state = optimizer.init(params)
    step = make_noisy_train_step(
        init_g, u, optimizer, noise_std=noise_std, reg_strength=reg_strength, regularizer=regularizer,
        per_axis_target_budget=per_axis_target_budget, pre_conv_max_rank=pre_conv_max_rank,
        convolution_max_rank=convolution_max_rank, convolution_tol=convolution_tol,
        convolution_method=convolution_method, rsi_max_rank=rsi_max_rank,
        rsi_sketch_size=rsi_sketch_size,
        gaussian_convolution=gaussian_convolution, forward_operator=forward_operator,
    )

    n = noisy_y.shape[0]
    key = jax.random.PRNGKey(seed)
    history: list[float] = []
    for i in range(n_steps):
        if batch_size is None or batch_size >= n:
            batch = noisy_y
        else:
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (batch_size,), 0, n)
            batch = noisy_y[idx]
        params, opt_state, loss = step(params, opt_state, batch)
        history.append(float(loss))
        if log_every and (i % log_every == 0 or i == n_steps - 1):
            if reg_strength:
                # Breakdown only computed at log time (forward-only, cheap,
                # infrequent) -- `loss` itself is nll + reg_strength*penalty
                # (see nll_noisy_tt), so printing just `loss` alone hides
                # whether the actual data-fit term is improving or the
                # penalty term is moving it around; see this print's use in
                # scripts/gl_gpu_noisy.py for why that distinction mattered
                # in practice (the two can move in opposite directions).
                g = rebuild_tt(init_g, params)
                raw_nll = -float(jnp.mean(log_density_noisy_tt(
                    g, u, batch, pre_conv_max_rank=pre_conv_max_rank,
                    convolution_max_rank=convolution_max_rank,
                    gaussian_convolution=gaussian_convolution, forward_operator=forward_operator,
                )))
                penalty_term = reg_strength * float(
                    regularizer(g) if regularizer is not None else
                    tikhonov_penalty_tt(square_tt(g), noise_std, per_axis_target_budget=per_axis_target_budget)
                )
                if (convolution_method == "rsi" and forward_operator is None
                        and gaussian_convolution is None):
                    print(
                        f"[fit_noisy_tt] step {i:5d}  RSI objective {float(loss): .6f}  "
                        f"exact spectral nll {raw_nll: .6f}  penalty {penalty_term: .6f}"
                    )
                else:
                    print(
                        f"[fit_noisy_tt] step {i:5d}  nll+penalty(minimized) {float(loss): .6f}  "
                        f"= nll {raw_nll: .6f}  +  penalty {penalty_term: .6f}"
                    )
            else:
                print(f"[fit_noisy_tt] step {i:5d}  nll {float(loss): .6f}")

    fitted = rebuild_tt(init_g, params)
    return fitted, history

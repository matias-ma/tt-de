"""Build a prescribed, known noise kernel U in TT/Tucker Chebyshev form.

U is fixed (not trained) -- it represents the known measurement-noise
distribution. Its declared domain must be wide enough around the signal's
domain that convolve_tt/convolve_tucker's fixed-integration-window safety
condition holds (see chebdCUR's `_convolution_safe_output_domain`); as a
rule of thumb this means U's domain should span roughly the signal's own
domain width plus a generous multiple of the noise's own scale.
"""

from __future__ import annotations

import jax.numpy as jnp
from chebttge import ChebTTOptions, tt_qtensge
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from chebtuckerge import ChebTuckerOptions, tucker_qtensge
from chebtuckerge.chebtucker import ChebTucker


def gaussian_noise_log_density(cov: jnp.ndarray):
    """log density of N(0, cov), batched over x of shape (n, d)."""
    inv_cov = jnp.linalg.inv(cov)
    log_det = jnp.linalg.slogdet(cov)[1]
    d = cov.shape[0]

    def log_p(x):
        quad = jnp.einsum("...i,ij,...j->...", x, inv_cov, x)
        return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + quad)

    return log_p


def gaussian_noise_density(cov: jnp.ndarray):
    log_p = gaussian_noise_log_density(cov)

    def p(x):
        return jnp.exp(log_p(x))

    return p


def gaussian_noise_domain(
    cov: jnp.ndarray,
    signal_domain: jnp.ndarray,
    margin_sigmas: float = 4.0,
    output_domain: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """A per-mode noise-kernel domain wide enough for convolve_tt/tucker's
    fixed-window safety condition ``a_u + b_f <= a_y`` and ``b_u + a_f >=
    b_y`` (see chebdCUR's ``_convolution_safe_output_domain``), where
    ``[a_f, b_f]`` is the *signal*'s own domain and ``[a_y, b_y]`` is the
    domain the convolved output ``q`` must be evaluable on.

    If ``output_domain`` is None (the common case: the noisy density is
    wanted on the same domain as the clean signal), this reduces to the
    original, deliberately generous rule of thumb -- extend the signal
    domain by its own width on each side, plus a `margin_sigmas * std`
    buffer -- kept as-is for exact backward compatibility.

    If ``output_domain`` is given (the signal's own domain is tighter than
    where the noisy *observations* actually land -- e.g. the true clean
    signal lives on [-2, 2] but noisy samples spread out to [-5, 5]), the
    minimal-sufficient kernel domain is derived directly from the safety
    condition above: this is generally *narrower* than the output-equals-
    signal case (required kernel width = output width + signal width,
    versus 2x signal width when signal and output coincide but the signal
    domain was forced wide to begin with), which also lowers the kernel's
    required Chebyshev degree for the same physical noise scale.
    """
    signal_domain = jnp.asarray(signal_domain, dtype=jnp.float64)
    std = jnp.sqrt(jnp.diag(cov))
    if output_domain is None:
        width = signal_domain[:, 1] - signal_domain[:, 0]
        lo = signal_domain[:, 0] - width - margin_sigmas * std
        hi = signal_domain[:, 1] + width + margin_sigmas * std
    else:
        output_domain = jnp.asarray(output_domain, dtype=jnp.float64)
        lo = output_domain[:, 0] - signal_domain[:, 1] - margin_sigmas * std
        hi = output_domain[:, 1] - signal_domain[:, 0] + margin_sigmas * std
    return jnp.stack([lo, hi], axis=1)


def build_diagonal_gaussian_noise_kernel_tt(
    variances: jnp.ndarray,
    signal_domain: jnp.ndarray,
    *,
    margin_sigmas: float = 4.0,
    n_coeffs: tuple[int, ...] = (17, 33, 65, 129, 257),
    n_starts: int = 16,
    gd_steps: int = 0,
    random_seed: int = 0,
) -> ChebTT:
    """Build an axis-aligned (diagonal-covariance) Gaussian noise kernel as
    an exactly rank-1 ChebTT, by fitting each axis's own 1D marginal
    Gaussian independently and composing the results -- no multi-dimensional
    cross-approximation pivot search at all.

    Why this exists (not just a simplification for convenience): convolve_tt's
    exact fixed-window convolution requires U's domain to be roughly 3x the
    *signal*'s domain width per mode (see ``gaussian_noise_domain``),
    regardless of the noise's own physical scale. When the noise is much
    narrower than the signal's spread (the common, realistic case -- e.g. a
    small measurement-noise kernel on a broad true distribution), the actual
    Gaussian density can underflow to an exact, bit-for-bit 0.0 over the vast
    majority of that required domain (this is not a resolution issue -- it
    happened on real data here with noise std ~2000 against a ~150000-wide
    required domain, i.e. tens of standard deviations out). Multi-dimensional
    cross-approximation's pivot search compares candidate-point magnitudes to
    decide on structure, so when the meaningful region occupies a
    vanishingly small fraction of the search volume (~(std/domain)^d), no
    practical number of candidates finds it by chance, and the search
    silently returns a degenerate constant (rank 1, degree 1) fit -- more
    n_starts does not fix this, since it's not a resolution shortfall, it's a
    needle-in-a-haystack search failure.

    Building each 1D marginal separately sidesteps the problem entirely: a
    single-mode Chebyshev fit only ever needs to resolve one smooth,
    well-behaved univariate function (exactly what this library already does
    robustly at similarly large domain-to-scale ratios elsewhere in this
    project), and a product of 1D functions has an a priori, search-free
    rank-1 decomposition -- the per-mode cores simply *are* the answer, no
    pivoting needed. This ignores any off-diagonal covariance; for the
    cryo-EM dataset this project was built against, the true correlations
    are weak (<0.25) and this was used as an approximation. For a fitted
    full-covariance Gaussian, use
    ``ftde.particle_convolution.fit_gaussian_kernel`` and
    ``TTConvolution.from_tt(..., zero_extension=True)``; the compact kernel
    no longer needs to span the entire fixed integration window.
    """
    variances = jnp.asarray(variances, dtype=jnp.float64)
    d = variances.shape[0]
    cov_diag = jnp.diag(variances)
    domain = gaussian_noise_domain(cov_diag, signal_domain, margin_sigmas=margin_sigmas)

    cores = []
    for mode in range(d):
        std = jnp.sqrt(variances[mode])

        def f(x, std=std):
            x0 = x[..., 0]
            return jnp.exp(-0.5 * (x0 / std) ** 2) / (std * jnp.sqrt(2.0 * jnp.pi))

        opts = ChebTTOptions(
            domain=domain[mode : mode + 1],
            max_rank=1,
            core_representation="coeff_tensor",
            coeff_core_n_values=n_coeffs,
            n_starts=n_starts,
            gd_steps=gd_steps,
            random_seed=random_seed + mode,
        )
        marginal_tt = tt_qtensge(f, opts, d=1)
        cores.append(marginal_tt.cores[0])

    return ChebTT(cores=tuple(cores), domain=domain, ranks=(1,) * (d - 1))


def build_diagonal_gaussian_noise_kernel_tucker(
    variances: jnp.ndarray,
    signal_domain: jnp.ndarray,
    *,
    margin_sigmas: float = 4.0,
    n_coeffs: tuple[int, ...] = (17, 33, 65, 129, 257),
    n_starts: int = 16,
    gd_steps: int = 0,
    random_seed: int = 0,
) -> ChebTucker:
    """Tucker analogue of ``build_diagonal_gaussian_noise_kernel_tt`` -- see
    its docstring for why this per-mode, search-free construction is needed.
    """
    from chebtuckerge.factors import ChebyshevCoeffFactor

    tt = build_diagonal_gaussian_noise_kernel_tt(
        variances, signal_domain, margin_sigmas=margin_sigmas, n_coeffs=n_coeffs,
        n_starts=n_starts, gd_steps=gd_steps, random_seed=random_seed,
    )
    d = len(tt.cores)
    factors = tuple(
        ChebyshevCoeffFactor(core.coeffs[0], domain=core.domain) for core in tt.cores
    )
    core_tensor = jnp.ones((1,) * d, dtype=jnp.float64)
    return ChebTucker(core=core_tensor, factors=factors, domain=tt.domain)


def build_gaussian_mixture_noise_kernel_tt(
    component_stds: jnp.ndarray,
    weights: jnp.ndarray,
    signal_domain: jnp.ndarray,
    d: int,
    *,
    margin_sigmas: float = 4.0,
    output_domain: jnp.ndarray | None = None,
    n_coeffs: int | None = None,
    n_starts: int = 16,
    gd_steps: int = 0,
    random_seed: int = 0,
) -> ChebTT:
    """An axis-isotropic K-component Gaussian-mixture noise kernel
    ``U(z) = sum_k weights[k] * N(z; 0, component_stds[k]^2 * I_d)``, built
    as an exactly rank-K ChebTT.

    Each component's per-axis 1D marginal is fit independently (the same
    "no multi-dim cross-approximation" trick as
    ``build_diagonal_gaussian_noise_kernel_tt``, and for the same reason:
    a multi-dim pivot search can silently degenerate when the noise is
    narrow relative to the ~3x-signal-domain-wide window ``convolve_tt``
    requires). The K independent rank-1 terms are then combined into a
    single rank-K ChebTT via the standard "diagonal core" embedding of a
    sum of rank-1 terms -- no generic TT-addition operator is needed: the
    first core places each component's own mixture weight, each middle
    core is block-diagonal across the K components (off-diagonal blocks
    zero), and the last core closes each component off. A fixed (not
    ladder) ``n_coeffs`` is used so every component's core has identical
    shape, which the block-diagonal embedding requires.
    """
    component_stds = jnp.asarray(component_stds, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if component_stds.ndim != 1 or weights.ndim != 1 or component_stds.shape != weights.shape:
        raise ValueError("component_stds and weights must be one-dimensional arrays of equal length")
    signal_domain = jnp.asarray(signal_domain, dtype=jnp.float64)
    if signal_domain.shape != (d, 2):
        raise ValueError(f"signal_domain must have shape ({d}, 2)")
    if component_stds.shape[0] == 0:
        raise ValueError("at least one Gaussian-mixture component is required")
    if bool(jnp.any(component_stds <= 0)):
        raise ValueError("component_stds must be strictly positive")
    if bool(jnp.any(weights < 0)) or not bool(jnp.isfinite(jnp.sum(weights))) or float(jnp.sum(weights)) <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    weights = weights / jnp.sum(weights)
    k_components = component_stds.shape[0]
    max_var = float(jnp.max(component_stds) ** 2)
    domain = gaussian_noise_domain(
        max_var * jnp.eye(d), signal_domain, margin_sigmas=margin_sigmas, output_domain=output_domain,
    )

    # A narrow component must be resolved on the *expanded* kernel domain,
    # whose width is generally several times the signal width.  A fixed
    # degree such as 33 can therefore produce a kernel that has the right
    # peak but the wrong integral/convolution.  The estimate below is a
    # conservative resolution rule for exp(-(x/std)^2/2); callers can still
    # request a larger or smaller fixed degree explicitly.
    if n_coeffs is None:
        domain_width = float(jnp.max(domain[:, 1] - domain[:, 0]))
        min_std = float(jnp.min(component_stds))
        n_coeffs = max(33, 2 * int(jnp.ceil(domain_width / (2.0 * min_std))) + 9)
        if n_coeffs % 2 == 0:
            n_coeffs += 1

    # mode_coeffs[mode][k] -> (n_coeffs,) 1D Chebyshev coefficients of
    # component k's Gaussian marginal along that mode.
    mode_coeffs: list[list[jnp.ndarray]] = []
    for mode in range(d):
        per_component = []
        for k in range(k_components):
            std = component_stds[k]

            def f(x, std=std):
                x0 = x[..., 0]
                return jnp.exp(-0.5 * (x0 / std) ** 2) / (std * jnp.sqrt(2.0 * jnp.pi))

            opts = ChebTTOptions(
                domain=domain[mode : mode + 1],
                max_rank=1,
                core_representation="coeff_tensor",
                coeff_core_n_values=(n_coeffs,),
                n_starts=n_starts,
                gd_steps=gd_steps,
                random_seed=random_seed + mode * k_components + k,
            )
            marginal_tt = tt_qtensge(f, opts, d=1)
            per_component.append(marginal_tt.cores[0].coeffs[0, 0])
        mode_coeffs.append(per_component)

    if d == 1:
        coeffs = sum(weights[k] * mode_coeffs[0][k] for k in range(k_components))
        return ChebTT(
            cores=(ChebyshevCoeffCore(coeffs[None, None, :], domain=tuple(domain[0])),),
            domain=domain,
            ranks=(),
        )

    cores = []
    for mode in range(d):
        if mode == 0:
            core = jnp.stack(
                [weights[k] * mode_coeffs[mode][k] for k in range(k_components)], axis=0
            )[None, :, :]  # (1, K, n_coeffs)
        elif mode == d - 1:
            core = jnp.stack([mode_coeffs[mode][k] for k in range(k_components)], axis=0)[
                :, None, :
            ]  # (K, 1, n_coeffs)
        else:
            core = jnp.zeros((k_components, k_components, n_coeffs), dtype=jnp.float64)
            for k in range(k_components):
                core = core.at[k, k, :].set(mode_coeffs[mode][k])
        cores.append(ChebyshevCoeffCore(core, domain=tuple(domain[mode])))

    ranks = tuple(k_components for _ in range(d - 1))
    return ChebTT(cores=tuple(cores), domain=domain, ranks=ranks)


def build_gaussian_noise_kernel_tt(
    cov: jnp.ndarray,
    signal_domain: jnp.ndarray,
    *,
    margin_sigmas: float = 4.0,
    max_rank: int = 8,
    n_coeffs: tuple[int, ...] = (17, 33, 65, 129),
    n_starts: int = 64,
    gd_steps: int = 20,
    random_seed: int = 0,
) -> ChebTT:
    cov = jnp.asarray(cov, dtype=jnp.float64)
    d = cov.shape[0]
    domain = gaussian_noise_domain(cov, signal_domain, margin_sigmas=margin_sigmas)
    opts = ChebTTOptions(
        domain=domain,
        max_rank=max_rank,
        core_representation="coeff_tensor",
        coeff_core_n_values=n_coeffs,
        n_starts=n_starts,
        gd_steps=gd_steps,
        random_seed=random_seed,
    )
    return tt_qtensge(gaussian_noise_density(cov), opts, d=d)


def build_pairwise_correlated_noise_kernel_tt(
    std: float,
    rho: float,
    signal_domain: jnp.ndarray,
    d: int,
    *,
    margin_sigmas: float = 4.0,
    max_rank: int = 3,
    n_coeffs: tuple[int, ...] = (17, 33, 65, 129, 257),
    n_starts: int = 64,
    gd_steps: int = 10,
    random_seed: int = 0,
) -> ChebTT:
    """A genuinely non-diagonal noise kernel: independent adjacent pairs of
    coordinates -- (0,1), (2,3), ... (one leftover single-axis marginal if
    ``d`` is odd) -- each pair correlated with correlation ``rho`` (2x2
    covariance ``std^2 * [[1, rho], [rho, 1]]``), built via an honest 2D
    cross-approximation per pair and assembled into the full d-dimensional
    TT by simple core concatenation. This is exact, not an approximation
    beyond each pair's own 2D fit: independent blocks of a product density
    are exactly a TT whose bond rank AT each block boundary is 1 (every
    ChebTT's own boundary ranks are already 1 by construction), so
    concatenating each block's cores with a rank-1 bond in between is just
    literally what that TT looks like -- no extra approximation step.

    WHY NOT a single full d-dimensional cross-approximation of a genuinely
    correlated d-dim Gaussian noise kernel: confirmed empirically (see this
    project's own history) that it silently degenerates to a trivial
    rank-1/degree-1 fit here, which then crashes downstream in
    convolve_tt -- the same "needle in a haystack" pivot-search failure
    documented in ``build_diagonal_gaussian_noise_kernel_tt``'s docstring:
    convolve_tt's fixed-window safety condition forces the kernel's
    declared domain to be roughly 2x the signal domain's own width
    regardless of the noise's physical scale, so a genuinely narrow noise
    std occupies a vanishingly small fraction of the d-dimensional search
    volume once d is more than ~2-3. A single 2D cross-approximation
    (confirmed empirically here to succeed reliably where the full d-dim
    case fails) doesn't suffer nearly as badly -- the same volume-fraction
    curse is far gentler in 2D -- so restricting genuine correlation to
    adjacent pairs keeps this robust while still producing an honestly
    non-diagonal covariance (as opposed to the fully independent-per-axis
    ``build_diagonal_gaussian_noise_kernel_tt``).

    ``n_coeffs``'s default max degree (257, not the more usual 129) is
    required, not cosmetic: at 129 the same wide-domain/narrow-scale ratio
    that breaks the full d-dim search also leaves small residual ringing in
    the 2D fit (confirmed empirically: individual density values as wrong
    as -2.6e-7 at degree 129, versus ~1e-14 -- genuine floating-point
    roundoff -- at 257). ``log_density_noisy_tt``'s eps-floor turns any
    negative value into an enormous, batch-dominating NLL, so this isn't a
    cosmetic accuracy nicety -- degree 129 here silently produces training
    losses off by many orders of magnitude.
    """
    signal_domain = jnp.asarray(signal_domain, dtype=jnp.float64)
    cov_pair = (std**2) * jnp.array([[1.0, rho], [rho, 1.0]])

    cores: list = []
    ranks: list[int] = []
    mode = 0
    block_idx = 0
    while mode < d:
        if mode + 1 < d:
            pair_tt = build_gaussian_noise_kernel_tt(
                cov_pair, signal_domain[mode : mode + 2], margin_sigmas=margin_sigmas,
                max_rank=max_rank, n_coeffs=n_coeffs, n_starts=n_starts, gd_steps=gd_steps,
                random_seed=random_seed + block_idx,
            )
            cores.extend(pair_tt.cores)
            ranks.append(pair_tt.ranks[0])
            if mode + 2 < d:
                ranks.append(1)  # rank-1 bond between this independent block and the next
            mode += 2
        else:
            # odd trailing coordinate: its own independent 1D marginal, same
            # search-free per-axis construction as build_diagonal_*.
            std_arr = jnp.asarray(std, dtype=jnp.float64)
            domain_1d = gaussian_noise_domain(
                (std_arr**2)[None, None], signal_domain[mode : mode + 1], margin_sigmas=margin_sigmas,
            )

            def f(x, std=std):
                x0 = x[..., 0]
                return jnp.exp(-0.5 * (x0 / std) ** 2) / (std * jnp.sqrt(2.0 * jnp.pi))

            opts = ChebTTOptions(
                domain=domain_1d, max_rank=1, core_representation="coeff_tensor",
                coeff_core_n_values=n_coeffs, n_starts=n_starts, gd_steps=gd_steps,
                random_seed=random_seed + block_idx,
            )
            marginal_tt = tt_qtensge(f, opts, d=1)
            cores.append(marginal_tt.cores[0])
            mode += 1
        block_idx += 1

    domain = jnp.array([core.domain for core in cores], dtype=jnp.float64)
    return ChebTT(cores=tuple(cores), domain=domain, ranks=tuple(ranks))


def build_gaussian_noise_kernel_tucker(
    cov: jnp.ndarray,
    signal_domain: jnp.ndarray,
    *,
    margin_sigmas: float = 4.0,
    max_rank: int = 6,
    n_coeffs: tuple[int, ...] = (17, 33, 65),
    n_starts: int = 64,
    gd_steps: int = 20,
    random_seed: int = 0,
) -> ChebTucker:
    cov = jnp.asarray(cov, dtype=jnp.float64)
    d = cov.shape[0]
    domain = gaussian_noise_domain(cov, signal_domain, margin_sigmas=margin_sigmas)
    opts = ChebTuckerOptions(
        domain=domain,
        max_rank=max_rank,
        core_representation="coeff_tensor",
        coeff_core_n_values=n_coeffs,
        n_starts=n_starts,
        gd_steps=gd_steps,
        random_seed=random_seed,
    )
    return tucker_qtensge(gaussian_noise_density(cov), opts, d=d)


def build_paired_gaussian_mixture_noise_kernel_tt(
    covariances,
    weights,
    signal_domain,
    *,
    margin_sigmas: float = 4.0,
    n_coeffs: int = 513,
    tolerance: float = 1e-13,
) -> ChebTT:
    """``U(z) = sum_c weights[c] N(z; 0, covariances[c])`` with 2x2-block covariances.

    Every component may be arbitrary within adjacent coordinate pairs
    (0, 1), (2, 3), ... and must be zero across pairs (``d`` even). Each
    pair's 2D Gaussian is interpolated on a tensor Chebyshev grid, and its
    coefficient matrix is split by a truncated SVD; no cross approximation
    or pivot search is involved. All components share one kernel domain, so
    the mixture is the block-diagonal TT sum: bonds inside a pair have
    rank sum_c rank_c, and bonds between pairs have rank equal to the
    number of components. The represented pair densities are checked against
    the analytic Gaussians at random points.
    """
    import numpy as np
    from scipy.fft import dctn

    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if covariances.ndim != 3 or covariances.shape[1] != covariances.shape[2]:
        raise ValueError('covariances must have shape (components, d, d)')
    n_comp, d, _ = covariances.shape
    if d % 2 or weights.shape != (n_comp,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('need an even dimension and one nonnegative weight per component')
    block = np.kron(np.eye(d // 2), np.ones((2, 2))) > 0
    if np.any(covariances[:, ~block] != 0):
        raise ValueError('covariances must vanish outside adjacent coordinate pairs')
    for c in covariances:
        if np.any(np.linalg.eigvalsh(c) <= 0):
            raise ValueError('every component covariance must be positive definite')
    weights = weights / weights.sum()
    widest = np.diag(covariances.diagonal(axis1=1, axis2=2).max(0))
    domain = np.asarray(gaussian_noise_domain(jnp.asarray(widest), jnp.asarray(signal_domain),
                                              margin_sigmas=margin_sigmas))
    t = np.cos(np.pi * (np.arange(n_coeffs) + 0.5) / n_coeffs)
    rng = np.random.default_rng(0)
    # pieces[c][m] = (left (r, n), right (r, n)) for component c and pair m
    pieces = []
    for c, cov in enumerate(covariances):
        per_pair = []
        for m in range(d // 2):
            (a0, b0), (a1, b1) = domain[2 * m], domain[2 * m + 1]
            x = a0 + (t + 1) * (b0 - a0) / 2
            y = a1 + (t + 1) * (b1 - a1) / 2
            sub = cov[2 * m:2 * m + 2, 2 * m:2 * m + 2]
            inv, det = np.linalg.inv(sub), np.linalg.det(sub)

            def density(u, v, inv=inv, det=det):
                q = inv[0, 0] * u**2 + 2 * inv[0, 1] * u * v + inv[1, 1] * v**2
                return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))

            coeff = dctn(density(x[:, None], y[None, :]), type=2) / n_coeffs**2
            coeff[0, :] *= 0.5
            coeff[:, 0] *= 0.5
            u, s, vt = np.linalg.svd(coeff)
            r = max(1, int(np.sum(s > tolerance * s[0])))
            left, right = (u[:, :r] * s[:r]).T, vt[:r]
            probe = np.column_stack([rng.uniform(a0, b0, 4000), rng.uniform(a1, b1, 4000)])
            probe[:2000] = rng.multivariate_normal(np.zeros(2), sub, 2000)
            vx = np.polynomial.chebyshev.chebvander((2 * probe[:, 0] - a0 - b0) / (b0 - a0), n_coeffs - 1)
            vy = np.polynomial.chebyshev.chebvander((2 * probe[:, 1] - a1 - b1) / (b1 - a1), n_coeffs - 1)
            represented = np.sum((vx @ left.T) * (vy @ right.T), axis=1)
            analytic = density(probe[:, 0], probe[:, 1])
            if np.max(np.abs(represented - analytic)) > 1e-9 * density(0.0, 0.0):
                raise FloatingPointError('paired mixture kernel needs more Chebyshev modes')
            per_pair.append((left, right))
        pieces.append(per_pair)
    cores = []
    for m in range(d // 2):
        ranks = [pieces[c][m][0].shape[0] for c in range(n_comp)]
        offsets = np.concatenate([[0], np.cumsum(ranks)])
        first = np.zeros((1 if m == 0 else n_comp, offsets[-1], n_coeffs))
        second = np.zeros((offsets[-1], 1 if m == d // 2 - 1 else n_comp, n_coeffs))
        for c in range(n_comp):
            left, right = pieces[c][m]
            first[0 if m == 0 else c, offsets[c]:offsets[c + 1]] = left * (weights[c] if m == 0 else 1.0)
            second[offsets[c]:offsets[c + 1], 0 if m == d // 2 - 1 else c] = right
        cores += [ChebyshevCoeffCore(jnp.asarray(first), tuple(domain[2 * m])),
                  ChebyshevCoeffCore(jnp.asarray(second), tuple(domain[2 * m + 1]))]
    return ChebTT(cores=tuple(cores), domain=jnp.asarray(domain),
                  ranks=tuple(c.coeffs.shape[1] for c in cores[:-1]))

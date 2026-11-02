"""Rank truncation for coefficient-tensor Chebyshev tensor trains."""

from __future__ import annotations

import jax.numpy as jnp
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore


def round_tt(
    tt: ChebTT,
    *,
    max_rank: int | None = None,
    tol: float = 0.0,
) -> ChebTT:
    """Compress a coefficient-tensor TT with sequential SVDs.

    ``max_rank`` is an absolute cap on every internal rank. ``tol`` is a
    relative per-unfolding Frobenius tolerance used to discard small
    singular values. Supplying neither leaves the TT unchanged. The
    truncation is performed in coefficient space, so the tolerance is a
    practical spectral approximation criterion rather than an exact bound
    in the continuous Chebyshev Gram norm.

    A fixed ``max_rank`` is the intended use inside a JAX-jitted likelihood:
    all resulting core shapes remain static. Tolerance-only truncation is
    useful for one-off compression after fitting.
    """
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive")
    if tol < 0:
        raise ValueError("tol must be nonnegative")
    if max_rank is None and tol == 0:
        return tt
    if any(not isinstance(core, ChebyshevCoeffCore) for core in tt.cores):
        raise TypeError("round_tt requires coefficient-tensor Chebyshev cores")

    cores = [core.coeffs for core in tt.cores]
    rounded = []
    for mode in range(len(cores) - 1):
        core = cores[mode]
        r_left, r_right, n_coeffs = core.shape
        # The coefficient index is the physical mode index in the unfolding:
        # rows are (left-bond, Chebyshev-degree), columns are right-bond.
        matrix = core.transpose(0, 2, 1).reshape(r_left * n_coeffs, r_right)
        u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
        rank = s.shape[0]
        if tol:
            tail = jnp.cumsum(s[::-1] ** 2)[::-1] - s**2
            keep = jnp.sum(tail > (tol * jnp.linalg.norm(s)) ** 2) + 1
            rank = min(rank, int(keep))
        if max_rank is not None:
            rank = min(rank, max_rank)
        rank = max(1, rank)
        rounded.append(u[:, :rank].reshape(r_left, n_coeffs, rank).transpose(0, 2, 1))
        transfer = s[:rank, None] * vh[:rank, :]
        cores[mode + 1] = jnp.einsum("ab,bcn->acn", transfer, cores[mode + 1])

    rounded.append(cores[-1])
    new_cores = tuple(
        ChebyshevCoeffCore(coeffs, domain=tuple(core.domain))
        for coeffs, core in zip(rounded, tt.cores, strict=True)
    )
    return ChebTT(cores=new_cores, domain=tt.domain)

"""Exact pointwise (Hadamard) product of Chebyshev functional tensors.

Needed to materialize ``g(x)^2`` as an actual ``ChebTT``/``ChebTucker`` object
(so the existing, validated ``convolve_tt``/``convolve_tucker`` in chebdCUR
can be applied to it directly), as opposed to ``squared_integral_tt`` which
only ever produces the scalar ``integral g(x)^2 dx``.

Uses the exact linearization identity ``T_i(x) T_j(x) = (T_{i+j}(x) +
T_{|i-j|}(x)) / 2`` -- pure index bookkeeping, no monomial-basis conversion,
so (unlike a Taylor/derivative expansion) this stays exact and stable at any
degree.
"""

from __future__ import annotations

from functools import cache

import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from chebtuckerge.chebtucker import ChebTucker
from chebtuckerge.factors import ChebyshevCoeffFactor


@cache
def _cheb_product_tensor(n_f: int, n_g: int) -> np.ndarray:
    """``T[k, i, j]`` such that ``T_i * T_j = sum_k T[k, i, j] * T_k``."""
    n_h = n_f + n_g - 1
    tensor = np.zeros((n_h, n_f, n_g))
    ii, jj = np.meshgrid(np.arange(n_f), np.arange(n_g), indexing="ij")
    k_sum = ii + jj
    k_diff = np.abs(ii - jj)
    np.add.at(tensor, (k_sum.ravel(), ii.ravel(), jj.ravel()), 0.5)
    np.add.at(tensor, (k_diff.ravel(), ii.ravel(), jj.ravel()), 0.5)
    return tensor


def chebyshev_product_core(core_f: ChebyshevCoeffCore, core_g: ChebyshevCoeffCore) -> ChebyshevCoeffCore:
    """Pointwise product ``h(x) = core_f(x) * core_g(x)`` as a new coefficient core.

    Both cores must share the same domain (this is a product of two
    functions of the *same* variable, not a convolution). Result has
    Kronecker-combined matrix indices and degree ``n_f + n_g - 1``.
    """
    if not (
        abs(core_f.domain[0] - core_g.domain[0]) < 1e-9
        and abs(core_f.domain[1] - core_g.domain[1]) < 1e-9
    ):
        raise ValueError(
            f"chebyshev_product_core requires matching domains, got {core_f.domain} and {core_g.domain}"
        )
    n_f = core_f.coeffs.shape[-1]
    n_g = core_g.coeffs.shape[-1]
    tensor = jnp.asarray(_cheb_product_tensor(n_f, n_g))
    h_coeffs = jnp.einsum("kij,abi,cdj->acbdk", tensor, core_f.coeffs, core_g.coeffs)
    r_lf, r_rf = core_f.shape
    r_lg, r_rg = core_g.shape
    h_coeffs = h_coeffs.reshape(r_lf * r_lg, r_rf * r_rg, n_f + n_g - 1)
    return ChebyshevCoeffCore(h_coeffs, domain=core_f.domain)


def square_tt(g: ChebTT) -> ChebTT:
    """Return a new ChebTT representing ``g(x)^2`` (pointwise square).

    Mode-``k`` rank becomes ``rank_k(g)^2`` (Kronecker-structured), degree
    per mode roughly doubles. Domain is unchanged (same as ``g``'s).
    """
    new_cores = tuple(chebyshev_product_core(core, core) for core in g.cores)
    new_ranks = tuple(r * r for r in g.ranks)
    return ChebTT(cores=new_cores, domain=g.domain, ranks=new_ranks)


def _kron_core_tensor_self(core: jnp.ndarray, d: int) -> jnp.ndarray:
    outer = jnp.tensordot(core, core, axes=0)  # (*ranks, *ranks)
    perm = []
    for mode in range(d):
        perm.append(mode)
        perm.append(d + mode)
    outer = jnp.transpose(outer, perm)
    new_shape = tuple(core.shape[mode] ** 2 for mode in range(d))
    return outer.reshape(new_shape)


def square_tucker(g: ChebTucker) -> ChebTucker:
    """Return a new ChebTucker representing ``g(x)^2`` (pointwise square)."""
    d = len(g.factors)
    new_factors = tuple(
        ChebyshevCoeffFactor(
            chebyshev_product_core(factor.core, factor.core).coeffs[0],
            domain=factor.domain,
        )
        for factor in g.factors
    )
    new_core = _kron_core_tensor_self(g.core, d)
    return ChebTucker(core=new_core, factors=new_factors, domain=g.domain)

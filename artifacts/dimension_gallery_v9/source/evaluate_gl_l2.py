"""Post-fit integrated L2 error using deterministic GL transfer quadrature.

No fitted coefficients or checkpoints are chosen using these diagnostics.
Integrates q^2 and p*q separately, with p^2 from the GL partition function.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.special import roots_legendre

from ftde.ginzburg_landau import GL_BETA, GL_LAM, ginzburg_landau_log_normalizer
from ftde.tt_density import squared_integral_tt


def relative_l2_from_terms(q2, pq, p2):
    """Reject failed integrals instead of turning NaN into a zero error."""
    if not np.isfinite([q2, pq, p2]).all() or p2 <= 0:
        raise FloatingPointError('Nonfinite or invalid density-error integrals')
    squared = (q2-2*pq+p2)/p2
    if squared < -1e-10:
        raise FloatingPointError('Negative squared L2 beyond roundoff; refine/check integration')
    return float(np.sqrt(max(0., squared)))


def integrated_terms(g, order, log_z):
    """Return integral q^2 and integral p*q for q=g^2/integral(g^2)."""
    if order < 2*max(c.coeffs.shape[-1] for c in g.cores)-1:
        raise ValueError('quadrature order must resolve the fourth power of the signal')
    d = len(g.cores)
    # Constant core scalings cancel in q, and avoid unnecessary range growth.
    cores = tuple(ChebyshevCoeffCore(c.coeffs/jnp.max(jnp.abs(c.coeffs)), c.domain)
                  for c in g.cores)
    scaled = ChebTT(cores=cores, domain=g.domain)
    log_norm = jnp.log(squared_integral_tt(scaled))
    t, w = roots_legendre(order)
    fourth = jnp.ones((1, 1, 1, 1))
    log_fourth = jnp.array(0.)
    cross = None
    log_cross = jnp.array(0.)
    previous = None
    edge_scale = GL_BETA*GL_LAM*(d+1)**2/2
    for core in cores:
        a, b = core.domain
        nodes = jnp.asarray(a+(b-a)*(t+1)/2)
        weights = jnp.asarray(w*(b-a)/2)
        values = core(nodes)
        # Four distinct signal bonds, never an r^8 transfer tensor.
        state = jnp.einsum('abcd,tai->tibcd', fourth, values)
        state = jnp.einsum('tibcd,tbj->tijcd', state, values)
        state = jnp.einsum('tijcd,tck->tijkd', state, values)
        state = jnp.einsum('tijkd,tdl->tijkl', state, values)
        fourth = jnp.einsum('t,tijkl->ijkl', weights, state)
        scale = jnp.max(jnp.abs(fourth))
        fourth = fourth/scale
        log_fourth = log_fourth+jnp.log(scale)
        site = jnp.exp(-GL_BETA*(1-nodes**2)**2/(4*GL_LAM))
        if cross is None:
            incoming = jnp.exp(-edge_scale*nodes**2)[:, None, None]
        else:
            edges = jnp.exp(-edge_scale*(previous[:, None]-nodes[None, :])**2)
            incoming = jnp.einsum('st,sab->tab', edges, cross)
        state = jnp.einsum('tab,tai->tib', incoming, values)
        cross = jnp.einsum('tib,tbj->tij', state, values)*(weights*site)[:, None, None]
        scale = jnp.max(jnp.abs(cross))
        cross = cross/scale
        log_cross = log_cross+jnp.log(scale)
        previous = nodes
    q2 = jnp.exp(log_fourth+jnp.log(fourth[0, 0, 0, 0])-2*log_norm)
    pq = jnp.exp(log_cross+jnp.log(jnp.sum(cross[:, 0, 0]*jnp.exp(-edge_scale*previous**2)))
                 -log_norm-log_z)
    return q2, pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.records:
        record = json.loads(path.read_text())
        saved = np.load(path.with_suffix('.npz'))
        domain = saved['domain']
        d = record['dim']
        log_z = ginzburg_landau_log_normalizer(d)
        p2 = float(np.exp(ginzburg_landau_log_normalizer(d, beta=2*GL_BETA)-2*log_z))
        results = {}
        for label in ('naive', 'deconvolved'):
            cores = tuple(ChebyshevCoeffCore(jnp.asarray(saved[f'{label}_{k}']), tuple(domain[k]))
                          for k in range(d))
            g = ChebTT(cores=cores, domain=jnp.asarray(domain))
            order = max(64, 2*max(c.coeffs.shape[-1] for c in cores)-1)
            q2, pq = map(float, jax.jit(lambda g=g, log_z=log_z, order=order: integrated_terms(g, order, log_z))())
            q2_fine, pq_fine = map(float, jax.jit(lambda g=g, log_z=log_z, order=order: integrated_terms(g, 2*order, log_z))())
            squared = (q2_fine-2*pq_fine+p2)/p2
            results[label] = {'relative_l2': relative_l2_from_terms(q2_fine, pq_fine, p2),
                              'relative_squared_l2': squared, 'integral_q_squared': q2_fine,
                              'integral_pq': pq_fine, 'integral_p_squared': p2,
                              'quadrature_relative_change': max(abs(q2-q2_fine), abs(pq-pq_fine))/p2,
                              'quadrature_orders': [order, 2*order]}
        destination = path.with_name(path.stem+'_integrated_l2.json')
        destination.write_text(json.dumps(results, indent=2)+'\n')
        print(path.stem, json.dumps(results), flush=True)
        jax.clear_caches()


if __name__ == '__main__':
    main()

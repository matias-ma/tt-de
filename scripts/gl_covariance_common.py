"""Prescribed anisotropic noise benchmark; no signal-density information."""

import json

import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.fft import dctn


def noise_spec(d, scale=1.):
    if d % 2:
        raise ValueError('This benchmark uses adjacent coordinate pairs')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('noise scale must be finite and positive')
    weights = np.array([.4, .35, .25])
    covs = np.zeros((3, d, d))
    for c, (short, long, angle) in enumerate([(.18, .50, .12), (.22, .55, 1.05), (.20, .45, -1.05)]):
        short, long = scale*short, scale*long
        for k in range(d//2):
            theta = angle+.19*k
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            covs[c, 2*k:2*k+2, 2*k:2*k+2] = rotation@np.diag([long**2, short**2])@rotation.T
    return weights, covs


def draw_noise(seed, n, weights, covs):
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(weights), n, p=weights)
    noise = np.empty((n, covs.shape[1]))
    for c in range(len(weights)):
        mask = labels == c
        noise[mask] = rng.multivariate_normal(np.zeros(covs.shape[1]), covs[c], mask.sum())
    return noise, labels


def noise_density(x, weights, covs):
    x = np.asarray(x)
    return sum(w*np.exp(-.5*np.einsum('...i,ij,...j->...', x, np.linalg.inv(c), x)
                        -.5*(x.shape[-1]*np.log(2*np.pi)+np.linalg.slogdet(c)[1]))
               for w, c in zip(weights, covs, strict=True))


def build_noise_tt(weights, covs, kernel_domain, *, n=385, max_rank=24, tol=1e-12):
    """Resolve each prescribed 2D covariance block by DCT and balanced SVD.

    Sum component TTs exactly by block-diagonal embedding. Only fixed known
    noise is approximated here, never the unknown signal or its square.
    """
    d = covs.shape[1]
    if d % 2 or covs.shape != (len(weights), d, d):
        raise ValueError('Noise builder requires even-dimensional covariance matrices')
    block_mask = np.arange(d)[:, None]//2 == np.arange(d)[None, :]//2
    if np.any(covs[:, ~block_mask] != 0):
        raise ValueError('This benchmark noise builder requires adjacent-pair block covariance; '
                         'represent a dense covariance kernel separately before using TTConvolution')
    components, singular_tails = [], []
    t = np.cos(np.pi*(np.arange(n)+.5)/n)
    for covariance in covs:
        cores = []
        for k in range(0, d, 2):
            box = kernel_domain[k:k+2]
            x = box[:, 0, None]+np.diff(box, axis=1)*(t[None, :]+1)/2
            xx, yy = np.meshgrid(x[0], x[1], indexing='ij')
            points = np.stack([xx, yy], axis=-1)
            vals = noise_density(points, [1.], covariance[None, k:k+2, k:k+2])
            coeff = dctn(vals, type=2)/n**2
            coeff[0, :] *= .5
            coeff[:, 0] *= .5
            left, s, right = np.linalg.svd(coeff, full_matrices=False)
            rank = min(max_rank, max(1, int(np.sum(s > tol*s[0]))))
            singular_tails.append(float(np.linalg.norm(s[rank:])/np.linalg.norm(s)))
            cores.extend([(left[:, :rank]*np.sqrt(s[:rank])).T[None, :, :],
                          (np.sqrt(s[:rank, None])*right[:rank])[:, None, :]])
        components.append(cores)
    merged = []
    for k in range(d):
        blocks = [c[k] for c in components]
        if k == 0:
            core = np.concatenate([w*b for w, b in zip(weights, blocks, strict=True)], axis=1)
        elif k == d-1:
            core = np.concatenate(blocks, axis=0)
        else:
            core = np.zeros((sum(b.shape[0] for b in blocks), sum(b.shape[1] for b in blocks), n))
            a = b = 0
            for block in blocks:
                ra, rb, _ = block.shape
                core[a:a+ra, b:b+rb] = block
                a, b = a+ra, b+rb
        merged.append(ChebyshevCoeffCore(jnp.asarray(core), tuple(kernel_domain[k])))
    return ChebTT(cores=tuple(merged), domain=jnp.asarray(kernel_domain)), singular_tails


def save_tt(path, g, **extra):
    np.savez(path, domain=np.asarray(g.domain), **{f'core_{k}': np.asarray(c.coeffs) for k, c in enumerate(g.cores)}, **extra)


def load_tt(path):
    saved = np.load(path)
    domain = saved['domain']
    return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(saved[f'core_{k}']), tuple(box))
                             for k, box in enumerate(domain)), domain=jnp.asarray(domain))


def save_record(path, record):
    """Keep run metadata inside one NPZ instead of loose JSON reports."""
    temporary = path.with_name(path.stem+'.build.npz')
    np.savez(temporary, metadata=np.asarray(json.dumps(record)))
    temporary.replace(path)


def load_record(path):
    with np.load(path) as saved:
        return json.loads(str(saved['metadata']))

"""Noisy density estimation on real cryo-EM latent-space data (recovar
embeddings), analogous to ref_notebooks/load_zs.ipynb but using ft-de's exact
convolution (no Monte Carlo, no n_components mixture) instead of TTDE's
MC-sampling deconvolution loss, and exact closed-form 2D marginals (no
grid/MC averaging over the other dimensions) instead of load_zs.ipynb's own
random-sampling marginal estimator.

Run scripts/cryoem_cache.py first (once) to extract small per-dimension .npz
caches from the ~900MB cryoem_data/cryoem_test.joblib. This script only reads
those. Figures are saved under figs/cryoem/, never displayed inline.

Usage: python scripts/cryoem_density.py <dim> <method>
  dim: 4, 10, or 20 (whichever cryoem_data/cache/dim{dim}.npz caches exist)
  method: tt or tucker
"""

import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cryoem_common import (
    FIGS_DIR,
    data_domain,
    fit_signal,
    kde_marginal_fn,
    load_marginal_cache,
    marginal_density_fn,
)


def run(
    dim: int, method: str, *, naive_steps: int | None = None, deconv_steps: int | None = None,
    rank: int | None = None, degree: int | None = None, reg_lambda: float = 0.0,
    convolution_backend: str = "spectral", quadrature_order: int | None = None,
    rsi_max_rank: int = 64, rsi_sketch_size: int | None = None,
):
    print(f"=== cryoem density estimation: dim={dim} method={method} rank={rank} degree={degree} "
          f"naive_steps={naive_steps} deconv_steps={deconv_steps} reg_lambda={reg_lambda} ===")
    points, cov_mean = load_marginal_cache(dim)
    signal_domain = data_domain(points)
    print(f"n_points={points.shape[0]}  domain widths={np.asarray(signal_domain[:, 1] - signal_domain[:, 0])}")

    points, naive_fit, _naive_hist, deconv_fit, _deconv_hist, domain = fit_signal(
        points, cov_mean, method, naive_steps=naive_steps, deconv_steps=deconv_steps,
        rank=rank, degree=degree, reg_lambda=reg_lambda,
        convolution_backend=convolution_backend, quadrature_order=quadrature_order,
        rsi_max_rank=rsi_max_rank, rsi_sketch_size=rsi_sketch_size,
    )

    dim_pairs = [(0, 1), (2, 3)]
    if dim >= 10:
        dim_pairs.append((8, 9))
    if dim >= 20:
        dim_pairs.append((0, 19))

    for (i, j) in dim_pairs:
        plot_pair(points, domain, naive_fit, deconv_fit, method, i, j, dim)


def plot_pair(points, domain, naive_fit, deconv_fit, method, i, j, dim, *,
              figure_dir=None, deconv_label=None):
    xmin, xmax = float(domain[i, 0]), float(domain[i, 1])
    ymin, ymax = float(domain[j, 0]), float(domain[j, 1])
    n_grid = 150
    xs = jnp.linspace(xmin, xmax, n_grid)
    ys = jnp.linspace(ymin, ymax, n_grid)
    XX, YY = jnp.meshgrid(xs, ys, indexing="xy")
    grid = jnp.stack([XX.ravel(), YY.ravel()], axis=1)

    m_deconv = marginal_density_fn(deconv_fit, method, (i, j))
    z_deconv = np.asarray(m_deconv(grid)).reshape(n_grid, n_grid)
    z_naive = None
    if naive_fit is not None:
        m_naive = marginal_density_fn(naive_fit, method, (i, j))
        z_naive = np.asarray(m_naive(grid)).reshape(n_grid, n_grid)

    kde_marg = kde_marginal_fn(points, (i, j))
    z_kde = np.asarray(kde_marg(grid)).reshape(n_grid, n_grid)

    pts = np.asarray(points)
    z_all = [z_deconv, z_kde] + ([z_naive] if z_naive is not None else [])
    vmin = min(z.min() for z in z_all)
    vmax = max(z.max() for z in z_all)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    axes[0].hist2d(
        pts[:, i], pts[:, j], bins=100, range=[[xmin, xmax], [ymin, ymax]],
        density=True, cmap="viridis", vmin=vmin, vmax=vmax,
    )
    axes[0].set(title="Observed noisy data", xlabel=f"x_{i}", ylabel=f"x_{j}", aspect="equal")

    axes[1].imshow(
        z_kde, extent=(xmin, xmax, ymin, ymax), origin="lower", aspect="equal",
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    axes[1].set(title="Gaussian KDE", xlabel=f"x_{i}", ylabel=f"x_{j}")

    if z_naive is None:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "naive fit skipped\n(--naive-steps 0)", ha="center", va="center", transform=axes[2].transAxes)
    else:
        axes[2].imshow(
            z_naive, extent=(xmin, xmax, ymin, ymax), origin="lower", aspect="equal",
            vmin=vmin, vmax=vmax, cmap="viridis",
        )
        axes[2].set(title=f"{method.upper()} without noise model", xlabel=f"x_{i}", ylabel=f"x_{j}")

    im_deconv = axes[3].imshow(
        z_deconv, extent=(xmin, xmax, ymin, ymax), origin="lower", aspect="equal",
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    axes[3].set(title=deconv_label or f"{method} with deconv", xlabel=f"x_{i}", ylabel=f"x_{j}")

    for ax in axes:
        ax.set(xlabel=f"$x_{i}$", ylabel=f"$x_{j}$")
    bar = fig.colorbar(im_deconv, ax=axes, orientation="vertical", aspect=12, pad=0.01, shrink=0.7)
    bar.set_label("Marginal density")
    fig.suptitle(f"Cryo-EM latent density  |  {dim} dimensions  |  $x_{i}$ and $x_{j}$",
                 fontsize=16, fontweight="bold")

    destination = FIGS_DIR if figure_dir is None else figure_dir
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / f"cryoem_dim{dim}_{method}_marg_{i}{j}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", nargs="?", type=int, default=4, help="4, 10, or 20 (whichever caches exist)")
    parser.add_argument("method", nargs="?", default="tt", choices=["tt", "tucker"])
    parser.add_argument("--naive-steps", type=int, default=None, help="0 to skip the naive (no-deconvolution) fit entirely")
    parser.add_argument("--deconv-steps", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--reg-lambda", type=float, default=0.0, help="Tikhonov regularization budget multiplier (0 disables, matching the old default)")
    parser.add_argument("--convolution-backend", choices=["spectral", "gaussian", "rsi"], default="spectral")
    parser.add_argument("--quadrature-order", type=int, default=None)
    parser.add_argument("--rsi-max-rank", type=int, default=64)
    parser.add_argument("--rsi-sketch-size", type=int, default=None)
    args = parser.parse_args()
    run(
        args.dim, args.method, naive_steps=args.naive_steps, deconv_steps=args.deconv_steps,
        rank=args.rank, degree=args.degree, reg_lambda=args.reg_lambda,
        convolution_backend=args.convolution_backend, quadrature_order=args.quadrature_order,
        rsi_max_rank=args.rsi_max_rank, rsi_sketch_size=args.rsi_sketch_size,
    )

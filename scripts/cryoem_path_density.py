""""Along path" density evaluation on real cryo-EM latent-space data,
analogous to ref_notebooks/load_zs_along_path.ipynb but using ft-de's own
squared-TT/Tucker density (naive and with our exact convolution/
deconvolution) instead of TTDE's MC-trained, n_components-mixture model.

The simulation assigns each (noisy) image to one of a fixed number of ground
truth conformational "path" states (simulation_info.joblib's
`image_assignment`). The true probability of each state is exactly known
(the fraction of images assigned to it); an anchor point z_i for state i is
the mean of the noisy latent points assigned to it. We evaluate each fitted
density at every anchor and compare its (renormalized) values against the
true per-state probabilities via relative L2 error -- the same accuracy
metric the reference notebook uses -- and compare against a naive Gaussian
KDE fit directly on the noisy points (no deconvolution, no tensor
decomposition at all) as a baseline.

Run at dim=4 (matching the reference notebook, which truncates the load_dim=10
embedding to 4 dims) and dim=10 (the full loaded embedding, no truncation) --
Tucker is skipped at dim=10, matching the precedent set in cryoem_density.py:
a Tucker core tensor has rank^d entries, so it doesn't scale to higher
dimension the way TT does.

Only noise1 is used: noise1 and noise3's simulation_info.joblib turned out to
have byte-identical `image_assignment` arrays (confirmed directly), so
running both is redundant -- see todo.txt.

Run scripts/cryoem_cache.py first (once) to build cryoem_data/cache/
path_noise1_dim{4,10}.npz from cryoem_data/cryoem_test.joblib +
cryoem_data/noise1/simulation_info.joblib. This script only reads those.

Usage: python scripts/cryoem_path_density.py [dim ...]
  dim: one or more of 4, 10 (default: both)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from cryoem_common import FIGS_DIR, density_fn, fit_signal, load_path_cache

from ftde.kde import make_gaussian_kde

NOISE = "noise1"
DEFAULT_DIMS = [4, 10]


def methods_for_dim(dim: int) -> list[str]:
    return ["tt"] if dim >= 10 else ["tt", "tucker"]


def true_path_probability(labels: np.ndarray):
    n_states = int(labels.max()) + 1
    prob = np.bincount(labels, minlength=n_states) / labels.shape[0]
    return prob, n_states


def path_anchors(points: np.ndarray, labels: np.ndarray, n_states: int):
    dim = points.shape[1]
    zs = np.zeros((n_states, dim))
    for i in range(n_states):
        zs[i] = points[labels == i].mean(axis=0)
    return zs


def relative_l2_error(prob_est: np.ndarray, prob_true: np.ndarray):
    """Same metric as the reference notebook: drop the two path endpoints
    (least-populated, noisiest states), renormalize the estimate over the
    rest to a probability vector, then take ||est - true|| / ||true||.
    """
    est = np.asarray(prob_est)[1:-1]
    true = np.asarray(prob_true)[1:-1]
    est = est / est.sum()
    return float(np.linalg.norm(est - true) / np.linalg.norm(true))


def run(
    dim: int, *, naive_steps: int | None = None, deconv_steps: int | None = None,
    rank: int | None = None, degree: int | None = None, reg_lambda: float = 0.0,
    deconv_lr: float | None = None,
    convolution_backend: str = "spectral", quadrature_order: int | None = None,
    rsi_max_rank: int = 64, rsi_sketch_size: int | None = None,
):
    print(f"=== cryoem along-path density evaluation: dim={dim} rank={rank} degree={degree} "
          f"reg_lambda={reg_lambda} deconv_lr={deconv_lr} ===")
    points, cov_mean, labels = load_path_cache(NOISE, dim)
    prob_true, n_states = true_path_probability(labels)
    zs = path_anchors(np.asarray(points), labels, n_states)
    print(f"n_points={points.shape[0]}  dim={points.shape[1]}  n_states={n_states}")

    results = {}

    kde = make_gaussian_kde(points)
    prob_kde = np.asarray(kde(zs))
    l2_kde = relative_l2_error(prob_kde, prob_true)
    results["kde (naive, no deconv)"] = (prob_kde, l2_kde)
    print(f"naive KDE                       l2_error={l2_kde:.4f}")

    for method in methods_for_dim(dim):
        _points, naive_fit, _naive_hist, deconv_fit, _deconv_hist, _domain = fit_signal(
            points, cov_mean, method, log_prefix=f"[dim{dim}/{method}] ",
            naive_steps=naive_steps, deconv_steps=deconv_steps, rank=rank, degree=degree,
            reg_lambda=reg_lambda, deconv_lr=deconv_lr,
            convolution_backend=convolution_backend if method == "tt" else "spectral",
            quadrature_order=quadrature_order, rsi_max_rank=rsi_max_rank, rsi_sketch_size=rsi_sketch_size,
        )

        if naive_fit is not None:
            prob_naive = np.asarray(density_fn(naive_fit, method)(zs))
            l2_naive = relative_l2_error(prob_naive, prob_true)
            results[f"{method} (naive, no deconv)"] = (prob_naive, l2_naive)
            print(f"{method:6s} naive (no deconv)        l2_error={l2_naive:.4f}")

        prob_deconv = np.asarray(density_fn(deconv_fit, method)(zs))
        l2_deconv = relative_l2_error(prob_deconv, prob_true)
        results[f"{method} (with deconv)"] = (prob_deconv, l2_deconv)
        print(f"{method:6s} with deconv l2_error={l2_deconv:.4f}")

    plot_path(dim, prob_true, results)
    return results


def plot_path(dim: int, prob_true: np.ndarray, results: dict):
    idx = np.arange(1, len(prob_true) - 1)
    prob_true_fixed = prob_true[1:-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idx, prob_true_fixed, label="true", color="black", linewidth=2)
    for label, (prob_est, l2) in results.items():
        est_fixed = prob_est[1:-1]
        est_fixed = est_fixed / est_fixed.sum()
        ax.plot(idx, est_fixed, label=f"{label} (l2={l2:.3f})", alpha=0.8)
    ax.set(xlabel="path state index", ylabel="probability", title=f"cryo-EM along-path density, dim={dim}")
    ax.legend()
    fig.tight_layout()

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGS_DIR / f"cryoem_path_dim{dim}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dims", nargs="*", type=int, default=None, help="one or more of 4, 10 (default: both)")
    parser.add_argument("--naive-steps", type=int, default=None, help="0 to skip the naive (no-deconvolution) fit entirely")
    parser.add_argument("--deconv-steps", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--reg-lambda", type=float, default=0.0, help="Tikhonov regularization budget multiplier (0 disables, matching the old default)")
    parser.add_argument("--deconv-lr", type=float, default=None, help="override the deconv-fit learning rate (default: same as naive fit's); lower (e.g. 1e-4) if using --reg-lambda > 0 at higher degree, see todo.txt")
    parser.add_argument("--convolution-backend", choices=["spectral", "gaussian", "rsi"], default="spectral")
    parser.add_argument("--quadrature-order", type=int, default=None)
    parser.add_argument("--rsi-max-rank", type=int, default=64)
    parser.add_argument("--rsi-sketch-size", type=int, default=None)
    args = parser.parse_args()
    dims = args.dims or DEFAULT_DIMS
    all_results = {
        dim: run(
            dim, naive_steps=args.naive_steps, deconv_steps=args.deconv_steps,
            rank=args.rank, degree=args.degree, reg_lambda=args.reg_lambda, deconv_lr=args.deconv_lr,
            convolution_backend=args.convolution_backend, quadrature_order=args.quadrature_order,
            rsi_max_rank=args.rsi_max_rank, rsi_sketch_size=args.rsi_sketch_size,
        )
        for dim in dims
    }

    print("\n=== summary (relative L2 error vs. true path probability) ===")
    for dim, results in all_results.items():
        for label, (_prob_est, l2) in results.items():
            print(f"dim={dim:<3d} {label:30s} l2_error={l2:.4f}")

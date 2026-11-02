"""One-time extraction: load the big cryoem_test.joblib once, pull out the
small per-dimension noisy points + averaged noise covariance (99% variance
filtering as in ref_notebooks/load_zs.ipynb), plus the "along path"
data (ref_notebooks/load_zs_along_path.ipynb: the load_dim=10 embedding
truncated to a chosen number of dims, paired with the noise dataset's
ground-truth conformational labels), cache each as a small .npz, then free
the ~1GB joblib object. cryoem_density.py / cryoem_path_density.py read only
the small cached files.

Path noise is marginalized after inverting the full precision matrix.
This corrects the historical notebook's conditional-covariance shortcut.

Only noise1's simulation_info.joblib is used: noise1 and noise3 turned out
to have byte-identical `image_assignment` arrays (confirmed directly), so
running both produced identical, redundant results -- see todo.txt.

Run once (or whenever cryoem_data/cryoem_test.joblib or the simulation_info
files change): `python scripts/cryoem_cache.py`.
"""

import gc
import sys
from pathlib import Path

import joblib
import numpy as np

DATA_DIR = Path("cryoem_data")
JOBLIB_PATH = DATA_DIR / "cryoem_test.joblib"
CACHE_DIR = DATA_DIR / "cache"

# Marginal-density caches (scripts/cryoem_density.py): each dim uses its own
# dedicated embedding (zdim_key f"{dim}_noreg"), matching load_zs.ipynb.
MARGINAL_DIMS = [4, 10, 20]

# "Along path" caches (scripts/cryoem_path_density.py): matches
# ref_notebooks/load_zs_along_path.ipynb, which always loads the load_dim=10
# embedding and truncates it to the first `dim` columns (a different,
# smaller-rank subspace of the *same* 10D embedding, not the separately
# fitted "4_noreg"/"10_noreg" embeddings used by the marginal-density caches
# above). dim=4 matches the reference notebook; dim=10 (no truncation, the
# full loaded embedding) pushes the same evaluation to higher dimension.
PATH_LOAD_DIM = 10
PATH_DIMS = [4, 10]
NOISE_DATASET = "noise1"


def marginal_noise_covariances(covariances_inv: np.ndarray, dim: int):
    """Marginalize Gaussian noise by slicing covariance, not precision.

    Slicing the precision before inversion instead gives conditional noise
    covariance and systematically understates uncertainty in retained axes.
    """
    return np.linalg.inv(covariances_inv)[..., :dim, :dim]


def _filter_by_variance(points: np.ndarray, covariances_inv: np.ndarray):
    """Invert per-point inverse-covariances and drop the top 1% by variance
    (trace of covariance) -- the same outlier rule used in both reference
    notebooks. Returns (points, covariances_mean, mask).
    """
    covariances = marginal_noise_covariances(covariances_inv, points.shape[1])
    variances = np.trace(covariances, axis1=1, axis2=2)
    threshold = np.percentile(variances, 99)
    mask = variances < threshold
    covariances_mean = np.mean(covariances[mask], axis=0)
    return points[mask], covariances_mean, mask


def extract_marginal_dim(embedding, dim):
    zdim_key = f"{dim}_noreg"
    points = np.asarray(embedding["zs"][zdim_key])
    covariances_inv = np.asarray(embedding["cov_zs"][zdim_key])
    points, cov_mean, _mask = _filter_by_variance(points, covariances_inv)
    return points, cov_mean


def extract_path_dataset(embedding, noise: str, dim: int, load_dim=PATH_LOAD_DIM):
    zdim_key = f"{load_dim}_noreg"
    points = np.asarray(embedding["zs"][zdim_key])[:, :dim]
    covariances_inv = np.asarray(embedding["cov_zs"][zdim_key])
    points, cov_mean, mask = _filter_by_variance(points, covariances_inv)

    sim_info = joblib.load(DATA_DIR / noise / "simulation_info.joblib")
    labels = np.asarray(sim_info["image_assignment"])[mask]
    return points, cov_mean, labels


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {JOBLIB_PATH} (this is ~900MB, may take a moment)...")
    embedding = joblib.load(JOBLIB_PATH)
    print("loaded. top-level keys:", list(embedding.keys()))

    for dim in MARGINAL_DIMS:
        points, cov_mean = extract_marginal_dim(embedding, dim)
        out_path = CACHE_DIR / f"dim{dim}.npz"
        np.savez(out_path, points=points, cov_mean=cov_mean)
        print(f"dim={dim}: points={points.shape} cov_mean={cov_mean.shape} -> {out_path}")

    for dim in PATH_DIMS:
        points, cov_mean, labels = extract_path_dataset(embedding, NOISE_DATASET, dim)
        out_path = CACHE_DIR / f"path_{NOISE_DATASET}_dim{dim}.npz"
        np.savez(out_path, points=points, cov_mean=cov_mean, labels=labels)
        print(
            f"path/{NOISE_DATASET}/dim{dim}: points={points.shape} cov_mean={cov_mean.shape} "
            f"labels={labels.shape} (n_states={labels.max() + 1}) -> {out_path}"
        )

    del embedding
    gc.collect()
    print("done, embedding freed.")


if __name__ == "__main__":
    sys.exit(main())

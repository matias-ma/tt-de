# Refreshed density-estimation gallery

The current PNGs live directly in `figs/ginzburg_landau/` and `figs/cryoem/`.
Checkpoints, noisy validation scores, pilot selections and numerical checks
live in `artifacts/figure_refresh/`, with later cryo-EM refinements in
`artifacts/cryoem_refinement/`. Earlier figures are preserved separately
in `artifacts/figure_refresh/previous_figs/`.

## Experiments

- GL: 6, 10, 14 and 20 dimensions; 100,000 training observations per case.
  Noise-free data and both prescribed noise families (paired Gaussian and
  Gaussian mixture), with noise standard deviations scaled by 1.5.
- Cryo-EM: the cached 4D, 10D and 20D marginal datasets, and the 4D projection
  of the 10D path embedding. The 10D marginal and path observations are
  identical, so their figures share one fit. The old Tucker comparison is
  superseded in this gallery by the improved functional TT estimator.

All fits optimize the Chebyshev coefficients of a squared, normalized
`chebttge.ChebTT`. Optimization uses physical Gram coordinates, gradient
clipping and cosine-decayed Adam. Baseline checkpoints minimize validation
NLL; the subsequent cryo-EM refinement uses regularized checkpoints as
described in [the refinement notes](cryoem_refinement.md).

Noisy fits start with an integrated curvature warm-up (8,000 steps,
25 coefficients, strength `3e-5`). Polynomial marginal references are fitted
from noisy observations and the prescribed noise TT. Degrees 2, 4, 6 and 8,
with shared or separate coefficients, are compared using noisy marginal
validation likelihood and parameter count. Each reference has a budget of
1,200 L-BFGS iterations; its termination status and gradient norm are saved.
The selected reference defines
the marginal Fisher filter, with visibility floor 0.01. Its strengths 0.1,
0.3 and 1 each receive 8,000 further steps, starting from the same warm-up.
The basis has 33 coefficients, increased if needed to resolve the reference
amplitude. The warm-up and Fisher candidates are compared using paired
noisy validation NLL, choosing the smallest common Fisher penalty within
one standard error of the best likelihood.

The full prescribed TT kernel enters the likelihood, including the paired
correlations and the mixture's shared component. The pilot and Fisher filter
use its one-dimensional marginals. Prepared convolution is checked against
direct convolution with twice the quadrature order.

GL training and validation seeds are 0 and 2; the 200 sampler chains define
the clusters for paired standard errors. Clean GL densities are used only
after selection, for figures and integrated L2 diagnostics. These diagnostics
do not select settings or checkpoints.

Cryo-EM reserves 4,096 randomly selected noisy observations for validation
(seed 0), using the remaining 94,904 for training. Per-axis means and standard
deviations from the training set condition the optimization; plots map back
to the original coordinates with the density Jacobian. The baseline uses
a diagonal approximation to the supplied mean noise covariance. The
[subsequent refinement](cryoem_refinement.md) increases TT rank, compares a
noise mixture with that approximation, and corrects noise marginalization
in the 4D path dataset while retaining the same observations. These example
kernel approximations are not restrictions of the TT likelihood API. All noisy
observations in the split are retained even when outside the latent signal
box. The separate naive baseline fits their observed density.

## GL results

All twelve GL cases completed. These are unsquared relative L2 errors of
the full latent density, evaluated only after model selection.

| Dimensions | Noise-free | Paired Gaussian | Gaussian mixture |
|---|---:|---:|---:|
| 6 | 0.0812 | 0.0950 | 0.1314 |
| 10 | 0.1093 | 0.1411 | 0.1360 |
| 14 | 0.1319 | 0.2307 | 0.1688 |
| 20 | 0.1611 | 0.2440 | 0.2359 |

Across GL candidates, the largest log-density change on the saved
quadrature check was `6.47e-13`. The largest relative change from refining
the figure's FFT grid was `1.38e-4`, and the largest clipped negative noise
mass was `3.13e-7`.


## Reading the figures

Every GL panel uses `[-2.5, 2.5]` on both axes, including the noise kernel.
The axes and filenames use zero-based coordinate indices.

Noise-free panels show the true GL marginal, training-data histogram, KDE
and ft-de. Noisy panels show the true latent marginal, noise kernel, true
convolved marginal, noisy training-data histogram, KDE and **deconvolved
ft-de**: the estimate of the underlying density, before reconvolution.
Truth and recovered density share one color scale; the three observed-density
panels share another, and the noise kernel has its own labeled color scale.
Titles report the unsquared full-dimensional relative L2 of the recovered
density. Small footers list ranks, basis counts, sample counts, optimizer and
step settings, regularization, pilot choice, kernel resolution and bandwidths.
Footers also report measured TT fitting wall time for the chosen candidate
run and the recorded candidate search. The search includes inherited fitting
stages and deduplicates shared checkpoints; naive cryo-EM baseline fitting
time is listed separately. These timings include compilation and validation
within fitting, but exclude initialization, KDE/kernel/pilot construction,
plotting, failed attempts and unrelated exploratory runs. They are not
end-to-end runtime estimates. The baseline and early refinement runs used
an NVIDIA A100; subsequent 20D/path refinement and continuation used an
NVIDIA H100, always one GPU with four CPUs and 32 GB host memory.

The KDE panels are exact marginals of the full diagonal-bandwidth KDE,
whose bandwidth is selected from training data. The fitted TT pair
marginals integrate the other coordinates exactly. Convolved panels use
zero-padded linear FFT quadrature on a larger integration grid, checked at
spacings 0.025 and 0.0125 before cropping. This is a plotting calculation;
training uses the differentiable TT convolution. Histograms divide by the
entire training sample count and bin area, so cropping does not inflate
their density values.

When a Fisher candidate selects step zero, the recovered density is the
warm-up retained unchanged (apart from padding the basis). Its caption
states this explicitly; trying a regularizer does not guarantee validation
will prefer its subsequent updates.

Tiny negative tails in cross-approximated noise kernels are clipped for
display only, with the removed probability mass recorded in
`plot_checks.json`. Rendering rejects a kernel if that mass exceeds `1e-5`.
This bounds the resulting convolution change by the removed mass times
the maximum signal marginal density. The likelihood uses the original
prescribed TT approximation.

Cryo-EM path figures are post-fit diagnostics using simulation labels.
Cryo-EM marginal figures include the supplied noise kernel, the data
histogram, KDE, observed-density ft-de and recovered latent ft-de. All use
the original coordinate units. The noise kernel represents a displacement,
so its coordinates are rescaled without adding the observation mean.
Normalized density values at path anchors are not the probabilities of
discrete conformational states; the figure labels make this distinction
explicit. Neither labels nor path scores enter fitting or model selection.

## Reproduction

Activate the `chebdGE` environment and obtain one allocation, following
`instructions.txt`:

```bash
salloc --partition=cryoem --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00
conda activate chebdGE
python scripts/refresh_density_gallery.py --job-id "$SLURM_JOB_ID"
```

The runner starts one `srun` process per fit and plot case and stages figures
in `artifacts/figure_refresh/gallery/`. Completed fits are reusable; a saved
case rejects changed CLI settings. Use a new `--artifacts` directory for
a new experiment configuration. Inspect `run_status.json` and the case
logs before publishing the staged gallery.
JAX compilation caches use node-local temporary storage by default, keeping
recompilable executables off the home-directory quota. Override
`CHEBTTGE_JAX_CACHE_DIR` to choose another cache location.

For the current refined cryo-EM gallery, follow the baseline fits with
`python scripts/refresh_cryoem_gallery.py --job-id "$SLURM_JOB_ID"` using an
allocation with sufficient time remaining. Its protocol and cached artifacts
are described in [the cryo-EM refinement notes](cryoem_refinement.md).

To redraw selected cases without training:

```bash
python scripts/refresh_density_figures.py artifacts/figure_refresh/gl_free_dim10 --figures figs
python scripts/refresh_density_figures.py artifacts/figure_refresh/gl_paired_dim10 --figures figs
```

For independent, full-dimensional GL error integrals after selection:

```bash
python scripts/evaluate_gallery_fits.py artifacts/figure_refresh/gl_*_dim*
```

The numerical plotting tests are in `tests/test_gallery_marginals.py`.
Historical investigation scripts and notes remain available, but their
older figure layouts are not the current gallery workflow.

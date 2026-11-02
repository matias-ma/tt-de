# tt-de

A functional tensor decomposition density estimator: fits a probability
density (optionally under known measurement noise) by minimizing negative
log-likelihood over the Chebyshev coefficients of a functional tensor-train
(TT) or Tucker decomposition, using
`chebdGE` (private dependency)'s cross-approximation and
convolution machinery. See `docs/` for the experimental history and current status.

The current [dimensional comparison](docs/dimension_gallery.md) runs Swirl
and Ginzburg–Landau in **4, 10 and 20 dimensions**, with and without the
existing mixture noise. The direct-training API and canonical core sweeps
now default to **full amplitude-Hessian regularization**, strength `0.001`.
The benchmark runner supplies the more specific recipes recovered from
the completed studies, including core sweeps and staged noisy training:

```bash
srun --jobid=YOUR_JOB_ID --ntasks=1 --cpus-per-task=4 \
  bash scripts/run_noise_diagnosis_gpu.sh -u scripts/run_dimension_gallery.py
```

The runner stages figures and publishes only after all twelve synthetic
cases finish. Previous figures are archived outside `figs`; cryo-EM keeps
its existing 4D/10D/20D noisy results. See `artifacts/dimension_gallery_v9`
for frozen selections, error integrals, and the final publication audit.
The current full-dimensional errors are summarized in the [figure index](figs/README.md);
noisy 20D Swirl reaches 0.781, so the 0.2–0.3 target remains open.

The [noisy regularization experiment](docs/marginal_regularization.md)
adds an exact 1D marginal-curvature penalty and compares it with weaker
joint smoothing, using observed validation data for selection. The follow-up
also tests [smoother random initialization](docs/smooth_initialization.md).
The preceding training study added [independent optimizer smoothing and
coarse-to-fine polynomial training](docs/training_geometry.md), with matched
noisy comparisons in 4D and 6D.

## How it works, briefly

- The default density is parametrized as `p(x) = g(x)^2 / Z`, where `g` is a
  coefficient-tensor `ChebTT`/`ChebTucker` (squaring guarantees non-negativity
  for any real coefficients) and `Z = integral g(x)^2 dx` is computed exactly
  in closed form (no quadrature) via a Chebyshev Gram matrix.
- The current workflow starts `g` with `init_tt_constant`: a constant plus
  a small random perturbation. Density learning happens in optimization of
  the TT coefficients against the observations. This initializer takes only
  a box, rank, polynomial size and seed; it fits no KDE or signal mixture.
  The older KDE initializer remains available for reproducing historical runs.
- For **noisy** observations under a known noise kernel `U`, the noisy density
  is `q = convolve(square(g), U) / Z` (same `Z`). The spectral backend
  integrates the polynomial kernel using Clenshaw, Gram matrices and DCT.
  For diagonal Gaussian noise, the optional `GaussianConvolution` backend
  integrates the analytic kernel with deterministic quadrature and contracts
  the two copies of `g` directly, avoiding the rank-squared convolution and
  SVD rounding. `TTConvolution` extends that direct contraction to a prescribed
  Chebyshev TT noise kernel, including non-Gaussian and correlated noise.
  RSI approximates the convolution as a compressed TT. Its interpolation rows
  are selected before JAX compilation and held fixed during optimization, so
  the compressed objective remains differentiable. Use `--convolution-backend rsi` in the GL and cryo-EM example scripts (`--backend rsi` in the cryo-EM
  benchmark); the benchmarks also check the approximation against direct
  convolution. RSI need not improve statistical accuracy or runtime at these ranks.
  See [the convolution investigation](docs/convolution_research.md) for
  research, accuracy measurements, cryo-EM experiments and reproduction commands.
  See [the general-kernel and noisy GL follow-up](docs/gl_convolution_followup.md)
  for the distinction between Gaussian noise and Gauss quadrature, and GL errors.
  `TTConvolution.prepare(...)` precomputes integration moments for repeated NLL
  evaluation with an arbitrary fixed TT kernel. The covariance-mixture benchmark
  uses 100,000 noisy GL observations and saves compact integrated-L2 and
  marginal figures under `artifacts/figure_refresh/previous_figs/archive/gl_covariance_mixture/`.
- The squared estimator is implemented once for TT and once for Tucker; no
  mixture-of-decompositions escape hatch.

The [noisy recovery investigation](docs/noisy_recovery.md) identifies two
defects in the legacy core regularizer and adds an integrated amplitude
derivative penalty (`AmplitudeSobolevPenalty`) plus paired noisy-validation
selection. It includes matched strong-noise GL experiments and reproduction
commands. Both direct and RSI likelihoods support the new penalty.

**NLL remains the default training objective**, with full-Hessian
regularization in the direct-training workflow. Legacy fitting APIs retain
their historical behavior for reproduction.
MMD and NLL-plus-MMD are experimental options that require explicit selection.

The [noisy-estimation diagnosis](docs/noise_diagnosis.md) checks convolution
values and gradients against analytic noise integration, tests noisy-data
restarts and optimizer continuations, and compares matched noise levels and
domain widths. It distinguishes fitting the observed density from recovering
the latent density. These controls retain NLL and the Chebyshev TT basis.

The [objective and capacity study](docs/objective_study.md) tests
smaller ranks and mode counts, multiscale Fourier kernel scores (MMD), and
likelihood/kernel combinations. It includes strong correlated and mixture
noise, with separate held-out likelihood and kernel-score selections.
`scripts/run_objective_study.py` reproduces the full experimental comparison;
the fitting API and CLI default to NLL. See the [current figures](figs/README.md).
All models remain single Chebyshev TTs trained from near-uniform starts.

The preceding [noise-free optimization study](docs/noise_free_optimization.md)
learns one Chebyshev TT from a nearly uniform initializer. It compares
Adam, full-data L-BFGS, canonical core sweeps, slope/curvature penalties,
conditional likelihood and a quadratic density score. It varies elementary
initialization seeds and perturbation sizes; no learned signal pilot is used.
Its core-sweep experiments can be reproduced with `scripts/run_noise_free_v2.py`.
The [previous direct study](docs/direct_optimization.md), signed/clipped
experiments, and older gallery scripts remain historical reproductions.

## Tucker alternative

TT remains the default. The native Tucker estimator now supports the same
squared-density likelihood, exact normalization, full-Hessian regularization,
Adam training and prepared prescribed-noise convolution. The convolution
contracts the dense core directly without materializing its square.
See [Tucker implementation and experiments](docs/tucker_density.md).
Run `python scripts/run_tucker_gallery.py` for the feasible-dimension GL/Swirl
comparison; its figures use `tucker_` prefixes alongside the TT figures.

## Historical optional experiments

These methods are retained for reproduction and are outside the current
direct-optimization workflow.

The [information and learned-reference follow-up](docs/noisy_information_recovery.md)
adds noisy marginal pilots, `ReferenceScorePenalty`, and `FisherVisibilityPenalty`.
The learned-reference information filter reaches integrated relative L2 **0.141** on the
same strong-noise GL example, with selection based on noisy data only.

An optional [positive-core Chebyshev TT](docs/positive_chebyshev_tt.md) fits
the density directly with interval-wide positivity constraints and exact
normalization. It retains the Chebyshev basis and prescribed-noise TT
convolution. `scripts/fit_positive_tt.py` trains this TT-only alternative;
`PositiveChebyshevTT` provides its density evaluation and model format.
The implementation notes and figure index report the recovery comparisons.

The historical [initialization study](docs/initialization_study.md) documents
retired fitted-pilot experiments. Their tangent/local-mixture estimators,
pilot-to-TT conversion, selection APIs, drivers, and dedicated tests have been
removed. The reports and saved results remain as historical evidence; their
old commands no longer run. The original KDE utilities remain in use by
legacy TT/Tucker examples, model-selection routines, and comparison plots.

Current initialization uses `init_tt_constant`: each core has one constant
Chebyshev path plus small random coefficients, damped by `1/(1+n^2)` for mode
index `n`. QR changes the internal factorization, then the amplitude is
normalized in physical L2. The earlier swirl/cryo-EM runs used rank 32 and
49 modes; the earlier capacity study compared ranks 5–32 and 17–49 modes. These use
seed 0 and perturbation scale 0.01. The initializer takes only the
supplied box and these settings; all density structure is learned afterward
by objective minimization.

## Install

Follow [INSTALL.md](INSTALL.md). Access to the private `chebdGE` dependency is required; its source is not distributed here.

## Repo layout

Start with [the figure index](figs/README.md) and the
[dimensional comparison](docs/dimension_gallery.md). PNGs live
directly in `figs/swirl`, `figs/ginzburg_landau`, `figs/cryoem`, and
`figs/convergence`, with no numerical sidecars or experiment subfolders.
The new synthetic matrix covers GL and Swirl in 4D/10D/20D, each with and
without noise. Cryo-EM is retained from the preceding study and is labeled
separately. The current figure index reports the observation-selected
models, with the original gallery preserved under
`artifacts/gallery_archive/before_dimension_gallery_v9`.

Compressed checkpoints, numerical records, and logs for the new matrix live
in `artifacts/dimension_gallery_v9`. The preceding objective study lives in
`artifacts/objective_v3`. `scripts/finish_objective_study.py` freezes the
observation-based choices before evaluation and figure generation. Plotting
entry points are `scripts/plot_objective_study.py`,
`scripts/plot_objective_capacity.py`, `scripts/plot_objective_noise.py`, and
`scripts/plot_objective_candidates.py`. Superseded figures are archived at
`artifacts/gallery_archive/before_objective_v3`; older galleries and numerical
artifacts remain available outside `figs`. The older
[gallery notes](docs/figure_refresh.md) and LaTeX write-up in `note/` describe
historical experiments.

The historical `srun` commands assume a Slurm allocation. On other systems,
run the Python drivers directly from this directory after installation.

```
src/ftde/
  tt_density.py, tucker_density.py     noise-free squared-TT/Tucker density model + NLL training
  cheb_gram.py                         closed-form Chebyshev Gram matrix (exact normalization)
  cheb_product.py                      exact pointwise square of a ChebTT/ChebTucker (square_tt/square_tucker)
  init.py, kde.py                      KDE utilities still used by legacy examples and comparisons
  simple_initialization.py             near-uniform TT start using only the box and seed
  direct_training.py                  current direct objectives and integrated regularizers
  kernel_score.py                     direct TT Fourier integrals and multiscale kernel discrepancy
  scored_training.py                  likelihood/kernel objectives with separate validation scores
  core_sweeps.py                      canonical sign-preserving Newton-CG core updates
  direct_lbfgs.py                     current full-data likelihood refinement
  full_batch_optimization.py           previous full-data TT likelihood refinement
  signed_tt_least_squares.py           direct signed TT objective; final clipping measured separately
  tt_smoothing.py                     optional mass-preserving spatial heat filter in the same TT basis
  tt_sampling.py                      polynomial conditional-CDF sampling and moments
  noise_kernel.py                      builds a fixed, known Gaussian noise kernel U
  noisy_tt_density.py, noisy_tucker_density.py   noisy density model (U * p_tilde) + NLL training
  gaussian_convolution.py              direct diagonal-Gaussian squared-TT contraction
  tt_convolution.py                    direct contraction with general TT noise kernels
  noisy_model_selection.py             automatic degree selection (1-SE rule) for the noisy estimator
  data.py                              synthetic data generators for demos/tests

tests/
  test_density.py       noise-free TT/Tucker regression tests (d=3, plus two harder d=5 TT examples)
  test_noisy_density.py noisy TT/Tucker regression tests (d=3), deliberately cheap
  test_hard_gpu.py      harder noise-free examples (d up to 10); GPU-scale, smoke-testable locally
  conftest.py           report()/header() diagnostic printing + run_all_tests() for direct script execution

scripts/
  run_noise_free_v2.py                  observation-only objective/penalty/initialization grid
  run_noise_free_followups.py          deterministic and longer optimization controls
  select_noise_free_v2.py              freeze choices before truth evaluation
  evaluate_noise_free_v2.py            post-freeze density-error diagnostics
  run_noise_free_cryoem.py             transfer to prescribed-noise cryo-EM likelihood
  demo_tt_2d.py, demo_tt_banana.py       noise-free demos
  demo_noisy_1d.py, demo_noisy_2d.py     noisy (deconvolution) demos
  gpu_noisy_hard.py                      harder noisy example; GPU-scale, smoke-testable locally
  cryoem_cache.py                        one-time: extracts small .npz caches from cryoem_data/
  cryoem_common.py                       shared fitting/plumbing for the two scripts below
  cryoem_density.py                      naive vs. deconvolved 2D marginals on real cryo-EM latent data,
                                          alongside the Gaussian KDE's own exact marginal (4-panel figures)
  cryoem_path_density.py                 legacy normalized anchor-height diagnostic (not state probabilities)
                                          at dim 4 and 10, naive KDE vs. naive/deconvolved TT (+ Tucker at dim 4)
  cryoem_diagnostics.py                  static summary plots of the dim=10 rank/degree/reg_lambda
                                          investigation (todo.txt) -- matplotlib only, no jax/GPU needed
  gl_gpu_noise_free.py, gl_gpu_noisy.py   GL fits and matched-scale marginal comparisons

cryoem_data/                              real cryo-EM latent-space data (not tracked in git; see below)
  cryoem_test.joblib                     recovar latent embeddings + per-point inverse covariances
  noise1/                                simulation_info.joblib (ground-truth conformational labels)
  cache/                                 small .npz caches built by scripts/cryoem_cache.py

ref_notebooks/                            original TTDE-based notebooks these scripts are modeled on
figs/cryoem/                              figures saved by the cryoem_* scripts (gitignored)
figs/ginzburg_landau/                     GL noise-free and noisy marginal figures (gitignored)
```

The `cryoem_*` scripts are a from-scratch reimplementation of
`ref_notebooks/load_zs.ipynb` and `load_zs_along_path.ipynb` using this
package's own machinery (single TT/Tucker, exact closed-form marginals and
convolution) instead of TTDE's Monte Carlo deconvolution and
n_components-mixture models. Run `python scripts/cryoem_cache.py` once
first, then e.g. `python scripts/cryoem_density.py 10 tt` or
`python scripts/cryoem_path_density.py 4 10`. `cryoem_common.fit_signal`'s
signal (rank, degree) default to `_rank_degree_for_dim(dim)` -- dim=4 and
dim>=10 need very different settings on this real dataset (see todo.txt);
pass `--rank`/`--degree` explicitly to override. `--reg-lambda` enables the
frequency-aware Tikhonov deconvolution regularizer (`ftde.noisy_regularization`);
0 (default) disables it -- see todo.txt for why it hasn't been shown to make
deconvolution beat the naive (no-deconvolution) fit on this dataset.

Those commands and defaults reproduce the legacy notebook comparisons.
Use the gallery documentation above for the current estimator settings.
Normalized density heights at path anchors are not state probabilities;
their agreement with label frequencies is not a density-recovery accuracy metric.

## Running tests

Full, per-observation Gaussian covariances are supported by
`ftde.particle_convolution`. `fit_gaussian_kernel(covariance)` fits the entire
Gaussian with chebdGE on a box scaled by its marginal standard deviations;
off-diagonal entries remain in the target. `TTConvolution.from_tt(...,
zero_extension=True)` supports a compact kernel by integrating on the exact
intersection with the translated signal box. Its default fixed-window
behavior is unchanged.

For repeated training, `kernel_moments` prepares the polynomial moments of
each particle's kernel, and `ParticleTTProblem` contracts them with the signal
TT while differentiating its coefficients. `RaggedMoments` and `stream=True`
keep variable-rank caches on disk and transfer only a minibatch. The optional
`GaussianKernelBank` reuses chebdGE cross points, recomputes every cross value
from the new covariance, validates the resulting kernel and its mass, and
falls back to adaptive fitting. Its sampled acceptance checks are not uniform
error bounds; audit likelihood convergence with independent higher-resolution
fits. Correlated kernels can require substantially higher ranks than diagonal
kernels. See `md-committor/scripts/48_igg_full_covariance.py` for the 4D IgG
comparison and the `test_particle_convolution.py` tests for independent cubature
and gradient checks.

Each test file can be run directly (no pytest needed) for full unconditional
diagnostic output:

```bash
python tests/test_density.py
python tests/test_noisy_density.py
```

or via pytest (add `-s` to see the diagnostics on passing tests too):

```bash
pytest tests/test_density.py tests/test_noisy_density.py -v -s
```

`test_hard_gpu.py` and `scripts/gpu_noisy_hard.py` are GPU-scale: run with
`FTDE_TEST_SCALE=smoke` locally to check they're exception-free (seconds,
does not validate accuracy), or at full difficulty (the default) once you
have GPU access, e.g. `pytest tests/test_hard_gpu.py -v -s` or
`python scripts/gpu_noisy_hard.py`.

**Note for this machine specifically:** it has no GPU. JAX cross-approximation
and NLL training compile/scale up quickly even at moderate rank/degree, so
keep local runs at low rank/degree/`n_starts`/steps, and don't run many
high-degree builds back to back in one process (e.g. avoid a bare
`pytest tests/` that pulls in `test_hard_gpu.py` alongside the rest) -- it can
exhaust memory even when no single test looks extreme in isolation.

## Code style

`ruff check src/ tests/ scripts/` should report no issues.

## Research documents and earlier implementation

The original [cryo-EM workshop paper/poster](note/cryoem_workshop.pdf) is preserved alongside the current research
write-ups and figures. The functional implementation is now the main package
in `src/`; the superseded implementation remains available in Git history.

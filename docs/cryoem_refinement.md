# Cryo-EM refinement

This documents the earlier rank-20 gallery. The current figures and subsequent
unsquared-density experiments are documented in
[Cryo-EM continuity](cryoem_continuity.md).

The original gallery used TT rank 8 in 4D and rank 10 in higher dimensions.
Those ranks were insufficient for these data. Increasing the number of
Chebyshev modes alone barely improved the 4D validation likelihood and
introduced extra local oscillations. Increasing rank improved the likelihood
substantially; stronger function-space curvature regularization then reduced
the disconnected patches in the recovered marginals.

The selected models still optimize Chebyshev coefficients of a normalized
squared `chebttge.ChebTT` through the prescribed TT convolution. No clean
density, simulation label, or path diagnostic enters fitting or selection.

## Controlled comparisons

All comparisons retain the original 94,904 training observations and 4,096
validation observations, standardization, latent boxes, KDE and naive
observed-density baselines. Starting from the previous selected fit, the
refinement compares:

- 49 amplitude modes at the previous rank, with the previous kernel;
- 49 modes and rank 20, with the previous kernel;
- 49 modes and rank 20, with a four-component noise mixture.

Both rank-20 alternatives use curvature strengths `3e-5` and `3e-4`.
Each receives 4,000 Adam steps, batch 256, learning rate `2e-4` decaying to
`2e-5`, and gradient clipping at 1. Initialization first canonicalizes the
TT in physical Gram coordinates, preserves its normalized density, then
adds small perturbations in the new rank directions. Simply padding ranks
with zeros would leave the new directions difficult to optimize.

Within each refinement run, checkpoints minimize validation NLL plus the
candidate's curvature penalty. The original fit and all refined candidates
are then compared by paired noisy validation NLL: select the smallest
common physical amplitude curvature within one paired standard error of
the best NLL. The common length scales come from the baseline noise kernel.
This is a model-selection heuristic, not a confidence guarantee after
searching candidates, and not a measurement of latent relative L2.

The 10D and 20D experiments also test continuation: start from the selected rank-20
fit, then run 3,000 further steps at learning rate `1e-4` with curvature
strengths `1e-4` and `3e-4`. These candidates enter the same selection rule.
This lets new rank directions learn useful structure before applying
stronger smoothing, instead of requiring those directions to grow from
small initial perturbations under the stronger penalty.

The completed comparisons selected the mixture kernel in all four cases,
with the moderate continuation penalty in 10D and 20D:

| Case | Previous noisy NLL | Refined noisy NLL | Previous curvature | Refined curvature |
|---|---:|---:|---:|---:|
| 4D | 3.611118 | 3.539891 | 15.730 | 9.867 |
| 10D | 10.500752 | 9.978320 | 758.773 | 110.531 |
| 20D | 25.246122 | 24.758697 | 11994.596 | 221.312 |
| 4D path projection | 3.623019 | 3.543929 | 100.156 | 42.172 |

The path baseline is rescored under the corrected Gaussian noise kernel.
Curvature is the common noise-scaled amplitude H² measure within each case;
it is not a latent-density accuracy metric. In 4D, with the original Gaussian kernel
held fixed, the rank-20 model already reaches **3.547391**, whereas the
49-mode rank-8 control reaches **3.609927**.

## Noise corrections

The source embedding supplies a noise precision matrix per observation.
The original experiments replaced these distributions by one Gaussian at
the average covariance. The supplied per-axis variances vary substantially:
their coefficient of variation ranges from about 0.33 to 0.59 in 4D and
reaches about 1 in higher dimensions. Averaging covariance does not preserve
the shape of a mixture of Gaussians.

The alternative kernel clusters log noise variances from the **training
rows only** into four groups. Its weights are group frequencies and each
component uses its group's mean variance. It remains one stationary kernel
function, passed to the same arbitrary-kernel TT likelihood. The estimator
does not condition on a per-observation covariance. This approximation still
omits within-component off-diagonal correlations and any dependence of
noise on latent state; it is not an exact model of all source uncertainties.
The covariance metadata constructs these particular example kernels; it is
not a new requirement of the estimator API.

A separate post-fit check compares marginal characteristic functions
against the Gaussian mixture specified by held-out noise metadata. On a
fixed frequency grid, the four-component approximation's aggregate error
is 0.185 times the average Gaussian's in 4D and 0.230 times it in 10D.
This assesses the noise approximation only; it is not a latent-density
error or a check of omitted noise correlations. The audit also verifies
that the represented mixture integrates to one.

Every component is represented on the same physical Chebyshev domain.
The representation is checked against analytic noise values, and numerical
tests check the mixture and its moments. An initial pilot incorrectly
combined coefficients from different domains; its mixture scores are
explicitly marked invalid in `dim4_resolution/INVALID_MIXTURE.json` and are
excluded from the published results. The corrected runs are `*_rank20`.

The 4D path dataset had an additional preparation bug: inverting the first
4-by-4 block of a 10D precision matrix gives conditional covariance, not
marginal covariance. `cryoem_cache.py` now inverts the full matrix first and
then takes the retained covariance block. On the existing gallery rows the
correct variances are 14–34% larger. Refinement retains the original rows
and split, corrects the kernel, and rescores the previous fit under that
corrected kernel before comparing candidates. Thus row filtering changes
do not confound this comparison. Metadata extraction verifies correspondence
with the cached rows; fitting also verifies alignment with its split.

## Artifacts and reproduction

Published figures remain flat in `figs/cryoem/`. Their previous versions
are in `artifacts/cryoem_refinement/previous_figs/`. The refinement artifacts
include per-candidate fits, likelihood vectors, training histories,
quadrature checks, noise-cluster settings, and selected gallery cases.
Plot footers include measured fitting wall times. The recorded-search total
includes ancestral warm-up/Fisher fits and the refinement/continuation
candidates, deduplicating shared files. Initialization, KDE/kernel/pilot
construction, plotting and failed/discarded exploratory attempts are
excluded; naive baseline fitting is timed separately. A100 and H100 stages
are mixed in these totals, so they are not a uniform-hardware benchmark.

Using the `chebdGE` environment and an existing allocation with one GPU,
four CPUs and 32 GB on the `cryoem` partition:

```bash
python scripts/refresh_cryoem_gallery.py --job-id "$SLURM_JOB_ID"
```

This reuses the original gallery fits in `artifacts/figure_refresh/`, runs
the noise audit and candidate comparisons, and renders the selected cases.
Completed candidates are reusable. Use a new `--artifacts` directory when
changing settings. To redraw one selected case without refitting:

```bash
python scripts/refresh_density_figures.py artifacts/cryoem_refinement/gallery_dim4 --figures figs
```

Tests in `tests/test_cryoem_refinement.py` cover noise marginalization,
rank/basis expansion invariance and mixture representation. They test
numerical identities and known noise functions, not a clean signal used
for estimator selection. Existing marginal-plot and noisy-selection tests
also pass. Training checks prepared likelihood against direct convolution
with doubled quadrature, rejecting discrepancies above `1e-7`.
All 25 completed refinement/continuation/rescoring records passed that
check; the largest recorded log-density difference was `2.07e-12`.
The three test files pass all 12 tests, and the changed scripts pass Ruff.

Cryo-EM has no clean density available for a relative L2 claim. The improved
likelihood, lower curvature and more continuous shapes provide useful
evidence, but cannot establish that every narrow feature is correct.

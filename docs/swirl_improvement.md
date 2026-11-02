# Curved-density experiments, September 26, 2026

> Historical report: the tangent/local-mixture estimators, pilot conversion,
> and their experiment scripts have been removed. References and commands
> below describe that retired code. Saved results remain available; see the
> [current direct-optimization workflow](noise_free_optimization.md).

The main problem is not simply the number of Chebyshev modes. The old 10D
swirl amplitude has sign changes through the populated curve, and its
high-rank continuations preserve these artificial zero crossings. A positive,
data-derived local pilot offers a better initialization. A separate local
Gaussian-mixture estimator also substantially improves the higher-dimensional
swirl and the cryo-EM observation likelihoods.

**The local-mixture results are a change of model representation.** They are
not evidence that the original squared-TT optimizer, by itself, achieved the
reported mixture errors. Figures label the two model families explicitly.
The existing generic TT-noise convolution remains available; the new local
estimator currently requires a prescribed Gaussian-mixture noise distribution.

## What the diagnostics establish

All of these are post-fitting diagnostics, never fitting inputs:

* The old selected 10D noisy fit (rank 64, 49 modes) changes amplitude sign
  **six times along the true curve**. Continuity forces a zero at each crossing,
  although the true density is positive there. This explains a concrete source
  of holes. It does not mean every marginal defect is a sign crossing.
* In physical orthonormal polynomial coordinates, the highest eight modes
  contain at most **0.0117%** of the old amplitude's norm squared at any axis.
  The positive diagonal-mixture pilot projected into the same 49-mode,
  rank-64 representation has no sign changes there and a maximum tail of
  **0.000175%**. More modes are not the leading explanation in this comparison.
* The old amplitude has up to **12.21%** of its Schmidt energy beyond rank 16,
  but only **0.0221%** beyond rank 32. The new pilot amplitude has **18.22%**
  beyond rank 16 and **1.493%** beyond rank 32. Rank 16 really is restrictive;
  merely allocating rank 64 does not make the old optimizer use it to learn
  the correct geometry.
* The same-capacity positive pilot starts at noisy validation NLL **6.270118**,
  versus **6.284202** for the old selected tensor. This uses only an estimated
  density learned from the observations. The known swirl path supplies no
  initialization, ordering, widths, or component centers.
* Starting from the learned **tangent** pilot improves this further: the
  rank-64, 49-mode squared TT starts at noisy NLL **6.269018**. Its full-density
  relative L2 is approximately **0.2223 ± 0.0009** (Monte Carlo SE), against
  **0.4870 ± 0.0020** for the old TT on the same importance sample. The old
  tensor's exact-quadrature result, 0.4878, agrees with that check. The diagonal
  pilot gives 0.3783 ± 0.0016; learning local directions matters before the
  tensor conversion as well.
* The same construction in 20D, using rank 64 rather than the old rank 32,
  gives **0.3067 ± 0.0030** relative L2, versus the old exact result **0.866**.
  The matched importance estimate of the old error is 0.8731 ± 0.0067. The
  20D comparison combines initialization and increased rank; the 10D one
  isolates the initialization change at fixed rank and modes.
  The 20D converted tensor scores **9.656060** on noisy validation observations.

These observations support a combined capacity/initialization problem, with
the latter dominating the remaining 10D gaps after expansion. They do not
constitute a proof of a unique optimization basin or a guarantee that all
remaining fine structure is statistically identifiable.

## Local adaptive density

`src/ftde/local_mixture.py` implements EM over both the latent signal component
and the prescribed noise-mixture component. Conditional Gaussian moments
integrate out the clean observation. This follows the extreme-deconvolution
construction of [Bovy, Hogg and Roweis](https://arxiv.org/abs/0905.2979), extended
here to a stationary mixture of noise covariances.

The candidate set is identical across all six swirl cases and four cryo-EM
cases:

* shared diagonal signal covariance, 64 or 128 centers;
* full signal covariances, 16 or 32 centers;
* locally elongated covariance `psi I + u_k u_k^T`, 16, 32 or 64 centers.

The last family is a mixture of local one-dimensional Gaussian directions,
with a shared transverse width. Directions, centers, widths and weights are
all learned. There is no supplied curve, topology, path parameter or state
assignment. It is an inductive assumption that the density can be covered by
locally elongated components, not knowledge of the actual generating path.

K-means uses at most 20,000 training observations for initialization. EM then
uses the whole training set, in memory-bounded chunks, for 300 iterations.
Every tenth iteration is scored on the existing validation split. Across the
best checkpoints, paired one-SE eligibility followed by the fewest parameters
selects the model. The covariance floor is `1e-4 * training variance`, not the
known clean tube width. The original data split and noise parameters are
retained. For cryo-EM the analytic mixture parameters are checked against the
frozen TT kernel coefficients before fitting.

### Swirl full-dimensional relative L2

| Data | Dimensions | Previous TT | Selected local mixture |
|---|---:|---:|---:|
| Noise-free | 4 | 0.102 | **0.0519** |
| Noise-free | 10 | 0.221 | **0.0689** |
| Noise-free | 20 | 0.373 | **0.0924** |
| Gaussian-mixture noise | 4 | **0.216** | 0.2600 |
| Gaussian-mixture noise | 10 | 0.488 | **0.2818** |
| Gaussian-mixture noise | 20 | 0.866 | **0.3980** |

All six local selections choose 32 tangent components. L2 uses exact Gaussian
overlaps and path quadrature, checked at orders 2,000 and 4,000; the relative
cross-integral changes are about 2e-13. These are errors of the full joint
density, not just the displayed two-coordinate projections.

The noisy 4D regression is real. Its validation NLL improves by 0.00160 ±
0.00087 paired SE, yet clean L2 worsens and the pretzel acquires some local
lumps. Thus even improved noisy likelihood does not guarantee improved clean
recovery. The comparison figures preserve the old tensor alongside the new
model, and the main figure explicitly reports both errors. No truth-based
choice is disguised as an observation-only selection rule.

The local models were also evaluated on **8,192 fresh observations per case**
(seed 73129), generated only after all selections were frozen. Their test NLL
gaps to the true observation density are:

| Dimensions | Noise-free gap ± paired SE | Noisy gap ± paired SE |
|---|---:|---:|
| 4 | 0.00400 ± 0.00079 | 0.00211 ± 0.00057 |
| 10 | 0.00676 ± 0.00118 | 0.00391 ± 0.00095 |
| 20 | 0.01294 ± 0.00157 | 0.00912 ± 0.00141 |

This split is evaluation only. The old validation split remains a reused
development split; its paired SEs are not post-selection confidence bounds.

The corrected **10D tensor** was separately evaluated on the same fresh split:
NLL improves from 6.275878 to **6.263369**, a paired improvement of
**0.012508 ± 0.001915**. The gap to the true observation density falls from
0.015493 to **0.002985**. Thus the tensor improvement extends beyond the
validation observations used to choose the pilot.

### Cryo-EM reruns

| Embedding | Previous TT noisy NLL | Local noisy NLL | Improvement ± paired SE |
|---|---:|---:|---:|
| 4D | **3.531477** | 3.536587 | -0.005110 ± 0.002029 |
| 10D | 9.857318 | **9.807122** | 0.050196 ± 0.006178 |
| 20D | 24.656633 | **24.554700** | 0.101933 ± 0.008346 |
| Path 4D | 3.539250 | 3.539228 | 0.000021 ± 0.002217 |

The 4D, 10D and 20D local selections use 32 full-covariance components; path
4D uses 16. Only the 10D and 20D main recovery panels are replaced. Existing
4D/path-4D recoveries remain because the alternative does not beat them by a
paired SE. All four reruns have comparison and training figures. The supplied
noise approximation is unchanged. These are observation-prediction gains,
not measurements of unknown cryo-EM clean-density error. Label-based oracle
figures are post-selection diagnostics against an estimated reference.
The local Gaussian mixtures have support on all of R^d, whereas the tensor
fits retain their existing finite latent boxes. This is another model-family
difference, especially for tails; likelihood changes are not an isolated
test of geometric orientation alone.

The oracle audit is a caution against calling these likelihood gains clean
recovery gains: the local model's pairwise L2 against the estimated discrete
oracle **worsens** in all three 10D projections (0.448→0.467, 0.463→0.585,
0.197→0.343), and in two of three 20D projections (0.372→0.412,
0.408→0.515, while 0.244→0.199 improves). The smooth local model and a
50-state, label-derived reference make different assumptions. Neither better
observed likelihood nor a smoother image certifies the latent cryo-EM density.

## Bringing the pilot back into the squared TT

`src/ftde/mixture_init.py` approximates the square root of a learned pilot by a
positive sum of separable Gaussian amplitudes, then projects those functions
onto Chebyshev modes and compresses in physical L2. A local tangent component
is first expanded with one-dimensional Gauss–Hermite quadrature. Bounded
nonnegative least squares uses draws from the learned pilot, not clean truth.
The coefficient upper bound prevents rarely sampled components from receiving
huge amplitudes. Implicit right Gram matrices avoid materializing dense
component-by-component TT cores during compression.

The output is an ordinary `ChebTT`, compatible with the existing normalized
square, regularizer, exact marginalization and generic TT convolution. The
pilot's parameterization does not constrain subsequent tensor optimization.
Pilot approximation and compression errors are recorded separately; conversion
is not assumed lossless.

For example, compressing a 128-component diagonal cryo-EM pilot to rank 64
loses much more amplitude energy than in swirl (reported bounds 0.40/0.58 in
10D/20D). Separate trials use 64 pilot components and 81 polynomial modes,
whose compression bounds fall below 3e-6. This is a distinct capacity issue
in transferring a very narrow local pilot to a global polynomial tensor;
those tiny compression bounds exclude polynomial projection error and do
not imply that the fitted density is accurate.
The 20D swirl amplitude least-squares solver reached its iteration cap with
a projected-gradient residual of 3.04e-7. Its finite, feasible approximation
was accepted under the recorded 1e-5 tolerance; the reported density errors
evaluate that actual approximation, not an assumed exact square root.

Two pre-existing workflow defects were fixed:

* `expand_tt` now handles a reduced QR when a bond rank exceeds the number
  of available basis directions. Re-expanding an already rank-64, 49-mode
  tensor previously raised a reshape error.
* Validation honors the training microbatch memory budget. Previously,
  high-rank training could be memory-bounded while validation reverted to
  256 observations and exhausted GPU memory.

Matched tensor continuation outcomes and post-fit errors are stored in
`artifacts/swirl_improvement/`; the small-batch comparison uses 400 steps,
batch 64, four-way gradient accumulation, rank 64, 49 modes, amplitude H2
strength 1e-4, and Sobolev preconditioning. Earlier 800-step, batch-256 attempts
were stopped for cost before producing a completed checkpoint. Failed and
stopped attempts are not counted as successful fits or local-EM search time.

Both completed 10D continuations retain their initial checkpoints: the old
warm start finishes at NLL 6.288668 (worse than 6.284202), and the new start
finishes at 6.275605 (worse than 6.269018). The benefit comes from learning
the pilot and its regularizing tensor projection, not from those extra Adam
steps. The existing paired one-SE/H2 rule selects the new tensor; H2 energy
also decreases from 68.49 to 58.48. Doubled-quadrature discrepancies are below
1.1e-12 in log density for both completed trials.

The cryo-EM tensor transfer is less successful. In 10D, 49 modes with a
128-component pilot reach NLL 9.872345; 81 modes with a 64-component pilot
reach 9.858689. The prior tensor (9.857318) is retained by paired one-SE/H2
selection. Thus the best cryo-EM likelihood gains above belong to the local
mixture representation; they must not be attributed to an improved tensor.
In 20D the corresponding tensor trials reach 24.656498 (49 modes) and
24.654893 (81 modes), against 24.656633 previously. The latter difference,
0.001740 ± 0.006850 paired SE, is inconclusive; the old tensor has lower
common H2 energy and is retained. All completed tensor trials pass their
doubled-quadrature checks (maximum log-density discrepancy 2.09e-12).

## Figures and reproduction

Figures remain in the flat `figs/swirl` and `figs/cryoem` directories. Existing
figure names and main panel layouts are retained, with explicit model labels.
Earlier images are archived outside `figs`, under
`artifacts/swirl_improvement/previous_figs`. Coefficients, histories, numerical
checks, and JSON/NPZ artifacts are all outside `figs`. GL figures are unchanged.

Inside one allocation (one GPU, four CPUs, 32 GB, `cryoem` partition), using
the `chebdGE` environment:

```bash
python scripts/run_local_density_suite.py --family swirl
python scripts/run_local_density_suite.py --family cryoem
python scripts/evaluate_local_mixture.py --selected-only artifacts/swirl_improvement/swirl_*/
python scripts/plot_local_density_improvement.py artifacts/swirl_improvement/swirl_*/ artifacts/swirl_improvement/cryoem_*/
python scripts/fit_mixture_initialized_tt.py \
  --base artifacts/swirl/swirl_mixture_dim10 \
  --pilot artifacts/swirl_improvement/swirl_mixture_dim10/selected.npz \
  --output artifacts/swirl_improvement/reproduction_tt10 \
  --steps 400 --batch-size 64 --microbatches 4
```

Use new output directories for changed settings. Run one large TT convolution
fit at a time to avoid competing GPU memory allocations. Reported timings
include compilation, initialization and validation for each local search,
but exclude truth evaluation, plotting, aborted work and the old TT search.

Seventeen targeted tests pass, covering independent Gaussian conditional
moments, marginal normalization, tangent-to-TT conversion, repeated rank
expansion beyond the polynomial mode count, convolution values and
coordinates, microbatch equivalence, and model selection. Ruff passes the
new and modified experiment files.

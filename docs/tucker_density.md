# Tucker density estimation

TT remains the default. `TuckerTrainingProblem` uses the same normalized
squared Chebyshev amplitude, exact physical integrals, Hessian penalty,
Sobolev whitening, Adam updates, optional degree curriculum, minibatch
stream, and held-out NLL checkpoint selection as `DirectTrainingProblem`.
It optimizes a dense Tucker core and all univariate coefficient factors.

```python
from ftde import init_tucker_constant, TuckerTrainingProblem, TuckerConvolution

initial = init_tucker_constant(domain, rank=5, modes=25, seed=0)
problem = TuckerTrainingProblem(initial)  # Hessian strength 0.001
fit, record = problem.fit_adam(train, valid, output, steps=4000)

# Prescribed kernel can be a ChebTT or ChebTucker; it is never trained.
forward = TuckerConvolution.from_tucker(initial, kernel).prepare(observation_domain)
problem = TuckerTrainingProblem(initial, forward=forward)
fit, record = problem.fit_adam(noisy_train, noisy_valid, noisy_output)
```

`save_tucker`/`load_tucker` preserve the native representation. The legacy
`fit_tucker` and `fit_noisy_tucker` APIs remain available; the latter now
accepts `forward_operator=forward`, like TT. `squared_marginal_tucker`
computes exact normalized marginals directly from the amplitude.
`marginal_strength` enables the same exact univariate density-curvature
penalty as TT, including its physical length scales.

## Convolution and multiplication

`../chebdGE/src/chebtuckerge/computations.py` already provides
`convolve_tucker`. No upstream change is needed. Its explicit construction
Kronecker-multiplies the cores; after squaring a signal with R dense core
entries, the squared core alone has R² entries. There is no Tucker analogue
of `chebttge.multiply_rsi` in that module.

The new direct operator avoids this explicit square. Per-coordinate
integrals of pairs of factors act on one copy of the dense core, carrying
the prescribed kernel's TT bond. A final inner product with the original
core gives the convolved squared amplitude. The core remains in Tucker
format throughout training; no SVD or rank truncation enters its gradients.
When a coordinate pair has a large internal noise bond and small external
bonds, the operator eliminates that bond in the pair's factor moments
first. The resulting pair matrix acts directly on the dense core, avoiding
the large core-times-noise-bond intermediate. This contraction is exact;
`dataclasses.replace(forward, pair_axes=False)` retains the sequential path
for value/gradient checks and timing comparisons.
Prepared pair contractions also intersect the kernel's exact structural
blocks. Independent mixture components are processed separately, with
addition when several components reach the same output bond. Setting
`pair_groups=None` on a prepared operator retains the dense paired path
for comparison. Tests cover values and gradients for both odd dimensions
and multiple correlated pairs, including the final mixture sum.

Prepared convolution first multiplies the small univariate factors using
`T_i T_j = (T_(i+j) + T_|i-j|)/2`, then contracts their coefficients with
precomputed kernel moments. This also avoids a polynomial-size squared
moment matrix for each observation. Preparation uses the kernel's full
polynomial degree and degree-exact integration. Exact disconnected kernel blocks (including mixture components) also reuse
TT's block contraction optimization. Both direct and prepared paths reject observations outside the kernel's safe convolution window.
Prepared Tucker evaluation uses the doubling identities for even and odd
Chebyshev modes, reducing the dependency depth from the noise polynomial's
degree to its logarithm. It evaluates the same complete polynomial and
retains finite derivatives at interval endpoints; it does not truncate the
noise kernel.

A Tucker noise kernel is converted once to an untruncated TT outside
training. Existing prescribed TT kernels can be used unchanged, which
avoids introducing a second noise approximation in the comparison.

## Reproduction and comparison limits

Run in the `chebdGE` environment, within one GPU allocation following
`instructions.txt`:

```bash
export JAX_ENABLE_X64=1
python scripts/run_tucker_gallery.py
python scripts/run_tucker_cryoem.py
python scripts/benchmark_tucker_convolution.py
python scripts/check_tucker_convolution.py
python scripts/summarize_tucker_gallery.py
python scripts/audit_tucker_gallery.py
python scripts/audit_dimension_gallery.py --publication
```

The initial gallery uses 100,000 observations, the existing 4D train/validation
splits and kernels, and the same sample seeds/noise scales for new dimensions.
It fits GL in 4D (mode rank 5) and 8D (rank 3), and Swirl in 4D (rank 8) and
6D (rank 4). Each recipe runs 4,000 Adam steps. Clean GL uses Hessian strength
0.001; clean Swirl uses 0.01 in 4D and 0.003 in 6D. Noisy cases compare
full-basis strength 0.001 with strength 1e-5 and the same degree curriculum
as TT. Checkpoints and recipes are selected by held-out NLL, frozen before
truth is evaluated. Cryo-EM uses the existing 4D observations and noise,
rank 8, 49 modes, and strong/weak Hessian alternatives.

These are decomposition comparisons with feasible Tucker ranks, not
parameter-count-matched experiments. The TT gallery's canonical core-sweep
optimizer and marginal-curvature continuation are TT-specific and are not
used by this Tucker driver. All Tucker fits use the shared Adam loop, and
recipe selection uses minimum validation NLL rather than the TT gallery's
one-standard-error/marginal-curvature tie-break. These differences are
recorded so the saved errors are not mistaken for a controlled test of
representation alone.

Tucker's core contains the product of its mode ranks. The initializer checks
a default 1,048,576-entry budget before allocation. This is a guard against
accidentally requesting TT-sized Tucker ranks, not a guarantee that every
allowed model is cheap: convolution also depends on batch and kernel ranks,
and exact fourth-moment diagnostics can be more expensive than fitting.

Artifacts live in `artifacts/tucker_gallery`, with rank follow-ups in
`artifacts/tucker_capacity`. `current.json` maps the published conditions to
their frozen source folders. PNGs are saved alongside TT figures with
`tucker_` prefixes. Existing TT figures are preserved. Native Tucker pair
marginals contract the integrated modes into a small pair covariance.
The higher-rank 4D error uses exact blocked Gauss quadrature rather than a
converted TT's quartic environment. In 6D, eight independent importance-QMC
replicates estimate the fourth moment, with numerical standard errors
reported explicitly; the Swirl cross integral uses native deterministic
product-measure contraction. Full-dimensional L2 is unsquared; density truth
is used only after selection. Convolution display refinement checks remain.

## Practical stopping point

The initial exact-evaluation gallery stopped at 8D GL (rank 3) and 6D
Swirl (rank 4). `capacity_screen.json` records the resource estimates:
the next even dimensions at those ranks would give untruncated diagnostic
TT bonds 243 and 256. A single fourth-moment environment would then occupy
26.0 GiB and 32.0 GiB, respectively, before contraction intermediates.
The completed dimensions need about 0.32 GiB and 0.125 GiB for that array.
These are allocation estimates, not measured peak memory or a mathematical
limit on Tucker. Using that diagnostic cost to restrict the initial Swirl
mode ranks was overly conservative. The follow-up removes this restriction
with blocked 4D integration and uncertainty-reported 6D integration.

Custom `--root` studies stage their figures beneath that root, so smoke tests
and capacity follow-ups do not overwrite the published comparison.

## Higher-rank follow-up

The low-rank Swirl fits underfit the narrow curved density. Tucker mode
rank limits each coordinate's function space and is not comparable to a
TT bond rank. At 12,000 Adam updates with the same additive penalty, 4D
rank 8 reaches held-out NLL 1.390 while rank 16 reaches 1.239. Rank 24 with
20,000 updates and the TT sweep objective (`objective='logpen'`) reduces
full latent L2 from 0.433 to 0.120, close to TT's 0.116. Longer training
alone therefore does not explain the improvement.

The follow-up tests ranks 8 and 10 in clean 6D, and ranks 16 (4D) and 6
(6D) with prescribed noise. All choices use held-out observations. See the
[completed comparison and reproduction settings](../artifacts/tucker_capacity/report.md)
and [current figures](../figs/TUCKER.md). The original fits below remain
archived; `publish_tucker_capacity.py` updates only the observation-selected
Swirl figures and checks their provenance.

The subsequent [noisy rank-10 study](../artifacts/tucker_noisy_rank10/report.md)
tests the same rank used by clean 6D. Stopping the earlier noisy comparison
at rank 6 was a run-budget choice, not a demonstrated feasibility limit.
The rank-10 core itself occupies only 7.63 MiB. Noisy fitting is more
expensive because quadratic moment contractions retain the complete core
for each observation and their intermediates must support differentiation;
clean evaluation contracts away coordinate indices. Gradient accumulation
can reduce the batch-dependent memory cost without changing the effective
batch or objective. Rank-10 clean/noisy timing and compiled temporary-storage
measurements are saved with the new study. The 1,048,576-entry initializer
guard is configurable and must not be interpreted as a mathematical limit.

The [4D tuning follow-up](../artifacts/tucker_4d_tuning/report.md) tests
larger mode ranks, Hessian strengths and refinement of observation-selected
checkpoints. `expand_tucker_rank` preserves the represented function by
zero-padding the dense core and extending each factor with physical-L2
orthogonal directions. This lets a higher-rank fit start from an existing
Tucker estimate. The original and runtime-amended tuning plans are retained;
new truth errors are computed only after final held-out selection.
The selected rank-24 fits reach full 4D relative L2 **0.11368 clean** and
**0.22460 noisy**, compared with TT's 0.11555 and 0.29605. Clean selection
uses minimum held-out NLL; noisy selection now matches TT's paired
one-standard-error marginal-smoothness rule. The strict minimum-NLL noisy
control is also frozen and evaluated in the report. TT remains the default.

Exactly separable blocks within a noise-mixture pair now use two individual
factor actions instead of forming a dense pair matrix. Correlated blocks
retain the existing pair contraction. This changes neither the prescribed
kernel nor the fitted objective. The real 4D rank-24 benchmark records
approximately 1.25× faster value/gradient calls on an A100, with gradient
differences around 1e-15; this is not a full-training speedup measurement.

## Initial low-rank results (archived)

All eight synthetic conditions and the 4D Cryo-EM transfer completed, with
27 PNGs linked in the [Tucker figure index](../figs/TUCKER.md).

| Benchmark | Dimension | Mode rank | Clean relative L2 | Noisy relative L2 |
|---|---:|---:|---:|---:|
| GL | 4 | 5 | 0.0392 | 0.2561 |
| GL | 8 | 3 | 0.0975 | 0.2003 |
| Swirl | 4 | 8 | 0.4330 | 0.5133 |
| Swirl | 6 | 4 | 0.8915 | 0.9274 |

The rank-4 6D Swirl experiment is computationally feasible but recovers the
full density poorly. These results do not establish a best attainable
Tucker error; rank and optimizer comparisons remain distinct from the
saved TT recipes. Cryo-EM selects the weak Hessian fit with noisy validation
NLL 3.5541; its unknown latent density is not assigned an accuracy score.

On the allocated H100, the 4D rank-8 polynomial-noise microbenchmark takes
1.445 ms for explicit squaring/convolution and 0.323 ms for the direct
value-and-gradient calculation (4.47x). The explicit squared core has
16,777,216 entries versus 4,096 in the amplitude core. This is a small
controlled microbenchmark, not an end-to-end training speedup claim.
Value discrepancies are at most 1.8e-15 and coefficient-gradient
discrepancies at most 7e-15 across its four cases. The five selected
real-kernel fits agree with unprepared polynomial quadrature to within
2.2e-14 in log density on probes including a tail observation.

Focused Tucker tests and the shared TT optimizer/convolution regression
checks passed. Exact commands are recorded in
`artifacts/tucker_gallery/validation.json`; numerical benchmark results,
convolution checks, and publication provenance are retained alongside it.

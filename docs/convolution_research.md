# Gaussian convolution investigation

The estimator minimizes noisy negative log-likelihood (NLL), optionally with
Tikhonov regularization, for `q(y) = integral g(x)^2 U(y-x) dx / integral g(x)^2 dx`.
Only noisy observations and the prescribed noise enter fitting. Path labels
are used only after training and checkpoint selection, to report accuracy.

## Research and implementation choice

1. **Stable spectral convolution matrices.** Xu and Loureiro (2018),
   [Spectral approximation of convolution operator](https://arxiv.org/abs/1804.08762),
   use symmetry and recurrence relations for polynomial convolution. The newer
   Liu, Deng and Xu (2024) [Fredholm convolution paper](https://arxiv.org/html/2405.08598v1)
   is closer to this repository's fixed integration window: it chooses stable
   recurrence directions and avoids subtracting two Volterra operators. This
   is a credible way to accelerate general polynomial-kernel convolution.
   However, the installed chebdGE already avoids monomial conversion and those
   unstable recurrences: it uses Clenshaw, Gram integration and DCT interpolation.
   Replacing it with a polynomial recurrence would still require resolving the
   narrow Gaussian on a broad kernel interval and would retain the rank-squared
   signal representation unless the contraction were also restructured.

2. **Positive deterministic quadrature with the analytic Gaussian.**
   [NIST DLMF 3.5](https://dlmf.nist.gov/3.5) gives Gaussian quadrature,
   its positive weights and polynomial exactness. Gauss--Legendre nodes on
   the signal interval retain its finite support exactly. A whole-line
   Gauss--Hermite rule would need special treatment of the box cutoff;
   evaluating the polynomial outside its signal domain changes the model.
   Fixed Legendre quadrature is differentiable in every trainable coefficient.
   We evaluate the prescribed Gaussian directly, avoiding polynomial kernel
   interpolation and output interpolation. Node doubling estimates resolution
   error; no claim of exact Gaussian integration is made.

3. **Contract the two copies of the original TT directly.** For each axis,
   form `M_ij(y) = integral T_i(x) T_j(x) Gaussian(y-x) dx`. Then propagate
   `S_kl = sum_pqij S_pq C_pki M_ij C_qlj`. Contract in that order; never
   materialize a rank-four transfer tensor or the rank-squared TT square.
   This is algebraically the same squared-TT forward model. It costs
   `O(B d (r^3 n + r^2 n^2 + Q n^2))`, with observation batch B, degree
   count n, rank r and quadrature order Q. Memory per mode is approximately
   `O(B (r^2 n + n^2 + Q n))`. This removes the need to differentiate an
   SVD truncation. It is the main implemented backend, `GaussianConvolution`.

4. **Positive transfer and scaling.** A second implementation (`positive=True`)
   computes `sum_t w_t A(x_t)^T S A(x_t)` directly, where every weight is
   positive. It avoids cancellation when forming weighted polynomial Gram
   matrices and provides a separate contraction check, at a higher cost
   `O(B d Q r^3)`. Both modes scale each core, each propagated state, and the
   Gaussian weights; logarithms carry the compensating factors. The
   normalization uses the exact Chebyshev Gram matrix with the same scaling.
   No observation density clipping is used to hide negative forward values.
   These are numerical protections, not statistical regularizers.

5. **Statistical ill-conditioning remains distinct.** Even an accurately
   evaluated Gaussian convolution suppresses high frequencies. Fan (1992),
   [Deconvolution with supersmooth distributions](https://onlinelibrary.wiley.com/doi/10.2307/3315465),
   establishes the associated slow nonparametric recovery rates. Faster,
   more accurate integration cannot by itself make an unregularized inverse
   problem identifiable from finite data. The existing regularizer remains
   available; this investigation does not replace NLL with supervised fitting
   of the true path probabilities.

FFT convolution was also considered. It would require padding, grid resolution,
interpolation back to observations and preservation of TT structure. For this
analytic separable Gaussian, direct one-dimensional integration avoids those
additional error sources. The principal comparison retains the existing diagonal, averaged covariance.
An additional experiment rotates observations into the eigenbasis of the
full prescribed covariance, where the same operator applies exactly. The
rotation has unit absolute determinant, so no density Jacobian is needed.
Heteroscedastic per-observation noise remains a separate modeling extension.

## Numerical validation

`tests/test_gaussian_convolution.py` checks analytic uniform-box convolution
(including outside observations), degree-64 Chebyshev integration against
independent adaptive SciPy quadrature, quadrature doubling, the positive
transfer implementation, finite-difference gradients, extreme TT rescaling,
and fitting from samples only. `scripts/benchmark_convolution_accuracy.py`
extends the comparison to dimension 100 and to dimension 20/rank 32/degree 64.
Its `--high-degree` option additionally tests dimension 10/rank 16/degree 256.
Random polynomial functions here are **operator validation fixtures only**;
no estimator is initialized or fit using an unknown ground-truth density.

The `degree` arguments inherited from this repository actually specify the
number of Chebyshev coefficients. For example `--degree 3` means degree 2.

## Cryo-EM protocol

`scripts/benchmark_cryoem_convolution.py` uses the existing dim-10 path cache,
a seed-0 80/20 split, the training-derived 1% trimmed box with 5% margin, and
KDE initialization (`n_starts=64`, `gd_steps=10`). Training and validation
observations outside that box are excluded to reproduce the existing
protocol. This is a limitation: selection of noisy observations changes the
statistical population. The new operator itself allows observations outside
the signal box and does not require this trimming.

Every run minimizes observation NLL. Every 100 steps it measures the same
analytic-Gaussian noisy NLL on up to 4,096 held-out observations (the naive
baseline uses its own observation-density NLL); the minimum
selects the checkpoint. Ground-truth labels and anchors are accessed only
after this selection. The KDE comparison uses the same training observations.
The original figure used all observations, so its KDE error (0.076) is not
numerically identical to this split's baseline. The path metric is retained
for compatibility, but density at a state centroid is not generally equal to
integrated state probability; it is a diagnostic, not a full joint-density
L2 measurement.

Results and configuration are recorded in `artifacts/convolution/*.json`, with
fitted cores in matching local `.npz` files (gitignored). Reproduce inside the
`chebdGE` environment on a single GPU allocation (partition cryoem, four CPUs,
32 GB host memory). Reuse the allocation with `srun --jobid=...`:

```bash
python -m pytest tests/test_gaussian_convolution.py -q
python scripts/benchmark_convolution_accuracy.py
python scripts/benchmark_cryoem_convolution.py --rank 16 --degree 3 \
  --output artifacts/convolution/gaussian_r16_n3.json
python scripts/benchmark_cryoem_convolution.py --rank 8 --degree 5 \
  --backend spectral --output artifacts/convolution/spectral_r8_n5.json
python scripts/benchmark_cryoem_convolution.py --rank 8 --degree 5 \
  --compare-spectral --output artifacts/convolution/gaussian_r8_n5.json
```

For the existing full-data plotting workflow:

```bash
python scripts/cryoem_path_density.py 10 --rank 16 --degree 3 \
  --convolution-backend gaussian --naive-steps 0
```

The legacy `spectral` default is preserved. To use the new operator directly:

```python
from ftde import GaussianConvolution, fit_noisy_tt
operator = GaussianConvolution.from_tt(init, noise_std)
fitted, history = fit_noisy_tt(init, None, noisy_y,
                              gaussian_convolution=operator)
```

For a convergence check, construct another operator with
`quadrature_order=[2*q for q in operator.orders]`. The default node-count rule
is a conservative heuristic, not an a posteriori accuracy certificate.
The direct backend currently supports TT with independent Gaussian noise.
Other kernels and Tucker continue through the existing spectral backend.
Nonzero Tikhonov regularization still explicitly forms the square for the
penalty, so its separate high-rank cost remains.

## Noise-model diagnostics

A fresh check of the dim-10 cache finds noise/observed variance ratios of
approximately `[.052, .087, .124, .135, .200, .239, .353, .453, .561, .840]`.
The largest off-diagonal noise correlation is 0.212. Thus the old statement
that noise is uniformly mild, based on dim-4, does not apply to the full
10D embedding. The largest generalized eigenvalue of diagonal noise
covariance relative to observed covariance is 0.849 before trimming and
0.945 after the existing box filtering. This does not prove a noise-model
failure, but makes recovery in the later coordinates sensitive and motivates
checking the data-selection assumption independently of numerical integration.
These measurements are saved in `artifacts/convolution/noise_diagnostics.json`.

The new `--keep-outside` benchmark option retains all noisy observations while
keeping the training-derived signal box. It removes the observation-selection
bias of dropping noisy tails. Its NLL must **not** be compared directly with
trimmed-data NLL: the validation populations differ. The legacy path metric
also extrapolates a clean polynomial for anchors outside its support; state
45 is outside the reference box. New benchmark records report both that
legacy score and a support-corrected score, plus the outside-anchor indices.


The `--noise-eigenbasis --keep-outside` experiment uses the **full averaged**
covariance without a polynomial approximation to its correlated kernel.
The TT is fit in rotated coordinates; the saved `.npz` includes `rotation`,
so physical points must be multiplied by that matrix before evaluation.
The KDE baseline remains in the original coordinates. A rotated signal box
is a different support choice, so this comparison also changes the model
geometry, not only the covariance approximation.

`scripts/evaluate_convolution_fits.py` reevaluates saved convolved models on
one common set of 4,096 untrimmed held-out observations, to make comparisons
across data-selection protocols meaningful. It never loads labels.

## Results (2026-09-19, A100, chebdGE environment)

The numerical operator tests were successful. Maximum absolute log-density
errors against a doubled-order positive-transfer reference were:

| Dimension | Rank | Polynomial degree | Maximum log error | Finite gradients |
|---:|---:|---:|---:|---|
| 3 | 4 | 16 | 3.83e-13 | yes |
| 10 | 8 | 4 | 4.10e-13 | yes |
| 10 | 16 | 32 | 3.91e-14 | yes |
| 20 | 32 | 64 | 2.56e-13 | yes |
| 100 | 4 | 4 | 3.27e-13 | yes |
| 10 | 16 | 256 | 4.05e-13 | yes |

These are empirical error measurements for the recorded fixtures, not a
uniform bound over all coefficient tensors, gauges or tail observations.
The one-dimensional degree-64 test also agrees with independent adaptive
integration. Increasing polynomial resolution of the *legacy* Gaussian
kernel reduces its error too: in the dimension-10/rank-8 fixture, kernel
coefficient counts 65, 129, 257 gave log errors 0.0384, 1.65e-7, 6.27e-13.
Thus the old convolution is accurate when its input kernel is well resolved.

On the real dim-10 cryo-EM model, doubling direct quadrature resolution gave
errors below 1e-12 in all recorded runs. At matched rank 8/coefficient count 5,
legacy and direct training produced path errors **0.3441494554** and
**0.3441494543**. Their held-out NLLs also agreed to 1e-10. This is direct
experimental evidence against forward-convolution error being the principal
cause of that fit's poor path score.

The new contraction nevertheless removes a major computational limitation.
The 1,500-step matched fits took about 325 seconds (spectral) and 23 seconds
(direct), including compilation and validation. These end-to-end runs shared
an allocation with some other experiments, so use the separate isolated
`benchmark_convolution_gradient.py` measurement for a controlled timing.
It compares a rank-16 model, batch size 512, identical coefficients and
observations **inside the legacy output domain**, with compilation excluded.
The recorded steady gradient times were 0.001784 seconds (direct) and
2.299629 seconds (spectral), approximately **1,289x faster** in this specific
rank-16/batch-512 benchmark. NLLs differed by 1.25e-8 and coefficient gradients
by 2.53e-7 relative L2, consistent with the degree-128 kernel approximation.
This ratio is not a universal speedup claim. Results, device, kernel
resolution and gradient agreement are saved in
`artifacts/convolution/rank16_gradient_timing.json`.

The previous rank-16 SVD failures were avoided: no SVD or explicit square
is used in the forward operator. Both rank-16 and rank-32 fits completed
without NaNs. The full-data workflow reproduced the old naive rank-16 path
score (0.110867); direct deconvolution at rank 16/coefficient count 3 reached
**0.168331**, compared with **0.338** in the old rank-8 figure. KDE remained
better at **0.075709**. This improvement comes from enabling a different
capacity, not from more accurately evaluating the same rank-8 fit.

The held-out experiments give a more nuanced comparison:

- Rank 16 / 3 coefficients: path error 0.166 at 1,500 steps, 0.157 at the
  checkpoint selected from 6,000 steps. This operating point was chosen
  from the repository's earlier rank investigation, not from a new fit to
  labels. It is not the model favored by noisy likelihood.
- Rank 16 / 9 coefficients, all observations: common held-out noisy NLL
  **94.393858**, compared with KDE **94.912792** and the matched rank-8
  baseline **95.386471**. Lower is better. The learned observation density
  improves substantially under the actual fitting objective. Its clean
  path error is 0.284 (KDE on that training set: 0.071).
- Rotating to use the full noise covariance produces almost the same
  held-out NLL (94.394892), but path error 0.443. This test did not help.
- Tikhonov regularization at rank 16 / 9 coefficients reduced path error
  from 0.258 to 0.251 in the trimmed-data runs, while common untrimmed
  NLL increased from 94.4263 to 94.4493. The regularized run also used a
  lower learning rate, so this is not an isolated causal estimate of the
  penalty's effect.
- Rank 32 / 5 coefficients gave path error 0.357; simply increasing rank
  further did not solve the problem.

Every checkpoint was selected by held-out observation likelihood. No
configuration was trained using path labels or anchor probabilities. The
path-best operating point and the likelihood-best operating point differ;
there is no claim that selecting the former is justified by likelihood.
The common validation set is reused across configurations and is therefore
a model-selection set, not an untouched final test set. These are seed-0
experiments; replication across seeds remains necessary for statistical
performance claims.

The implementation successfully improves numerical reliability and cost,
and enables useful higher-capacity observation-density fits. It **does not
make the clean cryo-EM path estimate beat KDE**. The evidence points to
remaining inverse-problem sensitivity, noise-model assumptions, and the
centroid-density/state-mass mismatch rather than a large forward-integration
error at the old working resolution.

Full configurations, losses, path vectors and errors are in
[the results table](../artifacts/convolution/summary.md).
[The comparison figure](../artifacts/gallery_archive/before_dimension_gallery_v9/archive/convolution/cryoem_comparison.svg) shows
both likelihood and path accuracy; the full-data comparison is saved locally
as `figs/archive/convolution/full_data/cryoem_path_dim10.png`.

Additional reproduction commands:

```bash
python scripts/benchmark_convolution_accuracy.py --high-degree \
  --output artifacts/convolution/operator_degree256.json
python scripts/benchmark_cryoem_convolution.py --rank 16 --degree 9 \
  --steps 6000 --keep-outside --output artifacts/convolution/gaussian_r16_n9_all.json
python scripts/benchmark_cryoem_convolution.py --rank 16 --degree 9 \
  --steps 6000 --keep-outside --noise-eigenbasis \
  --output artifacts/convolution/gaussian_r16_n9_fullcov.json
python scripts/evaluate_convolution_fits.py
python scripts/benchmark_convolution_gradient.py
python scripts/plot_convolution_benchmarks.py
```

Validation completed: all 17 existing noisy-density regression tests and all
7 new Gaussian-convolution tests passed on the allocated GPU. Repository-wide
`ruff check src/ tests/ scripts/` and `git diff --check` also passed. The full
noise-free GPU suite was not rerun because its implementation was unchanged.

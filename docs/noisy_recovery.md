# Noisy density recovery investigation

The estimator still optimizes the Chebyshev coefficients of a single
`chebttge.ChebTT`, with density `p = g² / integral(g²)`. No GL formula,
clean observations, symmetry, or known Markov structure enters its objective,
initialization, or checkpoint/hyperparameter selection.

## Results (2026-09-24)

On the original 10D paired-noise case (noise scale 1.5, 100,000 training
observations), noisy-only selection chose **25 coefficients, rank 10,
curvature strength 3e-5, step 8,000**. Integrated relative L2 improved from
**1.6059 to 0.4549** (3.53 times smaller). The metric printed in the original
figure improved from **2.1118 to 0.2588**. These are single-seed results,
not a claim of noise-free accuracy or performance on all noise models.

| Fit | Coefficients | Relative sample error | Squared sample error | Integrated relative L2 |
| --- | ---: | ---: | ---: | ---: |
| Original saved RSI | 17 | 1.4532 | 2.1118 | 1.6059 |
| Matched direct, legacy penalty | 17 | 1.1993 | 1.4383 | 1.4116 |
| Matched direct, no penalty | 17 | 1.2994 | 1.6885 | 1.5295 |
| Curvature, strength 1e-4 | 17 | 0.7670 | 0.5882 | 0.6484 |
| No penalty, larger basis | 25 | 1.9249 | 3.7051 | 2.7737 |
| Curvature, strength 1e-5 | 25 | 0.5344 | 0.2856 | 0.6614 |
| **Selected curvature, strength 3e-5** | **25** | **0.5088** | **0.2588** | **0.4549** |

The matched direct controls and all new candidates use batch size 256; the
historical RSI fit used 512. Thus the original-vs-new difference is a
pipeline comparison. The matched controls show a substantial regularization
benefit independently: increasing basis size without smoothing makes
recovery worse, while adding smoothing helps even at the original size.

The selected fit's excess noisy NLL is 0.000480 per observation, with paired
cluster standard error 0.000705, relative to the minimum-NLL candidate.
The minimum-NLL candidate is the unregularized 25-coefficient model, whose
integrated error is 2.7737. This is a concrete example of why choosing the
smallest noisy NLL alone is inadequate for latent-density recovery.

The selected fit's convolution log-density changes by at most 2.44e-13 on
the quadrature-doubling probe. Doubling the independent integrated-error
quadrature from 64 to 128 changes its integrals by 1.76e-11 relative to
`integral p_true²`. The largest such discrepancy across all evaluated models
is 2.44e-9, for the rough unregularized 25-coefficient fit. These numerical
errors are negligible compared with the observed recovery errors.

Figures: [matched marginals](../artifacts/figure_refresh/previous_figs/ginzburg_landau/noisy_recovery/paired_comparison.png),
[all recovery errors](../artifacts/figure_refresh/previous_figs/ginzburg_landau/noisy_recovery/recovery_errors.png),
[validation curves](../artifacts/figure_refresh/previous_figs/ginzburg_landau/noisy_recovery/validation_curves.png).
Local run data: [frozen selection](../artifacts/noisy_recovery_evaluation/selection.json),
[all metrics](../artifacts/noisy_recovery_evaluation/metrics.json),
[selected coefficients](../artifacts/noisy_recovery_n25_mid/h2_3e-05/fit.npz).

## What was wrong

The paired-noise figure reports **squared relative error on clean sample
points**, 2.1118, whose square root is 1.4532. It is a full 10D density-value
diagnostic, not the error of the displayed 2D marginal and not integrated
Lebesgue L2. Integrating out eight dimensions can hide severe density errors.
The noise-free script also reports squared sample error. For example, .01
in that convention is .1 after taking its square root, and still differs
from integrated relative L2. Comparisons with published estimators require
matching these definitions, data, and normalization conventions.

The saved forward-operator audit already establishes that RSI is accurate
on this example: relative density RMS discrepancy against direct contraction
is 8.93e-8 over 2,048 noisy validation observations. Increasing convolution
accuracy alone cannot explain/fix the recovery error. See
[the original audit](rsi_figures.md).

The old regularizer has two concrete problems:

1. It adds weighted squared entries of individual cores of `square_tt(g)`.
   These entries depend on TT gauge, rather than only on the represented
   density. Scaling every amplitude core by `a` leaves normalized `p`
   unchanged but multiplies this penalty by `a⁴`. On the failing saved fit,
   scaling cores by .01 changes the penalty from 1.2312e-5 to 1.2312e-13.
   The optimizer can therefore reduce the penalty without smoothing the
   density. General invertible bond transformations also change it.
2. Its weights are `expm1((sigma*pi*k/width)²)`, rescaled so the largest
   weight equals .5. For the paired case, sigma=.829759 and width≈6.33,
   giving degree-16 weight 1.19e-57, degree-24 weight 4.65e-34, degree-30
   weight 3.60e-10, degree-31 weight 1.13e-5, and degree-32 weight .5.
   Almost all the nominal regularization budget goes to the last degree
   of `g²`. This does not constrain many distortions the noise obscures.
   Chebyshev degree is also only a heuristic proxy for Fourier frequency:
   these polynomials are not eigenfunctions of finite-window convolution.
   In particular, increasing noise while keeping the maximum weight fixed
   makes the relative weights at lower degrees *smaller*, precisely when
   more regularization of unresolved structure is needed.

These defects are independently reproducible; they do not establish that
every earlier inaccurate fit was caused entirely by the regularizer.

## Replacement and selection

`AmplitudeSobolevPenalty` computes

```
R_m(g) = sum_k s_k^(2m) * integral (partial_k^m g)² / integral g²
objective = mean noisy NLL + lambda * R_m(g).
```

The experiment uses `m=2`, with length scales `s_k` equal to the known noise
marginal standard deviations. The derivative Gram matrices use physical
coordinates; all remaining modes are integrated by TT contractions. No
rank-squared signal is needed for this penalty. Normalized environments and
core rescaling protect its evaluation. Its value is invariant to amplitude
rescaling and invertible TT gauge changes, within floating-point accuracy.

This is an axiswise curvature seminorm of the amplitude. It omits mixed
derivatives and jumps at the signal-box boundary; it is not an exact inverse
of the correlated noise covariance. `m=1` is also supported and corresponds
to one quarter of interior density Fisher information when `s_k=1`.

Derivative roughness penalties for deconvolution are supported by
[Cai, Gu and Kenney](https://arxiv.org/abs/2105.11205) and
[Kent and Ruppert](https://arxiv.org/abs/2205.09800). Those papers penalize
derivatives of the density; our amplitude penalty is an adaptation compatible
with the squared-TT parametrization, not a reproduction of either estimator.
Amplitude-based density regularization is discussed in
[this review](https://stat.ethz.ch/Manuscripts/maechler/wp-Density-CS96.pdf).

Each candidate first selects its checkpoint by noisy validation NLL. Across
candidates, `select_by_paired_noisy_nll` chooses the smallest roughness among
models within one paired standard error of the minimum-NLL candidate. The
uncertainty is computed from differences on the **same observations**.
For this GL sampler, chain IDs are clusters, allowing temporal dependence
within a chain. This rule is a selection heuristic, not a confidence guarantee
after hyperparameter search. Clean truth is evaluated only after selection
has been written to disk.

## Why noise-free accuracy is not guaranteed

Although convolution is injective for an exact Gaussian and infinite data,
it suppresses a Fourier direction `omega` by
`exp(-omega.T @ covariance @ omega / 2)`. Finite-sample errors in the noisy
density get amplified when that factor is inverted. Optimizing longer does
not recover information absent from the observations; it can instead fit
poorly identified directions. This is the supersmooth deconvolution problem
studied by [Fan](https://onlinelibrary.wiley.com/doi/10.2307/3315465).
Smoothness trades some bias for reduced amplification and must be validated.

## Use with an arbitrary prescribed TT kernel

```python
from ftde import AmplitudeSobolevPenalty, TTConvolution, fit_noisy_tt

operator = TTConvolution.from_tt(initial_g, noise_tt)
penalty = AmplitudeSobolevPenalty.from_tt(
    initial_g, order=2, length_scales=noise_std,
)
fitted, history = fit_noisy_tt(
    initial_g, noise_tt, noisy_samples,
    forward_operator=operator,
    regularizer=penalty, reg_strength=3e-5,
    n_steps=8000, batch_size=256, lr=5e-4,
)
```

The strength above is an example, not an automatically calibrated universal
default. Hold out noisy observations to choose strength and stopping. The
benchmark additionally uses Gram-preconditioned parameters and cosine learning
rate decay; the minimal API example uses the library's existing optimizer.
The old core penalty remains available for reproducing previous experiments.
The existing GL driver accepts `--regularizer h2 --reg-lambda 0.00003`, with
either direct or RSI convolution. Nothing in `../chebdGE` was changed.

## Reproduction

Use the `chebdGE` environment and one GPU allocation as specified in
`instructions.txt`. The saved setup comes from the documented paired RSI
setup command in `docs/rsi_figures.md`. It contains the noisy-KDE initializer
and prescribed noise kernel, not a fitted GL truth model.

```bash
python scripts/benchmark_noisy_recovery.py --strengths 0.0001 0.001 0.01 --steps 8000
python scripts/benchmark_noisy_recovery.py --strengths 0 --legacy --steps 8000 \
  --output artifacts/noisy_recovery_controls
python scripts/benchmark_noisy_recovery.py --strengths 0.0001 0.001 0.01 \
  --n-coeffs 25 --steps 8000 --output artifacts/noisy_recovery_n25
python scripts/benchmark_noisy_recovery.py --strengths 0.00001 0 \
  --n-coeffs 25 --steps 8000 --output artifacts/noisy_recovery_n25_weak
python scripts/benchmark_noisy_recovery.py --strengths 0.00003 \
  --n-coeffs 25 --steps 8000 --output artifacts/noisy_recovery_n25_mid
python scripts/evaluate_noisy_recovery.py artifacts/noisy_recovery \
  artifacts/noisy_recovery_controls artifacts/noisy_recovery_n25 \
  artifacts/noisy_recovery_n25_weak artifacts/noisy_recovery_n25_mid
```

Training uses 100,000 noisy points, 2,048 noisy validation points, rank 10,
noise scale 1.5, and the original seeds. All candidates start from the same
function. The 25-coefficient candidates zero-pad the 17-coefficient
initializer, increasing representational freedom without adding information.
The comparison uses direct prepared convolution so it isolates the
regularization/basis effects; the matched legacy and unregularized controls
use the same operator, batches, optimizer, and stopping rule.

The weaker-penalty extension was specified before inspecting any clean
density error, because the first validation comparison preferred the smallest
positive strength in its grid. This checks the remaining interval down to
unregularized fitting rather than assuming the grid endpoint is optimal.
The intermediate strength 3e-5 was then added, still before any clean-density
evaluation, to refine the interval between nearly unchanged noisy likelihood
at 1e-5 and a detectable likelihood cost at 1e-4 in the original basis.

## Verification

Run in the `chebdGE` environment:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_sobolev.py tests/test_gaussian_convolution.py -q
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_noisy_selection.py -q
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_noisy_density.py tests/test_tt_convolution.py -q
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_gl_l2.py -q
```

These cover independent first/second derivative cubature, physical-domain
scaling, nonorthogonal bond gauge invariance, extreme amplitude rescalings,
finite-difference gradients, JIT evaluation, sample-only regularized fitting,
paired/cluster uncertainty, and the existing noisy-estimator and convolution
regressions, plus the independent integrated-L2 evaluator checks. The 40
distinct tests pass. The benchmark checks prepared
convolution against direct integration before fitting and doubled quadrature
after checkpoint selection. Python compilation and Ruff checks pass for
the changed implementation and experiment scripts.

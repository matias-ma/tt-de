# General noise kernels and noisy Ginzburg–Landau

## What the Gaussian assumption does, and does not do

The phrase “exact Gaussian convolution” was too strong: `GaussianConvolution`
evaluates the Gaussian noise density analytically, but integrates numerically.
It does not assume the unknown signal density is Gaussian. The diagonal
Gaussian was the prescribed noise model in the cryo-EM experiment, not a new
assumption about the signal.

There are two separate simplifications:

1. Independent noise has a product kernel, so every integral needed for a TT
   contraction is one-dimensional. Gaussian noise also has a convenient
   analytic formula, removing the error of approximating the noise kernel by
   polynomials. An independent non-Gaussian kernel has the same separability.
2. The substantial algorithmic change is contracting two copies of the original
   signal TT directly. We do not construct the rank-squared polynomial TT for
   `g**2`, interpolate the output convolution, or differentiate an SVD rounding.
   This works beyond Gaussian noise and beyond independent noise.

“Gauss–Legendre quadrature” does **not** mean the kernel is Gaussian. It is a
numerical integration rule applicable to general integrands. Its polynomial
exactness is described in [NIST DLMF 3.5](https://dlmf.nist.gov/3.5).

For a prescribed noise TT with core `U_k(z)[a,c]`, the new `TTConvolution`
builds the one-dimensional moments

```
M_k(y)[a,c,i,j] = integral T_i(x) T_j(x) U_k(y-x)[a,c] dx.
```

It propagates an environment with two signal bond indices and one noise
bond index. Rank-one noise recovers the independent-kernel case; larger noise
ranks carry correlations. Gaussian mixtures and non-Gaussian polynomial
kernels work through the same interface. The cost grows with noise TT rank:
an arbitrary high-dimensional kernel need not have a cheap TT representation.

If the signal has `n` coefficients and the kernel has `m`, the product has
degree at most `2(n-1)+(m-1)` in the integration variable. We use
`Q = n + floor((m-1)/2)` Gauss–Legendre nodes, sufficient to integrate that
**represented polynomial** exactly in exact arithmetic. Floating-point error
and approximation of the true noise by the kernel TT remain separate errors.
The supplied kernel must represent a nonnegative, normalized density; the
operator does not silently clip negative kernel approximations.

The kernel box must contain `y - signal_box` for each observation. The operator
checks this safe window and returns NaN outside it rather than extrapolating.
The analytic Gaussian backend has no corresponding finite kernel-box limit.

## Usage

```python
from ftde import TTConvolution
from ftde.noisy_tt_density import fit_noisy_tt

operator = TTConvolution.from_tt(initial_signal, noise_tt)
fitted, history = fit_noisy_tt(
    initial_signal, None, noisy_samples,
    forward_operator=operator,
    n_steps=6000,
)
```

The older `gaussian_convolution=` argument is retained for compatibility;
`forward_operator=` accepts either operator. The existing GL driver also accepts
`--convolution-backend direct`. It retains its existing kernel construction and
other settings; it is distinct from the controlled benchmark below.

## GL experiment protocol

The benchmark uses the repository's GL distribution with beta=1/8, lambda=.02
and h=1/(d+1), 20,000 noisy training samples, 2,048 separate noisy validation
samples, and 4,000 clean test samples. Sampling uses the repository's parallel
Metropolis chains with different seeds for these three sets; residual MCMC
dependence is possible. KDE bandwidth and TT initialization use
only noisy training samples. A checkpoint is selected by validation NLL, every
200 steps. The true GL density is first evaluated after both fits and checkpoint
selection have finished. Training minimizes noisy NLL with the existing
regularizer, not an L2 loss against the known density.

The naive and deconvolved fits start from the same rank-8 initialization.
Eight-dimensional fits use 17 coefficients and 6,000 Adam steps; the
20-dimensional fit uses 21 coefficients and 4,000 steps. Deconvolution uses
learning rate 0.0003, batch size 256, regularization multiplier 1, and the
existing per-axis target budget .5. These are single-seed runs, not uncertainty
estimates or a comprehensive hyperparameter search.

Noise cases:

- Gaussian: independent standard deviation .55, analytic Gaussian backend.
- Mixture: `0.7 N(0, .3² I) + 0.3 N(0, .9² I)`, a non-Gaussian joint noise
  distribution with one shared component label per observation. The general
  backend uses its rank-2 polynomial TT with 257 coefficients. This is the
  repository's isotropic GL noise model, not a product of marginal mixtures.
- Paired: Gaussian noise with standard deviation sqrt(.7*.3²+.3*.9²),
  correlation .5 within adjacent coordinate pairs. The general backend uses
  the existing pairwise kernel builder with maximum rank 12, whose error is
  measured separately against the prescribed noise. The initial maximum-rank-3
  run encountered a nonfinite training loss at step 134; its failure is recorded
  separately, not discarded as a completed density estimate.

All noisy observations enter the convolved fit, including observations outside
the signal box. The naive fit filters them because its density has finite box
support. The box is estimated from noisy training samples using the existing
robust-domain rule. Thus this compares useful estimator pipelines; it is not
a controlled head-to-head training comparison with the legacy convolution.

## Error convention and artifacts

We additionally compute the **integrated** relative L2 error
`||p_est - p||_2 / ||p||_2`, using deterministic integration after fitting.
See [the integrated error table](../artifacts/gl_convolution/integrated_summary.md)
and [the integrated error plot](../artifacts/gallery_archive/before_dimension_gallery_v9/archive/gl_convolution/gl_integrated_errors.png).

`evaluate_gl_l2.py` evaluates `integral p_est^2` by contracting four copies of
the fitted TT. It evaluates `integral p_est*p` by combining two TT copies with
the nearest-neighbor GL transfer kernel, and obtains `integral p^2` from the
GL partition function at twice the inverse temperature. The signal is zero
outside its box, while `integral p^2` includes the GL reference's full
integration window. This avoids Monte Carlo weighting ambiguity. Both 64- and
128-node quadrature are run; the JSON records the difference. The partition
function uses the repository's 2,000-point grid on [-3,3], with negligible GL
tails beyond that interval. The evaluator is independently checked against
two-dimensional tensor-product cubature. These integrated results cover the
two TT estimators; the KDE comparison below uses the existing sample metric.

The repository calls the following quantity “relative L2 error”:

```
E² = sum_i (p_est(x_i) - p_true(x_i))² / sum_i p_true(x_i)²,
x_i sampled from p_true.
```

The JSON records both this squared value and `E = sqrt(E²)`. The plot and
summary table use E. This is a relative error on held-out clean samples,
equivalently an estimate involving p-weighted integrals; it is **not** an
estimate of the unweighted Lebesgue L2 norm. KDE here is fitted to noisy data,
without deconvolution, as in the original comparison.

See [the recorded table](../artifacts/gl_convolution/summary.md) and
[the error plot](../artifacts/gallery_archive/before_dimension_gallery_v9/archive/gl_convolution/gl_errors.png). Per-run JSON files
in that directory record both error conventions, checkpoint histories,
quadrature orders and convolution checks. Local NPZ files contain the selected
TT coefficients. Doubling quadrature checks integration accuracy; comparison
with an analytic Gaussian mixture separately checks the represented kernel.
Agreement of operators is not itself evidence that the statistical inverse
problem has been solved.

Reproduce within the allocated GPU node and `chebdGE` environment:

```bash
python scripts/benchmark_gl_convolution.py --dim 8 --noise gaussian --output artifacts/gl_convolution/gl8_gaussian.json
python scripts/benchmark_gl_convolution.py --dim 8 --noise mixture --compare-spectral --output artifacts/gl_convolution/gl8_mixture.json
python scripts/benchmark_gl_convolution.py --dim 8 --noise paired --kernel-rank 12 --output artifacts/gl_convolution/gl8_paired.json
python scripts/benchmark_gl_convolution.py --dim 20 --noise mixture --n-coeffs 21 --steps 8000 --output artifacts/gl_convolution/gl20_mixture.json
python scripts/evaluate_gl_l2.py artifacts/gl_convolution/gl8_gaussian.json artifacts/gl_convolution/gl8_mixture.json artifacts/gl_convolution/gl8_paired.json artifacts/gl_convolution/gl20_mixture.json
python scripts/plot_gl_convolution.py
```

## Training-recipe fix (2026-09-19): Gram preconditioning closed most of the gap to gl_covariance_mixture

The numbers above (interpretation section) were reproduced with the *original*
training recipe and are now stale; they are kept for historical comparison.
Comparing this benchmark's results against `artifacts/gl_covariance_mixture/`
(a separate, unrelated experiment that reaches integrated relative L2 ~0.2 on
a similar noisy-GL target) showed a large, unexplained accuracy gap that
turned out to have nothing to do with the convolution operator itself
(already validated to floating-point accuracy above and in
`docs/convolution_research.md`). It was an optimization/regularization gap in
this benchmark's training loop, fixed by porting three ingredients already
validated in `benchmark_gl_covariance.py`:

1. **Gram/Cholesky-preconditioned coefficients.** Optimizing the raw
   Chebyshev coefficients gives high-degree modes a very different effective
   step size than low-degree ones (their Gram-matrix eigenvalues differ by
   orders of magnitude); optimizing `c @ chol(gram)` instead equalizes this.
   Without it, the deconvolved fit here picked its best checkpoint at ~20-25%
   of the training budget and then visibly degraded afterward (see the old
   `steps=6,000 (naive selected 200, deconv selected 1,400)` titles this
   script used to print); with it, checkpoints are selected much later in
   training and stay stable (e.g. `deconv selected 6,800` of 10,000 below).
2. **Cosine-decayed learning rate** instead of a fixed one for the whole run.
3. **`TTConvolution.prepare()`** to precompute integration moments once,
   making more steps/data affordable in the same wall-clock budget (this is
   a speed lever only -- it does not change the represented forward model).

`scripts/benchmark_gl_convolution.py` now does all three by default (disable
preconditioning with `--raw-coordinates` to reproduce the old behavior), plus
separately overridable `--naive-steps`/`--deconv-steps` (0 skips training and
reports the untrained KDE cross-approximation itself). Re-running the exact
commands above with the new defaults (`--n-train 100000 --steps 10000`, the
new script's own defaults) gave integrated relative L2:

| Config | Old | New |
|---|---:|---:|
| 8D Gaussian | 0.956 | 0.345 |
| 8D mixture | 0.871 | 0.144 |
| 8D paired | 0.951 | 0.349 |
| 20D mixture | 0.991 | 0.281 |

`scripts/run_gl_convolution_experiments.py` reruns these four configs plus a
rank / Chebyshev-degree / training-set-size scaling sweep (dim=8, mixture
noise) and one explicit `--naive-steps 0` example, into
`artifacts/gl_convolution_scaling/`; `scripts/plot_gl_convolution_scaling.py`
turns that into `scaling_summary.png` (error vs. each of rank/n_coeffs/n),
`training_curves.png` (validation NLL vs. step, naive vs. deconvolved -- the
naive fit's NLL visibly turns and gets worse after its early best checkpoint
since it has no noise model to protect it, while the deconvolved fit
plateaus), and `naive_steps_zero_comparison.png` (KDE vs. untrained-init
naive vs. trained naive vs. deconvolved, showing that at this noise level
training the naive/no-deconvolution model barely helps over the untrained
KDE cross-approximation it started from, while deconvolution is transformative).
These scaling runs use smaller `--n-train 20000 --steps 6000` than the four
validation configs above (to keep the sweep affordable), so their absolute
errors are not directly comparable to the validation table.

Operator tests cover a correlated non-Gaussian polynomial kernel against
independent tensor cubature, agreement of values and coefficient gradients
with the legacy convolution, and sample-only NLL training. These tests and the
Gaussian operator tests passed (10 tests), as did the existing noisy-density
tests (17 tests). The integrated GL error evaluator also passed its independent
tensor-cubature check (1 test).

## Interpretation

Integrated relative L2 errors for naive / deconvolved TT are 0.955602 / 0.441779
for 8D Gaussian noise, 0.871031 / 0.267524 for the 8D mixture, 0.950812 / 0.424542
for 8D paired noise, and 0.991318 / 0.462581 for the 20D mixture. Across these
fits, doubling quadrature changes either error integral by at most 4.2e-13
relative to `integral p^2`. These are single-run estimator comparisons, not
statistical confidence intervals.

The paired-noise run completes with kernel rank 12 after the rank-3 run fails.
On 2,000 prescribed-noise diagnostic samples, the rank-12 kernel has relative
sample error 8.97e-7 and no negative evaluated values; this is a sampled check,
not a proof of global positivity. Its convolution quadrature-doubling error is
1.82e-13 in log density. Changing the kernel rank changes the represented noise
approximation, which remains distinct from the accuracy of integrating that
representation.

For the completed 8D mixture fit, the general operator and legacy spectral
operator differ by at most 8.9e-14 in log density on the comparison points.
Comparison with the analytic mixture differs by 3.2e-12; quadrature doubling
differs by 1.9e-13. For the 20D mixture these last two differences are 1.6e-11
and 1.1e-12 respectively. Thus direct contraction provides a general route to
accurate convolution, but these examples do not demonstrate that the legacy
convolution was numerically inaccurate. Improvements over a naive noisy-data
fit establish the benefit of modeling noise; they do not isolate an accuracy
improvement caused by changing the convolution algorithm. The benefits of
avoiding the explicit TT square, output interpolation and SVD rounding remain
useful independently of that distinction.

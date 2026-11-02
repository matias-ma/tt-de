# Learning what to regularize in noisy TT estimation

This continues [the curvature-penalty investigation](noisy_recovery.md) on
10D GL, paired Gaussian noise at scale 1.5, 100,000 noisy training points,
and 2,048 noisy validation points. The final model remains a single squared
`chebttge.ChebTT`, trained by gradients of its Chebyshev coefficients through
the full correlated convolution likelihood. No GL density values or clean
samples enter fitting or selection.

## Information-filter result

The visibility-filter experiment improves further to **integrated relative
L2 .14106**, with strength **.3**, visibility floor **.01**, and step
**8,000**. That is **3.22 times smaller** than the previous .45489. Its
relative sample error is .14970, and its squared sample error—the convention
in the original figures—is **.022410**, versus the previous .258847.

Both within-family noisy selection and the combined noisy-only comparison
of the first eight candidates select this model. Its noisy validation NLL is
15.6267307. Among the reference-score candidates, even the minimum-NLL fit
has excess .0003358 with paired cluster SE .0003213, so it narrowly fails
the specified one-SE criterion. This boundary is close; selection should not
be interpreted as a decisive statistical separation of the algorithms.

| Visibility strength | Noisy validation NLL | Integrated relative L2 | Squared sample error |
| --- | ---: | ---: | ---: |
| .01 | 15.627535 | .197731 | .028117 |
| .03 | 15.627241 | .170629 | .023807 |
| .1 | 15.626799 | .152500 | .022855 |
| **.3, selected** | **15.626731** | **.141059** | **.022410** |

These results apply to the tested grid, not a claim that .3 or a .01
visibility floor is universally optimal. The marginal pilot, both strength
grids, the stopping comparison, and the common-complexity selection rule
were specified before looking at the new clean-density errors.

Artifacts: [Fisher family selection](../artifacts/noisy_fisher_evaluation/selection.json),
[Fisher metrics](../artifacts/noisy_fisher_evaluation/metrics.json),
[selected Fisher TT](../artifacts/noisy_fisher_fits/fisher_0.3/fit.npz),
[combined eight-candidate noisy selection](../artifacts/noisy_information_phase2/selection.json).

The stopping-rule comparison is complete. It selects steps 8,000, 6,800,
and 6,800 for reference strengths .003, .01, and .03. Their integrated
errors are .16445, .16941, and .18131. In particular, it rescues the .03
trajectory from retaining its initialization, but it does not beat the
Fisher model's noisy validation score. The combined **eleven-candidate**
comparison still selects **Fisher strength .3**.

Final outputs: [selection](../artifacts/noisy_information_evaluation/selection.json),
[all metrics](../artifacts/noisy_information_evaluation/metrics.json),
[selected checkpoint](../artifacts/noisy_information_evaluation/selected_fit.npz),
[marginal comparison](../artifacts/figure_refresh/previous_figs/ginzburg_landau/noisy_information/paired_comparison.png),
[recovery errors](../artifacts/figure_refresh/previous_figs/ginzburg_landau/noisy_information/recovery_errors.png).

## Completed reference-score experiment

Noisy-only selection chooses strength **.01, step 8,000**, reaching
**integrated relative L2 .16948**, compared with **.45489** for the previous
curvature fit (2.68 times smaller). The original RSI baseline was 1.6059.
This meets the requested .2 integrated-L2 target on this seed.

| Fit | Noisy validation NLL | Integrated relative L2 | Relative sample error | Squared sample error |
| --- | ---: | ---: | ---: | ---: |
| Previous selected curvature | 15.628066 | .454889 | .508770 | .258847 |
| Reference score, .001 | 15.627067 | .178018 | .165444 | .027372 |
| Reference score, .003 | 15.627122 | .164453 | .153791 | .023652 |
| **Selected reference score, .01** | **15.627428** | **.169475** | **.157524** | **.024814** |

The .003 candidate has the smallest clean diagnostic in this table, but it
was **not** chosen using that diagnostic. The selected .01 candidate has
excess noisy NLL .000362 with paired cluster SE .000432, and has the smallest
reference-score complexity among eligible candidates. The .03 run retained
its initial curvature checkpoint under minimum-NLL stopping; the separate
checkpoint experiment tests this behavior using noisy data.

These are full 10D errors. The figure's 2D marginals are visual diagnostics,
not the quantity used to select or report integrated L2. Signal quadrature
doubling changes the selected model's error integrals by 7.0e-10 relative to
`integral p_true²`.

Artifacts: [frozen noisy selection](../artifacts/noisy_reference_evaluation/selection.json),
[metrics](../artifacts/noisy_reference_evaluation/metrics.json),
[selected TT](../artifacts/noisy_reference_fits/reference_0.01/fit.npz),
[marginal comparison](../artifacts/figure_refresh/previous_figs/ginzburg_landau/noisy_reference/paired_comparison.png),
[reference and information spectrum](../artifacts/figure_refresh/previous_figs/ginzburg_landau/noisy_information/reference_and_information.png).

## Why change the previous penalty?

Convolution suppresses information; inversion amplifies sampling error.
Large Chebyshev degree is an imperfect indicator of this effect. Furthermore,
plain amplitude curvature penalizes a sharp feature even when many noisy
observations support that feature. The new approach learns a reference shape
from noisy marginal data, and penalizes departures from it.

This is an empirical-Bayes construction: the prior is estimated from the
training observations, then used softly in the full TT fit. It adds a
data-selected assumption of similar marginal shapes in this experiment.
It does not impose independence, identical fitted TT marginals, symmetry,
or a prescribed interaction graph on the final density.

## Marginal pilot

`fit_noisy_marginal_reference` fits finite exponential families

```
rho_k(x) proportional to exp(sum_j c_kj T_j(t_k(x)))
```

by the Gaussian-convolved **noisy** marginal likelihood. Training uses exact
Gaussian bin probabilities (1,600 bins per axis) and 160-point signal
quadrature. Validation uses continuous densities. Candidate log-polynomial
degrees are 2, 4, 6, 8, with either shared coefficients across standardized
boxes or separate coefficients. Parameter count breaks ties within a paired
one-standard-error rule, clustering observations by the sampler's 200 chains.

The noisy data selected shared degree 4: validation marginal NLL 1.6052740,
versus 1.6120842 for shared degree 2. Shared degree 8 was statistically
indistinguishable but more complex. Separate degree 6/8 runs reached their
iteration limit; their finite candidates were retained in the comparison
and were not selected. Their optimization status is recorded explicitly.

Doubling selected signal quadrature changes validation log density by at
most 5.8e-15. Re-fitting with 3,200 bins and 320-point quadrature changes it
by at most 4.3e-6. The pilot uses the known Gaussian **marginal** noise law;
the joint TT likelihood retains the prescribed correlated TT kernel.

We increase amplitude coefficient count from 25 to 33 because the learned
reference's polynomial projection improves from about 1.2% to 0.12% relative
amplitude error. This diagnostic uses the learned reference, not the true
signal. The previous selected curvature TT is zero-padded to initialize all
joint fits at exactly the same function.

## Two penalties

`ReferenceScorePenalty` computes

```
R_score(g) = sum_k sigma_k² integral [partial_k g
               - (g/2) partial_k log rho_k]² / integral g².
```

This is one quarter of weighted interior relative Fisher divergence between
the fitted density and the product reference. It preserves the reference's
sharp shape and penalizes departures and dependencies softly. Polynomial
quadrature constructs the exact derivative matrices, then TT contractions
evaluate their normalized quadratic forms. Like the curvature replacement,
this is invariant to amplitude scaling and invertible TT gauge changes.
It does not penalize jumps at the finite signal-box boundary.

The squared log-gradient distance is related to
[Hyvärinen's score-matching framework](https://www.jmlr.org/papers/v6/hyvarinen05a.html).
Here it is a regularizer around a learned reference, integrated against the
fitted density; this is not the paper's sample score-matching estimator.

`FisherVisibilityPenalty` instead builds an information filter around each
reference. For an L2-unit amplitude perturbation `h` orthogonal to
`sqrt(rho)`, the observed-to-complete information ratio is

```
lambda(h) = integral [U * (sqrt(rho) h)]² / (U * rho).
```

Conditional expectation is a contraction, so its eigenvalues lie in [0,1].
Whitening the physical Chebyshev Gram matrix makes this an ordinary symmetric
eigenproblem. The projected reference amplitude has penalty zero; each
orthogonal eigenmode receives bounded weight `alpha / (alpha + lambda)`.
There is no exponentially growing degree weight or inverse singular value
in the objective. Information-based model complexity is discussed in
[Ly et al.'s Fisher-information tutorial](https://arxiv.org/abs/1705.01064);
this particular filter and its TT implementation are our construction.

For the first learned marginal the leading retained-information eigenvalues
are approximately .552, .0954, .0206, .00320, .000470, .0000613. The
corresponding directions are combinations of Chebyshev coefficients.
Doubling quadrature from 256 to 512 changes the spectrum by about 5e-14.

The TT penalty sums the marginal filters. This is an approximation to the
joint information geometry: it does not compute the full correlated,
multivariate Fisher Hessian. `alpha` and overall strength remain regularization
choices, selected or checked with noisy data. This is a more directly
operator-based prior, not a parameter-free solution to deconvolution.

## Selection protocol

Joint fits use rank 10, 33 coefficients per core, Gram-preconditioned Adam,
8,000 steps, batch 256, cosine learning rate starting at 5e-4, and identical
batch seeds. Reference-score strengths are .001, .003, .01, .03. Visibility
filter strengths are .01, .03, .1, .3 with `alpha=.01`.

One set keeps minimum noisy-NLL checkpoints. A separate reference-score run
at .003, .01, .03 uses the paired one-SE rule within each trajectory, choosing
the smallest penalty among indistinguishable checkpoints. This tests whether
small noisy-NLL fluctuations cause premature stopping of regularization.

For comparing these different penalties, the evaluator recomputes a **common
reference-score complexity** for every candidate before applying the paired
one-SE rule. It never compares the numerical magnitudes of two different
penalties as if they had the same units. Selection is saved before the GL
truth or integrated L2 is evaluated. Reusing noisy validation for pilot,
stopping, and hyperparameters makes this a selection heuristic, not a formal
post-selection confidence guarantee.

## Library use

### What is assumed known, and arbitrary TT noise

**No covariance matrix of the unknown signal is required.** Like the
existing deconvolution estimator, this method requires the noise law to be
supplied. Knowing the noise law is distinct from knowing the clean density.
Fisher information here is a matrix of likelihood sensitivities to amplitude
coefficients, not an assumed covariance of the GL signal.

The original .141 experiment used the known Gaussian marginal scales as a
shortcut. Following the kernel-generality check, both the pilot and filter
also accept **the prescribed noise TT directly**, with no covariance or
Gaussian parameters:

The synthetic GL observation generator still uses its Gaussian covariance
to generate the benchmark data. With `--kernel-marginals`, that covariance
is not separately passed into pilot fitting or the Fisher regularizer.

```python
from ftde import fit_noisy_marginal_reference, FisherVisibilityPenalty

log_coeffs, noisy_scores, diagnostics = fit_noisy_marginal_reference(
    noisy_train, noisy_validation, signal_domain,
    noise_tt=noise_tt, degree=degree, shared=shared,
)
penalty = FisherVisibilityPenalty.from_tt(
    initial_g, log_coeffs, noise_tt=noise_tt, visibility_floor=.01,
)
```

Degree/sharing are still chosen with noisy validation. The implementation
contracts the known noise TT to its normalized univariate marginals. Pilot
bin probabilities come from exact Chebyshev antiderivatives of these
marginals; support crossings split the signal quadrature into smooth panels.
The Fisher construction also respects support intersections and splits the
observation integration at changes in overlap. Noise means/variances used
for the pilot's initialization are computed from the supplied kernel.

This supports non-Gaussian and correlated TT noise. Correlations remain in
the full joint likelihood; the **regularizer still approximates joint
information by marginal information**. It does not assume independent noise
and does not reconstruct an unknown noise law. A valid density approximation
and the forward operator's existing kernel-box coverage requirements remain
necessary. Materially negative marginal approximations are rejected; only
tiny spectral roundoff is clipped.

Independent checks cover a correlated non-Gaussian polynomial kernel, an
analytic uniform-noise Fisher eigenvalue of .5, Gaussian-TT/analytic-Gaussian
agreement, and adaptive quadrature of a fitted non-Gaussian pilot.

```python
from ftde import ReferenceScorePenalty, TTConvolution, fit_noisy_tt

# log_coeffs comes from noisy marginal pilot fitting/selection.
penalty = ReferenceScorePenalty.from_tt(
    initial_g, log_coeffs, length_scales=noise_std,
)
operator = TTConvolution.from_tt(initial_g, noise_tt)
fitted, history = fit_noisy_tt(
    initial_g, noise_tt, noisy_samples,
    forward_operator=operator, regularizer=penalty,
    reg_strength=strength, n_steps=8000,
)
```

For the visibility version, replace the penalty with
`FisherVisibilityPenalty.from_tt(initial_g, log_coeffs, noise_std,
visibility_floor=.01)`. The marginal-pilot function is also exported from
`ftde`. The example strengths and shared-pilot choice require validation on
new datasets; the previous penalty is still available.
Use the benchmark commands below to reproduce the reported optimizer,
including Gram preconditioning and cosine learning-rate decay; the short
library example uses the library's standard optimizer.

Both benchmark scripts accept `--kernel-marginals` to use the prescribed TT
instead of Gaussian parameters. Give such runs a new output directory so
they do not overwrite the original benchmark artifacts.

On the actual GL noise TT, the first-axis information spectrum differs from
the Gaussian shortcut by 2.2e-7, while doubling the TT-based quadrature
changes it by 5.3e-14. The noise TT is an approximation to the generating
Gaussian, so exact equality is not expected. Re-fitting the shared degree-4
pilot with this TT changes its mean noisy validation NLL by about 2e-8;
the largest per-observation marginal-log difference is 6.0e-5.

The full kernel-only replay reaches **integrated relative L2 .141115**
(Gaussian shortcut: .141059), and squared sample error **.022438**.
This replay keeps the already selected degree, rank, strength, and optimizer;
its checkpoint is again selected by noisy validation. The final integrated
quadrature discrepancy is 8.9e-10 relative to `integral p_true²`.
Thus the improvement survives removing separate Gaussian parameters from
the estimator inputs. This remains a single GL seed; the non-Gaussian checks
validate implementation correctness, not a recovery-rate claim on all laws.

Kernel-only artifacts: [selected TT](../artifacts/noisy_fisher_kernel_evaluation/selected_fit.npz),
[metrics](../artifacts/noisy_fisher_kernel_evaluation/metrics.json),
[pilot comparison](../artifacts/noisy_reference_kernel/pilot_comparison.json),
[operator audit](../artifacts/noisy_reference_kernel/filter_audit.json).

```bash
python scripts/fit_noisy_marginal_reference.py --kernel-marginals --degrees 4 \
  --output artifacts/noisy_reference_kernel
python scripts/benchmark_noisy_recovery.py --kernel-marginals \
  --reference artifacts/noisy_reference_kernel/selected.npz --fisher-floor .01 \
  --warm-start artifacts/noisy_recovery_n25_mid/h2_3e-05/fit.npz \
  --n-coeffs 33 --strengths .3 --output artifacts/noisy_fisher_kernel
python scripts/evaluate_noisy_recovery.py artifacts/noisy_fisher_kernel \
  --baseline artifacts/noisy_fisher_fits/fisher_0.3/fit.npz \
  --baseline-name gaussian_shortcut --no-plots \
  --output artifacts/noisy_fisher_kernel_evaluation
```

## Reproduction

Use the `chebdGE` environment and the GPU allocation procedure in
`instructions.txt`. All paths below are local artifacts.

```bash
python scripts/fit_noisy_marginal_reference.py
python scripts/benchmark_noisy_recovery.py \
  --reference artifacts/noisy_reference/selected.npz \
  --warm-start artifacts/noisy_recovery_n25_mid/h2_3e-05/fit.npz \
  --n-coeffs 33 --strengths .001 .003 .01 .03 \
  --output artifacts/noisy_reference_fits
python scripts/benchmark_noisy_recovery.py \
  --reference artifacts/noisy_reference/selected.npz --fisher-floor .01 \
  --warm-start artifacts/noisy_recovery_n25_mid/h2_3e-05/fit.npz \
  --n-coeffs 33 --strengths .01 .03 .1 .3 \
  --output artifacts/noisy_fisher_fits
python scripts/benchmark_noisy_recovery.py \
  --reference artifacts/noisy_reference/selected.npz \
  --warm-start artifacts/noisy_recovery_n25_mid/h2_3e-05/fit.npz \
  --n-coeffs 33 --strengths .003 .01 .03 --checkpoint-rule paired-one-se \
  --output artifacts/noisy_reference_checkpoints
python scripts/evaluate_noisy_recovery.py \
  artifacts/noisy_reference_fits artifacts/noisy_fisher_fits \
  artifacts/noisy_reference_checkpoints \
  --selection-reference artifacts/noisy_reference/selected.npz \
  --baseline artifacts/noisy_recovery_n25_mid/h2_3e-05/fit.npz \
  --baseline-name previous_curvature --baseline-label 'Previous curvature fit' \
  --output artifacts/noisy_information_evaluation \
  --figure-dir figs/ginzburg_landau/noisy_information
```

## Numerical verification

Tests compare relative-score contractions against independent derivative
cubature, check coefficient gradients by finite differences, and test
nonorthogonal gauge changes and extreme rescaling. The information spectrum
is checked against an independent Gaussian/Hermite analytic result:
eigenvalues `(1+sigma²)^(-m)` for unit signal variance. Tests also check the
reference nullspace, PSD and upper bounds, and quadrature refinement.

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest -q \
  tests/test_reference_score.py tests/test_fisher_visibility.py \
  tests/test_marginal_reference.py tests/test_noise_marginals.py tests/test_sobolev.py \
  tests/test_noisy_selection.py tests/test_noisy_density.py
```

Together with the pilot's independent adaptive-quadrature prediction check,
existing Sobolev, noisy fitting, and noisy selection tests, and the general
kernel checks, **35 tests pass**. Ruff checks also pass for the changed files.

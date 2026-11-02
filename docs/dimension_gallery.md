# Dimensional density comparison

This study reruns Swirl and Ginzburg–Landau (GL) in 4, 10 and 20 dimensions,
with noise-free observations and the existing prescribed mixture noises.
It uses one normalized squared Chebyshev functional TT, initialized near
uniform. Neither a fitted signal pilot nor clean density values enter fitting
or model selection. Existing observations and observation-derived boxes are
reused where available; missing cases use the existing samplers with training
seed 0 and validation seed 2. Each case has 100,000 training observations.

The current run's machine-readable results live in
`artifacts/dimension_gallery_v9`. `complete.json` marks completion of the full
matrix; `selection.json` freezes each case before its `evaluation.json` is
computed. The figure index is replaced only after all cases are complete.

## Current results

These are **unsquared full-dimensional latent relative L2 errors**, with
100,000 training observations per condition and unchanged noise scales.
All checkpoint and model choices use observations, before truth evaluation.

| Dimension | GL clean | GL noisy | Swirl clean | Swirl noisy |
|---:|---:|---:|---:|---:|
| 4 | 0.0367 | 0.1672 | 0.1155 | 0.2961 |
| 10 | 0.0815 | 0.2892 | 0.2272 | 0.5909 |
| 20 | 0.1027 | 0.3751 | 0.2561 | 0.7810 |

The **0.2–0.3 noisy 20D target is not reached**. Clean 20D Swirl improves
from the initial 0.509 to 0.256 after observation-selected penalty and rank
comparisons. Noisy 10D Swirl improves from about 0.710 in the referenced
previous figure to 0.591. Noisy 20D Swirl improves from 0.856 to 0.781 with
the selected rank-64 continuation. These are single-seed development
results, not a claim of optimality or an impossibility bound.

See the [figure index](../figs/README.md) for the displayed models.
`artifacts/dimension_gallery_v9/summary.json` records each displayed source,
including the separately frozen clean and noisy 20D follow-ups.

## Recovered defaults

The successful “Hessian thing” had two distinct parts: an integrated
**amplitude full-Hessian penalty**, including mixed derivatives, and
**Newton-CG canonical core sweeps** for noise-free likelihood. The latter
cannot be transferred unchanged to convolved likelihood.

`DirectTrainingProblem`, `NoisyDirectTrainingProblem`, and `fit_core_sweeps`
now default to `regularizer='hessian', strength=0.001`. Their parameterization
remains `p = g² / integral(g²)`. Legacy training APIs retain their earlier
defaults for historical reproduction. The reported Hessian energy is
`H(g) = integral sum_ij (ell_i*ell_j*partial_ij g)^2 / integral g^2`,
with `ell_i = 0.05 * domain_width_i`; mixed derivatives occur twice in this
sum. Adam minimizes `NLL + lambda*H` (plus the marginal penalty when enabled).
The canonical clean core sweeps instead minimize `NLL + log(1 + lambda*H)`.
Thus the same numeric strength does not define identical objectives for
the two optimizers. Specific benchmark recipes are in
`src/ftde/benchmark_defaults.py`:

| Case | Default comparison |
|---|---|
| Noise-free Swirl | Rank 32 / 32 / 64 in 4D / 10D / 20D, 49 modes, 20 canonical sweeps; Hessian strength 0.01 / 0.003 / 0.001 |
| Noise-free GL | Rank 5, 25 modes, 4,000 Adam updates; Hessian strength 0.001 |
| Noisy GL | Rank 10, 25 modes; weak Hessian 1e-5 with 9→17→25 modes versus full-basis Hessian 0.001 |
| Noisy Swirl | Rank-32 baseline, 49 modes; 20D adds the selected rank-64 continuation; weak Hessian 1e-5 with 13→25→49 modes versus strong-Hessian learning followed by weak-Hessian and marginal-curvature continuations |

The noisy curriculum unlocks modes at updates 0, 1,000 and 2,000, and runs
4,000 updates. Its checkpoints become eligible at 2,000. The strong Swirl
fit runs 4,000 updates; its weak continuation runs 2,000, then a 2,000-update
marginal continuation uses strength `0.003 / source_marginal_curvature`.
All starts use seed 0 and perturbation 0.01. Adam uses learning rate 0.001
for fresh fits, 0.0003 for the weak continuation and 0.0001 for the marginal
continuation. Noise-free batches have 1,024 samples; noisy batches have 256.
In 20D, the noisy gradient is accumulated over two 128-observation chunks,
then clipped and applied once. This preserves the sampled batch and objective
while reducing backward-pass memory.

These recipes transfer the previous studies' observation-selected settings.
They are not a claim of globally optimal tuning in the previously untested
dimensions. In particular, the 20D Swirl results are a new capacity test.
The staged noisy recipe now runs through the common direct trainer. Its
strong initial stage uses that trainer's minibatch stream, whereas the old
objective-study trainer split an extra random key for Fourier scoring. Thus
the new staged runs are recipe transfers, not bit-for-bit replays of the
old 0.280 Swirl checkpoint. The curriculum and noise-free core-sweep controls
do reproduce their earlier 4D/10D results.

Each run retains the checkpoint with lowest held-out NLL. Across noisy
recipes, the predeclared choice is the lowest exact marginal curvature
within one paired standard error of minimum validation NLL. GL uses 200
MCMC-chain clusters for this comparison. All alternatives remain in the
numerical records. Truth is evaluated after the choice, without reselection
on latent error. This is a development heuristic, not a post-search guarantee.

## Clean 20D penalty-transfer follow-up

The initial 20D clean Swirl fit reaches relative L2 0.509 despite good-looking
pair marginals. Its normalized Hessian energy is about 147, compared with a
much smaller value in lower dimensions. This motivates a separate comparison
of strengths 0.003, 0.001, 0.0003 and 0.0001 at the same rank, basis and
20-sweep budget. The original fit remains a frozen control.

This is a **development follow-up after inspecting the baseline**, not a
fully preregistered experiment. The new grid is fixed before evaluating its
new latent errors, and selection uses minimum held-out NLL. Artifacts live
in `artifacts/dimension_followup_v10`; run `scripts/run_dimension_followup.py`
inside the same GPU environment. A publication manifest lets the selected
follow-up replace the displayed 20D clean result while preserving the
original matrix and baseline evaluation. The final gallery uses the rank-64
follow-up below and records the source directory of every displayed result.

The completed comparison selects strength 0.001 by held-out NLL. Its full
relative L2 is **0.43690**, down from 0.50922 at 0.003. The weaker 0.0003 and
0.0001 controls reach 0.40696 and 0.39879 L2 but have worse validation NLL;
they are not substituted using their true-density errors. The benchmark
default for clean 20D Swirl is therefore now 0.001. The original matrix's
frozen protocol retains its old setting for reproduction.

A subsequent fixed-strength capacity control compares rank 64 with the
selected rank-32 fit, again using only clean held-out likelihood to select.
Rank 64 is selected and reaches **0.25610** relative L2, versus 0.43690 at
rank 32. Artifacts are in `artifacts/dimension_capacity_v11`. Increasing
rank therefore helps the current training procedure, and sets the clean
20D default to rank 64 at Hessian strength 0.001. These are achieved errors,
not lower bounds on what either rank can represent or learn. Reproduce this control
with `scripts/run_dimension_followup.py --capacity` after the penalty grid.

## Matched noisy 20D capacity continuation

The higher clean capacity motivates a separate, predeclared noisy comparison
in `artifacts/noisy_dimension_capacity_v12`. It starts **only from the
observation-selected noisy model**, with the clean model excluded. Rank 32
and rank 64 receive 1,000 additional Adam updates each, learning rate 0.0001,
the same 256-observation batches, and unchanged Hessian and marginal penalties.
Both starts are canonicalized in physical L2; only the added rank directions
receive a small perturbation. Gradients use two chunks at rank 32 and eight
at rank 64 to stay within one GPU.

Selection prefers the smaller rank within one paired standard error of the
best noisy validation NLL, then the lower marginal curvature within a rank.
Thus extra capacity must earn a detectable validation gain. The fixed budget
and rule were recorded before the initial noisy 20D truth evaluation. This
remains a development comparison, not an independent confirmatory study.
The current full gallery runner includes this continuation; it can also be
run separately with `scripts/run_noisy_dimension_capacity.py` after the
baseline noisy case is frozen. Results are published only after selection
and evaluation, with both raw studies preserved.

The completed matched comparison selects **rank 64**. Validation NLL is
9.87837 for the source, 9.85981 after the rank-32 continuation, and 9.78533
after the rank-64 continuation. Rank 64 beats the matched rank-32 control
by 0.07447 ± 0.00387 in paired validation NLL. The corresponding latent
relative L2 errors, evaluated afterward, are **0.85615, 0.84201, and 0.78103**.
The full default runner includes this observation-selected continuation;
it does not start a fresh noisy rank-64 fit using the clean solution.

This is evidence that the rank change helps the current noisy training
procedure. A 1,000-update continuation does not establish the best
attainable error at either rank or prove that optimization has converged.
The final 20D latent pair errors are 0.122 and 0.139; reconvolved pair
errors are 0.022 and 0.027. The main unresolved gap is joint-density recovery.

## Figures and errors

Each synthetic folder contains two pair-marginal figures, coordinates (0,1)
and (2,3), and one convergence figure per condition. Convergence figures show
the entire history, a final-half likelihood zoom, and Hessian energy after
the first update. Dots identify retained checkpoints; continuation stages
restart their update count. Noisy panels put true
and recovered latent densities side by side, then show the prescribed noise,
the true observed density, the reconvolved fit, and the observations.
Comparable panels use the same color scale. Histograms divide by the full
sample count, so cropping does not renormalize away missing mass.

The title reports **unsquared full-dimensional relative L2**:

`sqrt(integral (p_hat - p_true)^2 / integral p_true^2)`.

It is not a pixel error of the displayed 2D projection. The true Swirl
marginals for pairs (0,1) and (2,3) are identical across these dimensions;
additional coordinates change the joint density without changing those
projections. Swirl uses analytic
Gaussian/path integration and exact polynomial fourth moments. GL uses
transfer quadrature, repeated at twice the order. Each case also records
latent and reconvolved pair errors and a refinement check on the displayed
FFT convolution. The model uses prescribed TT convolution during training;
the FFT is only for display.

The baseline noisy 20D Swirl result makes this issue particularly visible:
its two displayed latent pair errors are **0.113 and 0.140**, and its
reconvolved pair errors are **0.020 and 0.029**, while its full latent error
is **0.856**. Improving or matching these two projections alone would not
solve the joint-density recovery problem.

The joint-tube diagnostic in `figs/convergence` samples 8,192 points from
each final 20D Swirl TT by exact polynomial conditional-CDF inversion.
It compares their nearest-curve distances with independent true samples,
strictly after selection. Exact TT marginal means check the sampler, and
20,001 versus 40,001 curve nodes check nearest-distance resolution.
This diagnostic does not assume the curve during fitting. Its numerical
record is `artifacts/dimension_gallery_v9/joint_geometry.json`; reproduce it
with `scripts/diagnose_swirl_joint.py` after the final selections.

The final [joint-tube plot](../figs/convergence/joint_tube_recovery.png)
shows broader joint recovery under noise:

| Sampled density | Median distance to curve | 95th percentile | Outside the true 95% tube |
|---|---:|---:|---:|
| Truth | 0.599 | 0.769 | 5.0% |
| Clean estimate | 0.609 | 0.848 | 9.4% |
| Noisy estimate | 0.708 | 1.024 | 33.0% |

These are latent samples, with no measurement noise added. The noisy estimate
puts appreciably more probability away from the joint tube, despite matching
the displayed pair marginals fairly well. The exact-moment sampler checks
are within one standard error for every coordinate, and refining the curve
grid changes checked distances by less than 7e-7. The distance diagnostic is
not another L2 metric or an estimate of a single Gaussian width.

The previous gallery is preserved in
`artifacts/gallery_archive/before_dimension_gallery_v9`. Cryo-EM retains
its existing noisy 4D/10D/20D fits and training figure; there is no clean
dataset with which to make a matched noise-free comparison or measure L2.

## Why fitting through convolution is still harder

The fitted parameter is indeed the underlying density `p`. However, noisy
observations constrain it through `q = K*p`. In the population, likelihood
minimization measures `KL(K*p_true || K*p)`, while noise-free likelihood
measures `KL(p_true || p)`. For a small perturbation `h`, the noisy excess
risk is approximately `0.5 * integral (K*h)^2 / q_true`. Thus latent errors
that convolution strongly attenuates make a small change in the loss.

This does not say the target 0.2–0.3 is impossible. With an identifiable
kernel, correct model and unlimited data, the true density remains the
population target of unregularized noisy likelihood. At finite sample size,
some details are weakly constrained, and fixed regularization changes the
optimum. Better optimization can recover information that the current fit
leaves unused; it cannot make the observed likelihood sensitive to every
latent detail. The classical distinction between well-defined population
deconvolution and slow finite-sample recovery with Gaussian errors is
discussed by [Fan (1991)](https://www.stat.cmu.edu/~jiashun/Teaching/F08STAT756/Lectures/Fan.pdf).

The Swirl kernel is unchanged: mixture weights 0.5/0.3/0.2, isotropic standard
deviation 0.22, alternating diagonal standard deviations 0.45/0.25, and a
standard deviation 0.5 component with adjacent-pair correlations ±0.6.
The narrow latent tube has standard deviation 0.14. The noise therefore
blurs the tube's transverse structure substantially, even where it leaves
the broad path visible. The common mixture component across all coordinates
and the full pair correlations are retained in training.

Full-dimensional relative L2 is also sensitive to small width errors that
are hard to see in a pair marginal. For illustration only, if the truth is
an isotropic Gaussian and the estimate has the correct center but a standard
deviation 5% too large in every coordinate, relative L2 is 0.114 in 4D,
0.237 in 10D and 0.405 in 20D. This follows exactly from
`error² = 1 + a^(-d) - 2*(2/(1+a²))^(d/2)`, with `a=1.05`.
It is not a model of Swirl or an impossibility bound; it explains why
visually good low-dimensional marginals can coexist with a large joint error.

The useful diagnostics are: how much latent error remains in the clean
20D fit (a capacity/optimization reference), how much extra error appears with
noise, how close the reconvolved fits are, and whether observation-equivalent
recipes disagree before convolution. Those separate capacity limitations,
optimization failures, smoothing bias and weak statistical identifiability.

## Independent noisy-data check

After selection is frozen, `scripts/diagnose_dimension_results.py` evaluates
all candidates on 8,192 new observations, using signal seed 852731 and noise
seed 852732. These scores never change the selected model. Paired standard
errors use the same observations for each comparison; GL retains MCMC-chain
clustering. Test records live beside each frozen case in `test_scores.json`.

For noisy 10D Swirl, the selected marginal continuation improves test NLL
by **0.01291 ± 0.00143** (one paired standard error) versus the strong-Hessian
fit, and improves latent L2 from 0.714 to 0.591. Thus this recovery gain also
predicts independent noisy observations better. Nevertheless, the selected
latent pair errors are about 0.165, its reconvolved pair errors are only
0.023–0.027, and its full joint error remains 0.591. Those three metrics
measure different objects and should not be substituted for one another.
The selected 10D estimate has `||p_hat||_2 / ||p_true||_2 = 0.6284`.
This concentration deficit alone implies relative L2 at least 0.3716 by
the reverse triangle inequality. It is consistent with broadened or weakened
latent peaks, not just small oscillations around a correctly concentrated
density. Differences in shape account for the rest of the measured error.
A separate post-selection evaluation against the true convolved joint gives
`KL(true observed || fitted observed) = 0.04991 ± 0.00303` nats per observation
(Monte Carlo estimate ± one standard error). Path quadrature at orders 512
and 1,024 agrees within 4.4e-13 in log density on 64 test points. The remaining
observed likelihood gap is measurable: these results do not establish that
the large latent error is an unavoidable statistical limit. This diagnostic
is available with `--truth-gap` and is never used for training or selection.

For noisy 20D Swirl, rank 64 improves independent test NLL by
**0.07078 ± 0.00267** versus the matched rank-32 continuation and by
**0.09272 ± 0.00304** versus its source. This validates the predictive gain
outside the model-selection data. Its full observed KL gap to truth is
**0.12447 ± 0.00510**, compared with 0.21720 for the source. Therefore the
observed joint fit still has measurable error too; close pair marginals do
not establish a close full observed density. This experiment does not
separate all remaining optimization, approximation, and statistical errors.

The noisy GL results expose a related selection ambiguity: in 4D the strong
and curriculum models differ by only 0.00013 ± 0.00036 in independent test
NLL, while their latent errors are 0.391 and 0.167. In 20D the curriculum
has a clearer test advantage, 0.00544 ± 0.00117, and latent error 0.375
versus 0.695. An accurate forward operator and a good noisy likelihood can
coexist with substantial uncertainty or bias in the recovered latent density.

## Reproduce

Use the `chebdGE` environment and one GPU allocation with four CPUs and
32 GB host memory, as specified by `instructions.txt`:

```bash
srun --jobid=YOUR_JOB_ID --ntasks=1 --cpus-per-task=4 \
  bash scripts/run_noise_diagnosis_gpu.sh -u scripts/run_dimension_gallery.py
```

The GPU wrapper enables float64 and disables GPU memory preallocation.
The driver skips completed fits, verifies frozen records, evaluates each
case in a separate process, and archives/publishes after all twelve finish.
An interrupted incomplete fit restarts its recipe; saved completed stages
are reused. To redraw a completed case:

```bash
JAX_ENABLE_X64=1 python scripts/plot_dimension_gallery.py swirl10_mixture
JAX_ENABLE_X64=1 python scripts/plot_dimension_gallery.py --publish
```

The second command requires all twelve evaluated cases. Figures and numbers
are staged separately, keeping checkpoints, logs and JSON out of `figs`.

Verification: the targeted direct-training, core-sweep, training-geometry and
marginal-regularization suite passes 32 tests on CPU with float64 enabled.
After adding gradient accumulation, all 31 tests in the new microbatch,
direct-training and training-geometry suites pass, including numerical
agreement with full-batch updates and validation of invalid chunk sizes.
The three conditional TT sampler tests also pass, including an independent
49-mode polynomial-CDF check and correlated-moment verification.
`scripts/audit_dimension_gallery.py` verifies frozen hashes, full training
budgets and evaluation provenance, then compares each selected noisy model's
prepared convolution with direct quadrature at twice the order on eight
validation points per case. These are numerical spot checks, not a global
error bound on the forward operator.

For the final Swirl fits, `--analytic-noise` additionally compares against
independent quadrature of the prescribed Gaussian noise formula at orders
128 and 256, including a tail observation. This also checks the polynomial
noise-kernel representation, not just the prepared integration machinery.
The 10D and 20D selected fits agree within 5.4e-14 and 7.1e-14 in log
density on these probes, respectively. Across the six selected noisy fits,
prepared-versus-refined polynomial quadrature differs by at most 2.1e-12.

The final publication audit passes for **12 synthetic cases and 42 PNGs**,
including retained Cryo-EM figures and two summary/diagnostic plots. It
checks frozen model/evaluation/test provenance, displayed image hashes,
allowed dimensions and file types, retained Cryo-EM hashes, and figure-index
links. Recheck without a GPU with:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python scripts/audit_dimension_gallery.py --publication
```

Its record is `artifacts/dimension_gallery_v9/publication_audit.json`.
The original 67 PNGs remain in the archived previous gallery.

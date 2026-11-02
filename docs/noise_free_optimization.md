# Direct Chebyshev TT optimization: noise-free study

This is the current experiment workflow. The estimator is a single functional
TT amplitude `a`, with Chebyshev coefficients, and density
`p(x) = a(x)^2 / integral a^2`. The initializer is a constant plus a small
smooth random perturbation, using only the box, rank, mode count and seed.
No KDE, learned tangent mixture, fitted curve, or surrogate density supplies
its structure. The retired tangent/local-mixture pilot implementations and
their experiment drivers have been removed. Original KDE utilities remain
for legacy TT/Tucker examples and comparisons; this workflow does not call them.

## What changed

`src/ftde/direct_training.py` precomputes the observation basis matrices for
noise-free training. It supports ordinary likelihood, likelihood augmented
with exact coordinate-conditional likelihoods, and a four-dimensional
quadratic density score. All use the same normalized squared TT.

Three function-space penalties are compared, with `s_k` equal to 5% of the
coordinate box width:

- slope: `sum_k s_k^2 integral (partial_k a)^2 / integral a^2`;
- axis curvature: `sum_k s_k^4 integral (partial_kk a)^2 / integral a^2`;
- full Hessian: `sum_kl s_k^2 s_l^2 integral (partial_kl a)^2 / integral a^2`.

The last includes mixed derivatives twice, as the Hessian Frobenius norm
requires. These are integrated properties of the represented function,
independent of TT gauge or overall amplitude scale. They do not smooth
plots after fitting. Chebyshev coefficients remain the saved representation;
Gram/Sobolev whitening only changes optimizer coordinates.

The default objective is empirical NLL plus the selected amplitude penalty.
The quadratic score is `integral p^2 - 2 mean p(x_i)`; the integral of the
fourth power of the amplitude is exact polynomial quadrature in 4D.
The composite alternative adds the average negative log conditional density
`-mean_ik log p(x_ik | x_i,-k)` to the joint NLL. Conditional denominators are
exact univariate Gram contractions. Its weight is a tuning parameter, not
an assertion that the conditional observations are independent.

## Optimization controls

- Minibatch Adam, with and without gradient norm clipping, and a longer run.
- Full-data L-BFGS, both from the elementary initializer and as continuation
  of an actually optimized TT. Chunking changes memory use only.
- Physical-L2 canonical core sweeps with damped Newton-CG subproblems.

The sweep objective is slightly different: `NLL + log(1 + lambda R(a))`.
For one active core `c`, let `E = integral a^2 + lambda * raw_roughness`.
Rescale the core to `E=1`, and minimize
`c^T Q c - 2 sum_i w_i log |v_i^T c|` within the current observation sign
sector. Its Hessian is positive semidefinite. A fraction-to-boundary line
search avoids every observed amplitude zero, and an Armijo check requires
actual descent. Rescaling by the resulting penalized norm preserves density
shape. The implementation checks that complete sweeps lower the full
training objective. This is not a claim that the whole TT optimization is
convex or globally solved. An Adam log-penalty control separates the change
of objective from the change of optimizer.

A selected continuation checkpoint at step zero means **that continuation
failed to improve validation**. Its source is a TT learned by thousands of
coefficient updates from the near-uniform start. It is not an external
estimator hidden in initialization. Flat-start learning curves are retained.

## Protocol

Swirl observations and train/validation splits are unchanged. The initial
4D grid varies penalty type/strength, objective, optimizer, initial random
seed and perturbation size. The minibatch seed stays fixed when varying the
initializer, so this isolates initialization effects. A subset transfers to
10D; noise-free 6D Ginzburg–Landau is the blobby control.

Checkpoints minimize unpenalized held-out joint NLL. Across recipes, choose
the smoothest full-Hessian amplitude within one paired standard error of
the lowest held-out NLL. This is a development-selection heuristic, not a
post-search confidence statement. It avoids comparing penalties in different
units as if their numerical magnitudes were interchangeable.

`scripts/select_noise_free_v2.py` freezes model and result hashes before
`scripts/evaluate_noise_free_v2.py` imports the synthetic truth. Known
curves/densities enter only final full-dimensional L2, test likelihood and
figure diagnostics. No setting is subsequently chosen from true-density
error. The synthetic selection also fixes which regularizers transfer to
cryo-EM. Cryo-EM continues to use its prescribed measurement-noise TT;
unknown clean-density accuracy cannot be measured. The convex core solver
is restricted to noise-free likelihood: after convolution, an observation's
likelihood is a quadratic form in the active core, and the same convexity
argument does not apply. Cryo-EM therefore transfers the full-Hessian
regularizer and logarithmic penalty form, using Adam followed by full-data
L-BFGS on the prescribed convolved likelihood. It does not silently apply
the noise-free core objective to noisy observations.

The allocation running the 20D cryo-EM refinement received SIGTERM before
its time limit; the cause was not identified. The last saved checkpoint was
iteration 20. `scripts/resume_noise_free_cryoem.py` restored that checkpoint
and restarted the remaining iteration budget. This resets L-BFGS's internal
history, so the result is not an uninterrupted 40-iteration benchmark.
The interrupted history and starting checkpoint are retained, and the
combined result explicitly records the restart. No synthetic fit was rerun
or reselected during recovery.

## Completed noise-free results

All choices below were frozen using observations before evaluating truth.
The table reports full-dimensional relative L2, not a two-dimensional image
error. The previous direct fits also started near uniform; their training
budgets differ. The previous 4D reference used held-out quadratic density
risk for recipe selection; the current primary protocol uses held-out NLL.
The new Adam baseline shares the current NLL selection rule.

| Case | Previous direct fit | New Adam baseline | Selected current fit |
| --- | ---: | ---: | ---: |
| Swirl 4D | 0.09164 | 0.09815 | 0.11555 |
| Swirl 10D | 0.29513 | 0.27360 | 0.22718 |
| GL 6D | — | 0.18442 | 0.16053 |

The selected swirl fits use flat-start canonical core sweeps with full-Hessian
regularization: lambda 0.01 in 4D (checkpoint 15 of 20 sweeps), and 0.003 in
10D (checkpoint 20 of 20). GL selects Adam with the full-Hessian penalty at
lambda 0.001, checkpoint 1,800 of 4,000 updates. These are development-tuned
settings, not universal constants.

**4D is a tradeoff, not an L2 improvement.** Independent test NLL improves
from 1.24893 for the previous direct fit to 1.23287, while L2 error increases.
The smoother fit spreads density more broadly. In 10D both improve: test
NLL goes from -1.66545 to -1.80915. Along the true 10D curve, the fraction
with estimated density below half of truth falls from 39.2% for the previous
direct fit (30.7% for the new Adam baseline) to 12.5%. Missing density has
been reduced, not eliminated. These curve diagnostics are evaluation only.
The weaker axis-curvature core-sweep control reaches **0.19159** L2 in 10D,
but has worse held-out and independent-test likelihood than the selected
full-Hessian fit. It is shown as an unselected control, not promoted using
knowledge of the true density. This reinforces the distinction between
likelihood selection and minimizing integrated squared density error.

The observations support two distinct limitations. In 4D, lowering the
training objective further can worsen held-out likelihood; stronger
regularization helps likelihood but can over-smooth under L2. In 10D,
canonical core updates substantially improve held-out likelihood, indicating
that optimizer conditioning/local progress matter. More modes or rank alone
are not the only issue. Squaring the amplitude does not inevitably produce
the old degree of spikiness: the representation and positivity transform
are unchanged in these improvements.

A finite-sample amplitude-sign check also argues against attributing every
artifact to sign changes: on 4,096 held-out observations, the minority-sign
fractions are 0.195% / 0.024% for the 4D baseline / selected fit and
0.269% / 0.073% in 10D. These are amplitude signs, not negative densities;
all reported densities are squares. Core line searches preserve signs at
training observations only, not throughout the box. The remaining ripples
and undershoot cannot be diagnosed from squaring alone.

The alternative conditional and quadratic scores did not beat the selected
sweep fits on held-out joint NLL in the tested budgets. All alternative
objective variants also receive supplemental post-freeze density-error
evaluation; these diagnostics never reselect the published fits. In 4D,
conditional weights 0.1 / 1 / 10 give L2 errors 0.10115 / 0.10573 / 0.11836,
versus 0.09815 for the ordinary Adam NLL baseline. The tested quadratic
scores give 0.17070 / 0.15676 and introduce substantial holes along the
curve. In 10D, conditional weight 1 gives 0.30254, versus 0.27360 for the
ordinary Adam baseline. These alternatives do not improve full-dimensional
L2 over that baseline either, within this tested grid and training budget.
The separate [objective comparison](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/objectives.png)
shows both criteria. Unlike NLL, the quadratic score remains finite when
the estimated density at an observation approaches zero. Thus it does not
penalize holes in the same way; this distinction is consistent with the
observed failures, but does not establish their sole cause.
Changing initialization seeds affected 10D
Adam more strongly than 4D Adam. No fitted density initializer or sum of TTs
was needed for the reported improvement. Longer runs do not certify global
convergence; 10D's selected sweep is at its budget limit.

The fixed selected sweep recipes were also rerun with initialization seeds
1 and 2, without changing their settings or reselecting the main fit.
The range of retained validation NLL across seeds 0/1/2 was:

| Case | Original Adam recipe | Selected core-sweep recipe |
| --- | ---: | ---: |
| Swirl 4D | 0.00423 | 0.000208 |
| Swirl 10D | 0.06946 | 0.002114 |

This supports keeping the elementary near-uniform initializer. Both solver
and regularizer differ between these recipes, and three seeds are limited
evidence; the experiment does not establish seed independence. Canonical
core gauges remove arbitrary TT scaling from the local norm, while local
Newton steps account for curvature. That is a plausible conditioning
mechanism, not a proof of global convergence.

Ten focused numerical tests pass: evaluation/normalization consistency,
independent Hessian integration, normalized conditionals, quartic quadrature
and derivatives, actual learning from an elementary start, full-data chunk
invariance, convolved-objective agreement, gauge preservation, local/global
quadratic agreement, and monotonically decreasing sweep objectives. An independent 193-node
Gauss–Legendre bounded-box convolution agrees with the selected swirl cross
integrals to less than `8e-13 * integral p^2`, checking the separate
Gauss–Hermite/path-quadrature evaluation.

## Completed cryo-EM transfer

All four cases were rerun with the two frozen swirl regularizer strengths,
0.01 and 0.003, using the prescribed convolution and the original data split.
Lower NLL is better. The table uses each case's selected new fit, not simply
the lowest value from any continuation.

| Case | Previous direct NLL | New selected NLL | New minus previous | Paired SE |
| --- | ---: | ---: | ---: | ---: |
| 4D | 3.534807 | 3.540925 | +0.006119 | 0.001438 |
| Path 4D | 3.541141 | 3.549259 | +0.008118 | 0.001568 |
| 10D | 9.907843 | 9.951340 | +0.043496 | 0.003707 |
| 20D | 24.761178 | 24.790089 | +0.028911 | 0.003844 |

All four select strength 0.003. The path case retains its Adam checkpoint;
the other cases retain L-BFGS refinements. The 20D refinement completed 40
updates across its two segments, with the restart described above.

The latent marginals are visibly smoother and more connected, but **no
case improves held-out noisy likelihood over the previous direct fit**.
The previous fit is a reference outside the new-recipe selection; calling
a model the selected new fit does not mean it beats that reference. The
paired standard errors are descriptive on a reused development split,
not post-search confidence guarantees. Unknown latent truth prevents a
clean-density accuracy claim. This transfer tests the selected penalties;
it is not an exhaustive retuning of their strengths for every noisy case.

The gallery contains 17 current PNGs and its index, with no numerical
sidecars. The audit verified 114 frozen-file hashes and 65 completed fits
(49 main noise-free fits, four initialization checks, and 12 cryo-EM fits).
The study's numerical artifacts occupy about 271 MiB of file contents
(273 MiB allocated on disk); the gallery occupies 5.6 MiB. Compilation
caches were directed to compute-node `/tmp`, not the home directory.
The task's temporary cache directories were removed after publication and
the replacement allocation was released. The final home-quota check was
32.6 GiB used against a 48.8 GiB soft limit (50 GiB hard limit).

## Literature and choices

[Good and Gaskins (1971)](https://academic.oup.com/biomet/article-abstract/58/2/255/263526)
provide the classical root-density roughness-penalty motivation.
Our full-Hessian contraction is a direct multivariate implementation choice.
[Han et al.](https://arxiv.org/abs/1709.01662) motivate canonical matrix-product
representations and local sweeping for likelihood training; the sign-preserving
convex Newton subproblem above is our derivation, not a reproduction of their
optimizer.
[Novikov et al. (2021)](https://proceedings.mlr.press/v161/novikov21a.html)
study TT density fitting with a quadratic objective and show substantial
initialization/optimizer sensitivity in their experiments. Here we also test
a quadratic score, but retain the user's squared Chebyshev representation.
[Varin, Reid and Firth (2011)](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A21n11.pdf)
provide the conditional/composite-likelihood motivation.

[Score matching](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
is an interesting alternative, but its boundary assumptions and the zeros
of a squared polynomial amplitude need care. We did not substitute the
unmodified whole-space score-matching formula on a bounded box.

## Files and reproduction

For new noise-free observations, the core optimizer can be used directly;
it does not require any of this study's historical fit artifacts:

```python
from ftde import density_tt, fit_core_sweeps
from ftde.simple_initialization import init_tt_constant

# train and valid are disjoint observation arrays; domain is the fixed box.
initial = init_tt_constant(domain, modes=49, rank=32, seed=0)
model, record = fit_core_sweeps(
    initial, train, valid, "artifacts/my_direct_fit",
    regularizer="hessian", strength=0.003, sweeps=20,
)
estimated_density = density_tt(model, query_points)
```

The strength in this example is the selected 10D swirl value, not an
automatic recommendation for every dataset. Choose strength using held-out
observations. For measurement noise, use `NoisyDirectTrainingProblem` with
the prescribed forward convolution and Adam/L-BFGS instead.

Current figures: [gallery](../artifacts/gallery_archive/before_dimension_gallery_v9/README.md). Compressed models, compact
numerical records and logs: `artifacts/noise_free_v2`. Historical galleries
were moved without duplication to `artifacts/gallery_archive/before_noise_free_v2`.
The global PNG ignore rule now has an exception for `figs`, so Git and normal
repository file searches can find the images.

Use the `chebdGE` environment and the repository's existing single-GPU
allocation policy. All reported runs use double precision (`JAX_ENABLE_X64=1`).
Set `JAX_COMPILATION_CACHE_DIR`, `CUDA_CACHE_PATH`,
`XDG_CACHE_HOME`, and `MPLCONFIGDIR` to node-local `/tmp` locations. No large
geometry array is persisted. The study enforces a 512 MiB fitting-artifact
budget for its grid and saves only best/last compressed coefficient snapshots.

```bash
export JAX_ENABLE_X64=1
export JAX_COMPILATION_CACHE_DIR=/tmp/ftde_v2_jax
export CUDA_CACHE_PATH=/tmp/ftde_v2_cuda
export XDG_CACHE_HOME=/tmp/ftde_v2_cache
export MPLCONFIGDIR=/tmp/ftde_v2_mpl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python scripts/run_noise_free_v2.py --case swirl4 --names baseline hessian_0.001 hessian_0.01 h1_0.01
python scripts/run_noise_free_v2.py --case swirl10 --names baseline hessian_0.001 hessian_0.01 h1_0.1
python scripts/run_noise_free_followups.py --case swirl4 --names sweep_hessian_strong_flat_long
python scripts/run_noise_free_followups.py --case swirl10 --names sweep_hessian_medium_flat_long
python scripts/run_noise_free_followups.py --case swirl4 --names sweep_flat lbfgs_continue lbfgs_hessian_continue
python scripts/run_noise_free_followups.py --case gl6 --names baseline hessian
python scripts/select_noise_free_v2.py swirl4 swirl10 gl6
python scripts/evaluate_noise_free_v2.py swirl4 swirl10 gl6
python scripts/plot_noise_free_v2.py
python scripts/plot_noise_free_ablation.py
python scripts/plot_noise_free_objectives.py
python scripts/check_noise_free_initialization.py
python scripts/plot_noise_free_initialization.py
python scripts/run_noise_free_cryoem.py dim4 path_dim4 dim10 dim20
python scripts/publish_noise_free_cryoem.py
python scripts/audit_noise_free_quadrature.py
python scripts/audit_noise_free_phases.py
python scripts/audit_noise_free_v2.py --archive-old
```

These commands show the entry points; the frozen per-case protocols and run
configuration files record the complete executed grid. Once frozen, adding
or replacing fitted models requires a new study directory, not overwriting
the frozen evaluation.

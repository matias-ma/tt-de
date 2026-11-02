# Objective and capacity study

**Default decision: retain NLL with the existing regularizer.** MMD and hybrid
objectives remain opt-in experiments. The results below document the completed
comparison, including experimental fits selected by its two validation criteria.

The representation remains one normalized squared Chebyshev functional TT,
initialized near uniform. All fitted structure comes from coefficient
optimization. The previous experiment does **not** establish that L2 is an
incorrect objective or that NLL is uniquely correct. At the population level,
both logarithmic and quadratic density scores identify the true distribution
when it is available in the model family. Their finite-sample behavior,
optimization geometry and treatment of approximation error differ.

For a correctly specified observation model, expected NLL minimizes
`KL(q_true || q_theta)`. A quadratic density score minimizes an unweighted
integrated squared error, emphasizing absolute errors in high-density
regions differently. A kernel score compares a chosen range of spatial
scales. Their common population target does not make their finite-capacity
projections, gradients, or finite-sample selections identical. Under noise,
all of these scores are applied to the observation model `q`; recovering
`p` adds the inverse problem. This is why "lowest noisy validation loss"
and "lowest latent L2 error" can disagree without an implementation error.

## Alternative tested: multiscale kernel score

Let `p_theta` be the latent TT density, `u` the prescribed noise distribution,
and `q_theta = u * p_theta` the observation model. We minimize the average
squared discrepancy between characteristic functions:

`D(theta) = mean_t |phi_p_theta(t) phi_u(t) - mean_j exp(i t.Y_j)|^2`.

This is a random-Fourier approximation to squared maximum mean discrepancy
(MMD), using a mixture of Gaussian kernels. The empirical Fourier means are
statistics of the observations, not an independently fitted density. There
is no KDE, pilot conversion, mixture of signal models, or post-fit smoothing.
The model Fourier integrals come directly from contractions of two copies
of the Chebyshev TT, with oscillatory weighted Gram matrices integrated on
the physical box. Known additive noise acts by multiplication, with no
unstable division by its characteristic function.

Training frequencies and validation frequencies are independent draws.
The kernel uses bandwidths 0.2, 0.5 and 1 times `sqrt(d)` in coordinates
scaled by training-observation standard deviations. This supplies several
length scales rather than choosing one visually from the true density.
Training uses 1,024 fixed frequencies and minibatches of 32; validation uses
512 independent frequencies. No large observation-by-frequency table is
persisted. Finite features yield a proper but generally non-strict score:
matching them alone does not identify every possible density. The ideal
Gaussian-kernel integral is characteristic. We test both pure MMD and
positive weighted sums of MMD and NLL; the latter retains the likelihood's
penalty for assigning nearly zero probability to observations.

Our working hypothesis is that the kernel term may help broad distributional
structure, while likelihood supplies finer local constraints. This is a
hypothesis to test, not a guarantee: kernel bandwidths can also hide narrow
features. Strong noise suppresses Fourier information for either objective.
No loss function can recreate information that the observations do not
identify without additional assumptions or regularization.

For the ideal population kernel score, direct Fourier substitution gives
`D_obs = integral w(t) |phi_u(t)|^2 |phi_p_theta(t)-phi_p_true(t)|^2 dt`.
Thus applying a kernel discrepancy to the noisy observations automatically
downweights the latent details that the noise most strongly suppresses.
The prescribed Gaussian mixtures have nonzero characteristic functions:
infinite-data identifiability is not the issue here. Finite-sample inversion
of very small Fourier multipliers is poorly conditioned. Inverse weighting
could compensate, but would amplify estimation noise and require its own
regularization; we did not silently apply that transformation.

The two GL noise laws have equal marginal variance (standard deviation
0.82976), but different Fourier attenuation. The mixture's narrow component
preserves more high frequencies despite its broad tails. The
[known-noise diagnostic](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/noise_information.png) uses only
prescribed noise parameters. Its `N^-1/2` line is an iid reference scale,
not a hard frequency cutoff or an uncertainty estimate for the MCMC data.

## Capacity, comparisons, and selection

The noise-free GL grid includes ranks 5, 10, 20 and 32, and 17, 25 and 49
Chebyshev modes. The swirl grid also checks smaller ranks/mode counts. Noisy
examples reuse the saved observation splits, domain boxes and prescribed
noise kernels: GL 6D with correlated Gaussian noise, GL 6D with a broad
two-scale Gaussian mixture, and swirl 4D/10D with a mixture including
anisotropy and alternating pair correlations.

Every fit starts from the elementary initializer and receives 4,000 Adam
updates. This matches update budgets, not wall-clock budgets. Saved timings
include contention between jobs sharing the one GPU and should not be read
as controlled speed benchmarks. The integrated full-Hessian amplitude regularizer is retained.
The existing derivative-aware coefficient scaling is also retained: its
per-axis Cholesky factors depend on lambda. Consequently the lambda sweep
changes both penalty strength and optimizer coordinates, not solely the
penalty at fixed Euclidean coordinates. Matched objective comparisons at
fixed lambda share the same scaling.
After the preliminary capacity comparison, alternative objectives and two
additional regularizer strengths are run at an observation-selected capacity.
Because the noise-free swirl held-out NLL preferred rank 32 and 49 modes,
an additional four-objective control group transfers that capacity to both
noisy swirl cases. This extension uses the saved observation-only capacity
selection, not true-density errors. The base plan has 53 fits and the
capacity extension adds eight, for 61 fits total.
The preliminary capacity choice uses an ordinary paired-one-SE heuristic;
the final comparison accounts for GL's MCMC dependence by clustering its
paired score differences by chain. These are development-selection
heuristics, not post-selection confidence guarantees.

Two checkpoints are retained per fit: best held-out NLL and best held-out
kernel score. Two final choices are reported separately, with a preference
for fewer actual coefficients within one paired standard error. Consequently
the new objective is not judged solely by the criterion it seeks to replace.
All candidate hashes and both choices are frozen before truth evaluation.
Full-dimensional latent L2 and pairwise images are final diagnostics, never
fitting inputs or grounds for changing the selections afterward.

Numerical artifacts live under `artifacts/objective_v3`; `figs` contains
figures and its index only. GPU compilation caches are node-local under
`/tmp/ftde_v3_*`, and fitting artifacts have a 512 MiB budget.

## Sources and other alternatives considered

- [Gneiting and Raftery (2007)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
  develop proper scoring rules, including logarithmic, quadratic and kernel
  scores. A different proper score is a defensible statistical choice, not
  automatically an improvement in density recovery.
- [Briol, Barp, Duncan and Girolami (2019)](https://arxiv.org/abs/1906.05944)
  study minimum-MMD estimation and the efficiency/robustness tradeoff from
  the kernel choice. Our TT implementation uses deterministic model integrals
  rather than simulated model samples.
- [Rahimi and Recht (2007)](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)
  provide the random Fourier feature construction for shift-invariant kernels.
- [Hyvärinen (2005)](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
  motivates score matching. For our bounded-box squared amplitude, log-density
  derivatives are singular at amplitude zeros, and integration-by-parts
  boundary terms need treatment. We therefore did not substitute the
  whole-space score-matching formula unchanged.
- [Novikov et al. (2021)](https://proceedings.mlr.press/v161/novikov21a.html)
  use a quadratic objective for TT density estimation. Their results do not
  imply that our squared-amplitude parameterization and optimizer must behave
  the same way. The earlier limited quadratic-score test remains a control,
  rather than evidence against all least-squares estimators.
- [Fan (1991)](https://www.stat.cmu.edu/~jiashun/Teaching/F08STAT756/Lectures/Fan.pdf)
  distinguishes ordinary-smooth and supersmooth errors in nonparametric
  deconvolution theory. Gaussian errors belong to the severely ill-conditioned
  supersmooth setting. These general rates are context for the difficulty,
  not a finite-sample lower bound for these particular low-rank TT examples.

## Completed noise-free results

These are full-dimensional Lebesgue relative L2 errors, evaluated only after
freezing both score-based selections. All capacity controls below use the
same near-uniform initializer, Adam update budget, data split and full-Hessian
penalty. The kernel weight is part of the objective, not a second estimator.

| GL 6D setting | Relative L2 |
| --- | ---: |
| Rank 32, 49 modes, NLL, lambda 0.001 | 0.14473 |
| Rank 32, 25 modes, NLL, lambda 0.001 | 0.12157 |
| Rank 10, 49 modes, NLL, lambda 0.001 | 0.13003 |
| Rank 10, 25 modes, NLL, lambda 0.001 | 0.07710 |
| Rank 10, 17 modes, NLL, lambda 0.001 | 0.07362 |
| Rank 5, 25 modes, NLL, lambda 0.001 | 0.05761 |
| NLL-selected: rank 5, 25 modes, lambda 0.01 | **0.05324** |

The previous selected GL error was 0.16053. The matched controls make the
capacity effect clearer than that historical comparison alone: at fixed
25 modes and penalty, increasing rank from 5 to 32 worsens both validation
likelihood and latent error. More modes also hurt rank 10 in this example.
At 25 modes, the corresponding training NLL improves from 5.87600 at rank
5 to 5.86069 at rank 32, while validation NLL worsens from 5.85914 to
5.87493. This is more specific evidence of an overfitting component than
latent error alone. The absolute train/validation gaps depend on the saved
splits (including GL MCMC variability); their trend across matched ranks is
the useful comparison. Increasing the mode count can worsen training NLL
too, so finite optimization and regularization also matter. Single-seed
comparisons do not establish a universal optimal rank.
The earlier and current models both use squared Chebyshev TTs; this is not
evidence that their representation assumed a Gaussian signal.

| Swirl 4D setting (rank 32, 49 modes, lambda 0.001) | Relative L2 |
| --- | ---: |
| NLL alone | 0.10547 |
| NLL + 10 times kernel discrepancy | 0.10182 |
| NLL + 100 times kernel discrepancy; NLL-selected | **0.09300** |
| Pure kernel objective; kernel-selected checkpoint | 0.29483 |

The hybrid improves density error modestly, but its independent test NLL
is 1.22215 versus 1.21957 for the matched NLL-only control. It is a tradeoff,
not a uniform improvement. The previous selected core-sweep fit had L2
0.11555, while an earlier direct fit reached 0.09164 under a different
protocol. Thus 0.09300 is not claimed as the best swirl result ever obtained.
Rank 10 underfits this swirl: both training and held-out likelihood are
substantially worse (at 49 modes, training NLL 1.57508 versus 1.22185 for
rank 32, and validation NLL 1.58867 versus 1.23862),
and its final L2 is about 0.43 even with 49 modes. Rank 20/49 reaches 0.19540.

The separate kernel-score selection chooses likelihood-trained models in
both noise-free cases: rank 32/25 in GL (L2 0.12157), and a weaker-penalty
rank 32/49 swirl checkpoint (L2 0.10790). Pure kernel training loses even
under this alternative validation criterion in the tested budget. Finite
frequency sets, bandwidth choice, regularization and optimization all limit
the scope of that finding; it does not invalidate kernel-score estimation.
The additional penalty-strength sweep was applied to NLL, not separately to
every kernel weight. In particular, fixing lambda does not fix the effective
regularization relative to differently scaled data losses. A broader kernel
bandwidth/weight/penalty search could change the pure-MMD result.

Four new numerical tests pass: Fourier integrals versus independent cubature
and quadrature refinement; gradient and proper-score identities; prescribed
noise multiplication; and actual kernel-objective learning from the flat
start. For the saved noise TTs, 64 Fourier probes including zero agree with
the analytic noise characteristic functions to at most 4.65e-7 (paired GL);
the other three noisy kernels agree to about 1e-12. This bounds a checked
source of discrepancy between the likelihood's polynomial noise kernel and
the kernel score's analytic prescribed-noise formula; it is not a uniform
error certificate at all frequencies.

## Strong-noise results

Both observation-based selections are shown, rather than picking whichever
looks closer to the known density. The following errors concern the full
latent density; a good-looking pair marginal or reconvolved fit does not
establish accurate recovery of all coordinates jointly.

For correlated-Gaussian GL, likelihood selection chooses rank 10, 25 modes,
`NLL + 10 MMD²`, lambda 0.001. Its latent relative L2 is 0.72397. MMD
selection chooses the matched NLL-only fit (0.75421). In the plotted `(0,1)`
window, the likelihood-selected reconvolved fit has relative L2 0.02877,
while its latent pair error is 0.2884. Strong agreement in observation space
therefore coexists with substantial full-dimensional latent error.

For broad-mixture GL, the choice of validation criterion makes a larger
difference. Likelihood selection chooses rank 10, 25 modes, NLL, lambda
0.0001 (latent L2 0.79599); MMD selection chooses rank 10, 49 modes, NLL,
lambda 0.001 (0.30901). Their independent noisy test NLLs are 9.41432 and
9.41351 respectively. These are two likelihood-trained TTs: the benefit here
comes from selecting capacity/regularization with another score, not from
training a replacement signal estimator. Neither criterion wins everywhere,
and this one-seed comparison is not a guarantee that MMD selection will
recover the better latent fit for an unknown density.

For noisy swirl 4D, likelihood selection chooses rank 32, 49 modes,
`NLL + 100 MMD²`, lambda 0.001 (latent relative L2 0.29476). MMD selection
chooses a rank-20/25-mode `NLL + 10 MMD²` checkpoint (0.39719). The compact
NLL baseline is 0.39322, while **NLL alone at rank 32/49 modes reaches
0.29839**. Thus most of this improvement comes from capacity; adding the
kernel term at matched capacity makes only a small further difference.
The independent noisy test NLLs of the large NLL and hybrid models are
3.92182 and 3.92151. These comparisons are descriptive, with a single fit
seed and no claim of statistically significant superiority.

For noisy swirl 10D, likelihood selection chooses rank 32, 49 modes,
`NLL + 10 MMD²`, lambda 0.001 (latent relative L2 **0.70992**). MMD
selection chooses the same capacity with weight 100 (**0.72179**). The
compact NLL baseline is 0.78907, while the matched large NLL-only control
is 0.71204. Again the objective contributes only a small change compared
with capacity. The large NLL and weak-hybrid independent test NLLs are
6.37324 and 6.37340, respectively; their small difference does not support
a clear likelihood improvement. Pure MMD remains poor here (0.98773 at
the large capacity). Strong-noise full-dimensional recovery remains
substantially incomplete despite learning a much better observation model
than the near-uniform start. In the displayed `(0,1)` window, the
likelihood-selected latent marginal error is only 0.16872 and its
reconvolved error is 0.04958. The much larger full-density error shows why
visually reasonable two-dimensional marginals are insufficient. This study
does not separate all remaining capacity/optimization limitations from the
ill-conditioning caused by noise.

## Results at a glance

Full-dimensional latent relative L2, evaluated after freezing each separate
observation-based selection. This table does not use truth to choose
between the two columns.

| Case | NLL selection | MMD selection |
| --- | ---: | ---: |
| GL 6D, noise-free | 0.05324 | 0.12157 |
| Swirl 4D, noise-free | 0.09300 | 0.10790 |
| GL 6D, correlated Gaussian noise | 0.72397 | 0.75421 |
| GL 6D, broad mixture noise | 0.79599 | 0.30901 |
| Swirl 4D, correlated mixture noise | 0.29476 | 0.39719 |
| Swirl 10D, correlated mixture noise | 0.70992 | 0.72179 |

These results support capacity tuning and retaining the kernel objective as
an option, rather than replacing likelihood universally. The hybrid's gains
are modest in the matched controls. Kernel-based selection can be useful
(GL mixture noise), but it also chooses worse latent fits in other cases.
One initialization seed and one set of Fourier draws per role were used;
robustness to their variation has not been established in this study.

## Verification and storage

- All **61 fits completed 4,000 updates**. Neither checkpoint criterion
  selected the initial state in any fit.
- **14 tests passed**, including four new Fourier-score tests and the
  existing direct-training/core-sweep tests. Ruff and `git diff --check` pass.
- The audit verified **366 frozen file hashes**, original data hashes,
  and **46 evaluated checkpoints**, including quadrature refinement.
- Independent bounded-box Gaussian quadrature checks all six selected
  swirl cross integrals. The largest difference relative to the true squared
  norm is 1.10e-08.
- The gallery has **34 PNGs and its README only**. Numerical artifacts
  occupy about 85.1 MiB, below the 512 MiB study budget.
  JAX/CUDA/Matplotlib caches were confined to the compute node's `/tmp`,
  then removed. The one-GPU allocation was released after the checks.

Audit details are in `artifacts/objective_v3/audit.json`; test output is in
`tests_complete.log` in that directory. The retained noise-free swirl 10D
and cryo-EM figures are from the preceding study and are identified in the
figure index; those cases were not refitted here.

## Reproduction

Within an existing allocation, run
`srun --jobid=JOB bash scripts/run_objective_gpu.sh scripts/run_objective_study.py`.
After the noise-free swirl capacity choice is saved, run
`srun --jobid=JOB bash scripts/run_objective_gpu.sh scripts/extend_objective_capacity.py`.
The frozen plan records the grid and per-fit configurations record all
executed settings. `select_objective_study.py` freezes the selections;
truth evaluation and figure publication are separate stages.
`finish_objective_study.py` waits for the planned counts, freezes each case,
evaluates it, and publishes its figures. `plot_objective_capacity.py` adds
the matched rank/mode comparison once both noise-free cases are evaluated.

For publication from completed runs, execute the following using the same
GPU wrapper as above:

```text
scripts/finish_objective_study.py
scripts/check_objective_generalization.py
scripts/plot_objective_capacity.py
scripts/plot_objective_noise.py
scripts/plot_objective_candidates.py
scripts/audit_objective_study.py
```

To refresh particular density figures without fitting, pass case names to
`plot_objective_study.py`, e.g. `gl6_paired swirl4_mixture`. Completed fits
are reusable with identical configurations; adding or changing a fit after
a case is frozen is rejected. Existing logs are appended rather than erased
when restarting a driver.

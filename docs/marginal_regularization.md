# Noisy density recovery and marginal curvature

This follow-up uses the same squared Chebyshev TT, prescribed noise kernels,
observations, domains and capacities as the previous noisy diagnosis. The
purpose is to distinguish excessive joint smoothing from oscillations in
one-coordinate density marginals.


## Completed results

Twenty-eight new fits completed across regularization, smooth-initialization,
and calibrated-penalty studies. All choices below were fixed from noisy
validation observations before evaluating truth. The penalty improves the
swirl comparison modestly; it does not close the noisy/noise-free gap or
solve the GL case.

| Case / fit | Full latent relative L2 | Mean 1D relative L2 | Marginal curvature | Observed validation NLL |
| --- | ---: | ---: | ---: | ---: |
| Swirl: original strong-penalty control | 0.29839 | 0.04767 | 3.582 | 3.865824 |
| Swirl: matched weak-penalty continuation, no marginal penalty | 0.29998 | 0.07204 | 11.640 | 3.863550 |
| Swirl: light calibrated marginal penalty | 0.27697 | 0.04638 | 4.476 | 3.863557 |
| Swirl: moderate calibrated penalty, one-SE smooth choice | 0.27985 | 0.05073 | 3.410 | 3.863627 |
| GL: original control, retained by one-SE smooth choice | 0.49898 | 0.19590 | 67.323 | 9.475640 |
| GL: minimum-validation-NLL fit | 0.96076 | 0.21104 | 131.616 | 9.474186 |
| GL: best marginal-penalty fit with actual updates | 0.80248 | 0.19251 | 10.957 | 9.479782 |

The light swirl penalty has slightly lower latent L2 than the moderate
penalty; truth does not override the predeclared one-SE smoothness choice. Mean 1D errors use the fixed
`[-2.6,2.6]` window, without renormalizing curves. Curvature integrates over
the entire fitted box. Full-dimensional L2 uses refined quadrature.

The cleanest isolation is the three calibrated swirl continuations: same
source TT, amplitude penalty `1e-5`, batches, whitening and optimizer. The
moderate marginal penalty reduces curvature by **70.7%**, full-density L2
by **6.7%**, and mean 1D error by **29.6%** relative to the matched no-marginal
continuation. Its independent noisy test NLL changes by only
`+0.0000415 ± 0.0001936` paired SE. The light penalty changes test NLL by
`+0.0000075 ± 0.0000952` and reduces latent L2 to 0.27697. These test
likelihood differences are too small to distinguish at this sample size.
This is direct evidence that similar convolved likelihoods can accompany
visibly different latent oscillations.

The moderate strength is `0.0003102343`; the light strength is `0.0001034114`.
Both retain update 400 of a 2,000-update continuation, rather than their
last iterate. Their source first learned under the old strong joint
penalty, then received a weak-penalty continuation. Starting from scratch
with the weak penalty is **not** an equivalent recipe: its latent L2 is
1.178, despite a superficially competitive observed likelihood. The
[smooth-initialization study](smooth_initialization.md) did not remove that
failure.

Relative to the original swirl control, the moderate choice improves full
L2 from 0.29839 to 0.27985 but slightly worsens mean 1D error, 0.04767 to
0.05073. The light penalty improves both slightly. The previous gallery's
longer-Adam fit had L2 0.28649, so the improvement over that published fit is
only about **2.3%** for the moderate choice, not a large breakthrough. The
first grid's observation-selected weak-Hessian continuation reached 0.26450
L2, but the cross-study one-SE rule did not select it. It remains a reported
control, not a truth-selected replacement.

GL remains unresolved. Plain minimum noisy NLL picks a fit with latent L2
0.96076; its test NLL gain over the old control is `0.001516 ± 0.001537`
paired chain-clustered SE. The predeclared smoothness rule retains the old
0.49898 fit. Stronger marginal smoothing removes artificial peaks, but also
broadens genuine peaks and worsens joint recovery. The lighter calibrated
GL continuations reject their updates and retain their weak-penalty source.
No new GL improvement is claimed. The older broader hyperparameter study's
likelihood-selected GL fit had L2 0.79599; it is a different reference from
the fixed-strength control used here.

Figures:

- [Final swirl 1D marginals](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/noisy_recovery_1d.png).
- [Final GL 1D marginals](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/noisy_recovery_1d.png).
- [Calibrated swirl pair marginals](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/calibrated_marginals_2d.png).
- [Regularization comparisons](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/marginal_regularization.png).
- [Calibrated comparisons](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/calibrated_marginals.png).

Frozen cross-study choices and all metrics are in
`artifacts/calibrated_marginals_v7/overall_selection.json` and `summary.json`.
The smoother swirl checkpoint is
`artifacts/calibrated_marginals_v7/swirl4_mixture/marginal_moderate/fit.npz`.
No global regularizer default is changed from this small development study.

## Why accurate convolution does not make the two problems equivalent

For a latent density `p` and known measurement kernel `K`, noisy likelihood
fits `q = K*p`. At the population level its excess risk is
`KL(K*p_true || K*p)`, rather than `KL(p_true || p)`. An accurate forward
operator evaluates that different risk accurately; it does not undo the
information lost by the observation channel.

For a small density perturbation `h`, local noisy excess risk is approximately
`(1/2) integral (K*h)^2 / q_true`. For Gaussian noise with standard deviation
`sigma`, a Fourier perturbation at angular frequency `omega` is attenuated
by `exp(-sigma^2 omega^2 / 2)`. At sigma 0.3 and omega 10, the amplitude
factor is about 0.011. Fine latent ripples can therefore change observed
likelihood very little. Infinite-data Gaussian convolution is injective;
its inverse is nevertheless unstable with finite samples. Even at infinite
sample size, fixed regularization or model misspecification can give different
clean and noisy optima. Exact equivalence requires an identifiable forward
map, a correctly specified model, and removal of estimation/penalty error. See
[Fan (1991)](https://www.stat.cmu.edu/~jiashun/Teaching/F08STAT756/Lectures/Fan.pdf)
for the statistical distinction between ordinary and supersmooth noise.

There is a second, algorithmic issue in this repository. The earlier
[diagnosis](noise_diagnosis.md) found that the fixed amplitude-Hessian
penalty outweighed the noisy-likelihood benefit of detailed latent
structure. Thus a fit can simultaneously be too broad in its joint density
and show small oscillations in its marginals. Increasing that same penalty
need not solve both problems.

## The new penalty

Let `p_k(x)` be the exactly integrated, normalized one-coordinate marginal.
We add

```
R_marginal = sum_k width_k * s_k^4 * integral (p_k''(x))^2 dx,
s_k = 0.05 * width_k.
loss = mean noisy NLL + lambda_H * R_amplitude_Hessian
                      + lambda_M * R_marginal.
```

The marginal is quadratic in the amplitude coefficients. Left and right
Gram contractions integrate every other coordinate, then univariate
Gauss–Legendre quadrature integrates the squared derivative exactly for
the polynomial degree. No rank-squared density TT is materialized.
The width factor makes the penalty dimensionless, and it is invariant
under TT gauge changes and overall amplitude scaling. It is a training
penalty, not a filter applied to the displayed curves.

The joint amplitude penalty retains weak control over interactions that
1D marginals cannot see. The marginal penalty alone does not constrain
all joint oscillations or dependence. Like the existing derivative
penalties, it measures interior derivatives and does not penalize the
jump from a nonzero box-boundary value to zero outside the box.

Use `marginal_strength=` in either `DirectTrainingProblem` or
`NoisyDirectTrainingProblem`. It defaults to zero, preserving existing
objectives. The strength is a tuning parameter, not a universal default.
`ftde.marginals.squared_univariate_grid` evaluates the exact normalized
1D marginal for inspecting a fitted model.

## Experiment protocol

- Swirl 4D: TT rank 32, 49 amplitude coefficients per coordinate.
- Ginzburg–Landau 6D: TT rank 10, 25 coefficients per coordinate.
- Start each continuation from the previous NLL-trained noisy control.
  No clean fit or fitted pilot initializes a noisy model.
- Six equal-budget continuations: amplitude-Hessian strength 0.001,
  0.0001, or 0.00001; and amplitude strength 0.0001 with marginal strengths
  0.001, 0.01, or 0.1. Each receives 2,000 Adam updates at initial learning
  rate 0.0003, batch size 256, identical batch seed and checkpoint cadence.
- Within each case, the best non-marginal and best marginal continuation
  recipe, selected by observed validation NLL, are each retrained for
  4,000 updates from the elementary near-uniform start at learning rate
  0.001. These test whether continuation determines the outcome; they are
  not equal-total-compute comparisons to the continuations.
- Final selection minimizes unpenalized noisy validation NLL across all
  candidates, including the original control. Freeze model, data,
  configuration and validation hashes before evaluating latent truth.
- A secondary rule, recorded while fitting and before truth evaluation,
  chooses the lowest exact marginal curvature within one paired standard
  error of the minimum validation NLL. This uses the existing selection
  utility, with 200 chain clusters for GL. Both choices are reported; it
  does not replace the predeclared minimum-NLL primary comparison.
- Evaluate full-dimensional latent L2, actual 1D and pair marginals,
  integrated marginal curvature, and independent observed test NLL.
  Test likelihood uses paired errors, clustered over the 200 chains for
  GL. Development selection reuses the existing validation set; the
  post-selection standard errors do not correct for search multiplicity.

A continuation selecting step zero retains its already trained noisy
source model because its new updates failed validation. It is not a
near-uniform fit. All result files retain the full learning trajectory.

The existing optimizer uses Sobolev whitening whose matrices depend on
the amplitude-penalty strength. Changing that strength therefore changes
both the objective and optimizer coordinates; the marginal-strength
comparisons at fixed amplitude strength share the same coordinates.
These finite-budget runs do not certify global optimization.

Artifacts are isolated in `artifacts/marginal_regularization_v5`; previous
frozen experiments and published comparisons remain available. The study
uses one GPU, four CPUs and 32 GiB on the cryoem partition. Fitting has a
256 MiB artifact limit and reuses node-local compilation caches.

## Reproduction

Within the `chebdGE` environment and a single GPU allocation:

```bash
bash scripts/run_noise_diagnosis_gpu.sh scripts/run_marginal_regularization.py
bash scripts/run_noise_diagnosis_gpu.sh scripts/evaluate_marginal_regularization.py
bash scripts/run_noise_diagnosis_gpu.sh scripts/plot_marginal_regularization.py
```

Prepend `srun --jobid=... --ntasks=1 --cpus-per-task=4` on this cluster.
The fitting driver refuses to modify a frozen study. Use a new artifact
root for a new study; evaluation can be resumed against the existing one.

## Numerical verification

Thirteen focused tests passed: the existing direct-training and squared-pair
marginal checks, plus independent dense cubature of the new penalty,
coefficient-gradient finite differences, TT gauge and amplitude scaling,
coordinate-unit invariance, exact univariate normalization, support handling,
and integration of the penalty into both clean and convolved losses.

## Lighter, calibrated marginal penalties

Before evaluating truth, the first grid showed that its marginal penalty
weights were too aggressive for GL: validation rejected every continuation.
The fitted GL marginal curvature was much higher than swirl's, making the
same numerical weight a larger objective contribution. A separate follow-up
in `artifacts/calibrated_marginals_v7` therefore uses each case's frozen
minimum-NLL fit as its source and defines

```
lambda_M = budget / R_marginal(source),   budget in {0, 0.001, 0.003} nats.
```

The zero-budget run is a matched control. Every run retains its source's
amplitude-Hessian strength and receives 2,000 Adam updates, learning rate
0.0001, batch 256, and the same batches/checkpoints. Calibration uses the
fitted density, not a latent truth or clean-data comparator. The budgets
are a tuning grid, not an estimated universal optimal penalty.

Both minimum validation NLL and the paired-one-SE smoothness choice are
frozen. A cross-study choice across the original regularization grid,
[smooth-initialization checks](smooth_initialization.md), and this calibrated
follow-up is also frozen before any new truth evaluation. The calibrated
figures' source is the preceding **selected weak-penalty fit**, whereas the
first-grid figures' source is the older strong-penalty control.

Run `scripts/run_calibrated_marginals.py` after the first study is frozen,
then `scripts/finish_noisy_followup.py` after all three studies are frozen.
The latter verifies frozen inputs, fixes cross-study choices, evaluates
all candidates, and plots the comparisons.

The final marginal-penalty comparison shows the lowest-validation-NLL
marginal recipe whose selected checkpoint actually made updates. Rejected
continuations retaining step zero are not counted as successful applications
of the new penalty. This diagnostic choice is frozen before truth evaluation
and is shown separately from the overall minimum-NLL and one-SE choices.

Final audit: 28 completed new fits, 130 distinct frozen-file hashes verified,
18 focused tests passed, and Ruff checks passed for all changed Python
modules/scripts. Numerical artifacts occupy about 45.5 MiB across the three
studies. Results use one training realization per case; initialization
repeats are not independent training-dataset replications.

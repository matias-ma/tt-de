# Why noisy Chebyshev TT recovery is harder

This study keeps **NLL**, one normalized squared Chebyshev functional TT,
and the elementary constant-plus-small-random-coefficient initialization.
The additional Gaussian forward implementation caches integration moments
of the *known measurement kernel*. It neither fits a pilot density nor
changes the signal basis. Its coefficient values and gradients are checked
against the existing direct Gaussian quadrature.

## Findings

The current convolution is numerically accurate on the audited cases.
The clearest algorithmic issue is that **convolution weakens the likelihood's
preference for fine latent detail while the fixed smoothness penalty keeps
penalizing that detail**. Thus poor latent recovery is not necessarily a
bad local minimum: the regularized objective can actually prefer a blurrier
latent density. The experiments do not show that the remaining error is
statistically unavoidable.

### Matched noise levels

All settings, the box, initialization, latent draws and standard-noise draws
are held fixed; each trainer receives only its own observations.

| Gaussian noise sigma | Full 4D latent relative L2 | Latent pair (0,1) L2 | Reconvolved pair (0,1) L2 | Estimated observed KL, nats |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.1069 | 0.0443 | 0.0443 | 0.02087 |
| 0.15 | 0.1744 | 0.0785 | 0.0394 | 0.01039 |
| 0.30 | 0.3739 | 0.1730 | 0.0385 | 0.00800 |
| 0.60 | 0.7404 | 0.4667 | 0.0161 | 0.00317 |

The pair errors use the same `[-2.6,2.6]^2` window, without renormalizing
the displayed densities. Observed KL is a 4096-observation Monte Carlo
estimate against the true **convolved** density, with standard errors
0.00286, 0.00169, 0.00199 and 0.00166 respectively. It is not latent KL.
The observed fit becomes closer by these metrics while latent recovery
deteriorates. Extra measurement noise did not help latent recovery here.

[Noise-strength metrics](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/noise_strength.png) ·
[density panels](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/noise_strength_marg_01.png).

### The fixed penalty can prefer the wrong latent detail

After freezing every model, we evaluated the frozen noise-free TT through
each noisy forward operator. This provides a same-capacity, detailed
comparator **only for diagnosis**. It never initializes, trains or selects
any noisy fit. Training NLL in this comparison uses all 100,000 noisy
observations; independent test data are also checked.

Values below are the clean-data comparator **minus** the noisy-trained TT:

| Noise sigma | Difference in noisy training NLL | Difference in penalty | Difference in total training objective |
| --- | ---: | ---: | ---: |
| 0.15 | +0.00069 | +0.00402 | +0.00472 |
| 0.30 | −0.00450 | +0.01064 | +0.00614 |
| 0.60 | −0.00452 | +0.01740 | +0.01288 |

At sigma 0.6, the detailed comparator fits the noisy training data **better**
by 0.00452 nats, yet its extra penalty of 0.01740 nats outweighs that
benefit. The actual objective therefore prefers the noisy fit, whose latent
L2 is 0.7404 instead of 0.1069. The comparator also improves independent
noisy test NLL by 0.00351 ± 0.00151 paired SE. At sigma 0.3 the same pattern
occurs, with a test NLL improvement of 0.00631 ± 0.00167.

This separates regularization bias from a representation barrier: a much
more accurate latent density already exists in the same TT rank, basis,
box and positivity parametrization. It does not prove that this comparator
can be recovered from the noisy data alone. Nor does it license choosing
lambda from its known latent error. The next useful algorithm experiment
is observation-only calibration of the regularizer under noise, while
keeping NLL and the Chebyshev TT. Blindly removing regularization is not
justified by this diagnostic.

[Data term versus penalty](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/noise_regularization_bias.png).

### Optimizers and initialization

| Original prescribed-noise case | Matched 4000-step Adam control | Fresh starts | Longer Adam, selected checkpoint | L-BFGS, selected checkpoint | L-BFGS final iterate, diagnostic only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Swirl 4D, r=32, 49 modes | 0.2984 | 0.3149 / 0.3173 / 0.3141 | **0.2865** | 0.2971 | 0.2953 |
| GL 6D, r=10, 25 modes | **0.4990** | 0.4103 | 0.4990 | 0.4990 | 0.4956 |

Entries are full-dimensional latent relative L2, evaluated after selection.
Bold entries are the validation-selected choices within the fixed-setting
optimizer study. GL's fresh-start fit has lower latent error, but validation
did not select it; truth is not used to override that choice.

The three fresh swirl seeds reach very similar fits. Their validation NLL
differences from the matched old control are within one paired SE. Longer
Adam improves latent L2 by about 4% relative, but its independent noisy test
NLL change is only +0.00016 ± 0.00080 paired SE. L-BFGS lowers the full-data
penalized objective from 3.890558 to 3.889826 for swirl, and 9.446968 to
9.445206 for GL. Neither run reaches a convergence certificate within 30
iterations; these are finite-budget controls, not a proof of global
optimality. The improvements found are small compared with the noise gap.

GL's continuations select step zero of the **continuation**, meaning the
already learned noisy fit. The original fit learned its density through
NLL optimization; no fitted pilot replaces that learning. The final-iterate
diagnostic confirms the continuation did perform coefficient updates.

The truth-free stability check finds that the fresh swirl fits differ from
the old fit by 11–12% in latent pair-(2,3) L2, versus about 3.6–3.7% after
convolution. Similar observed likelihoods can accompany more variable latent
densities. These differences between fits are distinct from truth errors.

[Optimization and test scores](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/noisy_optimization.png) ·
[truth-free stability](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/noisy_stability.png).

### Convolution accuracy and geometry

| Audit case | Maximum log-density difference from analytic noise | Relative coefficient-gradient difference |
| --- | ---: | ---: |
| Swirl 4D mixture | 2.04e−14 | 1.18e−13 |
| Swirl 10D mixture | 7.64e−14 | 6.78e−14 |
| GL 6D mixture | 1.81e−13 | 3.29e−12 |
| GL 6D paired Gaussian | 3.87e−6 | 3.14e−6 |

The paired-GL noise TT has a small approximation error. The polynomial
convolution itself agrees with unprepared and refined integration at
approximately floating-point accuracy. These are probe-based checks,
including observed tails, not uniform bounds over all possible models and
inputs. There is no numerical discrepancy on the scale of the recovery
errors. The independent full-dimensional GL truth convolution also passes
quadrature doubling (1.64e−13 in log density) and wider-domain checks
(5.68e−14).

Widening the noise-free GL box at 25 modes changes latent L2 from 0.0861
to 0.1009. Using 49 modes on the wider box improves it to 0.0626. Geometry
and capacity matter, but this control is far too small an effect to explain
the noisy GL error near 0.5 by itself.

[Convolution audit](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/convolution_accuracy.png) ·
[GL box and resolution control](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/noise_box_control.png).

The main mixture-noise figures were refreshed in place for
[swirl (0,1)](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/noisy_swirl4_mixture_marg_01.png),
[swirl (2,3)](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/noisy_swirl4_mixture_marg_23.png),
[GL (0,1)](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/noisy_gl6_mixture_marg_01.png) and
[GL (2,3)](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/noisy_gl6_mixture_marg_23.png).

## The distinction between fitting and deconvolving

Write the latent density as `p_theta = a_theta² / integral(a_theta²)` and
the observation density as `q_theta = K * p_theta`. The population noisy
NLL differs from `KL(q_true || q_theta)` by a constant. It does **not**
directly minimize `KL(p_true || p_theta)` or latent L2 error.

For a small normalized-density perturbation `p_theta = p_true + epsilon h`,
the local excess NLL is approximately

```
(epsilon² / 2) * integral (K*h)² / q_true.
```

Convolution can make `K*h` tiny even when `h` is appreciable. In Fourier
coordinates, Gaussian noise multiplies a density perturbation at angular
frequency `t` by `exp(-sigma² |t|² / 2)`. For example, `sigma=0.3` and
`|t|=10` attenuate it by about 0.011. This is an illustration of
conditioning, not a sample-size forecast for these particular TTs.

Equivalently, the observed parameter score is the posterior conditional
expectation of the clean score: `s_Y(y) = E[s_X(X) | Y=y]`. Taking a
conditional expectation reduces its variance, hence the Fisher information.
Changing the optimizer cannot restore statistical information that this
observation channel suppresses. Gaussian convolution is still injective
in the ideal infinite-data problem; practical instability is not the same
as exact nonidentifiability.

Full-dimensional L2 also magnifies small width errors. As a simple
illustration, a centered isotropic Gaussian estimate that is 20% too wide
has relative L2 error 0.2347 in 2D, 0.3722 in 4D and 0.6492 in 10D:
`sqrt(1 + r^(-d) - 2*(2/(1+r²))^(d/2))`, with width ratio `r=1.2`.
This illustrates why a plausible pair marginal need not imply accurate
joint recovery; it is not an attribution of the measured swirl error to
width alone.

The underlying density can therefore be wrong while its reconvolution
closely matches the data. More independent measurement noise does not add
information. It might smooth a nonconvex optimization landscape, but a
useful noise-continuation procedure must ultimately return to the original
data and noise law. More independent **observations** are different: they
can help, although Gaussian deconvolution has particularly unfavorable
nonparametric convergence rates. Increasing the sample count while retaining
the same fixed penalty on the **mean** NLL does not by itself remove the
regularization bias demonstrated above.

These distinctions agree with the primary literature:

- [Fan, 1991, *On the Optimal Rates of Convergence for Nonparametric Deconvolution Problems*](https://www.stat.cmu.edu/~jiashun/Teaching/F08STAT756/Lectures/Fan.pdf).
- [Fan, 1992, *Deconvolution with supersmooth distributions*](https://onlinelibrary.wiley.com/doi/abs/10.2307/3315465).
- [Calmon, Polyanskiy and Wu, *Strong Data Processing Inequalities for Input Constrained Additive Noise Channels*](https://arxiv.org/abs/1512.06429).

## Controls and safeguards

- Independent analytic Gaussian-block quadrature checks the existing
  Chebyshev noise TT and convolution, including correlated noise and
  globally shared noise-component labels. Twelve held-out observations per
  case include four observed tails. The reference uses Gauss orders 128
  and 256. Its quadrature-weight SVD is an integration device, not another
  signal estimator.
- Additional checks disable moment truncation, use unprepared integration,
  double polynomial quadrature, compare coefficient gradients and use a
  finite-difference directional derivative.
- A truth-free stability check compares the fitted pair densities to the
  previous fit, both before and after convolution. Latent pair distances
  use exact polynomial Gram contractions. Convolved distances use Parseval
  integration under the analytic known noise, refining frequency spacing
  from 0.2 to 0.1 and polynomial quadrature from 256 to 512 nodes; the
  frequency window is `[-40,40]^2`. These are **differences between fits**,
  not estimation errors against the true density.
- Original mixture-noise swirl: three near-uniform starts at rank 32 and
  49 modes; original GL mixture: an additional start at rank 10 and 25 modes.
  All use 4000 Adam updates, batch 256, initial learning rate 0.001 and
  Hessian penalty 0.001. New restarts share the same batch sequence.
- Both cases also continue the previous noisy NLL fit for 4000 Adam updates
  at learning rate 0.0003, and separately for 30 full-data L-BFGS iterations.
  These continuations use the identical objective and observations.
  Final L-BFGS iterates are also evaluated as explicitly unselected
  diagnostics, so a rejected continuation is not mistaken for an optimizer
  that made no coefficient updates. Numerically tied choices prefer the
  previous fit.
- A matched 4D swirl experiment uses independent Gaussian noise standard
  deviations 0, 0.15, 0.3 and 0.6. Every fit uses rank 32, 49 modes, the same
  box, initialization, batch sequence, regularizer and 4000-update budget.
  The box `[-3,3]^4` is frozen by rounding out the previous **noisy-only**
  domain. Matched synthetic latent draws are used only by the data generator;
  each noisy trainer reads only its own noisy observations.
- GL noise-free controls compare its original narrower box to the wider
  noisy-data box at rank 10, 25 modes and otherwise identical settings.
  This changes both polynomial resolution per physical unit and the
  regularizer's physical length scales (5% of box width). It is a combined
  geometry control, not an isolation of those two effects. A 49-mode
  wide-box control tests whether increasing resolution helps: the width
  ratio is about 1.7, so 49 is the next existing mode-grid size above
  `25 * 1.7`. This is a geometry-based choice, not a truth-selected setting.
- Every model/checkpoint choice uses observed validation NLL. File hashes
  and choices are frozen before new latent truth evaluation. Independent
  test NLL differences use paired standard errors (clustered by the 200
  MCMC chains for GL; IID for swirl). The Gaussian sweep also
  compares test NLL against the true *observed* density, avoiding misleading
  comparisons of raw NLL across different noise levels.

The old Adam controls use the previous study's batch random-key sequence;
the new seed restarts share a different, fixed sequence. Thus comparisons
among new seeds isolate initialization, whereas comparison to old Adam
also includes minibatch-order variation. Continuations are additional work,
not equal-compute substitutes. Thirty L-BFGS iterations and a few restarts
cannot certify a global optimum. The noise-strength curve is one matched
data realization at a fixed optimization budget and regularization setting;
it is not a minimax lower bound or an average over independently generated
training datasets.

The matched old controls are the previous study's **NLL-only, lambda=0.001**
fits at the same rank and mode count. They are not necessarily the fits
chosen by that study's broader hyperparameter/objective search. In
particular, the old GL gallery's likelihood choice used lambda=0.0001.
The refreshed mixture-noise figures explicitly show the current fixed-
lambda optimizer comparison; they do not claim a new global hyperparameter
selection or a comparison against every older gallery winner.

## Reproduction

Use the `chebdGE` environment and one allocation with one GPU, four CPUs,
32 GB memory and partition `cryoem`, as required by `instructions.txt`.
Run GPU fitting sequentially on that allocation:

```bash
bash scripts/run_noise_diagnosis_gpu.sh scripts/run_noise_diagnosis.py
bash scripts/run_noise_diagnosis_gpu.sh scripts/extend_noise_diagnosis.py
bash scripts/run_noise_diagnosis_gpu.sh scripts/check_noise_objective_bias.py
```

Use `srun --jobid=...` in front of these commands on the cluster. The
extension driver finishes with freezing, evaluation and plotting. The last
command is the explicitly post-selection clean-data comparison. These
training commands describe a fresh study; the saved study is frozen and
refuses to overwrite its training results. Use a fresh artifact root to
repeat training, or rerun evaluation/plotting against the saved checkpoints.
The
analytic-noise audit is `scripts/noise_convolution_audit.py`; truth
evaluation and figure publication use `scripts/evaluate_noise_diagnosis.py`
and `scripts/plot_noise_diagnosis.py`. All checkpoints, observations and
numerical records are under `artifacts/noise_diagnosis_v4`. Figures remain
in the existing `figs/swirl`, `figs/ginzburg_landau` and `figs/convergence`
directories. Compilation caches are node-local `/tmp/ftde_v4_*`, not in
the home directory. The study enforces a 256 MiB artifact budget.

Verification: 77 frozen data/model/configuration/validation files retain
their hashes. All 15 new optimization runs completed; every fresh start
selected a checkpoint after learning. The 17 targeted tests cover the
existing Gaussian/general-TT forward maps, cached moments and gradients,
the analytic correlated-noise reference against dense 4D cubature, and the
Parseval calculation against an analytic blurred constant. Ruff checks
pass for the new scripts and test file. Numerical artifacts occupy about
40.3 MiB; only PNG figures and the existing index are written under `figs`.
The node-local caches were cleared and the study's GPU allocation released
after verification.

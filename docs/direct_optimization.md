# Direct TT optimization from an elementary start

This previous study is superseded by the [current optimization study](noise_free_optimization.md).
Its figures are preserved in `artifacts/gallery_archive/before_noise_free_v2`.

The focus was learning the density by optimizing functional TT
coefficients. No KDE, fitted signal mixture, estimated curve, or previously
fitted density supplies the initializer. `init_tt_constant` constructs a
constant amplitude with a small smooth random perturbation using only the
box, rank, polynomial size and seed. The signed experiment likewise begins
with a uniform density plus a small random perturbation. All saved cores
use the Chebyshev basis.

The older mixture-pilot studies remain available as historical experiments.
Their step-zero swirl results do not demonstrate successful TT refinement
and are not the recommended workflow.

## Two objectives, with actual TT training

The first path retains `p = a_TT² / integral(a_TT²)` and minimizes the
negative log likelihood of `U*p`, where `U` is the prescribed noise. The
controls compare the existing minibatch Adam implementation with deterministic
full-data L-BFGS refinement of the same objective and curvature penalty.
L-BFGS uses a line search, and its evaluation chunks affect memory use only;
every gradient uses all training observations. Its starting checkpoint is
the TT already trained from the elementary start, not a separate estimator.

The second path directly represents a signed function `f_TT`, with
`integral(f_TT)=1`, and minimizes

\[
J(f)=\int (U*f)(y)^2\,dy
 -\frac{2}{N}\sum_{i=1}^N(U*f)(y_i)
 +\lambda\sum_k s_k^4\int (\partial_k^2 f)^2.
\]

Here `s_k` is the prescribed noise standard deviation in coordinate `k`.
For noise-free observations, `U` is the identity and `s_k` is 5% of the
coordinate box width. This sets the units of the curvature penalty without
using the unknown signal. The noise-free quadratic term uses exact physical
Chebyshev Gram matrices; its empirical term is simply the sample average
of the TT density.
The first two terms estimate the squared error of the convolved function,
up to a constant independent of the estimator. They remain defined when
either `f` or `U*f` is negative. A tiny `1e-10 * integral(f²)` ridge makes
the conditional linear solves nonsingular.

For four dimensions, the TT can be optimized as two middle cores with
fixed full boundary ranks. Alternating these cores gives constrained convex
quadratic subproblems. Conjugate gradient and a scalar constraint correction
solve them, with a QR gauge change between updates. The initializer fits
nothing; all observed structure is learned through this objective.
Some inner solves in the strong-negative-penalty control do not reach the
requested conjugate-gradient tolerance. Their residuals are recorded;
accepted steps must still decrease the actual objective and preserve mass.

The empirical coefficient array is the sample average of the **linear
forward operator**, computed using the prescribed TT noise kernel. It is
part of the objective, not a fitted surrogate signal density. The integrated
quadratic term uses the Gaussian overlap identity for the supplied noise,
with pairwise numerical integration over the latent box. This implementation
currently requires four dimensions and noise that is a Gaussian mixture
block diagonal in pairs, or no measurement noise. The mixture here specifies **measurement noise**;
there is no learned signal mixture.

## Allowing negative values and truncating at the end

The unconstrained signed runs apply no clipping during optimization. Their
final probability density is separately defined as

\[
p_+(x)=\frac{\max(f(x),0)}{\int\max(f(x),0)}.
\]

Clipping changes the model. If the signed function integrates to one and
has negative mass `m_minus`, its positive part integrates to `1+m_minus`.
The final density therefore rescales all positive regions as well as
removing the negative ones. Validation must convolve this **clipped joint
density**, not clip a displayed marginal or reuse the signed convolution.

An additional control adds `eta * integral(min(f,0)²)` to the training
objective. It still convolves the signed function. For negative `f`, this
penalty has derivative `2*eta*f`, so it pushes values upward instead of
removing their gradients. Semismooth Newton core updates and a line search
decrease the actual penalized objective while preserving mass. The penalty
uses joint tensor quadrature, which is an explicit four-dimensional
limitation. Final truncation and renormalization are still measured.

## Protocol and implementation

The swirl experiments retain the original 100,000 training observations,
4,096 validation observations, domains and prescribed correlated noise.
Neither training nor selection calls the known swirl density or its curve.
Selection uses noisy validation; the clean target enters only the later
evaluation and figures. Numerical artifacts live under
`artifacts/direct_optimization`; figures remain in the existing folders.
Final noisy validation uses 97 nodes per coordinate for joint clipping,
with a 129-node check on the displayed signed fits before truth evaluation.
Within the negative-penalty family, fits within one paired standard error
of the best noisy likelihood are compared by negative mass: less removal
means the final density is closer to the function actually optimized.
The strict best-likelihood penalized fit is also independently evaluated.
The overall new-model rule remains paired one-SE, then coefficient count.
Noise-free four-dimensional selection uses 129 nodes per coordinate and
an additional higher-order integration check after fitting.

Modes/ranks are tested at 25/16 and 49/32. The direct density TT has ranks
`(modes, middle_rank, modes)`. Squaring doubles the polynomial degree of an
amplitude, so equal mode counts do not imply equal density approximation
spaces. It can also produce effective density TT ranks up to the square of
the amplitude rank. A direct signed density with middle rank 32 therefore
has a different approximation class from a squared amplitude of rank 32;
the signed rank-64 control does not fully match that squared density class.
Timing and parameter counts are recorded rather than treating these
different algorithms as an exactly matched computational comparison.

Noise-free controls additionally separate mode and rank increases: the
squared TT uses 25/16, 49/32, 65/32 and 49/64, plus a second seed at 49/32.
All use 4,000 Adam updates with batch 1,024. A 49/32 fit is continued for
100 full-data L-BFGS updates. Signed fits use 30 alternating sweeps at
25/16, 49/32 and 49/64. Curvature strengths are varied using held-out observations.
Across noise-free methods, the comparison criterion is the held-out
least-squares risk of the final normalized density. Zero clipped
likelihoods are reported as such, never hidden with a probability floor.
The saved noisy controls continue separately, as requested.
The historical noise-free gallery fits used **no curvature penalty**;
`swirl_fits.py` applies that penalty only in its noisy branch. The new
49/32 controls include strength zero, `1e-5`, `1e-4`, and `1e-3`, with
full-data refinement at the latter two strengths. This separates a useful
regularization comparison from the positivity question. Changes against
the historical baseline combine initialization, training budget and
regularization; they should not all be credited to the initializer.
The second seed changes both the initial perturbation and minibatch draws;
it is a reproducibility check, not an isolated initialization experiment.
A noise-free 10D control uses 49 modes and ranks 32/64, 4,000 Adam updates with
batch 256 and 40 full-data L-BFGS updates at rank 32. These controls check the existence
of artifacts without convolution; the noisy/noise-free studies are not a
matched causal estimate of the effect of adding noise.

Relevant entry points:

- `ftde.init_tt_constant`: general-dimensional elementary amplitude start.
- `scripts/fit_direct_optimization.py`: squared-NLL and signed-loss controls.
- `ftde.full_batch_optimization.fit_noisy_tt_lbfgs`: full-data likelihood refinement.
- `scripts/refine_direct_lbfgs.py`: continuation of a directly trained TT.
- `scripts/score_direct_optimization.py`: measure final clipping and select
  on noisy validation, before any clean-target evaluation.
- `scripts/run_direct_noise_free.py`: noise-free capacity and optimizer controls.
- `scripts/evaluate_direct_noise_free.py`: separate `score` and `evaluate`
  stages, with model hashes frozen before any use of the known density.

For ordinary squared-TT learning on your own observations and prescribed
noise TT, the explicit initialization/training separation is:

```python
from ftde import (
    AmplitudeSobolevPenalty, TTConvolution,
    fit_noisy_tt, fit_noisy_tt_lbfgs, init_tt_constant,
)

start = init_tt_constant(domain, rank=32, modes=49, seed=0)
penalty = AmplitudeSobolevPenalty.from_tt(
    start, order=2, length_scales=noise_standard_deviations,
)
forward = TTConvolution.from_tt(start, noise_tt).prepare()
trained, history = fit_noisy_tt(
    start, noise_tt, train, forward_operator=forward,
    regularizer=penalty, reg_strength=1e-4,
    n_steps=4000, lr=1e-3, batch_size=256,
)
refined, last, validation_logs, diagnostics = fit_noisy_tt_lbfgs(
    trained, train, valid, noise_tt, penalty, strength=1e-4,
)
```

`domain`, `noise_tt`, the noise scales and the two observation splits are
inputs. The benchmark script additionally uses Sobolev coordinates during
Adam and retains its best regularized-validation checkpoint. The lower-level
`fit_noisy_tt` call above returns its last iterate, so it is a usage example,
not an exact reproduction of the benchmark's checkpoint policy.

Example commands in the `chebdGE` environment and an existing GPU allocation:

```bash
export JAX_ENABLE_X64=1
srun python scripts/fit_direct_optimization.py --method squared --strengths .0001
srun python scripts/fit_direct_optimization.py --method signed
srun python scripts/fit_direct_optimization.py --negative-strength 1 --strengths .00001 .0001 --sweeps 40
srun python scripts/refine_direct_lbfgs.py
srun python scripts/score_direct_optimization.py
```

## Measured outcomes

The near-uniform start does learn the swirl through TT optimization. The
useful improvement so far is modest and restricted to noise-free 4D:
full-dimensional relative L2 decreases from **0.1022 to 0.0916**, a 10.4%
relative reduction. This does not solve the higher-dimensional or noisy
examples. The selected new squared fits are:

| Case | Historical squared TT | New direct optimization | New independent test NLL |
|---|---:|---:|---:|
| Noise-free 4D | 0.1022 | **0.0916** | 1.2492 |
| Noise-free 10D | 0.2205 ± 0.0016 | 0.2950 ± 0.0022 | −1.6524 |
| Noisy 4D | 0.2162 | 0.3074 | 3.8831 |
| Noisy 10D | 0.4864 ± 0.0042 | 0.6910 ± 0.0051 | 6.3083 |

The error columns are relative L2 of the full joint clean density, not
marginal error. The 10D uncertainties are Monte Carlo standard errors from
independent defensive importance sampling; 4D uses checked tensor
quadrature. Test likelihood uses 8,192 newly drawn observations. It includes
measurement noise only in noisy cases. Historical fits have different
initializers and training budgets; this is not a matched-compute comparison.
These are the historical **squared TT** baselines, not the signal-mixture or
pilot-initialized results from the earlier gallery.
All selected new squared fits use 49 modes and maximum rank 32. Historical
noise-free fits have the same sizes; historical noisy fits have rank 64.

The 4D noise-free selected fit uses 49 modes and maximum rank 32. Adam's
retained checkpoint is update 3,200, followed by retained L-BFGS update 15
(100 were run). Its validation NLL moves from 1.24264 to 1.24184; independent
L2 moves from 0.09934 to 0.09164. The 10D noise-free continuation retains
L-BFGS update 40 and improves L2 from 0.33739 to 0.29503. These are genuine
updates to a TT learned from the elementary start. No separately learned
signal estimator supplies either continuation.
All these L-BFGS runs reached their configured iteration limits; numerical
convergence is not claimed. In noise-free 4D, the full training objective
continues decreasing to 1.2051 at update 100 while validation deteriorates
after the retained early checkpoint. In noise-free 10D, the last checkpoint
is retained and additional optimization may still help.

### What explains the spikes?

**Amplitude sign changes are not the whole explanation.** Every noise-free
4D control has zero sampled raw-TT sign crossings along the true curve.
The noise-free 10D rank-32 models also have zero crossings, yet their joint
density oscillates and is too low on parts of the curve. Squaring can create
troughs when the amplitude crosses zero, as happens in several noisy or
rank-64 fits, but the earlier blanket attribution to these zeros was too
strong. The curve diagnostic is a finite post-fit probe, not a proof of
positivity everywhere.
The 10D marginal plots can look reasonably close while the full joint
density remains inaccurate. Even after rank-32 full-data refinement, the
fitted density is less than half the truth at 39.2% of uniformly sampled
curve-parameter values. This measures points on a diagnostic curve, not a
probability-mass fraction. It suggests remaining misallocation of mass in
the joint curved tube that individual 2D projections can conceal.

**Regularization matters even without noise.** In the 49-mode/rank-32
noise-free controls, changing curvature strength from 0 through `1e-5`,
`1e-4`, and `1e-3` reduces integrated amplitude curvature from 2,244.7 to
48.7, 14.6, and 6.6. Their relative L2 errors are 0.1024, 0.1006, 0.0993,
and 0.0977. The global curvature measure includes regions away from the
curve; it should not be interpreted as a proportional reduction in visible
ridge ripples. Stronger smoothing plus L-BFGS gives 0.0946, close to the
weakly regularized selected fit, with lower curvature. It is a useful
control, but the frozen validation rule selects the 0.0916 fit. The old
noise-free gallery did not apply a curvature penalty.

**More modes or rank are not an automatic fix.** At the same 4,000-update
noise-free 4D budget and `1e-4` strength, 25 modes/rank 16 gives L2 0.2512;
49/32 gives 0.0993; 65/32 gives 0.1148; and 49/64 gives 0.1067. The larger
two models attain *lower full-training objectives* (1.2075 and 1.1996,
versus 1.2149 at 49/32), but worse held-out likelihood and density error.
That pattern indicates overfitting at this regularization setting. It is
not evidence that the optimizer simply failed to exploit the larger 4D
class. A second 49/32 seed gives L2 0.0971.

In noise-free 10D, the rank-64 run instead has a **worse training objective**
than rank 32 (−1.6021 versus −1.6917), despite its larger function class.
The rank-32 full-data continuation improves that objective to −1.7454.
This is evidence of unfinished optimization at the tested budget. It does
not establish whether more iterations, a better TT optimizer, or a different
regularizer would close the remaining gap. It also explains why increasing
rank alone is not a reliable recommendation.

### Training signed values, clipping only at the end

The user's proposed control was implemented directly. The signed-loss
gradients never pass through a clip, and the accepted core updates decrease
the objective while preserving signed mass one. Nevertheless, its selected
noise-free 4D result has L2 **0.1823**, removes negative mass **0.1461**, and
assigns zero probability to **107 of 8,192** fresh test samples after final
truncation. Its test NLL is therefore infinite. Allowing negative values
solves the dead-gradient issue but does not by itself make the final clipped
density accurate. The signed and squared objectives and approximation
spaces differ, so this is not an isolated causal comparison of positivity
maps.
The final normalization divides every positive value by 1.1461, a **12.7%
reduction**, even along the sampled curve where all raw TT values are
already positive. Its raw values follow the curve better before this global
rescaling, but they do not define a probability density because the function
is negative elsewhere. The [clipping diagnostic](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_free_dim4_clipping.png)
shows this change explicitly; curve values alone are not a joint-density
error metric.

The negative-part penalty keeps a gradient at negative values. In noisy
4D it reduces removed mass from **0.535** for the selected unconstrained
signed fit to **0.0041** at penalty strength 100, improving L2 from **0.747
to 0.303**. This remains worse than the historical 0.216 fit and is similar
to the new squared fit. In noise-free 4D the tested penalty did not rescue
the signed approach (L2 0.183). The signed solver remains an explicit 4D
experiment, not a general replacement for the likelihood optimizer.

### Additional spatial smoothing

Following the user's suggestion, `ftde.heat_smooth_tt` adds an optional
linear spatial filter to the already optimized signed TT. For each
coordinate, it evolves the Chebyshev coefficients by the Galerkin heat
equation `G du/dt = -K u`, where `G` is the physical mass matrix and `K`
integrates products of first derivatives. The diffusion time is
`sigma² / 2`. Natural no-flux boundaries conserve signed mass inside the
original box. The resulting function has the same Chebyshev modes and TT
ranks; no additional density estimator is learned. This differs from
free-space Gaussian convolution near the box boundaries, and a finite
Galerkin filter is not a positivity certificate. Residual negatives are
still truncated only at the end of this control.

Eight scales, from zero to 16% of each coordinate's box width, were fixed
before evaluating the new grid. The input is the observation-selected
unconstrained signed 4D TT. Held-out least-squares risk selects **zero
additional smoothing**. The smallest positive amount is statistically
indistinguishable: its excess validation risk is `4.3e-6`, with paired SE
`7.8e-5`. Higher-order quadrature preserves this ordering. No setting is
reselected using clean-density error.

| Smoothing sigma / box width | Negative mass | Full 4D relative L2 | Zero test predictions |
|---|---:|---:|---:|
| 0 | 0.1460 | 0.18230 | 89 |
| 0.0025 | 0.1407 | 0.18224 | 83 |
| 0.01 | 0.0940 | 0.21007 | 36 |
| 0.04 | 0.00672 | 0.69438 | 0 |
| 0.16 | 0 at evaluated quadrature | 0.98587 | 0 |

The test counts use a **new** independent set of 8,192 observations
(seed 61492), so the baseline count differs from the earlier 107. All
eight frozen controls are evaluated for the diagnostic figure. Strong
smoothing removes most negative mass but also broadens the narrow tube
enough to lose its density structure. The tiny L2 change at 0.0025 is not
a meaningful improvement and does not change the validation choice.
The option remains available, with three tests covering mass conservation,
known cosine-mode decay and composition of diffusion steps.

[Spatial-smoothing tradeoff figure](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_free_dim4_smoothing.png).
Reproduce with `scripts/run_direct_smoothing.py` followed by
`scripts/plot_direct_smoothing.py`. Original learned TT checkpoints remain
unchanged. The cryo-EM transfer uses direct likelihood training and curvature
regularization; this unsuccessful signed-TT postprocessor does not replace it.

### Figures and checks

- [Noise-free 4D marginals](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_free_dim4_marg_23.png),
  [training and density profiles](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_free_dim4_training.png),
  [regularization and capacity controls](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_free_dim4_controls.png).
- [Noise-free 10D marginals](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_free_dim10_marg_23.png)
  and [training and gaps along the curve](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_free_dim10_training.png).
- [Noisy 4D marginals](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_dim4_marg_23.png),
  [training and negative mass](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_dim4_training.png),
  and [noisy 10D marginals](../artifacts/gallery_archive/before_noise_free_v2/swirl/direct_optimization_dim10_marg_23.png).

All comparison figures have a companion coordinate 0–1 marginal in the
same folder. Previous figures are preserved. JSON, arrays, checkpoints and
logs are stored only under `artifacts/direct_optimization`.

Eighteen focused tests pass, covering the elementary initializer, full-data
objective chunking and refinement, signed objective and constrained solves, linear
convolution, Sobolev regularization, and exact TT sampling. The independent
noise-kernel checks reach relative discrepancies below `7e-11`; signed-loss
Gram matrices agree across quadrature orders to `2e-13` relative Frobenius
error. Increasing final clipping quadrature changes displayed 4D recovery
errors by at most `1.6e-5`. Frozen-model hashes tie each reported number to
the checkpoint selected before truth evaluation.

## Cryo-EM transfer protocol

Because noise-free 4D improved, the same direct-training controls are rerun
on all four saved cryo-EM cases: 4D, the 4D path projection, 10D and 20D.
Each uses its original 94,904 training and 4,096 validation observations,
domain and frozen measurement-noise TT. Noise is independently checked
against its saved Gaussian-mixture specification. No state labels or clean
oracle density are accessed.

Each new TT starts near uniform at 49 modes/rank 32. Two fits use 4,000
Adam steps, batch 256, learning rate 0.001, and curvature strengths `1e-4`
and `1e-3`. Paired one-standard-error validation with lower physical
amplitude curvature as the tie-break chooses an Adam fit; it then receives
40 full-data L-BFGS steps. The same rule compares the trained candidates
with the historical reference. These are reused **development-validation**
observations, not a fresh independent cryo-EM test set. Historical fits
use maximum ranks 32 in 4D, 64 in 10D and 48 in 20D, all with 49 modes,
and different training budgets. Neither a likelihood gain nor a smoother
picture establishes improved accuracy of the unknown clean joint density.

To reproduce the transfer using the existing data artifacts:

```bash
srun python scripts/run_direct_cryoem.py
python scripts/plot_direct_cryoem.py
```

<!-- direct-cryo-transfer:start -->
### Cryo-EM measurements

None of the new fits beats its historical reference under the fixed development-validation rule. The historical references remain unchanged.

| Case | Historical validation NLL | Selected new validation NLL | Retained optimization checkpoint |
|---|---:|---:|---|
| 4D | 3.5315 | 3.5348 | Adam 4000 + L-BFGS 35 |
| 4D path projection | 3.5392 | 3.5411 | Adam 4000 + L-BFGS 40 |
| 10D | 9.8573 | 9.9078 | Adam 4000 + L-BFGS 40 |
| 20D | 24.6566 | 24.7612 | Adam 4000 + L-BFGS 40 |

Every new fit learns from a near-uniform Chebyshev TT. Full-data refinement continues that trained TT; it does not import a separately learned signal density. The figures retain both curvature strengths so their visual and likelihood tradeoff is visible. These are development-validation comparisons, not claims about the unknown clean-density error.

- [4D comparison](../artifacts/gallery_archive/before_noise_free_v2/cryoem/direct_optimization_dim4.png).
- [4D path projection comparison](../artifacts/gallery_archive/before_noise_free_v2/cryoem/direct_optimization_path_dim4.png).
- [10D comparison](../artifacts/gallery_archive/before_noise_free_v2/cryoem/direct_optimization_dim10.png).
- [20D comparison](../artifacts/gallery_archive/before_noise_free_v2/cryoem/direct_optimization_dim20.png).
- [All cryo-EM training trajectories](../artifacts/gallery_archive/before_noise_free_v2/cryoem/direct_optimization_training.png).

The source data and prescribed noise are unchanged. The final audit checks all original 121 PNGs, the frozen model hashes, the numerical checks, and the 19 new figures. Eighteen core tests and three spatial-filter tests pass.
<!-- direct-cryo-transfer:end -->

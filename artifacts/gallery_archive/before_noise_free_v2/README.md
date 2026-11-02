# Density-estimation figures

**Current work: direct TT optimization from a near-uniform start.** All new
structure is learned by optimizing Chebyshev TT coefficients; there is no
KDE or fitted signal-mixture initializer. Noise-free 4D relative L2 improves
modestly from **0.1022 to 0.0916** using curvature regularization and full-data
L-BFGS after Adam. Larger rank/mode counts alone do not improve recovery.
The noisy and 10D cases remain worse than the historical squared fits.
Training a signed TT and clipping only at the end avoids dead gradients,
but its selected noise-free 4D error is **0.1823**, with zero test likelihoods
at some observations. See the [measurements and diagnosis](../../../docs/direct_optimization.md).

| Direct optimization case | Marginals | Diagnostics |
|---|---|---|
| Noise-free swirl, 4D | [spiral](swirl/direct_optimization_free_dim4_marg_01.png), [pretzel](swirl/direct_optimization_free_dim4_marg_23.png) | [training and profiles](swirl/direct_optimization_free_dim4_training.png), [regularization and capacity](swirl/direct_optimization_free_dim4_controls.png) |
| Noise-free swirl, 10D | [spiral](swirl/direct_optimization_free_dim10_marg_01.png), [pretzel](swirl/direct_optimization_free_dim10_marg_23.png) | [training and gaps](swirl/direct_optimization_free_dim10_training.png) |
| Noisy swirl, 4D | [spiral](swirl/direct_optimization_dim4_marg_01.png), [pretzel](swirl/direct_optimization_dim4_marg_23.png) | [training and negative mass](swirl/direct_optimization_dim4_training.png) |
| Noisy swirl, 10D | [spiral](swirl/direct_optimization_dim10_marg_01.png), [pretzel](swirl/direct_optimization_dim10_marg_23.png) | Full-dimensional independent error is shown in titles |

The [final-clipping diagnostic](swirl/direct_optimization_free_dim4_clipping.png)
shows how removing negative mass forces a global rescaling, reducing density
even where the optimized TT was already positive.
The [spatial-smoothing experiment](swirl/direct_optimization_free_dim4_smoothing.png)
also tests eight heat-filter strengths on that fixed signed TT. Very light
smoothing is nearly tied with no smoothing; stronger smoothing reduces
negative mass but worsens recovery by broadening the curved density.

<!-- direct-cryo-transfer:start -->
**Direct-training cryo-EM reruns:** None of the new fits beats its historical reference under the fixed development-validation rule. The historical references remain unchanged.

| Case | Historical validation NLL | Selected new validation NLL | Comparison figure |
|---|---:|---:|---|
| 4D | 3.5315 | 3.5348 | [Marginals](cryoem/direct_optimization_dim4.png) |
| 4D path projection | 3.5392 | 3.5411 | [Marginals](cryoem/direct_optimization_path_dim4.png) |
| 10D | 9.8573 | 9.9078 | [Marginals](cryoem/direct_optimization_dim10.png) |
| 20D | 24.6566 | 24.7612 | [Marginals](cryoem/direct_optimization_dim20.png) |

[Training trajectories](cryoem/direct_optimization_training.png) show the near-uniform starting point and subsequent TT updates. These comparisons reuse development validation; the clean cryo-EM density is unknown.
<!-- direct-cryo-transfer:end -->

The earlier studies below remain for comparison. In particular, the old
mixture and pilot results are different methods; they are not the new
direct-optimization results. All previous PNGs are preserved.

**Historical initialization study:** eight recipes × eight cases × three seeds, keeping
the normalized squared Chebyshev TT (rank 32, 49 modes). The general-purpose
choice by aggregate noisy-validation ranks is a noise-corrected Gaussian
with guided TT conversion; ordinary Gaussian is close. Neither wins on
every case. The swirl-specific tangent pilot lowers median relative L2
from **0.330 to 0.284 in 4D** and **0.696 to 0.368 in 10D**. Those tangent
fits retain their initial TT checkpoints. See the
[protocol and interpretation](../../../docs/initialization_study.md), including
the cases where KDE remains better and the limitations of likelihood-based
selection. These controlled, shorter runs are separate from the older
gallery fits below; all previous PNGs are retained.

- [Recovery errors and seed ranges](convergence/initialization_recovery.png)
- [Training trajectories](convergence/initialization_training.png)
- [Independent noisy-test likelihood](convergence/initialization_test_likelihood.png)
- [Construction cost and pilot-to-TT conversion loss](convergence/initialization_construction.png)
- [Longer-training controls](convergence/initialization_long_controls.png)
- [Initial-score and short-training selection](convergence/initialization_screening.png)

Marginal comparisons (the seed and checkpoint are chosen using noisy
validation; truth is shown only for final evaluation):

| Case | Figures |
|---|---|
| Correlated Gaussian, 4D | [coordinates 0–1](convergence/initialization_gaussian4_marg_01.png) |
| Multimodal, 4D | [coordinates 0–1](convergence/initialization_multimodal4_marg_01.png) |
| Banana, 4D | [coordinates 0–1](convergence/initialization_banana4_marg_01.png) |
| Ring, 4D | [coordinates 0–1](convergence/initialization_ring4_marg_01.png) |
| GL, 6D | [coordinates 0–1](ginzburg_landau/initialization_gl6_marg_01.png) |
| GL, 10D | [coordinates 0–1](ginzburg_landau/initialization_gl10_marg_01.png) |
| Swirl, 4D | [spiral](swirl/initialization_swirl4_marg_01.png), [pretzel](swirl/initialization_swirl4_marg_23.png) |
| Swirl, 10D | [spiral](swirl/initialization_swirl10_marg_01.png), [pretzel](swirl/initialization_swirl10_marg_23.png) |

The same swirl-selected initialization is tested on cryo-EM, with separate
comparison panels for [4D](cryoem/initialization_study_dim4.png),
[4D path](cryoem/initialization_study_path_dim4.png),
[10D](cryoem/initialization_study_dim10.png), and
[20D](cryoem/initialization_study_dim20.png). Their captions identify which
fit is selected. These use the historical development validation split;
they do not report clean-density errors. The 10D/20D comparisons include
longer continuations of the new candidates.
The selection rule retains the new 4D-path fit for lower roughness at
comparable likelihood and the new 20D fit for a modest likelihood gain;
the 4D and 10D cases retain their previous fits.

**Positive-core Chebyshev TT:** the new [4D](swirl/noisy_mixture_positive_dim4.png)
and [10D](swirl/noisy_mixture_positive_dim10.png) comparisons test a directly
fitted, normalized density TT with positive polynomial cores. The Chebyshev
basis and TT convolution are retained. Positivity checks pass, but recovery
is worse: relative L2 is **0.485 / 0.807**, versus **0.216 / 0.488** for the
original squared fits. This remains an optional experiment; existing models
and PNGs are preserved, and it was not applied to cryo-EM.
See the [4D](swirl/noisy_mixture_positive_dim4_training.png) and
[10D](swirl/noisy_mixture_positive_dim10_training.png) likelihood checks,
[4D](swirl/noisy_mixture_positive_dim4_ridges.png) and
[10D](swirl/noisy_mixture_positive_dim10_ridges.png) ridge profiles, and
[4D](swirl/noisy_mixture_positive_capacity_dim4.png) and
[10D](swirl/noisy_mixture_positive_capacity_dim10.png) degree controls.
[Implementation and full results](../../../docs/positive_chebyshev_tt.md) explain
the positivity constraints and the remaining limitations.

**Positivity research:** [map and gradient illustration](swirl/positivity_options.png)
compares squaring, clipping, softplus, and a coefficient barrier. It is an
explanatory figure, not a fitted-density result. The
[literature review and proposed positive-core TT algorithm](../../../docs/positivity_options.md)
describe a way to retain efficient normalization and convolution.

**Direct clipped-TT experiment:** the new
[density comparison](swirl/noisy_mixture_clipped_dim4.png),
[training and test likelihood](swirl/noisy_mixture_clipped_training_dim4.png),
and [ridge profiles](swirl/noisy_mixture_clipped_ridges_dim4.png) test
`max(f_TT, 0)` inside the normalized noisy likelihood. These are 4D reference
experiments with no learned signal mixture; existing figures remain intact.
See [implementation, limitations, and results](../../../docs/clipped_tt_experiment.md).

**September 26 update:** curved-density experiments are in the existing swirl
and cryo-EM folders. See [findings and reproduction](../../../docs/swirl_improvement.md).
Figures explicitly distinguish squared-TT fits from the new local-mixture
alternative. Earlier images are archived under
`artifacts/swirl_improvement/previous_figs`; numerical artifacts remain outside
this folder. GL figures are unchanged.

The local-mixture swirl errors are 0.052 / 0.069 / 0.092 without noise and
0.260 / 0.282 / 0.398 with noise in 4D / 10D / 20D. The noisy 4D result is
worse than the old TT's 0.216 despite slightly better noisy likelihood.
A positive local pilot converted back into the **same rank-64, 49-mode TT**
reduces the 10D noisy error from 0.488 to about 0.222. The initialization
comparison figures show this separately from the mixture alternative.

Cryo-EM local-mixture reruns improve 10D/20D noisy NLL to 9.8071/24.5547;
4D/path-4D retain their earlier recoveries. These are prediction improvements,
not established clean-density improvements; agreement with the estimated
label oracle worsens in several projections. Training, oracle, and new
`cryoem_local_comparison_*` plots document that distinction.

The historical recipe descriptions below document the preceding squared-TT
gallery. The swirl table and current figure titles identify the newer models
and their measured errors.

GL and cryo-EM refreshes are complete. All 48 GL and cryo-EM fit plots include measured fitting wall times in their footers.
The [swirl-pretzel benchmark](#swirl-pretzel) and the [cryo-EM label oracle](#cryo-em-label-oracle) were added later; see
[the benchmark notes](../../../docs/swirl_benchmark.md).

GL panels all use `[-2.5, 2.5]` on both axes. Histograms show training data.
The noisy ft-de panel shows the recovered underlying density, before reconvolution, with the same color scale as the true GL marginal.
Every GL title reports the unsquared, full-dimensional relative L2 error. Settings appear in small text below each figure.

[Run settings and reproduction](../../../docs/figure_refresh.md). Previous figures are preserved outside this folder in [the run archive](../../../artifacts/figure_refresh/previous_figs/).

## Ginzburg–Landau

Relative L2 errors; each entry links to the coordinate 2–3 marginal. Other coordinate pairs are alongside it in the same folder.

| Dimensions | Noise-free | Paired Gaussian | Gaussian mixture |
|---|---:|---:|---:|
| 6 | [0.0812](ginzburg_landau/noise_free_dim6_marg_23.png) | [0.0950](ginzburg_landau/noisy_paired_scale1p5_dim6_marg_23.png) | [0.1314](ginzburg_landau/noisy_mixture_scale1p5_dim6_marg_23.png) |
| 10 | [0.1093](ginzburg_landau/noise_free_dim10_marg_23.png) | [0.1411](ginzburg_landau/noisy_paired_scale1p5_dim10_marg_23.png) | [0.1360](ginzburg_landau/noisy_mixture_scale1p5_dim10_marg_23.png) |
| 14 | [0.1319](ginzburg_landau/noise_free_dim14_marg_23.png) | [0.2307](ginzburg_landau/noisy_paired_scale1p5_dim14_marg_23.png) | [0.1688](ginzburg_landau/noisy_mixture_scale1p5_dim14_marg_23.png) |
| 20 | [0.1611](ginzburg_landau/noise_free_dim20_marg_23.png) | [0.2440](ginzburg_landau/noisy_paired_scale1p5_dim20_marg_23.png) | [0.2359](ginzburg_landau/noisy_mixture_scale1p5_dim20_marg_23.png) |

## Cryo-EM

All marginal panels include the supplied noise kernel. See the
[stability experiments](../../../docs/cryoem_stability.md) for the current higher-rank,
stronger-smoothing fits and the resampled-density averaging comparison, the
[continuity and unsquared-fit notes](../../../docs/cryoem_continuity.md) for the earlier
rank, optimizer, and final-clipping experiments, and the
[earlier refinement notes](../../../docs/cryoem_refinement.md) for noise-model comparisons.

The current fits use maximum amplitude ranks 32 (4D), 64 (10D), and 48 (20D),
49 Chebyshev modes, the previous noise mixture built from supplied
uncertainties, and curvature regularization. The latest fixed-kernel
continuations retain NLL 3.531 (4D), and improve noisy validation NLL from
9.866 to 9.857 (10D) and 24.688 to 24.657 (20D). The selected 10D/20D fits
use H² strength 3e-4 and Sobolev preconditioning. These are likelihood
improvements, not clean-density L2 errors. Direct unsquared fits with joint
clipping were tested but did not improve validation likelihood. Averaging
three resampled 10D fits helped their predictions but did not beat the
selected single TT fit, so the gallery retains that single-density model.
Fine-scale structure remains uncertain, particularly in the higher-dimensional
reconstructions. Timings distinguish the chosen run, recorded candidate
search and naive baseline; preparation and plotting exclusions are labeled.

- [cryoem_dim10_marg_01](cryoem/cryoem_dim10_marg_01.png)
- [cryoem_dim10_marg_23](cryoem/cryoem_dim10_marg_23.png)
- [cryoem_dim10_marg_89](cryoem/cryoem_dim10_marg_89.png)
- [cryoem_dim20_marg_01](cryoem/cryoem_dim20_marg_01.png)
- [cryoem_dim20_marg_019](cryoem/cryoem_dim20_marg_019.png)
- [cryoem_dim20_marg_23](cryoem/cryoem_dim20_marg_23.png)
- [cryoem_dim20_marg_89](cryoem/cryoem_dim20_marg_89.png)
- [cryoem_dim4_marg_01](cryoem/cryoem_dim4_marg_01.png)
- [cryoem_dim4_marg_23](cryoem/cryoem_dim4_marg_23.png)
- [cryoem_path_dim10](cryoem/cryoem_path_dim10.png)
- [cryoem_path_dim4](cryoem/cryoem_path_dim4.png)
- [cryoem_path_training_dim4](cryoem/cryoem_path_training_dim4.png)
- [cryoem_training_dim10](cryoem/cryoem_training_dim10.png)
- [cryoem_training_dim20](cryoem/cryoem_training_dim20.png)
- [cryoem_training_dim4](cryoem/cryoem_training_dim4.png)

### Cryo-EM label oracle

`cryoem_oracle_dim{4,10,20}_marg_*.png` compare the deconvolved ft-de with a clean density built from the
simulation's 50 state labels (used only after fitting): data histogram, oracle ⊛ noise, discrete-state oracle,
continuous-path counterfactual, deconvolved ft-de. In the 10D and 20D embeddings the true clean density is a string of
separated beads (about 2.1 noise σ between states, clean state σ about 0.4 noise σ). See
[the benchmark notes](../../../docs/swirl_benchmark.md#4-the-cryo-em-label-oracle).

## Swirl-pretzel

A continuous tube around one curved path (spiral on (0,1), trefoil "pretzel" on (2,3)), with a sum of three
Gaussian noise covariances. Same panel layout as GL. `*_candidates_dim*.png` shows every searched candidate, with
density profiles along the true ridge. Relative L2 is full-dimensional; [details](../../../docs/swirl_benchmark.md).

| Dimensions | Noise-free | Noisy (Gaussian mixture) |
|---|---:|---:|
| 4 | [0.052, local mixture](swirl/noise_free_dim4_marg_23.png) | [0.260, local mixture](swirl/noisy_mixture_dim4_marg_23.png) (old TT: 0.216) |
| 10 | [0.069, local mixture](swirl/noise_free_dim10_marg_23.png) | [≈0.222, positive-pilot TT](swirl/noisy_mixture_dim10_marg_23.png) (old TT: 0.488) |
| 20 | [0.092, local mixture](swirl/noise_free_dim20_marg_23.png) | [≈0.307, positive-pilot TT](swirl/noisy_mixture_dim20_marg_23.png) (old TT: 0.866) |

Tensor L2 errors are importance estimates with Monte Carlo SE ≈0.001/0.003
in 10D/20D. Local-mixture L2 uses analytic Gaussian overlaps. See the
[initialization comparison](swirl/noisy_mixture_initialization_dim10.png)
and [amplitude-zero diagnostic](swirl/noisy_mixture_diagnosis_dim10.png).

## Sample-size convergence

[convergence.png](convergence/convergence.png): full-dimensional relative L2 against N = 1,000 to 300,000 for GL (10D,
noise-free and mixture noise, 2 training seeds) and the swirl-pretzel (4D, noise-free and noisy, 1 seed, rank-32
continuations only). The recipes are unchanged across N. The dotted KDE baseline is fitted to the same observations
(the noisy ones in noisy cases, so it is not deconvolved). Produced by `scripts/run_convergence_sweep.py`, then
`scripts/convergence_summary.py plot`; `scripts/convergence_table.py` writes the note's table.

| | 1,000 | 3,000 | 10,000 | 30,000 | 100,000 | 300,000 |
|---|---:|---:|---:|---:|---:|---:|
| GL noise-free | 1.633 | 1.345 | 0.442 | 0.180 | 0.111 | 0.071 |
| GL mixture noise | 0.746 | 0.528 | 0.258 | 0.214 | 0.138 | 0.130 |
| swirl noise-free | 0.490 | 0.314 | 0.276 | 0.178 | 0.102 | 0.089 |
| swirl noisy | 0.904 | 0.630 | 0.407 | 0.281 | 0.250 | 0.188 |

At N ≤ 3,000 the unregularized noise-free GL recipe overfits (error above 1).

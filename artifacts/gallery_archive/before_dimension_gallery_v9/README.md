# Current Chebyshev TT density figures

NLL remains the default objective. This gallery also preserves the completed
MMD/hybrid comparison; those experimental fits are labeled in the figures.

One normalized squared Chebyshev functional TT is fitted from a nearly
uniform, slightly perturbed start. All density structure comes from
coefficient optimization. There is no learned signal pilot, KDE initializer,
or mixture of signal estimators. The Gaussian mixtures below are **prescribed
measurement noise**.

## Optimizer smoothing and coarse-to-fine training (latest)

The 4D Swirl and 6D GL comparison separates the loss penalty from the
optimizer's Sobolev whitening, then tests staged polynomial-mode activation
alone and in combination. Ten new fits and two reused controls share a
weak loss penalty; the best new recipe by validation NLL is repeated
alongside a matched baseline with a second initialization seed.

Coarse-to-fine training substantially improves GL: full-density relative
L2 falls from 0.499 for the previous smooth choice to 0.185, selected by
the predeclared smoothness-within-one-SE rule. It also improves Swirl's
weak-penalty fresh fits in both seeds (about 1.14–1.18 to 0.34–0.35),
although the previous marginal-penalty Swirl fit remains better at 0.280.
Stronger whitening alone does not reliably help. GL's successful
curriculum has only been tested at initialization seed zero so far.

- [Validation, roughness and mode activation](convergence/training_geometry_training.png).
- [Post-selection errors and noisy test likelihood](convergence/training_geometry.png).
- **Factorial 1D marginals:** [Swirl](swirl/training_geometry_1d.png), [GL](ginzburg_landau/training_geometry_1d.png).
- **Pair marginals:** [Swirl](swirl/training_geometry_2d.png), [GL](ginzburg_landau/training_geometry_2d.png).
- **New choices versus previous smooth choice:** [Swirl](swirl/training_geometry_recovery_1d.png), [GL](ginzburg_landau/training_geometry_recovery_1d.png).
- [Implementation, protocol and results](../../../docs/training_geometry.md).

Selections are frozen before truth evaluation. The factorial baseline is
the previous weak-penalty fresh fit, not the stronger earlier published
recipe. The comparison against that earlier recipe is shown separately.

## Marginal smoothing and smooth random starts

Twenty-eight new noisy fits compare weaker joint regularization, exact
1D marginal-curvature penalties, and smoother random starts. The calibrated
swirl penalty reduces curvature 71% and full-density L2 from 0.300 to 0.280
against a matched continuation without that penalty, at essentially the same
noisy test likelihood. Improvement over the previous published swirl fit is
modest (0.286 to 0.280). That study did not improve GL; its smoothness-based
choice retained the previous control. Smooth random starts do not reliably prevent
roughness from growing during learning.

- **Actual 1D marginals:** [swirl](swirl/noisy_recovery_1d.png), [GL](ginzburg_landau/noisy_recovery_1d.png).
- **Calibrated penalty, pair marginals:** [swirl](swirl/calibrated_marginals_2d.png), [GL](ginzburg_landau/calibrated_marginals_2d.png).
- [Initial regularization grid](convergence/marginal_regularization.png) and [lighter calibrated penalties](convergence/calibrated_marginals.png).
- [Smooth initialization: training trajectories](convergence/smooth_initialization_training.png) and [post-selection errors](convergence/smooth_initialization.png).
- [Detailed regularization results](../../../docs/marginal_regularization.md) and [initialization controls](../../../docs/smooth_initialization.md).

Selections use noisy validation data and are frozen before truth evaluation.
The new comparisons are separate from the earlier galleries below.

## Previous noisy-estimation diagnosis

All new fits use NLL and a single squared Chebyshev TT. The main finding is
that convolution weakens sensitivity to latent detail while the fixed
smoothness penalty continues to penalize it. In the matched Gaussian-noise
sweep, full 4D latent relative L2 rises from 0.107 without noise to 0.740 at
sigma 0.6, even though the convolved pair fits remain close to the data.
Independent convolution checks find numerical errors far too small to
explain this gap. Optimizer changes give modest improvement on the original
noisy swirl: 0.298 to 0.286.

- [Matched noise levels](convergence/noise_strength.png) and [density panels](swirl/noise_strength_marg_01.png).
- [How the fixed penalty can outweigh the likelihood benefit of detail](convergence/noise_regularization_bias.png). This is a post-fit diagnostic; the clean-data comparator never trains or selects a noisy fit.
- [Restarts, Adam continuation and full-data L-BFGS](convergence/noisy_optimization.png).
- [Variation between fitted densities before and after convolution](convergence/noisy_stability.png), using no true latent density.
- [Independent convolution and gradient audit](convergence/convolution_accuracy.png).
- [GL box-width and polynomial-resolution control](ginzburg_landau/noise_box_control.png).

The swirl 4D and GL 6D mixture-noise marginal figures linked below now show
the current **fixed-lambda NLL optimizer comparison**, selected using noisy
validation observations. Their matched baseline is the old NLL-only fit at
lambda 0.001, not the old broader search's gallery winner. The separate
objective-comparison figures and other cases retain their previous results.

[Detailed findings and reproduction](../../../docs/noise_diagnosis.md).
Numerical artifacts are outside this gallery in `artifacts/noise_diagnosis_v4`.

## Objective and capacity study

- Noise-free GL 6D: [density and rank controls](ginzburg_landau/noise_free_dim6_marg_01.png), [second coordinate pair](ginzburg_landau/noise_free_dim6_marg_23.png), [training](ginzburg_landau/noise_free_dim6_training.png), [objectives](ginzburg_landau/objectives_gl6_free.png).
- Noise-free swirl 4D: [density and rank controls](swirl/noise_free_dim4_marg_01.png), [second coordinate pair](swirl/noise_free_dim4_marg_23.png), [training](swirl/noise_free_dim4_training.png), [objectives](swirl/objectives_swirl4_free.png).
- [Matched rank and mode-count comparison](convergence/capacity.png).
- [Training and full-dimensional error overview](convergence/objectives.png).
- [All candidate validation scores](convergence/objective_candidates.png).
- [How the prescribed noise suppresses Fourier information](convergence/noise_information.png).

The GL capacity controls improve with fewer ranks/modes; rank 5 with 25
modes is selected by held-out likelihood. Swirl needs more capacity, and a
likelihood-plus-MMD objective gives a modest improvement in its latent L2
error. Pure MMD training is not a universal improvement. MMD here means a
multiscale Fourier kernel discrepancy, computed directly from the TT.

## Strong-noise experiments

Each figure shows the true latent density, prescribed noise kernel, true
noisy density, observed histogram, and both the latent and reconvolved fits.
The paired-GL and 10D-swirl figures display the earlier separate likelihood
and MMD selections; the GL-mixture and 4D-swirl figures now display the
NLL optimizer comparison described above. Settings and full-dimensional latent errors are printed
on the figures. The noise laws and observation splits are fixed across
compared objectives.

| Case | Coordinates (0,1) | Coordinates (2,3) | Objectives |
| --- | --- | --- | --- |
| GL 6D, correlated Gaussian | [Figure](ginzburg_landau/noisy_gl6_paired_marg_01.png) | [Figure](ginzburg_landau/noisy_gl6_paired_marg_23.png) | [Comparison](ginzburg_landau/objectives_gl6_paired.png) |
| GL 6D, broad mixture | [Figure](ginzburg_landau/noisy_gl6_mixture_marg_01.png) | [Figure](ginzburg_landau/noisy_gl6_mixture_marg_23.png) | [Comparison](ginzburg_landau/objectives_gl6_mixture.png) |
| Swirl 4D, correlated mixture | [Figure](swirl/noisy_swirl4_mixture_marg_01.png) | [Figure](swirl/noisy_swirl4_mixture_marg_23.png) | [Comparison](swirl/objectives_swirl4_mixture.png) |
| Swirl 10D, correlated mixture | [Figure](swirl/noisy_swirl10_mixture_marg_01.png) | [Figure](swirl/noisy_swirl10_mixture_marg_23.png) | [Comparison](swirl/objectives_swirl10_mixture.png) |

Selections are frozen using held-out observations before the true latent
density is evaluated. Close agreement after convolution can still accompany
poor latent recovery. Comparison fits are retained alongside the selected fit;
the known truth is not used to decide which one to publish.

[Methods, results, literature, and reproduction](../../../docs/objective_study.md).
Numerical artifacts are in `artifacts/objective_v3`. Superseded figures are
archived outside this gallery at `artifacts/gallery_archive/before_objective_v3`.

## Latest available figures from the preceding study

These cases were **not rerun** in the objective/capacity study above. They
remain the latest available implementations for their cases, with their
original settings and diagnostics:

- Noise-free swirl 10D: [density](swirl/noise_free_dim10_marg_01.png), [second pair](swirl/noise_free_dim10_marg_23.png), [training](swirl/noise_free_dim10_training.png).
- Cryo-EM: [4D](cryoem/dim4.png), [path 4D](cryoem/path_dim4.png), [10D](cryoem/dim10.png), [20D](cryoem/dim20.png), [training](cryoem/training.png).
- Earlier [optimizer/regularizer comparisons](convergence/noise_free_optimization.png) and [three-seed initialization check](convergence/initialization.png).

See the [preceding study](../../../docs/noise_free_optimization.md) for their
interpretation and limitations. This folder contains only PNG figures and
this index; all checkpoints, numerical records, and logs live outside it.

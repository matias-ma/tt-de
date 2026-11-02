# Does a smoother random start prevent noisy-fit oscillations?

This is a controlled follow-up to the [regularization study](marginal_regularization.md).
The previous initializer was already near uniform with random coefficient
amplitudes decaying as `1/(1+k^2)`. That suppresses high polynomial degrees,
but does not make their derivative contributions negligible: derivatives
of high-degree Chebyshev polynomials can be large, particularly near the
box boundary.

Two new options in `init_tt_constant` retain the same box, coefficient
basis, TT core sizes, perturbation magnitude and random seed:

- `spectral_decay=2` changes the random coefficient amplitude decay to
  `1/(1+k^2)^2`.
- `spectral_scale=4` multiplies the original envelope by
  `max(exp(-0.5*(k/4)^2), 1e-6)`. The small floor keeps nonzero random
  paths rather than exactly zero-padding a lower-rank initializer.

These options only change initialization. They are not permanent mode
constraints or extra training penalties. Higher modes remain trainable.
Defaults preserve the previous initializer's arithmetic and random draws.
The represented amplitude is normalized in physical L2 after the same QR
canonicalization. The nominal random perturbation multiplier stays fixed, but filtering
also reduces its represented function-space magnitude. Smoothing can
reduce effective numerical rank even when
parameter dimensions are unchanged; this is part of its conditioning effect.


## Completed results

Smoother random initialization did **not reliably prevent oscillations**.
Both cases retain the default initializer under minimum observed validation
NLL. Full-dimensional latent L2, evaluated after selection:

| Case / seed | Default initializer | Faster polynomial decay | Low-pass initializer |
| --- | ---: | ---: | ---: |
| Swirl, seed 0 | 1.17815 | 1.17854 | 1.31862 |
| Swirl, seed 1 | 1.13620 | — | 1.19665 |
| GL, seed 0 | 0.96076 | 1.02180 | 0.86079 |

The GL low-pass run has lower latent L2 than its matched default, but worse
validation likelihood and slightly worse mean 1D marginal error (0.21928
versus 0.21104). It is not selected, and does not beat the older strong-penalty
GL fit (0.49898). Thus the results do not establish that initialization never
matters; they show no reliable repair from these particular smoother starts.

For swirl seed zero, initial amplitude-Hessian energy falls from `1.003e-5`
to `6.196e-11` with faster polynomial decay, yet final energy is about 206
versus 198 for the default. The roughness grows rapidly during optimization;
it is not simply inherited from a visibly oscillatory starting density.
The two seed pairs are limited evidence, not a proof about all local minima.

For swirl, the effective improvement came from the **training path**: first learn under
stronger joint regularization, then relax it, and apply a small marginal
curvature penalty. This remains coefficient optimization of one TT, not a
signal-density pilot initializer. Under the tested budgets, starting directly
with the final weak penalty led to much worse latent recovery.

- [Training likelihood and roughness](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/smooth_initialization_training.png).
- [Swirl 1D marginals](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/smooth_initialization_1d.png).
- [GL 1D marginals](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/smooth_initialization_1d.png).
- [Post-selection error comparison](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/smooth_initialization.png).

## Controlled comparison

For each noisy case, use exactly the same objective, amplitude penalty,
optimizer, 4,000-update budget, batch sequence and validation checkpoints as
the preceding study's `fresh_amplitude` run. That objective was selected
using noisy observations before any truth evaluation. Compare its default
seed-zero initializer with both smoother seed-zero alternatives. In swirl,
also run default and low-pass initializers with seed one, keeping the batch
seed zero for both. GL receives the two seed-zero comparisons.

All six new fits begin from the elementary constant-plus-random TT. No
learned density, clean data, or fitted pilot supplies initialization.
This isolates initialization within each matched comparison; comparisons
to regularization continuations have different optimization histories.

Select minimum noisy validation NLL as the primary choice, and retain the
same secondary lowest-marginal-curvature-within-one-paired-SE choice. GL
uses 200 MCMC-chain clusters. These are development-selection heuristics,
not post-search confidence guarantees. All model, data and configuration
hashes are frozen before latent density evaluation. This study, the preceding regularization grid, and its lighter calibrated
follow-up all finish observation-only fitting before any new latent truth
evaluation begins.

Artifacts: `artifacts/smooth_initialization_v6`. The study reuses the same
single GPU allocation and enforces a 128 MiB artifact budget. The matched
previous fit is called `original` in these records; here it means the
**weak-penalty fresh start**, not the earlier strong-penalty noisy control.

## Verification and reproduction

Five initializer tests check physical normalization, unchanged core shapes,
determinism, nonconstant random starts, invalid settings and a greater-than-
tenfold reduction of the represented initial amplitude's integrated
Hessian energy in the test case. Thirteen additional focused tests cover
the marginal penalty and existing training/marginal calculations.

Using `chebdGE` and the existing one-GPU allocation:

```bash
bash scripts/run_noise_diagnosis_gpu.sh scripts/run_smooth_initialization.py
bash scripts/run_noise_diagnosis_gpu.sh scripts/finish_noisy_followup.py
```

Prepend `srun --jobid=... --ntasks=1 --cpus-per-task=4` on this cluster.
The first command requires the preceding regularization study to be frozen.
The second requires all three follow-up studies frozen, evaluates all fits, and generates
separate regularization and initialization figures. A frozen study is not
modified by another fitting run.

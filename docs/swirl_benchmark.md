# Swirl-pretzel benchmark and the cryo-EM "bead" question

The deconvolved cryo-EM marginals look beaded: bright blobs strung along the
arms of the pretzel. There are two possible explanations. Either ft-de breaks a
continuous ridge into beads, or the clean cryo-EM density really is beaded. This
page tests both: (1) a new synthetic benchmark whose truth is a *continuous*
curved ridge, fitted with the same code and recipe as the current cryo-EM
gallery, and (2) a label-based oracle for the cryo-EM clean density itself.

Figures: [`figs/swirl/`](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/) and `figs/cryoem/cryoem_oracle_*`.

## 1. Benchmark design

`src/ftde/swirl.py`. A latent path parameter `t ∈ [0, 1]` with smooth density
`w(t) ∝ 1 + 0.6 sin(4πt + 0.4)` is embedded as `x = γ(t) + 0.14·ε`, `ε ~ N(0, I_d)`:

- coordinates (0, 1): an Archimedean spiral (1.6 turns), a **swirl**;
- coordinates (2, 3): a trefoil with three self-crossings, a **pretzel**;
- every further coordinate: a slow sinusoid-plus-drift of `t`, so other pairs
  are continuous arcs and loops.

This is the continuous analogue of the cryo-EM latent space (one
conformational path embedded in `d` dimensions), without discrete states.
Its brightness along the ridge is smooth by construction. The only sharp
features in its pair marginals are path crossings and fold caustics, and those
are genuinely present in the truth.

The density, exact pair marginals, `∫p²` and the true noisy observation
density are all path quadratures (`t`-quadrature error ≈ 2e-10). Sampling is
exact and i.i.d.: 100,000 training (seed 0) and 4,096 validation (seed 2)
observations. **The truth is used only in `swirl_figures.py` / `swirl_ripple.py`,
after fitting and selection.**

### Noise: a sum of covariances

A three-component zero-mean Gaussian mixture (`scripts/swirl_fits.py`):

| weight | covariance |
|---:|---|
| 0.5 | 0.22² I (narrow, isotropic) |
| 0.3 | diag(0.45², 0.25², 0.45², …) (anisotropic) |
| 0.2 | 0.5² [[1, ±0.6], [±0.6, 1]] within each adjacent pair, sign alternating (correlated) |

The per-axis noise std is 0.30–0.37, so the clean tube (σ = 0.14) is about 0.4 noise σ
wide. That ratio matches the cryo-EM oracle below (clean state σ ≈ 0.4 noise σ).
The spiral's turns are 2.7–3.3 noise σ apart, and the pretzel's inner lobes
about 2. That's hard enough that neither the histogram nor the KDE of the noisy data
shows the pretzel's inner structure, but not hopeless.

The prescribed kernel TT is built exactly by
`build_paired_gaussian_mixture_noise_kernel_tt` (new, `src/ftde/noise_kernel.py`).
Each component's 2×2 blocks are interpolated on a 513-point tensor Chebyshev grid,
split by SVD, and summed block-diagonally. There is no cross approximation. The
kernel TT has ranks 30 inside pairs and 3 between pairs, and is checked against
the analytic mixture to 1e-9 of its peak (7e-15 observed).

## 2. Fitting recipe (identical code to the gallery)

`scripts/swirl_fits.py`, one case per process:

1. **Base fit** from a KDE-initialized TT (rank 16, 25 modes, 64 starts):
   noise-free uses the GL noise-free recipe (unregularized NLL, 8,000 steps,
   lr 1e-3, batch 2048). Noisy uses the GL warm-up (amplitude H² 3e-5
   through the prescribed TT convolution, 8,000 steps).
2. **Capacity continuations** as in the current cryo-EM gallery
   (`docs/cryoem_stability.md`): `expand_tt` to ranks 32 and 64 with 49
   modes, 3,200 Adam steps, lr 2e-4 → 2e-5. Noisy candidates use amplitude
   H² strengths 1e-4 and 3e-4, Sobolev preconditioning and regularized checkpoints.
3. **Selection**: noise-free takes the minimum validation NLL (GL rule). Noisy
   takes paired noisy one-SE, then minimum common amplitude H² (cryo-EM rule).

The GL Fisher-visibility stage is omitted. The current cryo-EM figures come
from the H²/Sobolev continuations, and those are what this benchmark
reproduces.

Two infrastructure changes were needed:

- `fit_candidate(..., microbatches=m)` accumulates the gradient of the same
  256-row batch over `m` equal chunks with `lax.scan`. The gradient is identical
  (tested), and the rank-64 convolution backward pass no longer needs about 12 GB
  for one tensor. The default `m = 1` keeps every existing caller's code path.
- In 10D and 20D, the thin clean tube makes the LSCV KDE so concentrated that the
  initializer's pivot search collapsed to rank 1. `initialize()` broadens only the
  initializer bandwidth (×2, ×4, then the cryo-EM naive-baseline rule) until all
  ranks exceed one and every observation has finite log density. It needed ×2 in 10D
  and ×4 in 20D, noise-free only. This uses data only, and is recorded in
  `initialization.json` and in the figure footers.

## 3. Results

Relative L2 is the unsquared full-dimensional Lebesgue error
`‖q − p‖ / ‖p‖`. `∫q²` is exact Gauss–Legendre quadrature, `∫pq` is exact
Gauss–Hermite quadrature at each path node (cross-checked against 200k-sample
Monte Carlo), and `∫p²` is a double path quadrature. "Oracle NLL" is the true
(observation) density's NLL on the same validation set. It's a reference that
fitting can approach but never beat in expectation.

| case | selected | relative L2 | NLL gap to truth (nats) | search time |
|---|---|---:|---:|---:|
| noise-free, 4D | rank 32 | **0.102** | 0.033 ± 0.004 | 3 min |
| noise-free, 10D | rank 32 | **0.221** | 0.129 ± 0.007 | 16 min |
| noise-free, 20D | rank 64 | **0.373** | 0.388 ± 0.013 | 33 min |
| noisy, 4D | rank 64, H² 1e-4 | **0.216** | 0.0021 ± 0.0010 | 29 min |
| noisy, 10D | rank 64, H² 1e-4 | **0.488** | 0.0157 ± 0.0027 | 2 h |
| noisy, 20D | rank 32, H² 3e-4 † | **0.866** | 0.160 ± 0.008 | 2 h 23 min |

The NLL gap is the paired mean of `log q_true − log q_fit` over the validation set (noisy observations
in noisy cases). It's a reference that a fit can approach but not beat in expectation. Every candidate's L2 is in its case's `integrated_l2.json`, and all candidates are shown in
`figs/swirl/*_candidates_dim*.png`.

† In 20D the rank-64 noisy continuations were omitted for compute (about 6 h each on the shared GPU):
`swirl_mixture_dim20_r32` selects among the base fit and the two rank-32 fits of `swirl_mixture_dim20`, reused
through symlinks. The rank-32 continuations there also needed 4-way gradient accumulation
(`swirl_fits.microbatches`).


**Ripple** (`scripts/swirl_ripple.py`, `ripple.json`) isolates bead-like
errors. Each pair marginal is evaluated along the projected true path, the
truth profile is subtracted, and the high-pass part of the difference is kept
(Gaussian filter in `t`, width 0.012, about 1.5 tube widths of path). It is
reported relative to the norm of the truth profile. Genuine sharp features of
the truth cancel, and only oscillations that the truth lacks remain.

| case | candidate | ripple (0,1) swirl | ripple (2,3) pretzel |
|---|---|---:|---:|
| noise-free 4D / 10D / 20D | all candidates | 0.005–0.022 | 0.013–0.037 |
| noisy 4D | base (rank 16, 25 modes) | 0.024 | 0.054 |
| noisy 4D | selected rank 64, H² 1e-4 | 0.023 | 0.062 |
| noisy 4D | rank 64, H² 3e-4 | 0.016 | 0.042 |
| noisy 10D | base | 0.061 | 0.141 |
| noisy 10D | selected rank 64, H² 1e-4 | 0.046 | **0.183** |
| noisy 10D | rank 32, H² 1e-3 / 3e-3 / 1e-2 (extra sweep) | 0.027 / 0.019 / 0.008 | 0.133 / 0.121 / 0.097 |
| noisy 20D | selected rank 32, H² 3e-4 | 0.032 | 0.088 |

In the extra 10D sweep (`artifacts/swirl/swirl_mixture_dim10_h2sweep`, same base fit), stronger smoothing
removes the spiral beads but costs accuracy. The NLL gap to the truth grows from 0.030 to 0.036, 0.056 and 0.109
nats, and the relative L2 from 0.55 to 0.68, 0.80 and 0.90. The pretzel's misplaced segments persist and only blur.


The `*_candidates_dim*.png` figures show every searched candidate, with
ridge profiles along the true path.

## 4. The cryo-EM label oracle

`scripts/audit_cryoem_oracle.py`, figures `figs/cryoem/cryoem_oracle_dim*_marg_*.png`.
The simulation assigns every image to one of **50 discrete conformational
states** (counts 20 to about 4,900). For each state, the oracle uses the mean of its
noisy latent points and the PSD part of (within-state covariance − mean supplied
noise covariance), so its clean density is a 50-component Gaussian mixture. A
*continuous-path counterfactual* joins consecutive state means with segments
(interpolated weights and covariances). It's roughly what a smooth pretzel
would look like. Labels enter only here, after all fitting.

| embedding | consecutive-state spacing | clean state σ | within-state σ |
|---|---:|---:|---:|
| 4D | 1.24 noise σ | 0.65 noise σ | 1.19 noise σ |
| 10D | 2.16 noise σ | 0.43 noise σ | 1.08 noise σ |
| 20D | 2.05 noise σ | 0.41 noise σ | 1.07 noise σ |

In 10D and 20D, adjacent states are about 5 clean-state widths apart, so **the
clean density is itself a string of well-separated beads**, not a continuous
pretzel. Convolution with noise (σ ≈ 1 unit on this scale) merges them, which is
why the histogram and KDE look continuous. In the 4D embedding, states are only
about 2 clean widths apart, and the oracle is a lumpy but mostly connected ridge.

Oracle ⊛ noise reproduces the observed ft-de closely: 2D relative L2 0.02–0.14.
This validates the oracle and the supplied noise model. The deconvolved ft-de
reproduces the oracle's *qualitative* pattern in 10D and 20D: beaded arms, a
separate central cluster, and the two isolated blobs on the lower loops. It
does not match individual bead positions, though. Its 2D relative L2 is 0.20–0.46 against
the discrete oracle and 0.16–0.38 against the continuous counterfactual, and
pointwise L2 penalizes misplaced beads heavily. The within-state excess over the
supplied noise could also partly come from noise-model misspecification, in which
case the true states are even sharper.

## 5. Conclusion

**The beads are not simply an artifact of the estimator, and not simply real either.**

1. *Without noise, ft-de never fragments a continuous ridge* (4D, 10D, 20D; ripple ≤ 3.7% for every candidate).
   The TT representation, the Chebyshev basis and the optimizer do not by themselves produce beads.
2. *With cryo-EM-like noise in 4D, deconvolution recovers a continuous swirl and pretzel*, including inner
   pretzel lobes that no noisy histogram or KDE shows. It is within 0.002 nats of the true noisy likelihood, and
   the only residual is faint ripple.
3. *In 10D, deconvolution does fragment the ridges*: beads on the faint outer spiral arm, and boxy,
   axis-aligned pretzel segments with a spurious blob. This is the same visual signature as the cryo-EM fits.
   The artifacts are present in the rank-16 base fit and inherited by every continuation. The truth beats the
   selected fit by 0.016 ± 0.003 nats on *noisy* validation data, so this is an optimization or basin failure
   that the data can detect, not a limit of identifiability. Refitting from three chebdGE initializations
   (`swirl_mixture_dim10_initseed{1,2}`, rank 32, H² 3e-4) gives the same noisy NLL within noise (gaps 0.030, 0.033,
   0.031 ± 0.004) and the same L2 (0.588, 0.588, 0.600), but the artifacts move: a detached blob, streaks, or a broken
   arm. The average of the three normalized densities has the same noisy NLL (Δ = 0.0003 ± 0.0007) but a clearly
   better pretzel marginal (2D relative L2 0.18 against 0.21–0.26). So artifact *locations* are basin-dependent and
   invisible to the noisy likelihood (note figure `init_seeds.png`).
4. Across all candidates, held-out noisy NLL tracks the latent L2 error (see the note's `nll_gap_vs_l2.png`),
   so the selection rule chooses sensibly among what it is offered. Stronger smoothing trades beads for bias.
5. *The cryo-EM clean density is itself beaded* (section 4): 50 discrete states about five clean widths apart in
   the 10D and 20D embeddings. The ft-de beads there are qualitatively expected, but, given point 3, individual
   bead positions should not be trusted unless they recur across independent initializations.


## Reproduction

In the `chebdGE` environment, inside one allocation as prescribed by
`instructions.txt` (for example `salloc --no-shell` then `srun --jobid=…`):

```bash
python scripts/swirl_fits.py --dim 10 --noise mixture --output artifacts/swirl/swirl_mixture_dim10
python scripts/swirl_figures.py artifacts/swirl/swirl_mixture_dim10 --figures figs/swirl
python scripts/swirl_ripple.py artifacts/swirl/swirl_mixture_dim10
python scripts/audit_cryoem_oracle.py extract   # ~10 GB RAM; reads the 900 MB joblib
python scripts/audit_cryoem_oracle.py plot --figures figs/cryoem
```

Cases: `--dim {4,10,20} --noise {free,mixture}`. Completed candidates are
reused, and changed settings are rejected. `tests/test_swirl.py` covers
normalization, histogram agreement, `∫p²`, the paired kernel and microbatch
equivalence.

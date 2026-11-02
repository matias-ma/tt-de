# Cryo-EM continuity and the unsquared-density experiment

The subsequent 10D/20D improvements and current gallery are described in
[Cryo-EM stability](cryoem_stability.md). This page records the preceding stage.

The new gallery improves the continuity of the main cryo-EM curve by increasing
TT rank and improving optimization. It still optimizes Chebyshev coefficients
of a `chebttge.ChebTT`. The published density is its normalized square;
the panels are exact polynomial marginals, with no image smoothing.

All comparisons retain the previous gallery's noisy observations, training and
validation split (94,904 / 4,096), latent support, coordinate standardization,
and noise kernel. No clean density or state labels enter training or selection.
The cryo-EM kernel remains the previously constructed four-component stationary
noise mixture. This is an approximation based on the supplied uncertainty
metadata, not a claim that cryo-EM noise is an arbitrary unknown function.
The convolution operators themselves accept represented TT kernels.

## What helped

| Case | Previous noisy validation NLL | New NLL | Maximum amplitude rank | Selected continuation time |
|---|---:|---:|---:|---:|
| 4D | 3.539891 | 3.531477 | 32 | 45.2 s |
| 10D | 9.978320 | 9.866188 | 48 | 219.4 s |
| 20D | 24.758697 | 24.688191 | 32 | 215.2 s |
| Path 4D | 3.543929 | 3.539250 | 32 | 45.9 s |

NLL is measured in standardized coordinates; lower is better. These are
prediction scores on noisy observations, not clean-density errors. Repeated
development against this validation split means it is not an untouched test set.
The times are continuation fits from the previous selected density on one H100,
including compilation and validation, not end-to-end fits from scratch.
Figure footers separately report recorded candidate search and baseline time.
Search totals include ancestral fits, deduplicated by resolved paths, but
exclude initialization, kernel/KDE/pilot construction, plotting and failed or
discarded exploration. Earlier stages used different GPU models.

Selected fits use 49 amplitude modes per axis and relative amplitude H² penalty
strength 1e-4. The 10D fit uses 3,200 Adam steps, the others 2,400, with batch
256, cosine learning rate 2e-4 to 2e-5 and gradient clipping at 1. Checkpoints
use validation NLL plus their regularizer. Across candidates, selection uses
paired noisy one-standard-error eligibility, then the smallest common physical
amplitude H² penalty. This selects seed 19 for the 10D rank-48 fit.

The rank-20 continuation barely changed 10D NLL (9.972207), while increasing
rank to 32 reached 9.896137. Increasing modes alone from 49 to 81 retained the
initial checkpoint. A physical Schmidt decomposition of the rank-32 amplitude
puts up to 7.48% of its squared norm beyond rank 20 at a central split. Together
these results support a rank bottleneck; they do not establish a unique cause
for every remaining peak or rule out further basis/optimizer improvements.

For the larger 10D fits, the optimizer coordinates use the metric
`Gram + strength * derivative_weight * derivative_Gram` in each core.
This Sobolev preconditioner changes coefficient scaling, not the represented
density or objective. An invariance test checks this distinction. Stronger
smoothing alone was less successful: an H¹ strength of .01 broadened the
curve and worsened validation NLL to 9.999512. The new selected density does
not have lower global H² energy than the old one; more continuous ridges and
lower overall curvature are different properties.

Matched rank-48 runs with seeds 0 and 19 achieved NLL 9.865521 and 9.866188.
Their paired difference, .000667, is below its .001571 standard error.
The (0,1) marginal L1 difference between seeds is .0658, compared with changes
of .1937 and .1989 from the old fit. These runs share the same data and warm
start and vary minibatches and rank-expansion perturbations. They are not
bootstrap or independent-data replications. Several small peaks, particularly
in other projections, persist and are not established as real signal.

## Removing the square and clipping at the end

The proposed nonnegative clipping operation is `max(f, 0)`; `min(f, 0)` keeps
the nonpositive part. `scripts/test_cryoem_unsquared.py` implements a direct,
signed density TT, normalized by its signed integral. The new
`LinearTTConvolution` contracts the signal coefficients linearly with the
same generic TT noise kernel. No square is applied to the optimized density.

Clipping is applied to the **full joint four-dimensional density**, followed
by normalization, marginalization and reconvolution. Clipping a displayed
marginal would not evaluate the proposed model. Full tensor quadrature is
used only for this small-dimensional experiment, not proposed as a scalable
20D method. Its orders 65 and 97 provide a convergence check; clipping makes
the integrand nonpolynomial, so these are numerical approximations.

Six trials compare noisy-KDE and previous-recovery initialization, a
negative-mass penalty, raw-valid checkpoint selection, and final-iterate
selection. For a close starting comparison, the old recovered density was
converted once to a direct rank-64, 97-mode density TT. Physical L2 compression
has relative error bound .0321% before normalization; its initial noisy NLL
is 3.539897 versus 3.539891 for the original. Squaring that initial amplitude
only converts the old density into direct coefficients; the new optimization
is unsquared.

With unconstrained signed optimization, some convolved predictions become
negative. The training objective uses a linear continuation of `-log(q)` below
1e-12 to permit optimization to proceed, but invalid raw validation NLL is
reported as null, never made finite by flooring. Both final-iterate runs save
the final signed model and evaluate its actual clipped version, irrespective
of raw validation validity.

At learning rate 2e-4, the final model has negative-part integral 2.6693 and
positive-part integral 3.6693 despite signed integral 1. At learning rate
2e-6 these are .12057 and 1.12057. The respective raw convolved models have
9 and 16 nonpositive predictions among 4,096 validation observations.
After joint clipping and normalization, all 4,096 predictions are positive.
Full-validation NLLs are **4.719625** and **3.551256**, respectively, compared
with **3.539891** for their starting fit and **3.531477** for the newly selected
squared fit. The conservative unsquared fit's paired deterioration from its
starting fit is .011365 with standard error .001365. These likelihood scores
are computed after reconvolving the clipped joint density. The slower fit's
65-versus-97 quadrature NLL difference over all 4,096 rows is .0000375,
well below the measured deterioration.

Thus final clipping can produce a valid density, but it changes the model
whose convolved likelihood was optimized. Negative mass can improve a signed
model's apparent fit through cancellation; deleting it and renormalizing
changes that fit. These trials did not improve the recovery, and are retained
as experiments rather than replacing the public fit. This is not evidence
that every unsquared, positivity-constrained method must fail. Optimizing
the clipped model's own normalized likelihood would be a different objective
and require a scalable treatment of joint positivity and integration.

## Numerical checks and artifacts

All 13 squared candidates passed the doubled-quadrature convolution check;
the largest recorded log-density difference is 2.061e-12. Sixteen relevant
tests cover exact squared marginals, direct convolution values/gradients,
joint clipping on an analytically integrable signed example, coordinate
invariance, noise representation, and selection. Efficient squared marginals
contract two amplitude bonds without allocating rank-squared density cores.

Artifacts are in `artifacts/cryoem_continuity/`: candidate coefficients,
histories, per-observation likelihoods, `numerical_summary.json`,
`rank_spectra.json`, `unsquared_full_validation_comparison.json`, and
`unsquared_dim4*` diagnostics. Each experiment records
its settings. Squared candidates also record source and kernel hashes.
`dim10_repeat_seed19` used L2 preconditioning, so it is an optimizer/seed
control, not the matched Sobolev replication; the matching pair is explicitly
listed in `numerical_summary.json`.

The previous 15 cryo-EM PNGs are in `previous_figs/`; GL figures are unchanged.
To redraw the selected gallery using the `chebdGE` environment:

```bash
python scripts/refresh_density_figures.py \
  artifacts/cryoem_continuity/gallery_dim4 \
  artifacts/cryoem_continuity/gallery_dim10 \
  artifacts/cryoem_continuity/gallery_dim20 \
  artifacts/cryoem_continuity/gallery_path_dim4 --figures figs
```

To reproduce a selected continuation in a new output directory:

```bash
python scripts/refine_cryoem_continuity.py \
  --base artifacts/cryoem_refinement/gallery_dim10 \
  --output artifacts/cryoem_continuity/reproduction_dim10 \
  --rank 48 --modes 49 --steps 3200 --seed 19 --preconditioning sobolev
```

To reproduce the conservative unsquared final-iterate experiment:

```bash
python scripts/test_cryoem_unsquared.py \
  --base artifacts/cryoem_refinement/gallery_dim4 \
  --initial-amplitude artifacts/cryoem_refinement/gallery_dim4/selected.npz \
  --output artifacts/cryoem_continuity/reproduction_unsquared \
  --rank 64 --steps 3200 --lr 2e-6 --checkpoint-rule final --orders 65 97
```

Run GPU work through the existing one-GPU/four-CPU/32GB `cryoem` allocation,
following `instructions.txt`. The scripts do not request allocations themselves.

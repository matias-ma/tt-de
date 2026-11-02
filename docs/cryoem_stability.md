# Reducing residual cryo-EM beads

This experiment tests two changes while keeping the observation split, latent
box, coordinate normalization and supplied noise representation fixed:

1. Increase amplitude rank while applying stronger function-space curvature
   regularization. This gives the fit more capacity to represent a curved
   ridge while discouraging small ripples.
2. Resample the training observations, continue several TT fits, and average
   their normalized **joint densities**. This can reduce features that vary
   between fits without changing the prescribed noise kernel.

All fits optimize Chebyshev coefficients through `chebttge`. No clean-density
function, state frequencies or path labels enter training or selection.
The resampled fits share the current fitted initialization, so their spread
is a conditional sensitivity diagnostic, not a bootstrap confidence interval.
It can miss artifacts inherited from that initialization or the noise model.
The resampled members have rank 48. Their spread describes that ensemble,
not an uncertainty band for the separately selected rank-64 continuation.

## Rank-64 continuation result

The completed 10D rank-64 continuation improves noisy validation NLL from
9.866188 to 9.857318. The paired difference is -.008870 with standard error
.002615 on the existing validation split. The full-dimensional amplitude H²
penalty falls from 151.644 to 96.263 (36.5%). Its prepared convolution agrees
with doubled quadrature to 1.052e-12 in the recorded log-density check.
The continuation took 723.1 seconds on one A100 80GB, including compilation
and validation, from the previously fitted rank-48 starting point.

Exact marginal density Hessian energies are:

| Projection | Previous | Rank 64 | Reduction |
|---|---:|---:|---:|
| (0,1) | 71.3614 | 41.2637 | 42.2% |
| (2,3) | 85.0808 | 50.1077 | 41.1% |
| (8,9) | 36.8131 | 23.6632 | 35.7% |

These are smoothness diagnostics, not percentages of removed artifacts or
improvements in unknown clean-density error. Marginal entropy increases by
.106, .081 and .097 nats, respectively: the improvement includes some
broadening, not merely removal of oscillations. Isolated concentrations still
remain in the (2,3) projection. The rank-48 control and conditional bootstrap
experiments test whether extra rank and density averaging provide additional
benefit beyond this continuation.

The matched unresampled rank-48 control reaches NLL 9.862306 and amplitude
H² 95.326 after the same 3,200 steps and strength. Thus stronger smoothing
and additional optimization provide much of the curvature reduction; extra
rank provides a further improvement in fitting the noisy observations.
Rank 64 improves NLL over that control by .004988 (paired SE .001182),
with similar amplitude curvature. The existing paired one-SE/common-H²
selection rule chooses rank 64 among the old fit and these two unresampled
continuations. This selection is frozen before inspecting state-label
diagnostics; individual bootstrap realizations are not selected for publication.

## 20D continuation

Increasing the previous amplitude rank from 32 to 48 while using H² strength
3e-4 and Sobolev preconditioning improves noisy validation NLL from 24.688191
to 24.656633. The paired improvement is .031559 with standard error .003511.
Amplitude H² falls from 229.756 to 152.610 (33.6%). This fit is selected over
the previous one by the same paired one-SE rule. It takes 677.9 seconds on
one A100 for 2,400 continuation steps; its doubled-quadrature check differs
by at most 2.068e-12 in log density. Several gaps on the main (0,1) curve fill
in, although the recovered density remains uneven.
Exact marginal Hessian energy drops by 43.7%, 32.6% and 44.4% for (0,1),
(2,3) and (8,9), respectively. No matching rank-32 continuation control was
run in 20D, so this result supports the combined change rather than isolating
the contribution of each part.

## Density averaging

For fitted amplitudes `g_b`, the ensemble is

`p(x) = mean_b [g_b(x)^2 / integral(g_b^2)]`.

This is a nonnegative, normalized joint density. Its convolved prediction and
its marginals are the averages of those quantities for the members. We score
the mixture likelihood with log-sum-exp; averaging NLLs would score a different
object. No core or amplitude averaging is performed, since bond gauges and
amplitude signs should not affect the answer. No image smoothing is applied.

`src/ftde/density_ensemble.py` implements pointwise density, generic convolved
density and pair marginals. An `ensemble.json` manifest records members and
weights, so the result is usable beyond the plotted projections. Tests check
normalization, amplitude-sign invariance, support and convolved predictions
against independent numerical integration. The mixture convolution accepts
the existing generic TT convolution operators; it assumes no Gaussian form.

Density averaging can fill minima whose positions differ across members.
It cannot remove a systematic artifact present in every member, and it may
broaden real features. Therefore agreement on noisy observations and visual
smoothness alone do not certify the recovered latent density.

The three resampled fits reached validation NLLs 9.869044, 9.869640 and
9.871301. Their equal-weight density mixture reaches **9.866220**, improving
over the individual resampled fits but essentially matching the old fit's
9.866188. It remains worse than the selected rank-64 fit by .008902
(paired SE .001660), and is not consistently smoother across the three
checked marginals. It therefore stays an experimental alternative rather
than replacing the selected single TT density.

For projections (0,1), (2,3) and (8,9), the mean member L1 distance from the
mixture is .0526, .0549 and .0268. Pointwise variation is concentrated around
some local peaks and ridge positions. These are sensitivity diagnostics,
not calibrated errors against an unknown clean density. Each member draws
94,904 training rows with replacement, containing 59,980, 59,884 and 59,853
distinct original rows. The three continuation fits cost 1,225.0 seconds
total on the A100. Checkpoint steps are 2,400, 3,200 and 3,200, although all
three training runs execute the full 3,200-step budget.

## Comparisons and diagnostics

The current 10D gallery fit is the frozen starting point. The rank comparison
uses 49 amplitude modes, H² strength 3e-4, Sobolev optimizer preconditioning,
3,200 Adam steps, batch 256, and cosine learning rate 2e-4 to 2e-5. The training
set has 94,904 rows and validation has 4,096. Conditional bootstrap seeds
11, 23 and 37 resample only training rows, with replacement. Validation rows
are never added to training. Checkpoints use validation NLL plus the
regularizer, as in the existing gallery.

The fixed validation split has already been used in development. Paired
standard errors describe score differences on that split; they are not
post-selection confidence guarantees or independent test performance.

Comparison figures use a common density color scale for the old fit, best
single continuation and average of resampled fits. A separate panel shows
pointwise variation between the resampled fits. It is labeled as a sensitivity
diagnostic. Marginal Hessian energy measures small-scale variation in each
projection. It is integrated from exact pair-density polynomial coefficients,
including mixed derivatives, with noise-standard-deviation length scales.
It is not the fitted amplitude penalty, a full-joint curvature measure, or an
error against unknown truth. A finite-difference check, grid entropy and
marginal mass are also recorded to assess broadening and numerical issues.

The misleading normalized path-height diagnostic is not used for these
comparisons. Density values at individual state anchors do not measure state
probability or the transverse width of a recovered ridge.

After all fitting and single-model selection were complete, a metadata audit
counted 50 labeled conformations in the cached simulation, with unequal
counts (20 to 4,907 among 99,000 observations). No labels affected the models
or their selection. This is another reason not to equate every concentration
of density with an artifact. Labels alone do not establish whether a specific
estimated peak is real or whether the desired physical target is discrete or
continuous; the audit makes no clean-density accuracy claim.

## Artifacts and reproduction

Artifacts live in `artifacts/cryoem_stability/`. Each run records its settings,
source/kernel hashes, model coefficients, validation log densities and
training history. Bootstrap runs additionally record data hashes and sampled
row indices. The comparison saves its joint-mixture manifest and exact
marginal arrays. Measured continuation times include compilation and
validation, excluding initialization, plotting and previous fitting stages.

Using the `chebdGE` environment inside the existing allocation prescribed by
`instructions.txt`, a bootstrap member can be reproduced with:

```bash
python scripts/fit_cryoem_bootstrap.py \
  --base artifacts/cryoem_continuity/gallery_dim10 \
  --output artifacts/cryoem_stability/reproduction_bootstrap11 \
  --rank 48 --steps 3200 --strength .0003 --seed 11 --resample
```

The same command without `--resample` is the unresampled control. Use
`scripts/refine_cryoem_continuity.py` for the rank-64 control and
`scripts/compare_cryoem_stability.py` to compare candidates and average
specified members. Use new output directories when changing settings.

The selected 10D continuation can be reproduced with:

```bash
python scripts/refine_cryoem_continuity.py \
  --base artifacts/cryoem_continuity/gallery_dim10 \
  --output artifacts/cryoem_stability/reproduction_dim10 \
  --rank 64 --modes 49 --strength .0003 --steps 3200 --preconditioning sobolev
```

For the selected 20D continuation, change the base to `gallery_dim20`, the
output to a new directory, rank to 48, and steps to 2400. To redraw the two
selected cases from their saved coefficients:

```bash
python scripts/refresh_density_figures.py \
  artifacts/cryoem_stability/gallery_dim10 \
  artifacts/cryoem_stability/gallery_dim20 --figures figs
```

Gallery search timings cover the selected single-fit search and its ancestor
runs. The separate bootstrap experiment's cost is reported in its comparison
figures; these are not end-to-end timings from an uninitialized estimator.

The ten 10D/20D PNGs in `figs/cryoem` were updated after inspection. The other
five cryo-EM PNGs and all 33 GL PNGs are unchanged, verified by hashes in
`gallery_manifest.json`. Earlier cryo-EM images are in `previous_figs/`.
Experimental comparisons remain in the artifact directory, keeping the
public gallery flat with 48 PNGs.

Eleven relevant numerical tests passed, including independent integration
checks for mixture densities, marginal derivatives, and convolution, along
with the existing refinement and selection checks. Ruff passes changed code.
The high-rank nonadjacent marginal evaluator now reuses a transfer matrix
across grid points, with a bounded temporary size; equivalence tests pass.
The 20D (0,19) marginal on a 180×180 grid takes 1.83 seconds on one allocated
CPU in `marginal_performance.json`. Plotting is excluded from fit times.
The first rendering pass was stopped after a slow contraction; the complete
gallery was rerendered with the equivalent faster evaluation.

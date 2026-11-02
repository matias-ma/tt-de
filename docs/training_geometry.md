# Independent optimizer smoothing and coarse-to-fine training

`DirectTrainingProblem` and `NoisyDirectTrainingProblem` now accept
`precondition_strength`. This controls the Sobolev whitening of optimizer
coordinates independently of the loss regularizer's `strength`.
The default, `None`, preserves the previous coupling and arithmetic.
Changing whitening preserves the represented initial density and objective;
it changes the optimization trajectory. Adam in these coordinates is not
exact Sobolev gradient descent and need not monotonically reduce roughness.

For each core, physical coefficients `c` are represented by `p = c L`,
where `L L.T = G + beta D`. `G` is the physical Chebyshev Gram matrix;
`D` is the scaled first-derivative Gram for H1, or second-derivative Gram
for axis-H2/full-Hessian regularization. The full-Hessian loss also includes
mixed derivatives; the whitening metric remains separable across axes.
The statistical objective continues to use the independently chosen loss
coefficient, including any marginal-density curvature penalty.

`fit_adam` also accepts a `mode_schedule`: a sequence of `(update, counts)`
stages. Counts include the constant coefficient, and can be either an
integer shared across axes or a tuple with one count per axis. The
schedule must begin at zero, never shrink a subspace, and end with all
template modes active. Mode activation applies to the **amplitude**;
squaring it to obtain a density doubles the maximum polynomial degree.

```python
problem = NoisyDirectTrainingProblem(
    initial, forward, regularizer="hessian",
    strength=1e-5, precondition_strength=1e-3,
)
fit, record = problem.fit_adam(
    train, valid, output,
    steps=4000, batch_size=256, learning_rate=1e-3,
    mode_schedule=[(0, 13), (1000, 25), (2000, 49)],
    selection_start=2000,
)
```

This example requires a 49-mode template. A stage beginning at 1000
is activated after 1000 completed updates, before the next update. The
boundary checkpoint has the new mask but has not yet learned newly
unlocked coefficients. `selection_start` is optional, independent of the
schedule, and excludes earlier checkpoints from model selection while
retaining their diagnostics. It defaults to zero.

Inactive parameters, gradients and Adam updates are masked. Gradients
are masked before global clipping. Because `L` and its inverse are lower
triangular, a zero tail in `p` implies a zero tail in physical coefficients
`c`. Activation therefore enforces an actual nested polynomial subspace,
not merely a small step size. The initial parameters are projected into
the first subspace; newly unlocked coordinates start at zero. The TT
array sizes and nominal ranks are retained, though effective ranks can
decrease under a projection. Adam moments, minibatch random stream and
the global cosine learning-rate schedule are retained across stages.
The full-size contraction kernels are retained too; this implementation
changes the optimization path rather than reducing per-update compute.

The default `mode_schedule=None` follows the existing unrestricted path.
The independent whitening parameter also works with existing L-BFGS;
mode activation is implemented only for Adam.

## Matched experiment

`scripts/run_training_geometry.py` compares a 2×2 design on noisy Swirl
(4D, rank 32, 49 modes) and Ginzburg–Landau (6D, rank 10, 25 modes):

| Recipe | Whitening strength | Mode activation |
| --- | ---: | --- |
| Baseline | 1e-5 | Full basis throughout |
| Stronger whitening | 1e-3 | Full basis throughout |
| Coarse to fine | 1e-5 | Three nested subspaces |
| Both | 1e-3 | Three nested subspaces |

All runs use loss Hessian strength `1e-5`, no marginal penalty, the same
data and batches, 4000 updates, learning rate `0.001`, batch size 256,
and validation checkpoints every 200 updates. Selection begins at
update 2000, when full capacity is enabled in every run. Swirl activates
13 → 25 → 49 modes at updates 0, 1000, 2000; GL uses 9 → 17 → 25.
The initial seed is zero for the factorial comparison. The new recipe
with best noisy validation NLL is repeated with seed one alongside a
fresh matched baseline, keeping batch seed zero. These are limited
replication checks, not a many-seed reliability study.

The two seed-zero baselines are reused from the preceding study after
checking their frozen hashes, configurations, data hashes and selected
checkpoint eligibility. They already selected checkpoints after step
2000, so restricting the selection window does not change either fit.
This yields ten new fits and two reused controls. Previous strong-loss
and marginal-penalty fits have different objectives and training paths;
the weak-loss baseline here is not the best previously published fit.

Selection uses observed validation NLL. A secondary rule chooses the
lowest exact marginal curvature within one paired standard error of the
best NLL, with 200 MCMC-chain clusters for GL. Both choices are frozen
before latent truth evaluation. The same validation set has been reused
across exploratory studies, so these are development heuristics rather
than post-search statistical guarantees. Artifacts live in
`artifacts/training_geometry_v8`; preceding study files remain intact.

## Completed results

Coarse-to-fine training is the useful change in these experiments.
Stronger whitening alone does not reliably reduce oscillations or improve
latent recovery. All numbers below are full-dimensional latent relative
L2 errors, computed only after freezing both observation-based choices.

| Case / initialization seed | Weak baseline | Stronger whitening | Coarse to fine | Both changes |
| --- | ---: | ---: | ---: | ---: |
| Swirl 4D / 0 | 1.17815 | 1.16876 | 0.34284 | 0.33574 |
| Swirl 4D / 1 | 1.13620 | — | 0.35119 | — |
| GL 6D / 0 | 0.96076 | 0.98571 | 0.18530 | 0.18873 |
| GL 6D / 1 | 0.90170 | 1.01829 | — | — |

The Swirl minimum-NLL and smoothness rules both select coarse-to-fine
seed one. Its matched seed-one baseline has mean 1D marginal error
0.31988 and integrated marginal curvature 284.49; the selected curriculum
fit reduces these to 0.09269 and 13.17. The seed-zero comparison gives
the same qualitative result, with curvature falling from 341.32 to 13.44.
The benefit persists after unlocking all 49 modes. However, the new
Swirl fit does **not** beat the previous calibrated marginal-penalty fit:
full L2 is 0.35119 versus 0.27985, and mean 1D error is 0.09269 versus
0.05073. Those recipes also have different loss penalties and training
histories, so that comparison is a practical reference rather than the
matched experiment.

For GL, pure minimum validation NLL selects stronger whitening at seed
zero, whose latent error is 0.98571. The predeclared smoothness rule
instead selects coarse-to-fine seed zero: its validation NLL is only
0.000664 higher, with paired chain-clustered SE 0.000894. Its exact
marginal curvature is 38.52 rather than 128.38. This selection was made
before evaluating latent truth.

The resulting GL full-density error is **0.18530**, down from 0.96076
for the weak baseline and **0.49898 for the previous smooth choice**.
Mean 1D marginal error falls from 0.21104 to 0.06669 against the weak
baseline, and from 0.19590 against the previous smooth choice. The pair
marginals also lose the spurious bands and match the true structure much
more closely. The curriculum was tested at seed zero only in GL: the
predeclared seed-one repeat followed minimum NLL and therefore repeated
whitening instead. The two seed-zero curriculum variants agree closely,
but they are not independent initialization repeats.

Large latent improvements coexist with small noisy likelihood differences.
Against the weak seed-zero baseline, the selected Swirl fit's independent
test NLL changes by `-0.001724 ± 0.001118` (one paired SE), and the selected
GL fit's by `+0.000707 ± 0.000598` (one chain-clustered SE). Thus these
experiments support a much better training path for latent recovery,
especially in GL; they do not establish a large predictive-likelihood
improvement. No training or selection uses these test measurements.

- [GL recovery compared with the previous smooth fit](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/training_geometry_recovery_1d.png).
- [Swirl recovery compared with the previous smooth fit](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/training_geometry_recovery_1d.png).
- [Matched 1D fits: GL](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/training_geometry_1d.png), [Swirl](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/training_geometry_1d.png).
- [Pair marginals: GL](../artifacts/gallery_archive/before_dimension_gallery_v9/ginzburg_landau/training_geometry_2d.png), [Swirl](../artifacts/gallery_archive/before_dimension_gallery_v9/swirl/training_geometry_2d.png).
- [Training trajectories and mode activation](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/training_geometry_training.png).
- [All post-selection errors and noisy test comparisons](../artifacts/gallery_archive/before_dimension_gallery_v9/convergence/training_geometry.png).

## Verification and reproduction

The targeted suite checks invariance of densities, losses and physical
directional derivatives under whitening changes, exact polynomial
cutoffs, learning newly activated modes, unrestricted/full-mode
equivalence, and invalid inputs. It also exercises the existing direct
training, L-BFGS and marginal-penalty paths: 29 tests pass.
The 18 new geometry tests also pass on CPU; the integer-input cases were
checked after normalizing NumPy integer options for JSON output. Ruff passes
for all seven new or changed implementation, test and experiment files.
The real Swirl seed-one unrestricted replay matches the previous
implementation's entire 4000-update validation trajectory within
`7.55e-15`. Across both benchmarks, the largest coefficient in a locked
physical mode is `1.04e-15` (roundoff). All 59 frozen model, data, configuration
and source files passed their hash audit before evaluation.

Using `chebdGE` and an existing one-GPU allocation:

```bash
srun --jobid=... --ntasks=1 --cpus-per-task=4 bash scripts/run_noise_diagnosis_gpu.sh scripts/run_training_geometry.py
srun --jobid=... --ntasks=1 --cpus-per-task=4 bash scripts/run_noise_diagnosis_gpu.sh scripts/finish_training_geometry.py
```

The second command requires the entire training study to be frozen.
Training refuses to modify a frozen study. Figures are saved as
`training_geometry_1d.png` and `training_geometry_2d.png` in each case's
figure directory, plus comparison and trajectory figures in
`figs/convergence`.

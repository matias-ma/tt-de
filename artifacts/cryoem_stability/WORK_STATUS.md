# Cryo-EM stability work completed

Published improved single-TT fits in 10D and 20D. Noisy NLL improved from
9.866188 to 9.857318 and from 24.688191 to 24.656633; amplitude H2 declined
36.5% and 33.6%. Full settings, controls and limitations are documented in
docs/cryoem_stability.md.

Three conditional bootstrap continuations finished. Averaging their densities
improved predictions over individual resampled fits but did not beat the
selected rank-64 fit. Experimental coefficients, ensemble manifest and
sensitivity plots are retained in comparison_dim10 and the member folders.
They are not calibrated confidence intervals.

Published ten verified PNGs. Five other cryo-EM PNGs and all 33 GL PNGs are
unchanged. The public gallery remains flat with 48 figures. Earlier cryo-EM
images are in previous_figs; gallery_manifest.json records verified hashes.
Noise panels, settings and measured compute times are included.

Eleven relevant numerical tests passed. The latest three core tests also
validate the faster equivalent matrix-transfer marginal calculation.
Ruff and git diff --check passed. checks.json records convolution accuracy.
The first rendering pass was stopped for a slow contraction; render_fast.log
records the successful complete redraw. The new 20D marginal evaluation
was measured at 1.83 seconds on one allocated CPU.

State-label metadata was inspected only after all fitting and model selection.
No labels affected optimization or selection; no clean-density accuracy is
claimed. See postfit_label_metadata.json.

All work is complete. Allocation 14407868 (one A100, four CPUs, 32 GB) was
relinquished. No unrelated jobs were modified.

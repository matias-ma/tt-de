# Cryo-EM continuity work completed

Published all 15 updated cryo-EM figures. Their earlier versions are in
previous_figs. gallery_manifest.json verifies that all 33 GL figures are
unchanged. Selected cases are gallery_dim4, gallery_dim10, gallery_dim20,
and gallery_path_dim4. See docs/cryoem_continuity.md for findings and reproduction.

Thirteen squared candidates were compared with a fixed source, noise kernel,
observations and split. Matched rank48/Sobolev seeds confirm similar coarse
10D recovery, though fine features remain uncertain. The rank32/modes65
L2 repeat is NOT a matched Sobolev replication.

Six unsquared trials were completed, including actual final-iterate clipping
of the full joint 4D density, normalization and reconvolution. Full4096
validation audits at orders65/97 confirm the conservative final-clipped
run is worse than its starting squared fit. Unsquared trials remain outside
the public gallery. All raw negative predictions are reported, never floored
in validation scores. See unsquared_full_validation_comparison.json.

Sixteen relevant tests passed; Ruff passed all changed code files.
GPU allocation used: 14405209, one H100 / four CPUs / 32GB. No unrelated
jobs were modified. Allocation was relinquished at task completion.

# Completed cryo-EM refinement and timing refresh

All four cryo-EM cases completed; 4D/10D/20D and corrected 4D path projection.
Rank20,49modes,mixture noise selected in all; smoothing continuation selected
in10D and20D. Full results/timings/checks: run_summary.json and
numerical_checks.json. Both allocations14394472 and14403398 released.

All48publicPNGs verified and published:33GL,15cryo. Every plot includes
recorded fitting wall times. Folders remain flat. No clean density or labels
used for fitting/selection; labels appear only in final path diagnostics.

12 numerical/gallery/selection tests passed; relevant scripts pass Ruff.
25 refinement/continuation/rescoring runs passed doubled-quadrature checks,
maximum recorded log difference2.07e-12. Initial pilot mixture with mismatched
component domains is marked invalid and excluded. Current kernels use the
common-domain construction checked against analytic noise densities.

Baseline figures archived in previous_figs. Reproduction:
python scripts/refresh_cryoem_gallery.py --job-id EXISTING_ALLOCATION
See docs/cryoem_refinement.md and docs/figure_refresh.md for protocol,
measured-time exclusions, hardware and remaining accuracy limitations.

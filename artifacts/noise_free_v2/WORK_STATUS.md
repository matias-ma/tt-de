# Completed: direct Chebyshev TT optimization study

All 65 fits are complete: 49 main noise-free fits, four initialization
checks, and 12 cryo-EM fits. There are no pending fits or plots.
The gallery contains 17 current PNGs and its index, with no numerical
sidecars. The 140 previous PNGs are archived outside figs.

Results, methods, and limitations: docs/noise_free_optimization.md.
Figure index: figs/README.md.
Validation: 10 focused numerical tests passed; Ruff passed on the new
modules and scripts; the final audit verified 114 frozen-file hashes.

The original allocation 14570003 unexpectedly received SIGTERM. The 20D
refinement recovered from checkpoint 20 and completed 40 updates across
two L-BFGS segments. The reset of optimizer history is documented in the
report, result record, and figures. Its original progress and restart
checkpoint are preserved.

The replacement allocation 14582963 was released after publication and
cleanup of this task's four node-local cache paths. No task allocation
remains. User job 14452283 was not modified.

Study storage: about 273 MiB on disk; gallery: 5.6 MiB. The final home-quota
check was 32.6 GiB used against a 48.8 GiB soft limit. Existing home caches
were left untouched.

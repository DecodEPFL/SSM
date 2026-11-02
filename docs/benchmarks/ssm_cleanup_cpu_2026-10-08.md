# SSM cleanup: CPU comparison, 2026-10-08

Compared the source saved immediately before the cleanup with the refactored
package. Both versions ran in the same Python process, with one CPU thread.
Timings alternate version order to reduce drift between runs. Each number is
a median of seven batches of ten calls, after eight warmup calls per version.

Training measures forward plus backward, with gradients cleared between calls;
it excludes an optimizer step. Evaluation uses `eval()` and `no_grad()` with
warmed parameter caches. CUDA was unavailable and was not measured.

The 13 small cases use batch 4, length 128, feature width 8, state size 16,
and two layers. DeepSSM uses `LGLU2`; metric stacks use their residual branch.
The two larger L2N cases use batch 8, length 512, feature width 16, state size
128, and two layers. Times are milliseconds; small differences reflect run
variability rather than an established improvement or regression.

| Configuration | Training before | Training after | Eval before | Eval after |
| --- | ---: | ---: | ---: | ---: |
| `lru` | 4.592 | 4.652 | 1.623 | 1.592 |
| `l2n` | 5.450 | 5.280 | 1.617 | 1.423 |
| `defect` | 7.618 | 7.590 | 1.308 | 1.274 |
| `tv` | 5.340 | 5.340 | 1.700 | 1.620 |
| `tvc` | 5.647 | 5.653 | 1.816 | 1.741 |
| `l2ru` | 19.961 | 19.891 | 6.134 | 5.916 |
| `zak` | 7.466 | 7.395 | 2.041 | 1.888 |
| `l2nt` | 13.832 | 14.082 | 1.971 | 1.958 |
| `metric_l2n` | 7.135 | 7.225 | 1.281 | 1.296 |
| `metric_defect` | 8.951 | 8.846 | 1.035 | 1.041 |
| `context_ports` | 9.637 | 9.613 | 1.872 | 1.648 |
| `context_select` | 5.620 | 5.262 | 1.791 | 1.702 |
| `context_select_tvc` | 7.050 | 7.123 | 2.301 | 2.265 |
| `large_l2n_scan` | 57.543 | 54.396 | 17.670 | 15.248 |
| `large_l2n_conv` | 58.483 | 58.081 | 19.235 | 17.574 |

No meaningful slowdown was observed in these paired CPU cases. The clearest
improvement is L2N cached inference: about 12% for the small case and 14% for
large-state scan (9% for large-state convolution). DeepSSM now asks for only
the final state, avoiding a physical-coordinate solve for the full trajectory.
Forward certificate scaling also avoids storage diagnostics it does not use.

Correctness was checked separately against the saved version with fixed seeds:
outputs, input gradients, parameter gradients, and checkpoint keys agree across
all 13 small configurations. The complete test suite ran 152 tests: 150 passed
and two CUDA tests were skipped. Additional interface tests cover final-state
gradients, empty sequences, layouts, explicit RNN state, the legacy dense interface,
difference-context streaming through an encoder, and certificate freezing.

The numerical scan, FFT, CUDA capture, static-layer, and REN source files are
unchanged. Public import aliases, configuration field order, parameter names,
and checkpoint layouts remain compatible.

[Raw timing medians](ssm_cleanup_cpu_2026-10-08.json).

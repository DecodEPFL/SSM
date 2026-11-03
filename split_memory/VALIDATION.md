# Split-memory implementation validation

Validation date: 28 October 2026. All implementation/tests/new artifacts are
inside `split_memory/`. External source changes are limited to
`Test_files/run_benchmarks.py`, `Test_files/benchmark_ui.py`,
`direct_spending/ui_settings.py` and `direct_spending/ui_layout.py` for the
existing benchmark/UI registry and a prediction-axis range fix. Existing model mathematics, datasets, splits,
training entry points, cancellation and saved-run viewers are preserved.

## Results

| Checks | Passed | Skipped |
| --- | ---: | ---: |
| New focused model/integration/native-widget checks | 24 | 0 |
| Existing repository regression suite | 186 | 2 |
| Prior UI and spending integration checks | 17 | 1 |

The two repository skips are CUDA-only checks. The prior UI suite's optional
native test was skipped in that invocation; the new native controls/workflow were
actually exercised in the focused suite. Detailed logs are in `validation/`.

Focused checks cover structured scan/reference/chunk agreement through 8,193
steps, exact/near-extreme gates, nonzero initial states, bounded memory, normalized
row sums after optimizer updates, empirical contraction, exact K>=T outputs and
all parameter/input/state gradients, finite differences, tightening finite-sweep
gradients, adaptive and assembled-chunk residuals, causality including adaptive
prefix freezing and batch independence, suffix padding/empty samples, internal-gap
fallbacks, rho=0, cap failure, FP32/FP64/BF16 CPU arithmetic, optional generic
energy composition, checkpoint/solver reconstruction, actual dataset integration,
UI/defaults round trips, visibility of predictions outside the target range, and `aot_eager` compilation of the fixed tensor kernel.

A deliberate one-sweep/tolerance-1e-12 failure produced the existing error row
with `ConvergenceError` and measured failure diagnostics. This expected negative
case appears in the test log. It was not reported as successful convergence.

## Actual existing-UI workflow

The test selected `split_memory` in the real Tk model list, set its controls,
saved the generated configuration, invoked the existing Run button, ran one
native Cascaded Tanks epoch, processed the normal progress/completion path,
loaded the report into the existing results/details/plot widgets, and verified a
real prediction/loss image object. The checkpoint reloaded through
`load_checkpoint`; physical prediction metrics matched the report within 1e-6.
The UI processes were closed after validation. No personal defaults were written.
The native-defaults check used a temporary file and a fresh window.

The test uses the application's normal Tk main loop. A preliminary manual
`update()` event-pumping check stalled in this macOS/Tk build and was closed;
the corrected normal-loop workflow passed. No user process or unrelated run was
stopped. The final report explicitly takes solver diagnostics from the complete
validation trajectory, rather than the last short plotting preview. Prediction axes now expand to include out-of-range predictions, keeping initial poor fits visible. Dense cell counters include evaluated padding and separately report valid-token evaluations.

**Visual inspection remains unavailable**, per the user's earlier instruction.
Programmatic widget/geometry checks and an actual Tk event-loop workflow are not
claimed as screenshot/manual visual verification. Manual cancellation interaction
was not re-tested; that existing path was unchanged.

## Recorded one-epoch smoke result

- Hardware: Apple M1 Max, 32 GiB RAM, macOS 15.6.1, Python 3.13.9,
  PyTorch 2.9.0.
- CPU FP32, seed 0, trajectory batch 1; native train/validation length 1,024 each.
  The native validation metric skips 50 conditioning samples; model state includes
  those inputs. No trajectory was split between training and validation.
- H=2, d_c=4, M=1, rho=.7, initial retention=.95; scan memory, 32 fixed sweeps,
  tolerance 1e-4, required convergence; 61 trainable parameters.
- Physical validation RMSE: 2.08472447; NRMSE: 0.983364259.
- Existing CPU training-loop timer: 0.0823299885 s. It includes epoch
  validation and in-loop plot preview, excludes initial loading/construction and
  final artifact export, and is one measurement rather than a steady-state
  throughput benchmark. No accelerator synchronization is needed for this CPU run.
- Complete validation residual: 8.64267349e-07; corresponding computed-
  memory state-error bound: 2.88089109e-06, below 1e-4.
- 32 sweeps plus one verification pass: 33792 valid-token
  cell evaluations. Measured maximum absolute row sum: 0.699931026.
- Maximum head norm: 0.494276941; measured memory-bound
  excess: 0.

This single-epoch result verifies integration. It establishes no accuracy,
architecture-ranking, retention-learning or speed advantage. The analytical
contraction/residual bounds are not interval-arithmetic guarantees. The run's
ordinary JSON/Markdown reports, PNG plots, configuration, UI log and tagged
checkpoint are under `validation/ui_smoke/`. `configs/ui-smoke.json` is the
portable relative-output version for loading through the existing UI or CLI.

## Floating-point audit and backend limits

For pure isometric retention with no writes (B=2,H=2,T=8193, angle=.013, seed 41),
FP32 scan-versus-loop max absolute difference was
5.40018082e-05; the scan's maximum
absolute head-norm drift was 0.000212907791.
The focused FP32 long-isometry test uses explicit absolute/relative slack 2e-3.
FP64 scan-versus-loop difference was 1.24344979e-14,
with head-norm drift 6.10400619e-13
and test slack 2e-11. Damped/random-write cases use tighter dtype-dependent
slack. These measured deviations are retained, not hidden by a post-scan clamp.
The independent FP64 K=T=9 gradient audit's largest absolute difference across
inputs, both incoming states and every parameter was
1.58761893e-14.
Details and per-parameter errors are in `validation/numerics.json`.

MPS FP32 fixed/adaptive/sequential forward/backward checks passed with CPU operator
fallback **disabled**. PyTorch warned that padding more than three dimensions
uses its View Ops implementation on MPS, with possible performance implications;
no native/fused performance claim is made. CPU FP64 checks do not imply MPS FP64
support. MPS full-dataset training and CUDA execution were not validated. There
were no CUDA devices available. Runtime details are in `validation/runtime.json`.

Internal mask gaps use a clearly reported exact sequential fallback. Newton/DEER,
implicit differentiation, custom kernels and full-model suite compilation are not
implemented; the pure fixed kernel's compiled output/backward check passed using
`aot_eager`. All training gradients are through the implemented explicit graph,
including finite sweeps and discrete adaptive-freezing choices. See README.md
for residual/memory-error scope, complexity, initialization and claim limitations.

## Review and reproduction

`integration.patch` records the four existing-file wiring changes. New model
files are `config.py`, `state.py`, `memory.py`, `solvers.py`, `model.py`,
`benchmark.py`, `ui.py` and the lazy package initializer, with contained tests.
The source hashes and check counts are in `validation/summary.json`.

From SSM-clean, run:

```sh
python -m unittest discover -s split_memory/tests -t . -v
BENCHMARK_UI_TK_TESTS=1 python -m unittest split_memory.tests.test_integration.NativeUITests -v
python Test_files/benchmark_ui.py
```

The next optimization decision needs profiling at the user's target device,
length and widths: separate memory scan, precomputed projections, K dense reader
passes and backward. This smoke timer does not identify a measured bottleneck
or justify a custom kernel/Newton/implicit-backward speed claim.

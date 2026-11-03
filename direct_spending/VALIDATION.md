# Direct-spending integration validation

Implementation and new tests are contained in `direct_spending/`. Existing runner/UI wiring is limited to `Test_files/run_benchmarks.py` and `Test_files/benchmark_ui.py`. The earlier `energy_bounded/` files match their recorded hashes, and unrelated tracked changes match the baseline. A separate 2,000-epoch `spending_gru` run with gamma 7 appeared under the existing `Test_files/benchmark_runs/` during this work. Its artifacts were left untouched and excluded from these validation claims.

## Initial implementation checks

| Suite | Passed | Skipped |
| --- | ---: | ---: |
| New focused checks | 32 | 1 |
| Existing repository checks | 186 | 2 |
| Earlier projection-wrapper compatibility | 29 | 1 |

Skips are unavailable CUDA/MPS checks. CPU FP32/FP64, lower-precision storage proposals, and CPU BF16 autocast backward are covered. Source parsing and `git diff --check` pass. Detailed logs and source hashes are in `validation/`.

Focused checks cover the exact vector unit-ball head, zero/unit/intermediate spending, batch-independent prefix accounting, zero/fixed/learned beta, beta initialization once per trajectory, positive-beta zero-input responses, affine/reference agreement through 8,192 steps and large dynamic ranges, masks, causality, full/chunk/step/core-cache equivalence, gradients and finite boundary conventions, learned-beta gradients/underflow, AMP-consistent initialization, all core/beta configuration paths, parameter-budget factory support, negative initial-context values, tagged checkpoints, saved UI configurations and single/batch commands.

## Existing-data smoke job

Loaded `configs/ui-smoke.json`, generated with the existing UI command builder, through `Test_files/run_benchmarks.py --config ...`. The job used the existing cached Cascaded Tanks data: one complete 1,024-sample training trajectory, one complete 1,024-sample validation trajectory, and the native 50-sample metric initialization window. It completed one epoch with static MLP, zero beta, gamma=1, CPU, and diagnostics. No dataset, split, benchmark entry point, comparison campaign or tuning sweep was added.

The ordinary report JSON/Markdown, prediction/loss PNG, summary PNG, storage PNG, history and `direct_spending_v1` checkpoint were produced. Checkpoint reconstruction reproduces the saved physical prediction metrics. Both PNG artifacts decode and were inspected as generated plots; that does not verify native UI rendering.

The smoke job reported maximum normalized prefix violation 2.411e-09 and maximum normalized energy-identity residual 2.089e-08, within the explicit FP32-return tolerance of 2e-6. These are finite-precision audits, not interval-arithmetic guarantees.

The certificate is in standardized, centered coordinates. At gamma=1 the actual target prefixes require more initial energy than beta=0 allows. The runner reports this initialization incompatibility and leaves gamma/beta unchanged. The one-epoch result verifies integration, not accurate fitting or model ranking.

## UI cleanup follow-up (28 October 2026)

The model list now shows only `direct_spending` for this family. Its four
`spending_*` aliases remain in the CLI registry. The first box contains shared
run/training/evaluation/device/output settings. The second box shows tabs only
for selected models, with their architecture, initialization and gain controls.
Feedforward dimensions, TCN/Transformer controls, learned-beta context width,
per-channel gates and TV/TVC CUDA graph controls appear only where applicable.
Hidden fields retain values for switching back and are omitted from commands.
GRU/LSTM width/layer controls now reach the real runner factory, retaining the
old 64/2 defaults. The only external source changes are still the existing UI
and runner files; no existing test sources were changed.

**Save as defaults** atomically writes the complete configuration to
`direct_spending/configs/ui-defaults.json`. A fresh UI restores it automatically.
Tests cover invalid run values, corrupt files, failed writes, rollback after a
failed load, old single-alias migration, and explicit rejection of a multi-core
alias batch that cannot be represented by one core selector. No personal default
file was created by validation; the tests use temporary files.

- All **43 focused checks passed**, including the real native-data one-epoch
  integration check and the new UI/defaults checks.
- Existing repository suite: **186 passed, 2 skipped** (CUDA unavailable).
- Native Tk checks constructed the real widgets, visited every model, checked
  conditional visibility for all four spending cores, tested beta/context
  controls, restored defaults in a fresh window, and checked field bounds and
  action/log availability at 900x650 and 1120x920.
- The focused checks ran with host access needed by Tk. The basic optional MPS
  model forward/backward check also passed in that environment. The earlier
  restricted-environment run had reported MPS unavailable. This does not add a
  GPU training/throughput validation claim.
- The previous `energy_bounded/` files and all unrelated source changes were
  preserved. No benchmark run artifacts were modified.

Logs: `validation/ui-cleanup-tests.txt` and
`validation/ui-cleanup-regression-tests.txt`. The current source hashes and
check counts are recorded in `validation/summary.json`.

## UI verification limitation

**Visual UI verification remains unavailable**, as requested by the user after
computer use reported missing permissions. Native widgets and geometry were
checked programmatically; no screenshot or human-style visual inspection is
claimed. Actual manual Run/Stop/progress and saved-run navigation remain
unverified. No backend/configuration test is represented as visual success.

CUDA remains untested. The new family uses the portable differentiable affine
scan and explicitly skips suite compilation. Storage/output-feedback spending
maps are outside its scope. Transformer KV caching is real, but context is
unbounded and appending copies the cache; full attention can exceed memory for
long native trajectories.

## Run in the current UI

Restart `python Test_files/benchmark_ui.py` from SSM-clean. Set the shared options
in the first box and choose `direct_spending` once in Models. Set its core and
operator parameters in the Direct spending tab under Model options, then Run.
Use Save as defaults to reuse the complete configuration on subsequent launches.
Load configuration still accepts `direct_spending/configs/ui-smoke.json`. The
Results / saved runs tab has Open saved run and Storage trace. See `README.md`
for configuration, coordinate, state and mathematical semantics.

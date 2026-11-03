# Output projection + learned linear path — validation, 2026-11-03

Implemented `r_t = NN(u_0:t, context_0:t) + D u_t`, then one energy projection
after the complete sum. D is a native bias-free `nn.Linear`, learned and
zero-initialized by default; optional identity initialization is exposed.
Context conditions the neural branch only and never supplies energy.

Select **direct_spending → Model options → Readout → Output prediction + linear**.
Restart an already-open UI. The conditional linear-initialization control,
parameter count, save-as-defaults, CLI, result labels, storage plots and
checkpoint loading are wired. Original spending mode/defaults and checkpoints
retain their behavior. New checkpoint family/tag is
`energy_projection` / `energy_projection_v1`; new run directories append
`--projected-linear` and cannot overwrite a spending-mode directory of the
same core/beta choice. Existing `energy_bounded/` sources were not modified.

## Checks

- New focused suite: **12 passed**, including native Tk and MPS checks. These
  cover an independent online oracle; per-sample prefix identity/inequality;
  combined neural/linear clipping; exact signed identity with D=I and zero neural
  residual; full/chunk/step outputs and all states for all five cores; arbitrary
  masks, NaN padding and state gradients; persistent core-only context and
  causality; learned gamma/beta/D/core gradients; finite-difference gradients;
  finite zero/switching gradients; checkpoint dtype/family; and actual parameter
  matching/counts. An intentionally corrupted scan triggers a reported reference
  fallback and reproduces the independent reference outputs without refill.
- Long refill/saturation tests: FP64 length 16385 with proposals of magnitude
  `1e200`, and explicit FP32 length 8193 with magnitude `1e30`, passed without
  fallback. MPS length 4097 with `1e30` proposals also passed forward/backward.
- Native MPS forward/backward plus chunk/step equivalence passed for all cores
  with learned gamma and context, with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
  CPU FP64 and BF16 autocast checks passed. MPS FP64 is explicitly unsupported;
  CUDA hardware was unavailable.
- Original direct-spending compatibility suite: **55 run, 51 passed, 4 skipped**
  in the restricted CPU process (two opt-in Tk tests and two accelerator tests).
  The full original UI suite plus the then-current new suite separately passed
  **25 checks** on the host, including both original native Tk tests.
- Repository regressions: **188 run, 186 passed, 2 skipped** (CUDA-only), using
  `PYTHONPATH=src` to exercise this checkout rather than an older installed copy.
- The actual new-mode parameter-count button reports **603** parameters for
  width-8, one-layer LSTM, one input/output, zero beta and learned gamma:
  601 neural, one D entry, one gamma parameter. Narrow-window controls passed
  programmatic geometry checks. **Visual UI verification remains unavailable**,
  as previously requested.

The optimized sequence path uses the existing work-efficient associative scan
with max-plus pairs. It caps proposal energy at the reachable prefix supply
before scanning and uses scaled geometry for enormous proposals. It avoids the
large-cumsum subtraction used by a reflected-sum formulation. Energy and returned
output precisions have separate audit tolerances. Under-budget proposals remain
unchanged; clipped outputs have a conservative rounding guard. The real-arithmetic
identity is audited against actual returned outputs; floating-point identity
errors include that small deliberate numerical dissipation. No epsilon budget,
refill, implicit initial-energy learning, or detached reservoir state is used.

## Reproducible native-data comparison

A two-epoch LSTM pilot ran first, followed by the following exploratory
**single-seed, 50-epoch** comparison on the existing Cascaded Tanks split.
Each pair starts with identical neural-core weights; the new D starts at zero.

| Core | Readout | Parameters | Validation RMSE | Training + per-epoch validation |
| --- | --- | ---: | ---: | ---: |
| LSTM | Spending fractions | 4401 | 2.826064 | 8.912 s |
| LSTM | Output prediction + linear | 4402 | 2.098742 | 8.851 s |
| Transformer | Spending fractions | 4529 | 21.518249 | 1.452 s |
| Transformer | Output prediction + linear | 4530 | 2.213754 | 1.570 s |

Settings were fixed before fitting: seed 0, width 16, two layers, two attention
heads, dropout 0, gamma **10 fixed**, beta **0**, standardized coordinates,
parameter matching off, AdamW rate 0.001, weight decay 0, gradient clip 1, and
the existing 50-epoch cosine schedule. One native 1024-sample training trajectory
and one native 1024-sample validation trajectory are used; the official first
50 validation samples initialize state and are excluded from scoring. Trajectory
batch size is 32, giving one optimizer update per epoch. No trajectory slicing,
target-derived context, temporal split leakage or gamma adjustment is introduced.
The necessary target-prefix initial-energy audit returns zero at gamma 10 for
both splits. Checkpoint selection uses validation, as in the existing runner.

The new readout improved these short fits, but errors remain substantial.
These are neither tuned comparisons nor repeated-seed architecture rankings.
Different readouts also interpret initial raw outputs differently even though
the neural weights match. Equal epochs do not imply equal compute across cores.

At selected new-mode checkpoints, clipping fraction and worst normalized
positive prefix violation were **0**. Normalized identity errors were
`1.73e-15` for LSTM and `1.87e-15` for Transformer. Thus the constraint was inactive
on these validation trajectories at the selected checkpoints; this experiment
mainly assesses the change in output parameterization in that regime. Original
spending-head saturation was 0 for LSTM and 0.07715 for Transformer; it is a
different diagnostic from proposal clipping. Its worst positive prefix violation
was at most `9.98e-9`, within the existing FP32 tolerance of `2e-6`.

All four trained checkpoints reloaded and reproduced predictions. Full histories,
metrics, certificates and timing medians/protocols are recorded in
`validation/comparison/comparison.json`; the entry point is `comparison.py`.

## Timing and persistent state

Apple M1 Max, macOS 15.6.1, Python 3.13.9, PyTorch 2.9.0, two CPU threads.
Neural computation FP32, energy auto-FP64; no AMP or compilation. The separate
steady-state timings exclude two warmup calls, use medians of three calls,
batch 2, and omit plotting and optimizer steps. CPU operations complete before
each timer stops. They are short microbenchmarks, not confidence intervals.

| Core / readout | Inference T=128 / 2048 | Forward + backward T=128 / 2048 |
| --- | ---: | ---: |
| LSTM / spending | 2.831 / 34.196 ms | 20.349 / 338.636 ms |
| LSTM / projected + linear | 2.876 / 34.181 ms | 19.905 / 337.107 ms |
| Transformer / spending | 1.368 / 51.573 ms | 3.746 / 91.740 ms |
| Transformer / projected + linear | 1.637 / 52.428 ms | 3.960 / 93.414 ms |

Core-only timings isolate readout overhead. For new-mode LSTM, core inference
was 2.006 / 32.731 ms and core forward/backward was 19.382 / 329.984 ms.
For new-mode Transformer these were 0.943 / 52.463 ms and 1.974 / 90.410 ms.
Small apparent improvements from adding a wrapper are timing noise; no speedup
is claimed. Throughputs in tokens/s are in the JSON.

Cached step latency uses batch 1, one warmup step and the median of 20 timed
steps after initial contexts 0, 128 and 1024. New-mode LSTM end-to-end medians
were 269.8, 267.9 and 269.3 microseconds; core-only 57.9, 56.5 and 52.7.
New-mode Transformer end-to-end medians were 442.5, 435.3 and 506.1 microseconds;
core-only 225.0, 231.0 and 254.1.

After those 21 appended tokens, LSTM persistent state used 272 bytes (256 core
and 16 reservoir/initial-beta bookkeeping), independent of context. Transformer
state used 5421, 38317 and 268589 bytes, showing growth of its real KV caches;
reservoir/bookkeeping contributes 16 bytes. These are unique tensor-storage
bytes, not peak training RAM or optimizer memory. The D weights add four bytes
in this SISO FP32 example and no recurrent cache.

The useful next profiling target is the single-step readout: its masking,
norm/projection and safety checks add noticeable overhead for a small core.
Profiling should identify the operations before specializing, compiling or
fusing them. Current measurements do not establish a particular kernel bottleneck.

## Native runner smoke with active clipping and learned gamma

A separate three-epoch width-8, one-layer Transformer job used the actual CLI,
gamma initialized at 0.5 within `[0.1,2]`, beta=0, and active proposal clipping.
It saved ordinary reports/storage plots and a distinct new checkpoint. There
were 643 parameters; fitted gamma was **0.4993749559**, validation RMSE
**2.436228698**, clipping fraction **0.744140625**, positive prefix violation
**0**, and normalized identity error **7.35e-7**, within the FP32 output tolerance.
Checkpoint reload reproduced metrics and the fitted certificate.

This smoke intentionally verifies clipping/gain plumbing, not good prediction:
at that fitted gamma, exact target prefixes require about 777.54 initial energy
on training and 714.76 on validation, while configured beta is zero. The runner
reports the incompatibility; gamma/beta were not increased or refilled to hide
it. See `validation/native-smoke-summary.json` and `validation/native_smoke/`.

## Files and reproduction

New implementation, focused tests, comparison and documentation are contained
in `direct_spending/output_projection/`. Shared integration edits are limited
to `direct_spending/config.py`, `model.py`, `benchmark.py`, `ui.py`,
`ui_results.py`, `diagnostics.py`, `README.md`, and the existing runner
`Test_files/run_benchmarks.py`. Native core implementations are reused, and
existing `energy_bounded/` sources remain unmodified. Saved UI defaults were preserved;
task runs wrote only their separate output folders.

See [README.md](README.md) for UI/Python usage, equations, scan details and all
test/comparison commands. The native CLI smoke is reproducible with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
python Test_files/run_benchmarks.py --benchmarks Cascaded_Tanks \
  --models direct_spending --spending-readout projected --spending-linear-init zero \
  --spending-core transformer --d-model 8 --n-layers 1 \
  --spending-gamma 0.5 --spending-gamma-mode learned \
  --spending-gamma-min 0.1 --spending-gamma-max 2 --spending-beta-mode zero \
  --spending-diagnostics --param-budget off --epochs 3 --lr 0.001 --seed 0 \
  --device cpu --no-show --no-gif --no-plot \
  --out direct_spending/output_projection/validation/my_native_smoke
```

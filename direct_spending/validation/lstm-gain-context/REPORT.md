# LSTM, bounded learned gamma, parameter counts and contextual inputs

Implemented in SSM-clean, 2026-11-03. Restart an already-open benchmark UI to
load the new controls. Existing saved UI defaults and user benchmark outputs
were preserved.

## What changed

- Direct spending has an `lstm` core and CLI alias `spending_lstm`. It uses
  native `nn.LSTM` between learned input/output projections, with explicit
  `(hidden, cell)` state. Dense chunks call the library module once; masking
  reuses the existing batch-row compaction adapter. Full, chunked, single-step,
  padded and resumed execution preserve both states and storage gradients.
- **Count parameters** in **Model options** opens a scrollable, asynchronous
  snapshot. It loads selected dataset dimensions and uses the actual model
  factory, parameter-budget policy and effective width, reporting trainable
  and total parameter counts. It does not train or write benchmark results.
  All 15 visible model selections were counted successfully on Cascaded Tanks.
- Gamma now has **fixed** (default) and **learned** modes. Learned mode adds one
  scalar logit, with `gamma_min + (gamma_max-gamma_min)*sigmoid(logit)` and
  explicit positive bounds. Ordinary autograd passes through the gain and
  actual filtered outputs. At least FP32 is used for the gain; inward-rounded
  representable endpoints keep numerical saturation within the declared
  limits. There is no energy floor or refill and no new objective penalty.
  Saved reports/certificates include initial/fitted gamma, mode, limits and
  the uniform `gamma_max` ceiling. The final target audit uses the selected
  checkpoint's fitted gamma. Learned-gamma run folders have a distinct suffix.
  Old fixed-gamma checkpoints load without added gain tensors. Parameter dtype
  is preserved, including exact fitted-gamma round trips for FP64 checkpoints.
- The Python model API accepts `context=` with
  `SpendingConfig(context_features=k)`. Context reaches only the causal core;
  reservoir norms use only the external energy input. Sequence/chunk context
  is `(B,T,k)`, step context is `(B,k)`, with matching device/dtype. This is
  separate from the existing once-per-trajectory beta context. Benchmark UI
  datasets retain zero separate context features; no contextual data are
  synthesized from targets or future observations.

Gamma is a global scalar. Hold model weights fixed through a streamed
trajectory, and reset state after optimizer updates. A checkpoint's fitted
gamma certifies that fixed operator; the configured ceiling supplies the
uniform bound over learned parameter values. Initial-energy allowances and
standardized versus physical coordinates keep their existing meaning.

## Native-module audit

| Core | Native PyTorch computation retained or added |
| --- | --- |
| MLP | `nn.Linear`, `nn.GELU`, `nn.Sequential` |
| GRU | `nn.GRU` plus input/output projections |
| LSTM | `nn.LSTM` plus input/output projections; no custom cell or time loop |
| TCN | `nn.Conv1d` with native dilation, linear projections and GELU |
| Transformer | `functional.scaled_dot_product_attention`, linear projections, LayerNorm, GELU and dropout |

Transformer-specific code implements causality, valid-token positions and real
per-layer KV caches. Replacing it with prefix recomputation would lose cached
inference. TCN code supplies its per-layer causal history buffers. The storage
scan retains the repository's differentiable associative scan over PyTorch
tensor operations: it implements the spending recurrence rather than a standard
neural layer. No custom recurrent gates, attention softmax, CUDA kernels,
compilation dependency or speculative fused replacement was added. Actual
kernel selection is controlled by PyTorch and its backend; this audit does not
claim every native operation is fastest on every device. Masked GRU/LSTM/TCN
execution still compacts rows, while dense chunks use native sequence kernels.

## Validation

- Focused direct-spending suite: **55 run, 51 passed, 4 skipped** in the
  restricted CPU process. Skipped cases were two opt-in native Tk tests and
  two accelerator checks.
- Host native UI suite plus existing all-core accelerator check: **15 passed**.
  This includes conditional LSTM/gamma controls, saved-default migration,
  narrow-window geometry and the actual parameter-count button. Its final
  scrollable-dialog version was independently rerun: **1 passed**, reporting
  **602 trainable parameters** for width 8, one-layer LSTM, one input/output,
  zero beta and learned gamma. No training subprocess was started by counting.
- Learned-gamma/context MPS check: **1 passed**, covering all five cores with
  finite forward/backward, context gradients and prefix audits.
  Both accelerator commands used `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Repository regressions: **188 run, 186 passed, 2 skipped** (CUDA unavailable),
  using `PYTHONPATH=src` to test this checkout rather than the environment's
  older installed package.
- CPU FP64 gradient/certificate checks, BF16 autocast checks, hard gain-limit
  saturation, zero external input, arbitrary masks, causal context, legacy
  checkpoint loading and learned-gain checkpoint round trips passed. MPS was
  checked in FP32; MPS FP64 remains unsupported. CUDA was unavailable.
- Visual UI verification remains **unavailable**, as requested. Native Tk
  controls were checked programmatically.

## Small end-to-end smoke

The final native Cascaded Tanks run used seed 0, the existing full-trajectory
split and standardizer, width 8, one LSTM layer, beta exactly zero, initial
gamma 10 within `[0.1,20]`, 10 epochs of AdamW, rate 0.001, gradient clip 1,
no parameter matching, no dropout/AMP/compile or live training monitor, and two CPU threads.
There is one 1024-sample training trajectory and one 1024-sample validation
trajectory; validation excludes its official 50-sample initialization prefix
from the score but retains it in core/storage initialization. Batch size 32
still means one optimizer update per epoch.

Apple M1 Max, macOS 15.6.1, Python 3.13.9, PyTorch 2.9.0; core FP32 and
automatic storage FP64. The model has **602 parameters** (601 neural and one
gain parameter). Fitted gamma was **9.972662925720215**. Validation RMSE was
**21.941260898910546** in physical units; this short fit is poor and is only
a wiring/gradient/certificate smoke, not an accuracy or architecture ranking.
The training/evaluation loop took approximately **1.37 s**, without a
controlled performance timing protocol. Worst normalized positive prefix
energy violation was **0**, and identity error was **2.30e-8**, below the
existing FP32 audit tolerance of 2e-6. Checkpoint reload reproduced metrics
and the actual fitted-gamma certificate.

See `smoke-summary.json`, `smoke/report.json` and `smoke/report.md` for measured
values and settings; test logs and `all-model-counts.json` are alongside this
report. No claim of improved accuracy or throughput follows from this smoke.

## Context guarantee

For context `c` entering only the dimensionless proposal,
`z = core(v,c)`, storage telescopes exactly as before:

```text
sum ||y_t||² <= beta + gamma² sum ||v_t||²
```

Consequently, finite beta and `v in l2` imply `y in l2`, uniformly over
bounded, causally supplied contextual sequences. Context may have infinite
duration and infinite total energy, including a constant signal, because it
never refills the tank. The core must remain well-defined on finite inputs;
the certificate does not bound its hidden states or prevent numerical overflow.

Merely bounded energy inputs do not imply square-summable outputs. For example,
`v_t=1`, `p_t=1`, beta=0 gives `y_t=gamma` indefinitely. Context alone cannot
produce nonzero output with zero external input and zero initial energy.
Do not concatenate persistent context into the reservoir's energy input.
The guarantee applies in the selected certificate coordinates; affine
de-standardization can add a nondecaying output offset. Select physical
coordinates when a physical-signal zero-state bound is required.

```python
from direct_spending import DirectSpendingModel, SpendingConfig
config = SpendingConfig(core="lstm", gamma=2., context_features=3)
model = DirectSpendingModel(1, 1, config)
y, state = model(v, context=c)  # v: (B,T,1), c: (B,T,3)
```

## Files and reproduction

All implementation additions are in `direct_spending/`. Existing wiring changes
are limited to `Test_files/benchmark_ui.py` and `Test_files/run_benchmarks.py`.
Changed contained sources: `config.py`, `cores.py`, `model.py`, `storage.py`,
`benchmark.py`, `ui.py`, `ui_layout.py`, `README.md`, `tests/test_models.py`,
`tests/test_ui.py`; new sources: `gain.py`, `parameter_count.py`,
`ui_parameter_count.py`, `tests/test_extensions.py`. The existing
`energy_bounded/` implementations were reused without modification.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s direct_spending/tests -t . -v
BENCHMARK_UI_TK_TESTS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest direct_spending.tests.test_ui \
  direct_spending.tests.test_models.ModelTests.test_supported_accelerators \
  direct_spending.tests.test_extensions.ExtensionTests.test_accelerator_learned_gain_context_forward_and_backward -v
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python Test_files/run_benchmarks.py --benchmarks Cascaded_Tanks \
  --models direct_spending --spending-core lstm --d-model 8 --n-layers 1 \
  --spending-gamma 10 --spending-gamma-mode learned \
  --spending-gamma-min 0.1 --spending-gamma-max 20 --spending-beta-mode zero \
  --spending-diagnostics --param-budget off --epochs 10 --lr 0.001 --seed 0 \
  --device cpu --no-show --no-gif --no-plot \
  --out direct_spending/validation/my_lstm_gain_smoke
```

Further optimization should be measured: native RNN packing for large padded
batches or Transformer cache append/mask costs are possible targets. No timing
here establishes either as the current bottleneck. Gamma/head conditioning
and optimization schedules need controlled accuracy/time experiments before
changing defaults.

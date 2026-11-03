# Generic energy-bounded sequence models

This folder contains the implementation, tests, legacy benchmark bridge, synthetic
task, and measured comparison. It is independent of the SSM architecture registry.
No existing repository source needs modification. The construction is explicit;
it is **not claimed to reproduce an identified paper**.

## Construction and guarantee

A causal core first computes the complete raw sequence `v = core(u)`. The wrapper
then filters its final readout, including any core output residual. There is no
reservoir-state or filtered-output feedback into the core.

For each sample separately, with feature dimensions summed only:

```text
a_t = gamma² ||u_t||²     q_t = ||v_t||²     b_t = E_t + a_t
y_t = v_t                    if q_t <= b_t
      sqrt(b_t/q_t) v_t       otherwise
E_{t+1} = b_t - ||y_t||² = max(E_t + a_t - q_t, 0)
```

At zero budget the output is zero. There is one scalar tank for the **entire
output vector**, with no allocation across channels and no sharing across batch
rows. `gamma > 0` is fixed, checkpointed, and absent from optimizer parameters.
Default initial energy is exactly zero; no learned initialization or refill exists.

In real arithmetic, for every prefix:

```text
E_{T+1} + sum(t=0..T) ||y_t||² = E_0 + gamma² sum(t=0..T) ||u_t||²
```

Consequently zero initial energy certifies a zero-state finite induced l2 gain
at most gamma, for every finite, well-defined causal core parameter choice.
Supplying nonzero initial energy gives the **additive E_0 allowance** above.
This does not imply hidden-state stability, contraction, incremental gain,
accuracy, or universality over gain-bounded operators. Outputs cannot prevent
hidden-state overflow.

## API and state

```python
import torch
from energy_bounded import EnergyBoundedModel, GRUCore, detach_state, state_to

model = EnergyBoundedModel(GRUCore(2, 3, width=24), gamma=1.0)
u = torch.randn(4, 100, 2)                       # batch, time, features
y, state = model(u)                             # fresh independent trajectories
a, state = model(u[:, :37])
b, state = model(u[:, 37:], state)               # preserve BOTH states
state = model.init_state(4)                      # explicit reset
sample, state = model.step(state, u[:, 0])       # repository convention: state first
y, state, info = model(u, diagnostics=True)
```

`WrapperState(core, energy)` contains an opaque core state and energy `(B,1)`.
The structural `CausalCore` protocol requires `d_input`, `d_output`,
`init_state(batch_size, device=..., dtype=...)`, `forward(u, state=None, mask=None)`
returning `(raw, updated_state)`, and `step(state, u, mask=None)`. Step inputs
are `(B,features)`; sequence inputs are strictly `(B,T,features)`, including `T=0`.
Arbitrary adapters can implement this contract; fixed-size tensor state is not
required. Modules retain no trajectory state between calls.

Keep state on the same device/dtype as the core. `state_to` moves tensor-tree
states, preserves boolean/integer metadata, and keeps reservoir energy at least
FP32. Moving FP64 state to MPS requires explicit `energy_dtype=torch.float32`.
These helper functions support tensors, containers and dataclasses; genuinely
custom objects need their own state adapter. Default wrapping does not inspect
or transform opaque state. `check_core_state=True` optionally audits supported
state trees for nonfinite values, adding work proportional to their size.

State remains attached to autograd across chunks. Use `detach_state(state)` only
at an intentional truncated-BPTT boundary. Diagnostics are opt-in and retain
energy `(B,T+1,1)`, raw norms `(B,T,1)`, clipping fraction, and actual backend.
Detach them explicitly when logging across training iterations.

`mask` is boolean `(B,T)`, **True for valid**, on the input device. Invalid
tokens pause logical time, emit zeros, neither inject nor spend energy, and do
not change semantic core state. Left padding, interior holes, empty rows, NaN
padding, and later resumption are supported. GRU/TCN use a per-row compacted
masked reference path; dense batches use native sequence kernels. Transformer
positions count valid tokens. Its physical cache retains padding slots, which
remain masked keys; memory therefore grows with physical context including pads.
Independent trajectories must be reset and must never be split across data splits.

## Reference, scan, and numerical limits

`EnergyReservoir(...)(u, v, energy=None, mode="reference"|"scan")` always returns
`(y, final_energy, diagnostics_or_None)`. Its `step(energy,u,v)` uses the reference.
The default sequence mode is the portable cumsum/cummin scan:

```text
d_t = a_t - q_t
S_t = E_init + cumsum(d)_t
E_{t+1} = S_t - min(0, S_0, ..., S_t)
```

Shift the energies to obtain `E_t`, then project outputs in parallel. The initial
zero in the running minimum is essential. This path has no Python time loop,
O(BT(m+p)) additional work, O(BT) scalar intermediates, and O(B) persistent
energy. Sequence-core complexity remains the core's own. The sequential fallback
retains the same work order but has Python/kernel-dispatch cost proportional to T.

`accumulation="auto"` uses FP64 norms/energy on CPU and CUDA, FP32 on MPS.
Explicit `float32`/`float64` policies are available. Outputs preserve raw FP32/FP64
dtype; FP16/BF16 proposals produce FP32 outputs. Under autocast, norms are cast
before squaring. Finiteness of valid inputs, raw outputs and reservoir arithmetic
is checked by default; overflow raises `FloatingPointError`. This can also detect
a core failure, but cannot prove that an unobserved internal state stayed finite.

Safe branches substitute **ones only inside inactive arithmetic expressions**,
never into the available budget. Positive square-root arguments and denominators
are selected before evaluation. Zero-budget output has zero gradient; the
unclipped branch (including positive-budget equality) uses the identity
gradient; exhaustion uses zero derivative. Running-minimum ties use PyTorch's
finite subgradient convention. Reference/scan gradients are checked away from
kinks; at kinks different finite subgradients can be chosen.

Clipped scaling is rounded inward by `1 - 4*eps(output_dtype)`. The tracked
recurrence can discard that rounding-size surplus rather than crediting it to
future outputs. No artificial energy is created. Accordingly the energy identity
is numerical, not bitwise: tests audit actual returned outputs in FP64 with
explicit normalized tolerances (typically 2e-6 for FP32, 1e-10 in stable FP64
cases; FP32 accumulation permits 2e-5). This is not interval-arithmetic certification.

The scan checks its local recurrence and a cumulative roundoff-risk envelope.
If cancellation is suspect, it **warns** and uses the reference, or raises with
`cancellation="raise"`. Tests cover 32,768-token scans, long saturation/refill,
and raw proposals of magnitude 1e100, which deliberately take the fallback.
FP32 energy can trigger fallback earlier on long/high-dynamic-range sequences.
CPU/CUDA FP64 is a precision/performance choice, not a hidden backend fallback.

An alternative associative pair is `(a,b)` representing `max(E+a,b)`:
`(a1,b1) star (a2,b2) = (a1+a2, max(b1+a2,b2))`, identity `(0,-inf)`.
Pairs `(d_t,0)` can avoid global subtraction, but no custom tree scan is included.
Installed PyTorch 2.9's prototype associative-scan documentation states no
autograd and CUDA code-generation restrictions. Portable differentiable ops
are used instead; measured forward/backward timings are in the report.

## Cores and costs

| Core | Dynamics and sequence path | Persistent streaming state / limitation |
| --- | --- | --- |
| `GRUCore` | Learned input/output projections and native `nn.GRU`; nonlinear gates | O(B*layers*width); sequential temporal dynamics |
| `TCNCore` | Residual dilated `Conv1d`, GELU, causal left history, no time normalization | Per-layer **input** buffers; receptive field `1+(kernel_size-1)*sum(dilations)`; default 15 valid tokens |
| `TransformerCore` | Pre-norm, sinusoidal absolute positions, supported SDPA, causal masks for multi-token chunks | Real per-layer KV caches and valid-token offsets, O(B*layers*context*width); no truncation/window |

Transformer full attention has quadratic context work; each cached step only
projects new tokens and attends to previous keys. Appending uses concatenation,
so it also copies O(context) cache elements per call. It does not recompute full
prefix representations. No in-place/preallocated inference cache optimization
is provided. Dropout defaults to zero; equivalence tests disable it. Nonzero
training dropout consumes randomness differently across chunk partitions.

## Run and existing trainer integration

Requirements: the repository's `torch>=2.2`, Python>=3.9; tests use standard
`unittest`. The measured host uses PyTorch 2.9.0/Python 3.13.9. No new external
data or dependency is required. Run from the **SSM-clean repository root**:

```bash
python -m unittest discover -s energy_bounded/tests -t . -v
python -m energy_bounded.experiment --steps 12 --repeats 3 --warmup 1 \
  --lengths 32 256 --contexts 16 128 --output energy_bounded/results/smoke.json
python -m energy_bounded.experiment --seeds 0 1 2 --steps 120 \
  --lengths 64 1024 --contexts 16 128 512 1024 --warmup 2 --repeats 7 \
  --output energy_bounded/results/comparison.json
python -m energy_bounded.summarize energy_bounded/results/comparison.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

`experiment.py` follows the existing contained-experiment convention, uses Adam,
and trains through the actual returned outputs. `BenchmarkAdapter` also connects
both raw and wrapped cores to the existing `SystemIDTrainer` without modifying
the benchmark registry:

```python
from pathlib import Path
from Test_files.Benchmark import SystemIDTrainer, TrainingConfig
from energy_bounded.benchmark_adapter import BenchmarkAdapter

trainer = SystemIDTrainer(
    BenchmarkAdapter(model),
    TrainingConfig(device="cpu", normalize_data=False,
                   save_dir=Path("energy_bounded/results/legacy_trainer")),
    normalizer=None,
)
# u_train/y_train and u_val/y_val are DIFFERENT complete zero-initialized trajectories.
trainer.train_epoch(u_train, y_train)             # legacy (T,features) accepted
trainer.validate(u_val, y_val, n=0)              # audit the whole trajectory
```

The legacy trainer bridge is tested with actual forward/backward and validation
for all six arms. Use separate complete trajectories; the generic CLI makes
trajectory-level splits explicit. Centering input or denormalizing output with
a nonzero mean changes the physical-unit certificate, so leave normalization
off unless the complete transformed map is analyzed. Existing benchmark factory
and UI choices remain untouched.

The synthetic teacher is a nonlinear zero-initialized contraction-filtered
system with a proven gain upper bound .9. Gamma is fixed at 1 **before fitting**.
There is no free initial-condition response. Training/validation/test use disjoint
trajectory IDs, and evaluation is deterministic. Longer trajectories measure
forced-response prediction, not autonomous autoregressive rollout. Raw/wrapped
pairs share initial parameter hashes and every training batch. All cores receive
the same width, optimizer, steps, and learning rate, with no tuning sweep; model
sizes and actual compute costs differ and are reported. This is a small initial
comparison, not evidence of a universal architecture winner.

See [REPORT.md](REPORT.md) and [results/comparison.json](results/comparison.json)
for errors, clipping, prefix audits, timings, throughput, cache growth, seed
variation, hardware, and the next measured optimization opportunity. Results
are CPU-only; CUDA and MPS were unavailable and their tests explicitly skip.

# Split memory with a contractive nonlinear reader

`split_memory/` contains this implementation, tests and validation artifacts.
The existing nonlinear benchmark runner and UI are reused. External edits are
limited to their registry/configuration/mask/checkpoint wiring and the shared
model-options registration, plus a plot-range correction so predictions outside
the target range remain visible. The energy-reservoir models remain separate.
This implements the supplied handoff, not a reproduction or an originality claim.

## Existing UI and commands

Restart the existing UI from the SSM-clean root:

```sh
python Test_files/benchmark_ui.py
```

Select **split_memory** once in **Shared run settings → Models**. Its tab under
**Model options** has the independent **Memory heads (2D)** and **Computation width**,
memory radius/initial retention, reader and memory evaluation choices. The
recurrent reader additionally shows rho, solver, and the solver's relevant
tolerance/fixed-sweep/maximum-sweep controls. **Require tolerance** is on by
default. Disabling it explicitly permits approximate training; evaluation still
reports residuals and convergence. Pointwise/linear ablations hide recurrence
options. Ordinary training/device/output options stay in the first box.

**Save as defaults**, **Save configuration…** and **Load configuration…** include
these parameters. Existing defaults/config files acquire the new defaults when
loaded. No personal default file is created by installation/tests. Results,
prediction/loss plots and diagnostics use the existing saved-run views.

For a reproducible one-epoch wiring check, load `configs/ui-smoke.json` in the UI,
or use the same existing runner:

```sh
python Test_files/run_benchmarks.py --config split_memory/configs/ui-smoke.json
```

Its Cascaded Tanks job uses native whole-trajectory train/validation splits,
CPU FP32, seed 0, two memory heads, computation width 4, rho .7, M=1, initial
retention .95, 32 fixed sweeps, tolerance 1e-4, parameter matching off and no GIF.
This is an integration check, not an accuracy comparison or recommended fit.
The UI's usual parameter-budget default can increase the actual reader width;
the report/checkpoint records the effective dimensions. For the linear ablation,
parameter matching adjusts head count instead of the unused computation width.

Examples through the existing runner:

```sh
python Test_files/run_benchmarks.py --models split_memory --benchmarks Cascaded_Tanks \
  --epochs 1 --param-budget off --split-memory-heads 4 --split-computation-width 16 \
  --split-solver adaptive --split-tolerance 1e-4 --split-max-sweeps 64 \
  --split-diagnostics --device cpu --no-show --no-gif --out split_memory/validation/example

# Same parametrization, exact sequential evaluation instead of Jacobi:
python Test_files/run_benchmarks.py --models split_memory --epochs 1 \
  --split-solver sequential --split-memory-backend sequential --no-show --no-gif
```

Reader choices are `recurrent`, `pointwise`, `linear`; memory backends are `scan`
and `sequential`. Solver choices are `fixed`, `adaptive`, `sequential`.
No additional training entry point, dataset, dashboard or comparison campaign is
introduced. Nothing automatically attaches an energy wrapper.

## Equations and defaults

For each of H heads, the input alone produces

```
a = sigmoid(gate(x))
theta = pi * tanh(angle(x))
v = M*z/sqrt(1 + ||z||_2^2)
A = a * R(theta)
b = (1-a)*v
m_after = A*m_before + b
u = P_m*flatten(m_after) + P_x*x + bias
c_after = tanh(W_c*c_before + u)
y = P_out*c_after + output_bias
```

One block is implemented. Persistent states are `m: (B,H,2)` and `c: (B,d_c)`;
default initial states are zero. H and d_c are independent. The default is H=8,
d_c=16, M=1, rho=.7, retention=.95, row delta=1e-4, fixed 32 sweeps, tolerance
1e-4, adaptive cap 64, and required convergence. These are starting settings,
not empirically established recommendations. No initial state is learned.

`||v_j||_2 <= M` implies each head stays within
`max(||m_init,j||_2,M)` in real arithmetic. The flattened memory has the
corresponding sum-of-squared-head-bounds limit. A configured a=1 is isometric
with no write; an ordinary fixed sigmoid gate below one eventually forgets.
Unrestricted additive/isometric writes are not implemented.

```
W_c[i,j] = rho_effective * V[i,j] / (delta + sum_k |V[i,k]|)
rho_effective = rho * (1 - 8*machine_epsilon)
```

This gives `max_i sum_j |W_c[i,j]| <= rho < 1` in real arithmetic. Tanh is
1-Lipschitz, so the computation update is uniformly rho-contractive **in c**,
with input/memory held fixed. Gradients flow through the row normalization.
Rowwise algebraic rescaling prevents overflow in its norm calculation; the
proposal normalization similarly avoids squaring huge z. Sensitive arithmetic
uses FP32 minimum (FP64 for double models/inputs), with autocast disabled for
the block. FP16/BF16 parameters/inputs therefore do not make the state arithmetic
low precision. There is no CPU fallback hidden inside model evaluation.

Initialization uses a unit diagonal raw V plus Gaussian mixing with std `.02/d_c`,
preserving meaningful self-recurrence. P_m/P_x weights have std
`.1/sqrt(input_width)` and zero forcing bias. Gate/angle weights have std
`.05/sqrt(d_in)`; gate bias is logit(initial retention), angle bias zero. Write
weights have std `.3/sqrt(d_in)` and zero bias. Readout weights have std
`.5/sqrt(readout_width)` and zero bias. These choices aim to avoid immediate tanh
saturation; their modeling quality has not been established.

## Memory scan, state and masks

Affine composition applies pair 1 then pair 2:

```
(A1,b1) star (A2,b2) = (A2*A1, A2*b1+b2)
m_after[t] = A_prefix[t]*m_init + b_prefix[t]
```

Transitions are stored as real `(a*cos(theta),a*sin(theta))` pairs; complex
multiplication is expressed using real tensors. No dense 2H-by-2H matrices are
materialized. The existing recursive adjacent-pair associative scan is reused:
O(BTH) arithmetic/scalar storage and O(log T) levels. It is not a prototype
PyTorch associative-scan API. Tree order can differ numerically from the loop.

```python
import torch
from split_memory import SplitMemoryConfig, SplitMemoryContractiveRNN
from split_memory import detach_state, state_to

model = SplitMemoryContractiveRNN(2, 3, SplitMemoryConfig())
x = torch.randn(4, 100, 2)
y, state = model(x)                 # configured sequence solver
y_ref, exact_state = model.reference(x)
y_next, state = model.step(state, torch.randn(4, 2))  # exact one-token recurrence
y_diag, state, info = model(x, diagnostics=True)
```

`forward(x,state=None,mask=None)` returns output/state; diagnostics or
`return_trajectory=True` add a third dictionary. The latter explicitly includes
memory/computation traces. Otherwise traces are not retained as diagnostics.
`init_state(batch,device=...,dtype=...)` / `reset` initialize both states.
`step(state,x)` follows the repository's state-first single-step convention.
`detach_state` is an explicit truncated-BPTT choice; nothing silently detaches
state or scanned memory. `state_to` preserves at least FP32 state precision.

Both states carry across chunks. Exact reference full/chunk/step paths implement
the same causal recurrence within floating tolerance. A step performs one memory
update and one computation-cell evaluation per valid token, with no whole-prefix
solver/recomputation. Persistent streaming state costs O(B(H+d_c)).

Boolean masks have shape (B,T). Invalid inputs, including NaN padding, are
ignored. Suffix padding is excluded from stopping norms and final-state gathering;
an empty sample preserves its incoming state. Internal/left gaps trigger a
reported `sequential_mask_fallback` for the batch, following packed-event pause
semantics. No strict contraction is asserted over identity padding transitions.
Masks do not reset trajectories; reset explicitly at independent boundaries.
The suite's initialization window masks only scored loss/metrics, not the genuine
conditioning inputs or valid model updates. Native trajectory splits are reused.

## Solvers, residuals and gradient semantics

Fixed Jacobi starts at C=0 and performs K full-time sweeps:

```
predecessor = concat(c_init[:,None,:], C[:,:-1,:])
C_next = tanh(predecessor @ W_c.T + u)
```

Memory, u and W_c are computed once outside the sweep loop. Incoming c is clamped
at every sweep; no circular roll or future-dependent warm start is used. All
returned c values lie in [-1,1], so zero guesses have E0<=1. A conservative
sufficient K is `ceil(log(1/tol)/(-log(rho)))`; rho=0 needs one sweep and tol>=1
allows zero. K>=T recovers the valid sequential trajectory in exact arithmetic.
Fixed mode has no device synchronization inside its K loop.

A final recurrence evaluation measures the residual of the **returned** iterate:

```
R = max_valid |C - tanh(W_c*predecessor(C) + u)|
state_error_bound = R/(1-rho)
```

That pass is counted. Diagnostics report sweeps, `cell_passes` (evaluations per
token position), `cell_evaluations` (B*T times passes, including evaluated padding),
and `valid_cell_evaluations` (valid tokens times passes). Dense kernels evaluate
padding proposals but discard their updates and exclude them from residuals. Sequential
evaluation uses one pass, plus a diagnostic verification pass when requested.
Online `step` uses exactly one. Required convergence raises `ConvergenceError`
with diagnostics if the bound exceeds tolerance; the existing benchmark's error
row includes those diagnostics. Disabling required convergence is explicit.
An unchecked fixed solve reports convergence/bound as unknown, never achieved.

Adaptive mode reuses each verification proposal as the next iterate. It freezes
only **causal prefixes** whose inclusive prefix maximum residual is at most
`tol*(1-rho)`. All predecessors freeze with a frozen token, preserving its
residual certificate and preventing future inputs/batch peers from choosing a
different past iterate. Thus it is not simply a future-dependent global stopping
count applied to the whole returned sequence. It synchronizes per iteration,
and may cost more than fixed sweeps. At the cap it reports the measured residual
and success/failure. Its initial zero iterate has the additional E0<=1 bound,
which is used explicitly for tol>=1. The last verification certifies the old
returned iterate; sweeps count accepted updates and passes count sweeps+1.

Residual bounds assume exact memory. With flattened-memory infinity-norm error
delta_m, add `||P_m||_inf*delta_m` to R before dividing by 1-rho. The measured
memory-to-computation row-sum norm is reported. Finite-precision residuals, row
sums and memory-bound deviations are diagnostics, **not interval arithmetic**.

Finite-K full and chunkwise Jacobi are different truncated graphs. They approximate
the same recurrence; compare assembled residuals/tolerances rather than requiring
bitwise chunk invariance. A chunk's incoming c may itself be approximate.

All training uses ordinary autograd through memory preprocessing/scan, the actual
finite sweeps and readout. There is **no implicit backward, Newton or DEER**.
Adaptive prefix-freezing/stopping decisions are discrete and not differentiated;
adaptive training is supported with that explicit meaning. Forward agreement
alone does not ensure correct fully recurrent parameter gradients (e.g. a zero
initial guess can already match a zero-valued target trajectory). Use fixed K>=T
or sequential BPTT for exact small gradient checks. No T-independent parameter-
gradient error bound is claimed.

## Ablations and checkpoints

The pointwise reader uses `c=tanh(u)` over the same memory, with no W_c.
The linear ablation reads `P_out*flatten(m)` directly; its unused c is carried
unchanged. They are configuration variants under one UI model entry.

`split_memory_v1` checkpoints store input/output dimensions, all architecture and
solver configuration, effective width/head count, parameter dtype/weights,
normalizer, training history and metadata. They reconstruct via:

```python
from split_memory.benchmark import load_checkpoint
model, payload = load_checkpoint("path/to/model.pt", solver="sequential")
```

Solver/backend overrides reuse exactly the same weights. Other checkpoint tags
are rejected rather than reinterpreted. Existing energy/SSM checkpoint formats
are unchanged. The generic Python `EnergyBoundedModel(core, gamma=...)` adapter
can wrap this core explicitly if selected; that optional composition is tested,
but is not the default or a new UI combination.

## Efficiency, support and limitations

There is no measured speed/accuracy/novelty advantage. Idealized temporal span
is O(log T+K); actual fixed hardware pays linear-in-T work and memory traffic.
Ignoring projections, dense recurrent work is O(BTK*d_c^2). No-grad working
storage is O(BT(H+d_c)); ordinary sweep autograd can retain O(BTK*d_c)
activations. Step evaluation also recomputes weight normalization. Profile the
scan, projections, recurrent sweeps and backward before choosing a fused kernel,
cached inference parametrization, checkpointing or another solver.

The pure fixed kernel is tested with `torch.compile(..., backend='aot_eager',
fullgraph=True)` for outputs/gradients. The suite explicitly skips whole-model
compilation because validation, diagnostics and adaptive stopping remain eager.
This does not establish Inductor/fused-kernel performance. CUDA/MPS and visual
UI validation status, actual smoke metrics/timings and tests are in VALIDATION.md.

Only c is uniformly contractive. Bounded memory need not forget uniformly.
The model has no prescribed l2-gain certificate, long-horizon universality,
robust length-independent symbolic tracking guarantee or proven prediction
advantage. A single memory/reader cascade can remain nonexpansive in a weighted
product norm when the coupling is uniformly Lipschitz; the split does not
automatically escape a nonexpansive tracking obstruction. Stacks require their
own analysis. A fixed K reader is also a depth-K causal, weight-tied temporal
feedforward graph over memory features; future comparisons must isolate added
depth from recurrence. Iteration counts are T-independent only for the stated
bounded-initialization uniform state-error criterion. Whole-trajectory l2 error
and parameter-gradient accuracy require separate accounting. Memory retention
does not prevent interference, numerical drift, poor learned selection or hidden
overflow; valid inputs, incoming/memory states, normalized weights, projected forcing and
returned outputs are checked for nonfinite values; this is not a proof excluding
every possible internal floating-point overflow.

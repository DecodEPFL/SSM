# neural-ssm

**Train state-space sequence models with PyTorch, with optional guarantees on their input/output gain.**

Use `neural-ssm` for nonlinear system identification, streaming predictions, and
learning models for control. Start with a small recurrent model, train it with
ordinary PyTorch operations, then add a gain bound or context when you need it.

[First model](#your-first-model) · [Gain bounds](#add-a-gain-bound) ·
[Model choices](#choose-a-recurrent-core) · [Metrics](#learn-metrics-for-the-whole-stack) ·
[Context](#add-context) · [Tutorials](#continue-with-the-tutorials)

## Install

The core package requires Python 3.9+ and PyTorch 2.2+.
From a terminal:

```bash
git clone https://github.com/DecodEPFL/SSM.git
cd SSM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the examples below in Python. They use small synthetic sequences and require
no dataset download. For the plotting and benchmark tools, also install:

```bash
python -m pip install -e ".[experiments]"
```

For CUDA, install the PyTorch build matching your machine before installing the
package. The project does not install a CUDA runtime itself.

## Your first model

Follow these steps in order, in the same Python session.

### 1. Create a short sequence

```python
import torch
from neural_ssm import DeepSSM

u = torch.randn(1, 8, 1)  # one sequence, eight time steps, one input feature
```

Every sequence uses the shape **`(batch, time, features)`**. For example,
`(4, 100, 3)` means four independent sequences, each with 100 samples and three
input features.

### 2. Build a model and obtain predictions

```python
model = DeepSSM(
    d_input=1, d_output=1, param="lru",
    d_model=4, d_state=8, n_layers=1,
)
y, state = model(u)
print(y.shape)  # torch.Size([1, 8, 1]): one prediction per time step
```

This is a stable LRU recurrent baseline. Its weights are trainable; it does not
have a prescribed whole-model gain bound.

| Setting | Meaning in this example |
| --- | --- |
| `d_input=1`, `d_output=1` | One input and one output value per sample |
| `d_model=4` | Four features inside the network |
| `d_state=8` | Eight memory coordinates in each recurrent layer |
| `n_layers=1` | One recurrent block |

`state` contains one final memory tensor per layer. Here, `state[0]` has shape
`(1, 8)`. LRU memory is complex-valued; pass the returned tensor back unchanged.

### 3. Train with a familiar PyTorch loop

For a tiny example, ask the model to predict `0.3` times its input:

```python
target = 0.3 * u  # replace this with your measured output sequence
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
model.train()

for step in range(20):
    optimizer.zero_grad(set_to_none=True)  # clear previous gradients
    prediction, _ = model(u)              # start a fresh sequence
    loss = (prediction - target).square().mean()
    loss.backward()                       # differentiate through the sequence
    optimizer.step()                      # update the weights
```

With your own data, supply input and target tensors with matching batch and time
axes. The target's last dimension is `d_output`.

### 4. Continue a sequence across calls

Pass the first chunk's final state into the second chunk:

```python
model.eval()  # disable training-time dropout
with torch.no_grad():
    first, state = model(u[:, :4], detach_state=True)
    second, state = model(u[:, 4:], state=state, detach_state=True)
y_streamed = torch.cat((first, second), dim=1)
```

This produces the same predictions as processing the complete sequence, up to
floating-point differences. An explicit `state` continues from that memory;
without it, a call starts a fresh trajectory by default.

For training across chunks, keep `detach_state=False` (the default). Use `True`
at a truncated-backpropagation boundary to discard the preceding gradient
history. `reset_state=False` instead continues from the model's stored internal
state; `model.reset()` clears that state.

## Add a gain bound

Keep the same dimensions, and choose a certified recurrent core and bounded
feedforward branch:

```python
bounded = DeepSSM(
    1, 1, d_model=4, d_state=8, n_layers=1,
    param="defect", ff="LGLU2", gamma=1.0,
)
y, state = bounded(u)
print(bounded.certified_gain_bound().item())  # at most the requested gamma
```

`gamma=1.0` means that, **starting from zero state**, the output sequence's
squared energy cannot exceed the input sequence's squared energy. More generally:

$$
\sum_t \|y_t\|^2 \leq \gamma^2 \sum_t \|u_t\|^2.
$$

This is an upper bound enforced by the parametrization throughout training.
It is not an estimate from sampled inputs, and the model's actual gain can be
smaller. It bounds the full sequence, rather than each output sample separately.

Use a certified core from the table below and one of `LGLU2`, `BLGLU2`
(`BudgetedLGLU2` is an alias), `MBLIP`, or `TLIP` for `ff`. Ordinary `MLP`/`GLU`
and the historical `LGLU`/`LMLP` choices require `gamma=None`. Keep
`learn_x0=False` for the zero-state statement; a nonzero initial state needs an
additional storage-energy term, available through `storage_value(state)`.

A zero-state gain bound and an incremental bound are different guarantees:
the latter controls differences between two input/output trajectories.
`MetricDeepSSM` below provides an incremental certificate. Input-dependent
selective cells do not automatically provide that stronger guarantee.

## Choose a recurrent core

Each `DeepSSM` block combines a recurrence that carries memory with a feedforward
branch that mixes features at each time step. `param` selects the recurrence;
`ff` selects the feedforward. LTI cores keep their matrices fixed over time;
selective cores can change their matrices at each sample.

The following modes apply to `DeepSSM`:

| `param` | What it provides | Execution modes |
| --- | --- | --- |
| `lru` | Stable complex diagonal LTI baseline; use `gamma=None` | `scan`, `conv`, `loop` |
| `defect` | Certified real-block LTI core using thin defect completion | `scan`, `loop` |
| `l2n` | Certified LTI core using full-system norm normalization | `scan`, `loop`; `conv` with identity storage only |
| `tv` | Certified selective diagonal dynamics depending on input/context | `scan`, `loop` |
| `tvc` | Selective dynamics with an additional direct input/output term | `scan`, `loop` |
| `l2ru`, `zak` | Legacy certified LTI constructions | `scan`, `loop` |
| `l2nt` | Legacy dense normalized LTI construction | `loop` |

`scan` evaluates the recurrence in parallel over time. `loop` is the sequential
reference. Where supported, `conv` evaluates the LTI response using FFT
convolution. Compare supported modes on your sequence sizes and device.

To reuse settings across models, collect them in a configuration:

```python
from dataclasses import replace
from neural_ssm import SSMConfig

config = SSMConfig(d_model=4, d_state=8, n_layers=1,
                   param="defect", ff="LGLU2", gamma=1.0)
normalized = DeepSSM(1, 1, config=replace(config, param="l2n"))
selective = DeepSSM(1, 1, config=replace(config, param="tv"))
```

When you pass `config`, it supplies the model settings; make changes in the
configuration itself. `gamma` is the fixed whole-stack target. `train_gamma`
controls the cells' internal gain parameters, and `train_ff_lip` controls the
feedforward gain parameters.

### Allow dense state storage while keeping block scan

The state metric $P$ measures energy in the recurrent memory. Both `defect`
and `l2n` support a full, learned $P$ without making their recurrence matrix
$A$ dense:

```python
full_l2n = DeepSSM(1, 1, config=replace(config,
    param="l2n", l2n_state_metric="full"))
full_defect = DeepSSM(1, 1, config=replace(config,
    defect_state_metric="full", defect_max_radius=1.0, defect_factor_margin=0.0))
```

Identity storage is the default and has cheaper matrix preparation. Full storage
allows the small blocks to share a dense certificate, covering generic strictly
gain-bounded LTI input/output maps. This statement concerns the **LTI core**;
it does not claim completeness of the nonlinear network. Exact larger Jordan
chains may require larger blocks. The defect settings above remove fixed pole
and contraction margins from that construction.

Identity L2N uses rotation pairs and requires even `d_state`. Full L2N uses
general real 2×2 blocks and accepts an odd state size with a final scalar; use
`scan` or `loop`. Defect's default block size is two. Its full storage supports
block sizes one through four, with a final scalar for odd sizes when the block
size is two. Larger identity-storage blocks are also available through
`defect_block_size`.

Full storage adds dense preparation and backward work; the small-block recurrence
stays fast. See the [defect construction](docs/defect_parametrization.md) and
[measured full-metric comparison](docs/benchmarks/l2n_full_metric_cpu_2026-10-09.md)
for the details and computational tradeoffs.

## Learn metrics for the whole stack

`MetricDeepSSM` learns feature-energy metrics $Q$ shared between neighbouring
layers. This can reduce the conservatism of multiplying separate scalar layer
bounds, and gives a prescribed **incremental** gain for the complete sequence
map. Here, `gamma=1.0` bounds output differences by input differences when the
two executions start from equal states.

```python
from neural_ssm import MetricDeepSSM

metric = MetricDeepSSM(
    1, 1, d_model=4, d_state=8, n_layers=2,
    param="defect", gamma=1.0, state_metric="full",
)
y = metric(u, return_state=False)  # predictions only
```

This is a separate architecture: an LTI core, an activation, and an optional
coupled residual feedforward in each layer. Built-in cores are `defect` and
`l2n`; `ff` is `residual` (the default) or `none`. Activations are `tanh` (the
default), `relu`, `arctan`, and `identity`. Execution modes are `scan` and `loop`.
Use `MetricSSMConfig` when collecting these settings in a configuration.

The feature metric $Q$ and state metric $P$ have different jobs. Feature
metrics are learned by this stack; `state_metric="full"` additionally enables
dense state storage inside either LTI core. A tighter certificate does not
promise an exact gain calculation or better prediction accuracy.

State returns differ from the ordinary stack:

| Call | Return |
| --- | --- |
| `model(u)` or `controller(u, z)` | `(y, final_states)` |
| `metric(u)` | `(y, state_trajectories)`, including the initial state |
| `metric(u, return_state=False)` | `y` |
| `metric(u, return_state=False, return_last=True)` | `(y, final_states)` for streaming |

For streaming a metric model, pass those final states into the next call with
`state=final_states`; the same `detach_state` choices apply. Keep certificate
parameters in float32 or float64. Read the
[metric tutorial](Test_files/Tutorial_MetricDeepSSM.py) for a gradual introduction,
or the [architecture guide](docs/metric_transfer_ssm.md) for the equations.

## Add context

Context is a second sequence, such as an operating condition or reference.
Start by adding it as an extra input feature:

```python
from neural_ssm import ContextualDeepSSM

z = torch.ones(1, 8, 1)  # same batch and time axes as u
controller = ContextualDeepSSM(
    1, 1, 1, ssm_config=config,
    context_modes="input", context_filter="none",
)
y, state = controller(u, z)
```

The positional dimensions are `d_input`, `d_context`, and `d_output`. Here,
`"none"` leaves context unchanged. With the `input` port, the gain certificate
applies to the **combined input** `[u, filtered_context]`; context contributes
its own energy and may produce output even when `u` is zero.

Introduce the other ports one at a time, then combine them using a tuple such
as `context_modes=("gate", "select")`:

| Port | What context changes |
| --- | --- |
| `input` | Adds filtered context to the input features |
| `gate` | Attenuates residual branches with gates in `[0, 1]` |
| `mixer` | Applies a context-dependent output matrix with a fixed norm cap |
| `select` | Schedules the recurrent dynamics; requires `tv` or `tvc` |

The [context tutorial](Test_files/Tutorial_ContextualSSM.py) covers these ports,
finite/decaying windows, difference filtering, context encoders, and streaming.
For chunked filtering, `time_offset` counts samples already processed;
difference filtering also needs the previous raw context sample via
`previous_context`.

A fixed-context incremental certificate needs `select_input="context"`,
context-only gates/mixing (`gate_include_disturbance=False`,
`mixer_include_disturbance=False`), and no additive `input` port. Both runs must
use the same exogenous context and initial state. Check
`controller.incremental_gain_bound()` for this guarantee and
`controller.gain_diagnostics()` for the zero-state certificate and filter details.
These diagnostics do not establish stability of an arbitrary feedback loop.

## Run efficiently

Move the model and its inputs to the same device. For inference, combine
`eval()` with disabled gradients to reuse prepared matrices safely:

```python
model.eval()
with torch.inference_mode():
    y, state = model(u)
```

Training and evaluation with gradients rebuild the required computation graph.
Caches are invalidated after parameter updates, checkpoint loads, or dtype/device
changes; see [parameter caching](docs/parameter_cache.md).

`torch.compile(model)` is optional and has a warm-up cost. Keep shapes stable
and measure it on your hardware. Compile regression tests cover the main stacks;
they do not guarantee a speedup with every backend. For fixed-shape CUDA
selective scans, set `use_cuda_graph=True` in `SSMConfig`. `REN` remains eager.

## Continue with the tutorials

The lessons start with small examples and add one concept at a time. Each uses
numbered `# %%` cells, which supporting editors can run individually.

| Start here | What you will learn |
| --- | --- |
| [DeepSSM tutorial](Test_files/Tutorial_DeepSSM.py) | Construction, training, streaming, current/legacy cores, and feedforward choices |
| [Context tutorial](Test_files/Tutorial_ContextualSSM.py) | Each context port separately, filters, streaming, and combinations |
| [Metric tutorial](Test_files/Tutorial_MetricDeepSSM.py) | Shared metrics, state returns, training, and energy checks |

Run a complete lesson from the repository root:

```bash
python Test_files/Tutorial_DeepSSM.py
python Test_files/Tutorial_ContextualSSM.py
python Test_files/Tutorial_MetricDeepSSM.py
```

The [written walkthrough](docs/ssm_tutorial.md) follows the same progression.

## Run experiments

With the `experiments` extra installed, discover available datasets and models:

```bash
python Test_files/run_benchmarks.py --list
```

Then run a short CPU comparison on the Cascaded Tanks benchmark:

```bash
python Test_files/run_benchmarks.py \
  --benchmarks Cascaded_Tanks --models defect l2n metric_defect metric_l2n \
  --device cpu --epochs 10
```

The runner downloads the selected benchmark data, evaluates the official
validation split, and saves figures and reports under `Test_files/benchmark_runs/`.
Use `--device cuda` on a CUDA machine. `--mode auto` selects convolution for LRU
and identity L2N, and scan for the other current SSMs. Add
`--l2n-state-metric full --defect-state-metric full` to enable dense storage in
both ordinary and metric models. `--metric-ff` controls the metric stack's
feedforward independently of `--ff`.

For a desktop form, run `python Test_files/benchmark_ui.py` (requires Tk support).
For the standalone delayed-cue robot navigation task, click **Delayed navigation…**
in that window, or run `python -m delayed_navigation.ui` directly. Load
[`delayed_navigation/configs/ui-l2n-comparison.json`](delayed_navigation/configs/ui-l2n-comparison.json)
in the example window. It exports final PNG/PDF comparisons and optional GIFs.
The [navigation guide](delayed_navigation/README.md) explains the exact tank,
bistable memory, GRU/LSTM/L2RU/L2N/REN baselines, metrics and initial CPU results.

For a synthetic four-way timing comparison of identity/full L2N and defect,
run `python scripts/compare_defect_vs_normalization.py`. Add
`--l2n-state-metric full --defect-state-metric full` to compare only full storage.
The retained [L2RU/ZAK comparison](scripts/compare_l2ru_vs_zak.py) covers legacy
constructions.

## Package map and development

```text
src/neural_ssm/
├── ssm/
│   ├── config.py, registry.py   shared settings and model construction
│   ├── layers.py               DeepSSM and residual blocks
│   ├── metric_layers.py        MetricDeepSSM and its feature metrics
│   ├── contextual.py           context ports and filters
│   └── cells/                  lti/, selective/, legacy/
├── static_layers/              ordinary and bounded feedforward layers
├── rens/ren.py                 robust acyclic recurrent equilibrium network
└── utils/                      scan, Lyapunov, state, and cache helpers
```

Public imports include `DeepSSM`, `SSMConfig`, `MetricDeepSSM`, `MetricSSMConfig`,
`ContextualDeepSSM`, and `REN`. Standalone cells are also exported from
`neural_ssm`; see the [package guide](docs/package_structure.md) for canonical
imports, compatibility aliases, and adding a cell.

For development:

```bash
python -m pip install -e ".[dev,experiments]"
python -m pytest
```

## Research reference

For the original L2-bounded SSM parametrizations, cite Leonardo Massai,
Muhammad Zakwan, and Giancarlo Ferrari-Trecate,
[**L2RU: a Structured State Space Model with prescribed L2-bound**](https://arxiv.org/abs/2503.23818).

The metric-transfer approach has a separate precedent in
[**LipSSM: Structurally Lipschitz-Bounded Cascaded State-Space Model via Metric Transfer between Consecutive SSM Layers**](https://arxiv.org/abs/2609.30973).
The package's current core choices and metric residual layers extend beyond the
original L2RU paper; consult the linked construction guides for their assumptions.

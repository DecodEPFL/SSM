# Build your first SSM, one step at a time

Start with [Tutorial_DeepSSM.py](../Test_files/Tutorial_DeepSSM.py). Read its
numbered sections in order. Each section runs directly and explains the lines
it introduces. Editors that support `# %%` cells can run one section at a time.
The examples use tiny synthetic sequences and need no dataset download.

Install once from the project root:

```bash
python -m pip install -e .
```

## 1. Make a short input sequence

Each sample has one input value. We put eight samples into one sequence:

```python
import torch
from neural_ssm import DeepSSM

u = torch.randn(1, 8, 1)  # (number of sequences, time steps, input features)
```

The batch dimension lets you process several independent sequences together.
For example, `(4, 8, 1)` would mean four sequences of length eight.

## 2. Build a model and obtain predictions

Start with the stable LRU baseline and one input/output feature:

```python
model = DeepSSM(d_input=1, d_output=1, param="lru")
y, state = model(u)
print(y.shape)  # (1, 8, 1): one prediction for every input sample
```

`y` contains predictions. `state` contains the final recurrent memory of each
layer; we will use it when continuing the sequence in step 6.

## 3. Choose how much memory the model has

`d_model` sets the feature width between layers. `d_state` sets each layer's
recurrent memory size. `n_layers` sets how many recurrent blocks to stack:

```python
model = DeepSSM(
    d_input=1, d_output=1, param="lru",
    d_model=4, d_state=8, n_layers=1,
)
y, state = model(u)
print(state[0].shape)  # (1, 8): one sequence, eight memory coordinates
```

Input/output width, feature width, and memory size can all be different.

## 4. Request a whole-model gain bound

Choose a certified cell and bounded feedforward branch together:

```python
model = DeepSSM(
    d_input=1, d_output=1, d_model=4, d_state=8, n_layers=1,
    param="defect", ff="LGLU2", gamma=1.0,
)
print(model.certified_gain_bound())
```

Starting from zero state, `gamma=1.0` bounds the sequence's output energy by its
input energy. The LRU baseline uses `gamma=None`; it does not expose a prescribed
whole-stack gain certificate.

To preserve generic full LTI expressivity while keeping a fast block scan, let
different memory blocks share a full state metric:

```python
model = DeepSSM(
    1, 1, d_model=4, d_state=8, n_layers=1,
    param="defect", ff="LGLU2", gamma=1.0,
    defect_state_metric="full",  # learn the shared memory energy P
    defect_max_radius=1.0,       # allow poles anywhere strictly inside the unit disk
    defect_factor_margin=0.0,    # allow the full open range of thin contractions
)
y, state = model(u, mode="scan")
```

Full storage adds dense matrix construction once per forward, outside the time
loop. The original cheaper option is `defect_state_metric="identity"` (the
default); with small blocks it restricts the LTI family at a prescribed gain.
Full storage supports blocks of size one through four. With size two, an odd
`d_state` uses a final real scalar. Exact larger Jordan chains require a larger
realization block; the generic modal systems are covered by size two.

Returned states stay in the block recurrence coordinates. Their energy need
not be their squared norm:

```python
core = model.blocks[0].lru
P = core.storage_matrix()                  # inspect this layer's learned metric
energy = core.state_energy(state[0])       # x.T @ P @ x for each sequence
stack_energy = model.storage_value(state)  # include the stack's storage weights
```

The LTI-core completeness statement covers strict gain below the prescribed
bound; it does not assert completeness of the nonlinear residual stack. A limit
such as `defect_max_radius=0.999` intentionally excludes some near-unit poles.
At open limits, the implementation only guards saturation at machine precision.

## 5. Train with ordinary PyTorch operations

Here the desired output is simply `0.3` times the input. Replace this target
with your measured labels when using your own data:

```python
train_u = torch.randn(4, 32, 1)  # four training sequences
train_y = 0.3 * train_u
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
model.train()
```

One training update is five lines:

```python
optimizer.zero_grad(set_to_none=True)  # discard the previous gradients
prediction, _ = model(train_u)        # start each sequence from its initial state
loss = (prediction - train_y).square().mean()  # mean squared error
loss.backward()                       # compute gradients
optimizer.step()                      # update parameters
```

The runnable tutorial repeats this update in a small loop. Later sections show
how to train through connected chunks and construct a target that needs memory.

## 6. Continue through a second chunk

Use `eval()` to disable dropout and `no_grad()` when gradients are unnecessary:

```python
model.eval()
with torch.no_grad():
    first, state = model(u[:, :4], detach_state=True)
    second, state = model(u[:, 4:], state=state, detach_state=True)
y = torch.cat((first, second), dim=1)
```

The returned state is the bridge between chunks. An explicit `state` overrides
the reset setting; without one, the default call starts a fresh trajectory.
For training through the boundary, use `detach_state=False`. Use `True` at a
truncated-backpropagation boundary or for inference. `DeepSSM` defaults to
`False`, so the examples pass this choice explicitly.

## 7. Change one model choice at a time

Once the first example is clear, collect its repeated settings in a configuration:

```python
from dataclasses import replace
from neural_ssm import SSMConfig

config = SSMConfig(d_model=4, d_state=8, n_layers=1,
                   param="defect", ff="LGLU2", gamma=1.0)
```

Then change only the recurrent cell:

```python
l2n = DeepSSM(1, 1, config=replace(config, param="l2n"))
y, state = l2n(u)
```

Next, enable a dense certificate inside normalization. The block scan still
runs in the original state coordinates:

```python
full_l2n = DeepSSM(1, 1, config=replace(config, param="l2n", l2n_state_metric="full"))
y, state = full_l2n(u, mode="scan")
```

This mode learns general real blocks and dense state storage, covering generic
strictly gain-bounded LTI maps. It also supports odd state sizes. Matrix
preparation and backpropagation cost more; inference caches the prepared
matrices. Use `scan` or `loop` for this mode. The original identity L2N keeps
its rotation blocks and FFT convolution. `MetricDeepSSM` exposes the same choice
through `state_metric="full"`, and a contextual stack inherits
`SSMConfig.l2n_state_metric` from its core configuration.

Or keep the cell and change the nonlinear branch:

```python
mixed = DeepSSM(1, 1, config=replace(config, ff="MBLIP"))
y, state = mixed(u)
```

The later sections of `Tutorial_DeepSSM.py` give explicit constructions for
current and legacy cells, larger defect blocks, bounded and ordinary feedforward
branches, and deeper stacks. These tables are a reference for those examples:

| `param` | What it changes | Execution modes |
| --- | --- | --- |
| `lru` | Stable complex diagonal LTI baseline | `scan`, `conv`, `loop` |
| `l2n` | Identity: even rotation pairs. Full: general blocks and optional scalar | `scan`, `loop`; `conv` for identity only |
| `defect` | Thin defect-completion LTI; identity or full state metric | `scan`, `loop` |
| `tv` | Input/context-dependent diagonal dynamics | `scan`, `loop` |
| `tvc` | Selective dynamics with a direct term | `scan`, `loop` |
| `l2ru`, `zak` | Legacy certified LTI constructions | `scan`, `loop` |
| `l2nt` | Legacy dense normalized LTI construction | `loop` |

| `ff` | Whole-stack certificate with `gamma` |
| --- | --- |
| `LGLU2` | yes |
| `BLGLU2` / `BudgetedLGLU2` | yes |
| `MBLIP`, `TLIP` | yes |
| `MLP`, `GLU` | use `gamma=None` |
| Historical `LGLU`, `LMLP` | use `gamma=None` |

`gamma` is the fixed whole-stack target. `train_gamma` controls the cells'
internal gains; `train_ff_lip` controls feedforward budgets. Set either to
`False` to fix the corresponding internal budget. Keep `learn_x0=False` for
the pure zero-state gain statement. A nonzero initial state adds initial
storage energy that must be accounted for separately.

## 8. Add a second sequence for context

Read [Tutorial_ContextualSSM.py](../Test_files/Tutorial_ContextualSSM.py) next.
Its first example simply adds context alongside the ordinary input:

```python
from neural_ssm import ContextualDeepSSM

z = torch.ones(1, 8, 1)  # one context value per sample
controller = ContextualDeepSSM(
    d_input=1, d_context=1, d_output=1,
    context_modes="input", context_filter="none", param="lru",
)
y, state = controller(u, z)
```

The tutorial then introduces each port separately before combining them:

| Port | What context does |
| --- | --- |
| `input` | joins the input after filtering; contributes its own input energy |
| `gate` | attenuates residual branches with weights in `[0, 1]` |
| `mixer` | maps core features through a bounded output matrix |
| `select` | schedules TV/TVC recurrent matrices |

Later steps explain finite/decaying windows, difference filtering, streaming,
context encoders, and a custom filter. For streaming, `time_offset` counts
samples already processed. Difference filtering also needs
`previous_context=z[:, last_sample:last_sample+1]` from the previous chunk.
If a context encoder is present, pass that previous sample in its **raw** form.

With `select_input="context"`, matrices depend on context alone. An incremental
bound compares two inputs using the same exogenous context and initial state.
Gates/mixers must also be context-only (`gate_include_disturbance=False`,
`mixer_include_disturbance=False`). The additive `input` port is excluded from
that wrapper certificate. Input-dependent selection (`"both"`) retains the
zero-state gain bound; the wrapper reports `inf` for its incremental bound.

## 9. Try the metric-transfer architecture

Read [Tutorial_MetricDeepSSM.py](../Test_files/Tutorial_MetricDeepSSM.py) after
the basic model. It starts with a small model and one prediction call:

```python
from neural_ssm import MetricDeepSSM

metric = MetricDeepSSM(1, 1, d_model=4, d_state=8, n_layers=1)
y = metric(u, return_state=False)
```

This stack learns feature metrics internally and certifies an incremental gain.
Its tutorial explains the state returns before introducing training, streaming,
and energy checks. Its built-in cores are `defect` and `l2n`; its feedforward
choices are `residual` and `none`. Supported activations are `tanh`, `relu`,
`arctan`, and `identity`.

The default return is `(output, trajectories)`. For streaming, request
`return_state=False, return_last=True`. Keep metric parameters in float32 or
float64; autocast can wrap the forward pass while preparation retains that
precision. The [metric architecture guide](metric_transfer_ssm.md) gives the
underlying equations when you are ready for them.

## 10. Move from toy inputs to your own data

Convert a measured scalar sequence to `(1, time, 1)`:

```python
# measured_input and measured_output are your arrays of scalar samples.
u = torch.as_tensor(measured_input, dtype=torch.float32).reshape(1, -1, 1)
y_target = torch.as_tensor(measured_output, dtype=torch.float32).reshape(1, -1, 1)
```

Use these tensors in the training lines from step 5. Multifeature measurements
use `(batch, time, features)` instead. The existing
[benchmark runner](../Test_files/run_benchmarks.py) handles benchmark downloads,
official validation splits, plotting, and model comparisons.

Use `scan` for parallel sequence evaluation and `loop` to inspect sequential
recurrence. LRU/L2N also offer FFT `conv`; compare it with `scan` for your actual
sizes and device. `eval()` together with disabled gradients activates safe
parameter caches. See [parameter caching](parameter_cache.md) and the
[configuration fields](../src/neural_ssm/ssm/config.py) for further options.

To run the complete lessons:

```bash
python Test_files/Tutorial_DeepSSM.py
python Test_files/Tutorial_ContextualSSM.py
python Test_files/Tutorial_MetricDeepSSM.py
```

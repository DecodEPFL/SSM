# Exact storage-based direct spending

The model family also offers **Output prediction + linear**: the backbone
predicts in output units, a learned bias-free `D u` path is added, and the whole
proposal is projected into the available energy budget. Select it with
`--spending-readout projected` or the **Readout** selector in the existing UI.
The original spending-fraction mode remains the default and keeps its checkpoints.
See [output_projection/README.md](output_projection/README.md) for equations,
streaming/precision semantics, commands, and measured validation.

This implements the self-contained construction supplied for Daniele Martinelli's
draft, *Complete and Free Parametrizations of All Finite l2 Operators Using
Storage-Based Neural Networks*, equations (5), (6), and (10). It is a distinct
model family from `energy_bounded/`; that earlier projection model and its files
remain unchanged. Only the existing nonlinear benchmark runner and Tk UI are
wired outside this directory.

## Use the existing UI

From the **SSM-clean repository root**, with the existing environment active:

```bash
python Test_files/benchmark_ui.py
```

On the implementation host, the installed interpreter is
`/Users/leo/miniforge3/envs/SSM/bin/python` (PyTorch 2.9.0, Python 3.13.9).

1. The first box, **Shared run settings**, holds benchmark/model selection,
   training/evaluation settings (epochs, trajectory batch size, learning rate,
   weight decay, gradient clipping, seed, report metric, parameter budget), device,
   plotting, compilation and the output directory. These describe the run,
   independently of architecture.
2. Select **direct_spending** once in **Models**. The **Model options** box below
   shows one tab per selected model; choose its core only in the **Direct spending**
   tab: `static_mlp`, `gru`, `lstm`, `tcn`, or `transformer`. Width/depth and all architecture,
   initialization and gain settings live here. TCN uses kernel/dilations instead
   of a layer count; attention heads/dropout appear only for the Transformer.
3. Choose **Gamma**: `fixed` (default) uses **Fixed gamma** (finite, >=0).
   `learned` uses **Initial gamma** plus explicit positive **Minimum gamma**
   and **Maximum gamma**, with `min < initial < max`. Older models have their own
   **Gain limit (gamma)** in their tabs, so no second gain control appears in a
   spending-only run.
4. Choose **Initial energy**: `zero`, `fixed`, or `learned`. Fixed beta appears
   only in fixed mode. Learned mode shows the initial energy level and optional
   declared context constants; beta MLP width appears only with context.
5. Choose **Certificate units**. `standardized` preserves the suite's existing
   normalization and certifies normalized, centered signals. `physical` uses
   identity normalization for this model and certifies its physical signals.
6. Leave **Storage evaluation** at `scan`, or choose `reference` for the online
   recurrence. Enable **Storage diagnostics** to save its trace and summaries,
   then **Run**. **Stop** uses the existing process cancellation path.

When several models are selected, settings with the same name are shared across
those models, as in the existing runner. GRU/LSTM tabs now expose their actual
recurrent width/layer settings (`--rnn-hidden`, `--rnn-layers`), retaining defaults
64/2. Parameter matching can adjust width. Inactive options retain their values
for switching back, but are omitted from the run command. Settings scroll when
needed; action buttons and results remain reachable.

**Count parameters**, at the top of **Model options**, opens a responsive
snapshot for the current selection. It uses each selected dataset's actual
input/output dimensions and the same factory and parameter-budget policy as
training, reporting trainable/total counts and effective width. It loads dataset
dimensions (and may retrieve an uncached dataset), but performs no training and
does not overwrite benchmark results. Close cancels an unfinished count.

The learning-rate label previews the parsed value: `3e-3` shows `0.003`, whereas
`3e3` shows `3000`. The launch log repeats this value. Incomplete numeric entries
such as `3e-` name their field and prevent launch or saving defaults; entries
are never silently repaired. Restart an already-open UI to load code changes.

A gain certificate does not guarantee smooth training. The spending decision
changes both the current output and the energy available to later outputs, and
the hard unit-ball head has switching boundaries. On Cascaded Tanks this runner
uses one complete training trajectory, so increasing trajectory batch size does
not create additional independent examples. A smaller learning rate can reduce
oscillation while also slowing fitting; keep the certificate and initialization
policy fixed when comparing optimization settings.

For a small CPU check of the reported width-16, three-layer Transformer,
gamma 10, learned-beta-0.1 configuration (one seed, not a tuning recommendation):

```bash
python -m direct_spending.validation.learning_rate_check \
  --output /tmp/spending-rate-check.json --lrs 0.003 0.0003 --epochs 80
```

This uses the existing trainer and native splits, trains through the actual
spending outputs, disables plotting, and writes only the requested JSON. Its
80-epoch cosine schedule differs from a 2000-epoch run; it is a short diagnostic.
See `validation/learning-rate-check.md` for the measured results and limitations.

**Save as defaults** stores the current selections, all parameters, toggles and
output directory in `direct_spending/configs/ui-defaults.json`, inside this
contained folder. The next launch restores them automatically. This is separate
from **Save configuration…** / **Load configuration…**, which use a file of your
choice. Saves replace files atomically; invalid run parameters or write errors
leave the previous file intact. To return to built-in defaults, remove or rename
`ui-defaults.json` before opening the UI. No defaults file is created until you
click the button. Saved defaults affect the UI; CLI defaults stay unchanged.

For a minimal existing-dataset job, use **Load configuration** and choose
`direct_spending/configs/ui-smoke.json`: one epoch, Cascaded Tanks, static MLP,
width 4, zero initial energy, gamma 1, CPU, and diagnostics. This is a wiring
smoke configuration, not a recommended fit or a performance study.

Completed standard reports appear in **Results / saved runs**. Select a row for
**Run details / storage** and **Prediction and loss plots**. **Storage trace**
shows the optional saved trace; **Open saved run** loads an existing `report.json`.
**Save configuration** and **Load configuration** round-trip the selected models,
all settings, toggles, and output directory. The same saved file runs through the
existing CLI with `--config`. Named comparison registrations
`spending_static_mlp`, `spending_gru`, `spending_tcn`, and `spending_transformer`
remain CLI aliases for batch comparisons and are absent from the model list.
An older UI file containing one alias is migrated to `direct_spending` plus its
core. A multi-core alias batch gets an explicit message to run it via the CLI;
the UI does not silently collapse multiple cores into one.

**Visual UI verification is unavailable.** The native computer-use tool reported
that permissions were not granted, and the user explicitly chose to finish with
visual verification marked unavailable. The UI command/config paths and the
actual Tk widgets are checked programmatically, including conditional visibility,
window bounds, model switching and startup restoration. These checks do not
establish visual layout or manual interaction success.

## Exact construction and certificate

The independent causal core emits a dimensionless `z_t` in R^(n_y). The fixed head
and scalar storage produce the actual output returned to training/evaluation:

```text
p_t       = z_t / max(1, ||z_t||_2)
S_initial = beta(xi_0)
d_t       = gamma^2 ||v_t||_2^2
available = S_previous + d_t
y_t       = p_t sqrt(available)
S_t       = (1 - ||p_t||_2^2) available
```

Norms sum over features only. Every trajectory has its own scalar storage `(B,1)`;
storage is neither per channel nor shared across batch rows. The head is the
closed unit-ball head, including norm one at finite parameters. It is not
coordinatewise tanh, a smooth open-ball head, a sigmoid spending fraction, or
budget-dependent clipping of an arbitrary physical prediction. Nothing is
added after `y_t` that could invalidate its certificate.

In real arithmetic, independently at every prefix:

```text
S_{T+1} + sum(t=0..T) ||y_t||^2 = beta(xi_0) + gamma^2 sum(t=0..T) ||v_t||^2
```

`zero` is a separate exact beta=0 option and gives a pure zero-state induced
l2-gain bound <=gamma. `fixed` supplies an additive configured finite beta>=0.
`learned` uses softplus of a scalar logit when no context is declared, otherwise
`Linear(context_dim,beta_width) -> Tanh -> Linear(beta_width,1) -> softplus`.
The documented default initial learned energy is **0.1**. Its final affine
weights start at zero and bias at inverse-softplus(0.1); the scalar uses the
same logit. Only learned mode creates beta parameters.

Initial context in the suite is a user-declared constant vector, shared across
its trajectories and fixed independently of inputs and targets. The direct
model API also accepts explicit per-trajectory initial context. Context conditions
beta; the provided neural cores start from zero/empty state. No input/target
samples are secretly extracted to initialize beta. The learned value can grow;
its **actual** value is logged in run details and checkpoints. There is no hidden
cap on beta, beta regularizer, refill, or post-validation policy adjustment.
User-selected optimizer weight decay retains the suite's ordinary semantics.

### Optional learned gain

Fixed gamma remains the default and old fixed-gamma checkpoints load unchanged.
Learned mode adds one scalar parameter `g`:

```text
gamma = gamma_min + (gamma_max - gamma_min) * sigmoid(g)
```

The initial logit is chosen from the configured initial gamma. Computation uses
at least FP32; representable endpoints are rounded inward and the final gain is
clamped to these endpoints to keep roundoff within the declared range. This
does not put a floor on reservoir energy. Autograd differentiates through gamma
and the actual spending outputs; no detached gain or new loss penalty is used.

For fixed trained weights, the original identity holds with the **fitted gamma**.
The configured `gamma_max` also gives a uniform conservative gain ceiling over
all learned parameter values. Reports, run details and checkpoints distinguish
the initial value, fitted value, mode, limits and ceiling; `model_config.gamma`
is the initialization, while `certificate.gamma` is the fitted value. The target
feasibility audit is recomputed at the selected checkpoint's fitted gamma.
Gamma is a global scalar, not an input-dependent or per-token predictor.
Hold weights/gamma fixed across a streamed trajectory; reset states after an
optimizer update. Fitted bounds from different runs need to be compared explicitly.

```bash
python Test_files/run_benchmarks.py --models direct_spending --spending-core lstm \
  --spending-gamma-mode learned --spending-gamma 2 \
  --spending-gamma-min 0.1 --spending-gamma-max 5 \
  --spending-beta-mode zero --benchmarks Cascaded_Tanks \
  --out direct_spending/validation/my_lstm_learned_gain
```

This certificate does not imply internal neural-state stability, incremental
gain, contraction, or prediction accuracy. Finite neural implementations are
not thereby universal over gain-bounded operators.

## Coordinates, warm-up, and native datasets

The default suite standardizer is fit only on training data. In standardized
mode the certificate applies to `(v-mu_v)/sigma_v` and
`(y_prediction-mu_y)/sigma_y`. The mean added back to physical predictions
**does not have a physical-coordinate zero-state certificate**. Physical mode
uses an identity standardizer for direct spending, so no uncertified centering,
rescaling, readout, or output residual follows the storage output. Other models'
normalization and defaults are unchanged.

Every complete native trajectory starts storage once. The official initialization
window still feeds the causal model and deposits/spends energy; it is excluded
only from the existing validation loss/metrics. The certificate covers sample 0
through the complete trajectory. A scored suffix has its carried storage at the
metric start, not a newly zeroed tank. Optional diagnostics record that value.
Padding has a distinct validity mask and cannot be confused with warm-up samples.

Datasets with a nonzero initial-condition free response can be incompatible with
zero initial energy at a selected gamma. The runner reports the necessary initial
energy for exact **target prefixes** in the chosen coordinates and exposes the
explicit initialization choice. It never changes gamma/beta to hide a mismatch.
This audit is a necessary energy check, not a promise that an architecture can fit
the target. Native datasets, trajectory splits, losses, metrics, optimizer,
validation checkpoint selection, and run-report formats are preserved.

## State and supported spending maps

`DirectSpendingModel.forward(v, state=None, mask=None, initial_features=None, context=None)` uses
`(batch,time,features)` and returns `(y, SpendingState)`. The state contains an
opaque core state, storage `(B,1)`, and the initial beta `(B,1)` for honest logging.
`init_state` explicitly initializes/resets. `step(state,v)` takes `(B,features)`
and follows the repository's state-first convention. `diagnostics=True` adds a
third result. Empty chunks are accepted.

Beta is evaluated once at reset; chunk continuation rejects a new initial-context
argument. Preserve both core and storage state across chunks. No state is silently
detached. `direct_spending.detach_state` marks an explicit truncated-BPTT boundary;
`state_to` moves core tensors and preserves at least FP32 storage. FP64 storage
must explicitly be converted to FP32 when moving to MPS.
The learned beta MLP evaluates at parameter precision outside CPU/CUDA autocast,
so explicit reset and reset inside an AMP forward produce the same initial energy.

| Core | Causal implementation | Streaming state / limitation |
| --- | --- | --- |
| static_mlp | Shared MLP with GELU and final unconstrained projection, parallel over time | No neural state; only storage remembers history; restricted ablation |
| gru | Existing learned projections and native `nn.GRU` sequence kernels | Explicit per-layer recurrent state; step uses the same GRU kernel path |
| lstm | Learned projections around native `nn.LSTM`, one module call per dense chunk | Explicit `(hidden, cell)` tuple per layer; no custom recurrent cells or time loop |
| tcn | Existing residual dilated causal Conv1d/GELU with left history | Per-layer input buffers; receptive field `1+(kernel-1)*sum(dilations)`, default 15 valid tokens |
| transformer | Existing modest pre-norm causal SDPA model with sinusoidal positions | Real per-layer KV caches, valid-token position offsets, correct chunk masks; unbounded context |

The GRU, TCN and Transformer implementations are reused from `energy_bounded/`
without changing their behavior. Only their unconstrained readouts are used as
dimensionless z; the earlier projection reservoir is never invoked.

Masks are boolean `(B,T)`, True for valid tokens. Invalid positions emit zero,
use the identity storage transition `(a,b)=(1,0)`, and pause semantic core time.
NaN padding, interior holes, left padding, empty rows, and resumption are tested.
GRU/LSTM/TCN use per-row compaction for masked data and native dense kernels otherwise.
Transformer caches retain physical padding slots as permanently masked keys,
while positions count valid tokens only. Cache append copies O(context) elements;
it does not recompute full prefixes. Full attention has quadratic context work
and may exceed memory on long native benchmark trajectories. No implicit window
or truncation changes the operator. Disable dropout for deterministic chunk
equivalence (default zero); nonzero training dropout changes random draw order.

Storage/output feedback into spending cores is outside this implementation's
scope. Such feedback would generally invalidate precomputed scan coefficients;
it is never silently routed through this scan.

## Affine scan and numerics

Let `a_t=1-||p_t||^2` and `b_t=a_t*d_t`. Then
`S_t=a_t*S_previous+b_t`. Chronological composition is
`(a1,b1) star (a2,b2)=(a2*a1,a2*b1+b2)`, identity `(1,0)`.
For inclusive prefixes `(A_t,B_t)`, `S_t=A_t*S_incoming+B_t`.
Shift storage to recover each incoming budget and compute all outputs in parallel.

The scan reuses the repository's differentiable adjacent-pair recursive
`associative_scan` utility. It performs **O(BT) scalar work**, O(log T) parallel
depth, and O(BT) scalar intermediate storage. Feature reductions/scaling add
O(BT(m+n_y)) work. There is no optimized-path Python loop over time and no
cumulative-product division; exact zero coefficients are supported. The reference
uses the transparent multiplicative recurrence and serves single-step inference.
Core costs remain independent of the storage scan. No new custom kernels,
compilation, prototype associative-scan API, or performance study is introduced.
The suite explicitly skips `torch.compile` for this family.

Auto precision uses FP64 energy/reductions on CPU/CUDA and FP32 on MPS. Explicit
FP32/FP64 choices are available, with no silent FP64-to-CPU transfer for MPS.
FP64 on MPS raises a clear error. Outputs preserve FP32/FP64 raw precision, and
FP16/BF16 proposals produce FP32 outputs. Norm calculations scale z to avoid
overflowing an otherwise finite large proposal while evaluating the exact same
unit-ball head. In the normalized branch the mathematically exact fraction one
is recorded explicitly, preserving exact `a=0` rather than roundoff-sized storage.
The reference uses multiplication rather than subtracting almost equal energies.

The square root evaluates a safe positive argument even in its inactive branch;
available energy zero has a zero-gradient convention. At unit norm, the head
chooses the normalized branch's finite gradient for both p and its squared norm;
the model is not globally smooth.
No positive budget floor or epsilon energy is added. Softplus is evaluated in
energy precision; a learned beta that numerically underflows to zero raises an
error rather than acquiring a floor or silently becoming zero mode. Valid input,
raw-core, storage and output nonfiniteness raises an error; optional
`check_core_state=True` audits tensor-tree states. This is detection, not a proof
against hidden-state overflow.

Actual returned outputs are audited with normalized, dtype-appropriate tolerances:
tests use up to 2e-6 for FP32 returns and 1e-12 in stable FP64 stress cases.
Floating-point tree orders need not be bitwise equal to online evaluation.
The existing runner enables PyTorch's MPS fallback environment globally; unsupported
MPS operations may therefore follow that existing policy. This layer adds no CPU
compute fallback. Explicit CPU copies serialize diagnostics/checkpoints only.

## Expressivity qualifications

The abstract completeness statement concerns arbitrary causal unit-ball-valued
spending maps. It is not an exact completeness or approximation theorem for any
fixed finite MLP, GRU, LSTM, TCN, or Transformer.

For static `p=f(v)`, a zero-input tail has the fixed direction `f(0)` and geometric
storage decay. If `f(0)=0`, output stops immediately despite remaining storage.
Even a hypothetical static `f(v,S)` cannot implement a signed unit delay exactly:
opposite first impulses with zero first output leave identical scalar storage,
then require opposite next outputs at identical arguments.

There is also a zero-bias continuity limitation for continuous neural cores with
memory. At scalar gamma=1 and beta=0, exact `y_0=v_0` requires
`p_0(v_0)=sign(v_0)` for nonzero inputs. A continuous spending core cannot realize
that discontinuity. Uniform global relative/induced-gain approximation also fails
at arbitrarily small bipolar inputs, even though absolute dataset error can be
small. These restrictions do not weaken the energy guarantee. The exact head is
retained, and mathematical tests illustrate the distinctions without adding any
benchmark corpus or fitting campaign.

## Files, checks, saved models

`config.py`, `cores.py`, `storage.py`, `model.py`, and `state.py` implement the method.
`benchmark.py` connects model factories, options, certificates and tagged
`direct_spending_v1` checkpoints. `ui.py` and `ui_results.py` embed conditional
controls and read-only views of the existing suite's reports/plots in its current
window. `diagnostics.py` exports the optional storage plot. Wiring is limited to
`Test_files/run_benchmarks.py` and `Test_files/benchmark_ui.py`; an integration patch
is supplied for review. Configurations, validation logs, and the minimal native-data
smoke artifacts are contained here. There is no separate training entry point.

```bash
python -m unittest discover -s direct_spending/tests -t . -v
python -m unittest discover -s energy_bounded/tests -t . -v
PYTHONPATH=src python -m unittest discover -s tests -v
python Test_files/run_benchmarks.py --config direct_spending/configs/ui-smoke.json
# Existing runner, arbitrary supported spending core; no new suite:
python Test_files/run_benchmarks.py --models direct_spending --spending-core gru \
  --spending-gamma 1 --spending-beta-mode zero --spending-storage scan \
  --benchmarks Cascaded_Tanks --out direct_spending/validation/my_run
```

Normal jobs save model family/core/hyperparameters, gamma, beta mode/architecture/
context, storage implementation, coordinate choice, standardizer, history and
weights. Run directories include spending core and beta mode; learned-gamma jobs
append `--learned-gamma` so they do not overwrite a fixed-gamma job's artifacts.
Restore with
`direct_spending.benchmark.load_checkpoint(path, device="cpu")`, which returns
`(suite_model, payload)`. `payload['normalizer']` reconstructs the existing
`Standardizer`; `run_benchmarks.simulate` returns physical predictions as usual.
The tagged loader rejects older projection/SSM checkpoints; their formats and
identifiers are not repurposed. A plain PyTorch state_dict alone is not a substitute
for the accompanying model configuration.
Loading preserves the stored parameter dtype, including the learned gain's
precision, so its fitted certificate is reproduced.

See [VALIDATION.md](VALIDATION.md) for the original check results and
[validation/lstm-gain-context/REPORT.md](validation/lstm-gain-context/REPORT.md)
for the LSTM/gain/context additions, native-module audit and current CPU/MPS
checks. Visual UI verification remains unavailable; native Tk controls are
checked programmatically. No comparative-performance claim is made.

## Core-only contextual signals

The Python API accepts a separate sequence context, configured by
`SpendingConfig(context_features=k)`. Every core receives `[v_t, context_t]`,
but **only `v_t` supplies reservoir energy**. Context has shape `(B,T,k)` for
sequences and `(B,k)` for `step`; pass the matching chunk alongside each input
chunk. It must match input device/dtype. Padding pauses both signals' core time
and supplies/spends no energy. This is separate from `initial_features`, which
conditions beta once at reset.

```python
from direct_spending import DirectSpendingModel, SpendingConfig

config = SpendingConfig(core="lstm", width=16, layers=2,
                        gamma=2., context_features=3)
model = DirectSpendingModel(d_input=1, d_output=1, config=config)
y, state = model(v, context=c)  # v: (B,T,1), c: (B,T,3)
```

For every causal, well-defined context-conditioned core, the same bound holds:

```text
sum ||y_t||^2 <= beta + gamma^2 sum ||v_t||^2
```

Thus `v in l2` and finite beta imply `y in l2`, uniformly over bounded,
causally available context signals. Context may persist forever and need not
be square-summable. Its energy is absent from the right-hand side. This is a
conditional family of maps `v -> y`, indexed by context; it does not certify an
incremental gain with respect to context or hidden-state boundedness. Finite
neural computations remain necessary; nonfinite raw proposals are reported.
The interface does not prevent a caller from externally constructing noncausal
context from future data; such a construction loses the causal-map interpretation.

If the energy-supplying input is merely bounded, square summability is **not**
guaranteed: `v_t=1`, `p_t=1`, beta=0 yields `y_t=gamma` forever. With zero external
input and zero beta, context alone produces exactly zero output. Concatenating
context into `v` instead of using `context=` would count its energy and could
lose the desired conclusion for nondecaying context.

The existing benchmark datasets do not supply separate contextual channels;
their UI experiments retain `context_features=0`. Use the Python API for
explicit contextual data; no target samples or future observations are
automatically injected as context.

# Package structure

Start with the public API:

```python
from neural_ssm import DeepSSM, MetricDeepSSM, ContextualDeepSSM, SSMConfig, MetricSSMConfig, REN
```

The source is grouped by concepts. Cells have separate files because their
parametrizations differ. Small helpers stay with the code they explain.

```text
neural_ssm/
├── ssm/
│   ├── layers.py             SSL, DeepSSM, certificates, simple baselines
│   ├── metric_layers.py      MetricDeepSSM, feature metrics, weighted residual FF
│   ├── config.py             model settings, grouped by purpose
│   ├── registry.py           cell/feedforward choices and construction
│   ├── common.py             shared gain and storage algebra
│   ├── contextual.py         context wrapper, filters, gates, output mixing
│   ├── cells/
│   │   ├── common.py         shared LTI/selective algebra
│   │   ├── lti/              lru.py, l2n.py, defect.py
│   │   ├── selective/        tv.py, tvc.py
│   │   └── legacy/           l2ru.py, lruz.py, l2nt.py
│   └── experimental.py       unregistered expert-mixture prototypes
├── static_layers/
│   ├── generic_layers.py     ordinary GLU/MLP and shared LayerConfig
│   └── lipschitz_mlps.py     bounded feedforward networks and their primitives
├── rens/ren.py               robust recurrent equilibrium network
└── utils/
    ├── scan.py               parallel, convolution, and CUDA scan kernels
    ├── lyapunov.py           block-pair state metrics and weighted completion
    └── runtime.py            input shape, streaming state, inference caches
```

## Follow a forward pass

1. `SSMConfig.param` chooses the recurrent cell through `registry.py`.
2. In `layers.py`, `DeepSSM` encodes the input, runs its `SSL` blocks, and decodes
   the result. Each block applies a recurrent residual and a feedforward residual.
3. Open the chosen cell file to see its parametrization and recurrence. Shared
   sequence kernels live in `utils/scan.py`.
4. `SSMConfig.ff` chooses the static branch through the same registry. GLU/MLP definitions live
   in `generic_layers.py`. Bounded choices live in `lipschitz_mlps.py`.
5. If context is used, `contextual.py` wraps the same stack. It keeps filtering,
   residual gates, output mixing, and selective conditioning in one place.

The larger files have a short reading map and named sections. They keep the
construction, its assumptions, and its computation together.

For the new `MetricDeepSSM`, start with `metric_layers.py`. Its configuration,
feature metric, coupled residual feedforward, layer, and stack live together.
It reuses the defect/l2n cell implementations and the same scan/runtime helpers.
The residual FF stays with the metric stack because its weights depend on that
stack's learned `Q`; the standalone bounded MLPs remain in `lipschitz_mlps.py`.
See [the architecture guide](metric_transfer_ssm.md) for the equations.

## Ordinary versus bounded feedforward layers

| File | Contents |
| --- | --- |
| `generic_layers.py` | `LayerConfig`, conventional `GLU`, conventional `MLP` |
| `lipschitz_mlps.py` | capped linear maps, Sandwich primitives, `TLIP`/`LMLP`, bounded gates, GroupSort branches, `MultiBranchLipMixer` |

`DeepSSM` accepts `LGLU2`, `BLGLU2`/`BudgetedLGLU2`, `MBLIP`, and `TLIP` as
certified feedforward branches. The historical `LGLU` and `LMLP` constructions
remain available, but the stack does not accept them for its global certificate.
The module and class comments explain this distinction.

```python
from neural_ssm.static_layers.generic_layers import LayerConfig, GLU, MLP
from neural_ssm.static_layers.lipschitz_mlps import TLIP, L2BoundedGLUv2
from neural_ssm.ssm.cells.lti import DefectL2SSM
from neural_ssm.ssm.cells.selective import RobustMambaDiagSSM
```

## Current, legacy, and experimental cells

Current fixed-matrix cells are `lru`, `l2n`, and `defect`. Current selective cells
are `tv` and `tvc`. They have sibling folders because a selective recurrence is
not generally LTI.

`l2ru`, `zak` (the historical configuration key for `lruz`), and `l2nt` remain
wired as legacy constructions. The benchmark runner and UI label them clearly.
`Benchmark.py` defaults to `defect`; explicit legacy configurations still work.

The expert-mixture prototypes are in `ssm/experimental.py` and are imported
only when requested. Two small namespace initializers preserve older explicit
experimental imports. Raven and transformer implementations are removed.

## Compatibility and speed

Earlier import paths resolve through explicit aliases in package initializers.
They point to the same classes/functions; there are no forwarding files or
runtime model wrappers. Repository code uses the canonical modules or public API.

Registered parameter names, checkpoint keys, model initialization, and numerical
kernels retain their behavior. Configuration field order is also preserved.
Inference caches, shared-power scans, FFT convolution, CUDA scan capture, and
training graph capture retain their implementations. See
[parameter caching](parameter_cache.md) for the cache lifecycle.

## Adding a cell

Put the implementation in its family folder and register its factory and
certificate metadata in `ssm/registry.py`. Export it from the family initializer.
Keep shared algebra in `cells/common.py` and use the existing runtime/scan
helpers. Update certificate/recurrence checks and the benchmark choices together.

The [tutorial guide](ssm_tutorial.md) walks through construction, training,
state continuation, context, and execution choices. Run the examples with:

```bash
python Test_files/Tutorial_DeepSSM.py
python Test_files/Tutorial_ContextualSSM.py
python Test_files/Tutorial_MetricDeepSSM.py
```

Tutorial filenames and benchmark entry points remain unchanged:
`Test_files/Tutorial_DeepSSM.py`, `Test_files/Tutorial_MetricDeepSSM.py`, `Test_files/Tutorial_ContextualSSM.py`,
`Test_files/run_benchmarks.py`, and `Test_files/benchmark_ui.py`.

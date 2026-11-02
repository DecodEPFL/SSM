# Parameter construction and caching

Audit date: 2026-10-05. Parameter caching applies to gradient-free evaluation.
Training and grad-enabled evaluation construct fresh tensors on each forward,
so gradients flow through the constraints and no previous autograd graph is reused.
Each sequence constructs its fixed matrices once, before running the recurrence.

| Construction | Reused in gradient-free evaluation | Recomputed for each input |
| --- | --- | --- |
| `l2n` | Normalized contraction and energy-coordinate matrices; LU factors of `S`; dense matrices for loop/export | Input projections, scan/convolution, state-coordinate solves using the cached LU factors |
| `defect` | Orthogonal/defect factors assembled into `(A_blocks, B, C, D)` | Input projections and block scan |
| `l2ru`, `zak` | Certified state-space matrices, including norm, Cholesky, and solve results | Recurrence and output projections |
| `l2nt` | Normalized contraction and dense state-space matrices | Sequential recurrence |
| `tv`, `tvc` | Spectrally capped input/output projection weights | Selector network, time-varying coefficients, closed-form per-channel normalization, diagonal scan |
| REN, `mode="l2stable"` | Constrained and effective state-update matrices | Sequential nonlinear recurrence |
| DeepSSM | Capped encoder/decoder weights and their norms for gain/storage diagnostics | Residual updates and gain-budget arithmetic |
| Sandwich layers | Cayley factors | Input-dependent activations and projections |

TV/TVC coefficients depend on the current input or exogenous context. Reusing
those across sequences would change the model. Their normalization uses
vectorized scalar formulas for 2-by-2 matrices, rather than an SVD at each step.
The projection spectral norms are the expensive, input-independent part that
can be cached.

## Cache lifecycle

`utils/runtime.py` contains `EvalCacheMixin`, which tracks tensor identity, in-place mutation version, dtype,
device, and both parameters and registered buffers. It also tracks CPU/CUDA
autocast settings. Caches refresh after ordinary optimizer updates, checkpoint
loading, parameter replacement, or changes to fixed gain buffers. A train/eval
toggle or device/dtype conversion clears them. Cached values are runtime data
and do not enter `state_dict`.

Use both evaluation mode and disabled gradients:

```python
model.eval()
with torch.inference_mode():
    y, state = model(u, mode="scan")
```

Construct the model outside `inference_mode`. Tensors created inside that context
have no mutation counter, so a module constructed there conservatively bypasses
parameter caching. Evaluation with gradients also bypasses it.

During `torch.compile` inference, version checking and cache lookup run outside
the compiled numerical regions. Use the default `fullgraph=False`; these eager
boundaries preserve cache invalidation and avoid repeated normalization. Training
bypasses the caches and retains the existing single-graph capture behavior.

## Corrections from the audit

- Projection, encoder/decoder, and Sandwich caches previously could keep stale
  weights after updates or checkpoint loading, retain the old dtype/device,
  and detach weight gradients during evaluation. They now share the guarded cache.
- The common LTI cache previously tracked only parameter mutation counters,
  missing fixed buffers, replacement identities, and dtype/device changes.
- `l2n` previously refactorized its fixed coordinate transform on every scan
  call and rebuilt dense matrices on every loop call. It now reuses LU factors
  and dense matrices, without storing an explicit inverse.
- REN rebuilt its constraints for every inference sequence. Its standard
  gain mode now reuses them. Custom IQC modes retain live computations because
  their external `Q/R/S` inputs are not registered cache dependencies.
- Gain/storage diagnostics now reuse capped encoder/decoder weights and norms.
- Optional power-iteration projections now preserve the starting vector for
  autograd and record the updated auxiliary buffer version when caching.
- CUDA scan graph keys now track both inputs, device, strides, and gradient
  requirements. Capture samples preserve each input's gradient flag. A failed
  signature falls back to eager execution without repeated capture attempts.

## Remaining optimization opportunities

Caching parameters does not establish optimal execution performance. LRU still
recomputes inexpensive elementwise derived parameters. LTI scan powers and FFT
kernels are built for each sequence; caching them would need length-specific
entries and a memory policy. TV/TVC also recompute cheap fixed scalar transforms.
These should be profiled before adding cache complexity. `l2nt` and REN remain
sequential, and the dense scans used by some cells have higher arithmetic and
memory costs than diagonal/block scans.

Regression coverage is in `tests/test_parameter_cache.py`, including call counts
for actual cells, fresh results after lifecycle changes, gradients, precision,
and scan/conv/loop agreement. CPU checks are available locally. CUDA capture and
backward checks are included but require CUDA hardware to run.

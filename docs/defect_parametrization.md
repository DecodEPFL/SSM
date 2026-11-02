# Defect-completion LTI parametrization

`param="defect"` adds the structured contraction construction alongside the
existing `param="l2n"` normalization-based cell. Existing defaults and checkpoints
are unchanged. The standalone cell is `neural_ssm.DefectL2SSM`.

## Use it

```python
import torch
from neural_ssm import DeepSSM, SSMConfig

config = SSMConfig(
    param="defect",
    d_model=16,
    d_state=64,
    n_layers=2,
    ff="LGLU2",
    gamma=1.0,
    train_gamma=False,
    defect_block_size=2,
    defect_max_radius=0.999,
    defect_factor_margin=1e-3,
    rho=0.9,
)
model = DeepSSM(d_input=3, d_output=2, config=config)
u = torch.randn(4, 128, 3)
y, states = model(u, mode="scan")
```

For defaults, `DeepSSM(3, 2, param="defect", gamma=1.0)` also works.
In `Test_files/benchmark_ui.py`, select **defect** from the model list. Its
options panel exposes block size, contraction limit, factor margin, and
initialization. Choose **auto** or **scan** execution; select **l2n** as well
for a side-by-side comparison. The runner also accepts `--models defect l2n`
and `--defect-block-size 4` on the command line.
Use `mode="loop"` for a sequential reference. `mode="conv"` is rejected:
general real blocks are not complex scalar multipliers.

Set `defect_block_size=4` to allow interactions within groups of four states,
or set it equal to `d_state` for an unrestricted dense state matrix. The block
size must divide `d_state`. Larger blocks increase computation. The standalone
cell uses the names `block_size`, `max_radius`, and `factor_margin`.

## Full state storage with the same fast recurrence

The original default is `defect_state_metric="identity"`. To remove the
independent-block storage restriction at a prescribed gain, choose a full metric:

```python
from neural_ssm import DefectL2SSM

cell = DefectL2SSM(
    d_state=9, d_input=3, d_output=2,
    state_metric="full", max_radius=1.0, factor_margin=0.0,
)
y, states = cell(torch.randn(4, 128, 3), mode="scan")
P = cell.storage_matrix()       # (9, 9), in the returned state coordinates
energy = cell.state_energy(states[:, -1])  # one terminal energy per sequence
```

For `DeepSSM`, the corresponding configuration fields are
`defect_state_metric`, `defect_max_radius`, and `defect_factor_margin`.
For `MetricDeepSSM`, use `MetricSSMConfig(state_metric="full", max_radius=1,
factor_margin=0)`. Feature metrics and the state metric are separate choices.
The benchmark UI has an identity/full state-storage selector; the runner accepts
`--defect-state-metric full --defect-max-radius 1 --defect-factor-margin 0`.

Full storage supports block sizes one through four. Size two also supports an
odd state dimension: the final real scalar is padded internally by a zero dummy
mode, which is absent from returned states, learned parameters, and exports.

### How the full metric is constructed

Learn a packed upper-triangular factor `R` with a softplus-positive diagonal.
It has exactly `n*(n+1)/2` parameters and defines `Q=R.T @ R > 0`. Solve

\[
P-A^\top P A=Q.
\]

Every stable `A` gives `P>0`. The equation splits into block-pair problems with
four unknowns per pair for size-two blocks. We solve only the upper block pairs
and reflect their transposes; no `n²`-unknown Kronecker system is assembled.
An implicit backward reuses their LU factors for the adjoint Lyapunov equation,
then contracts only the diagonal state-block gradients. This avoids retaining
and differentiating the full block-pair coefficient construction. Higher
derivatives rebuild a differentiable adjoint solve.
Initially `Q=(1-rho²)I`, giving `P=I` for the initial scaled-orthogonal blocks.

Reuse the existing thin `X,Y,Z` contractions and their low-rank defects. Define

\[
T=R^{-\top}A^\top P,\qquad H^\top H=P+T^\top T,
\]
\[
B/\gamma=H^{-1}X,\qquad C=YR,
\]
\[
D/\gamma=-YT H^{-1}X+\Delta_{Y^\top}Z\Delta_X.
\]

Triangular solves implement the inverses in these formulas. Only `H` needs a
dense Cholesky in the forward construction; neither `P` nor a dense
energy-coordinate `A` is factored or assembled. The positive sum defining `H`
avoids subtracting nearly equal dense matrices. State-sized SVDs and numerical
matrix square roots are absent from both forward and backward.

For the proof, choose `S.T @ S=P` conceptually. The matrix

\[
J=\begin{bmatrix}
SAS^{-1} & SH^{-1}\\
RS^{-1} & -TH^{-1}
\end{bmatrix}
\]

is orthogonal: its block columns have unit Gram matrices and zero cross Gram.
Contractive block completion with the free `X,Y,Z` yields the stated weighted
bounded-real inequality. Orthogonal choices of the defect factors are absorbed
into `X,Y`, so this is a complete completion, not a restricted central choice.

Consequently, the full-metric size-two construction covers generic stable real
LTI input/output maps of the chosen order with gain strictly below `gamma`.
Generic modal coordinates use real scalars and conjugate pole pairs; the full
bounded-real metric exists in those coordinates. Exact larger Jordan chains
are outside size-two blocks. Fixed `max_radius<1` or `factor_margin>0` deliberately
restrict coverage further. Open limits have a machine-precision saturation guard
so rounded `tanh` values do not create unit poles or singular defect solves.

For fixed small blocks, dense construction costs `O(n³)` and `O(n²)` working
memory. Applying metric-dependent factors to the thin projections costs
`O(n²*(input_width+output_width))`. The recurrence retains its existing block
scan complexity. Evaluation caches the constructed matrices and metric; training
builds a fresh graph. Full metrics require float32/float64 parameters and their
construction runs at parameter precision under AMP.

## Identity-metric construction and optimizations

The cell constructs a contraction

\[
K=\begin{bmatrix}A&B/\gamma\\C&D/\gamma\end{bmatrix}
\]

and runs in energy coordinates, with storage `V(x) = ||x||²`. Unlike `l2n`, it
needs no learned dense change of coordinates, spectral norm of the assembled
system matrix, numerical SVD, eigendecomposition, or matrix square root.

Write `A = Ua diag(sa) Va.T` blockwise. Absorbing the inner orthogonal factors
into two freely learned contractions `X` and `Y` gives

\[
B/\gamma=U_A\operatorname{diag}(d_A)X,\qquad
C=Y\operatorname{diag}(d_A)V_A^\top,
\]
\[
D/\gamma=-Y\operatorname{diag}(s_A)X+
\Delta_{Y^\top}Z\Delta_X,
\qquad \Delta_M=(I-M^\top M)^{1/2}.
\]

The implementation uses:

- **Learned signed singular values.** `s = limit * tanh(raw)`; the state limit
  is `max_radius`, and the limits for `X,Y,Z` are `1-factor_margin`.
- **Stable scalar defects.** Evaluate `sech(raw)` using bounded exponentials,
  then `sqrt((1-limit²) + limit²*sech(raw)²)`. For `limit=1`, return `sech`
  directly to avoid underflow through squaring. Never subtract `tanh²` from one.
- **Thin CWY orthogonal factors.** Householder products use a triangular solve
  on the smaller rank dimension. No explicit matrix inverse is formed.
- **Exact thin defects.** For an orthonormal basis `U`, apply a defect as
  `value + U @ (diag(d-1) @ (U.T @ value))`. The identity complement preserves
  full rank even when input or output dimensions exceed the state dimension.
  Apply these updates to the thin factors of `Z`, not to dense identity matrices.
- **Cheap state factors.** Each 2-by-2 block uses two rotation angles. Larger
  blocks use batched CWY factors. The state rotations are absorbed as above.
- **A shared-power LTI scan.** Pair adjacent affine steps and square the block
  transition once per tree level. No copies of transition matrices across batch
  and time, and no dense state matrix on the forward path.
- **Evaluation caching.** Cache derived matrices only in evaluation mode with
  gradients disabled. Parameter/buffer changes, checkpoint loading, and changes
  in dtype/device invalidate the cache. Training always uses live parameters.

For state dimension `n`, block size `b`, batch `B`, and sequence length `T`, the
recurrence costs `O(T B n b + n b² log T)` work with `O(log T)` tree levels.
Dense input/output projections add their usual costs. For small blocks and
`n >> input_width, output_width`, setup avoids state-sized cubic operations.

Raw CWY columns are initialized randomly. A zero column has an explicit
identity-reflection convention, which preserves orthogonality but is
discontinuous at zero. Do not initialize all raw factors to zero. Adding an
epsilon to every nonzero column norm would break the exact construction.

## Guarantees and scope

The weighted bounded-real certificate yields

\[
\|x_{t+1}\|_P^2+\|y_t\|^2
\leq\|x_t\|_P^2+\gamma^2\|u_t\|^2,
\qquad P=I\text{ for identity storage}.
\]

The zero-state induced gain is at most `gamma`; LTI differences have the same
bound for equal initial states. A nonzero initial state contributes its initial
storage energy. `contraction_rate_bound()` returns `max(abs(sa))`, bounded by
`max_radius`. Singular values are not generally eigenvalues.

With identity storage, small independent blocks restrict expressiveness at a prescribed gain: some
low-gain systems need storage terms coupling different blocks. Larger blocks
can capture these interactions. Without the finite margins and with one
unrestricted block, the underlying strict construction covers stable LTI maps
below the gain bound up to a change of state coordinates. The configured
margins deliberately restrict the boundary further.

This is an LTI implementation. It does not add state-dependent nonlinear
feedback or an input-dependent selector. Pointwise contractivity of an
input-selected matrix alone would not certify incremental gain.

The standalone cell supports trainable `gamma`, in which case its current value
is the bound, not its initial value. `DeepSSM` retains its existing global gain
budget logic. Learned initial states cannot be used to assert a zero-state gain
bound for a trajectory starting from those states.

`compute_block_matrices()` returns `(A_blocks, B, C, D)`.
`compute_z_matrices()` and `compute_dense_matrices()` return `(A, B, C, D, P)`
for inspection/export; these diagnostic calls intentionally allocate dense
matrices. `P=I` for the default identity metric. Returned states always use the
same block recurrence coordinates; `state_energy()` applies the right metric.
`DeepSSM.storage_value()` and `MetricDeepSSM.storage_value()` also use it.

## Tests and comparison

From the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_defect_ssm.py -v
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/compare_defect_vs_normalization.py --output /tmp/defect-results.json
python scripts/compare_defect_vs_normalization.py --l2n-state-metric full --defect-state-metric full \
  --state-sizes 128 256 512 --widths 16 --include-optimizer \
  --output /tmp/full-metric-results.json
```

The comparison runs **timings only** by default, comparing four labeled curves:
`l2n(I)`, `l2n(P)`, `defect(I)`, and `defect(P)`. Both
`--l2n-state-metric` and `--defect-state-metric` accept `identity`, `full`, or
`both` (the default). The default
input/output width is 16 for every requested state size.

It prints median times and ratios for matching storage choices, for construction, construction plus backward,
uncached forward, cached inference, and complete forward/backward. A ratio
above one favors defect. It saves both JSON and CSV.
The full/full run compares the same generic strictly gain-bounded LTI family,
using the same real block scan. The initial transfer maps and parameterizations
differ. Comparing a full-metric curve against an identity-metric curve instead
compares different representable families.
See [the full-metric CPU measurements](benchmarks/defect_full_metric_cpu_2026-10-09.md)
for the earlier identity-normalization reference, and
[the full/full measurements](benchmarks/l2n_full_metric_cpu_2026-10-09.md) for
the new equally expressive normalization comparison.

To investigate why a variant is slower on your machine, include construction
and optimizer timings and record separate profiler traces:

```bash
python scripts/compare_defect_vs_normalization.py \
  --state-sizes 512 --widths 16 --workload all --include-optimizer --profile \
  --output /tmp/ssm-comparison.json
```

`--profile` saves a Chrome-compatible forward/backward trace for each variant
and case, and prints its busiest operators. Profiling runs separately from the
timed samples. CUDA traces include GPU kernels; MPS traces cover CPU dispatch.
JSON records both total parameters and `parameters_with_grad`, counting
parameters with an allocated gradient tensor in the measured workload.
This count includes zero gradients: L2N's `S` can receive a zero gradient through
the transformation of a zero initial state, even though it does not change the
zero-state transfer map. The existing implementation also converts its stored
final state through `S` when only predictions are requested.

Use `--l2n-state-basis identity` to omit `S` and its coordinate conversions.
The default `learned` setting retains the current implementation. Both choices
use identical initial LTI matrices, so this toggle isolates coordinate overhead
without changing the zero-state transfer map or input/output expressivity.
The cells use their native sequence kernels: L2N's rotations use a complex
diagonal scan, while general defect blocks use the real shared-power scan.
JSON records the engine used. Construction timings isolate matrix preparation
from these sequence-kernel differences.

### Full-metric normalization

Enable `state_metric="full"` on `Block2x2DenseL2SSM`, or
`l2n_state_metric="full"` in `SSMConfig`/`DeepSSM`. `MetricDeepSSM` uses its
existing `state_metric` setting for either built-in core.

Full L2N learns a packed upper-triangular factor `L`, with `P=L.T @ L`.
It uses free real 2×2 state blocks and an optional final scalar, so both distinct
real poles and conjugate pairs are available. Unlike the original L2N's `S`,
this factor participates inside normalization:

\[
K_P=\begin{bmatrix}L A L^{-1}&L B_0\\C_0 L^{-1}&D_0\end{bmatrix},\qquad
\alpha=\max(1,\|K_P\|_2).
\]

The final matrices are `A/alpha`, `gamma*B_0/alpha`, `C_0/alpha`, and
`gamma*D_0/alpha`. A machine-precision guard makes normalization strict at the
boundary. Already admissible strict contractions remain unchanged; always
normalizing to the boundary would lose that interior-surjectivity argument.
Any generic stable gain-bounded LTI map has a small-block modal realization
and a dense bounded-real certificate in those coordinates. With that certificate,
its weighted norm is strictly below one, so the construction returns it unchanged.
Exact larger Jordan chains remain outside both small-block constructions.

Implementation uses one right triangular solve for the weighted state columns
and one full system spectral norm. Neither a dense `P`, its Cholesky, nor a
Lyapunov solve is needed during the training forward. The dense energy-coordinate
state matrix is needed only inside the norm; the recurrence and `step()` use
the original real blocks. Cached inference reuses the physical projections.
State energy is evaluated as `||L x||²`, without assembling `P`.
Preparation and backward remain `O(n³)` for narrow ports, with `O(n²)` memory;
full/full runtime ordering must be measured. Full L2N supports float32/float64
parameters and keeps its certificate preparation at parameter precision under
autocast. The complex rotation convolution remains identity-mode only.

Avoiding SVD does not make full storage cheap: full defect also forms dense
`R.T @ R`, solves the block-pair Lyapunov equations, constructs a dense Gram
matrix, factors it, and runs triangular solves. Construction and backward can
cost more than the existing identity-storage normalization baseline, even though
both use fast small-block scan after their matrices have been prepared.
The spectral-norm loss also uses a simpler singular-value-only backward than
the general derivative through all SVD outputs. PyTorch's implementation uses
`U @ diag(gS) @ V.T` when the singular vectors are not outputs of the loss;
it still forms a dense product. See [the PyTorch 2.11 backward implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/autograd/FunctionsManual.cpp).

To plot forward plus backward time against the internal state dimension:

```bash
python scripts/compare_defect_vs_normalization.py --state-sweep \
  --output /tmp/defect-state-sweep.json
```

This measures `n = 6, 8, 16, 32, 64, 128, 256, 512, 768, 1024`, holding the
input/output width at 16, batch size at 4, and sequence length at 128. The first
point is 6 because the identity L2N reference requires even state dimensions. It uses one CPU thread,
five warmups, and 20 timed repetitions by default. Each repetition includes
parameter construction, gradient reset, forward, loss, and backward; optimizer
updates are excluded. Training reconstructs the matrices rather than using the
inference cache.

Alongside JSON/CSV, this saves `*-forward-backward.png` and
`*-forward-backward.pdf`. The curves show medians, with 25th–75th percentile
bands; JSON retains every timing sample. Override the dimensions with
`--state-sizes`, `--widths`, `--batch-sizes`, and `--lengths`. Different
width/batch/length combinations get separate panels. Add `--plot` to other
comparisons, or `--workload all` to retain all timing categories in a sweep.
See the [CPU state sweep](benchmarks/defect_state_sweep_cpu_2026-10-05.md)
for a completed run.

To also measure and plot complete training steps, including Adam updates:

```bash
python scripts/compare_defect_vs_normalization.py --state-sweep --include-optimizer \
  --output /tmp/defect-adam-state-sweep.json
```

This saves both the forward/backward plot and an additional
`*-training-step.png` / `*-training-step.pdf` plot. Adam uses learning rate
`0.001` (override with `--optimizer-lr`) and the same implementation for both
cells. Each timed step includes gradient reset, parameter construction, forward,
loss, backward, and an actual optimizer update. A fresh cell starts from the
same initialization as the forward/backward measurement, then its weights
evolve through warmup and timed steps on the same synthetic input batch.
Optimizer creation and first-use buffer allocation are excluded; complete-step
timing performs at least one warmup even with `--warmup 0`. These measure step
cost, rather than time to reach a particular accuracy.
See the [CPU sweep including Adam](benchmarks/defect_adam_state_sweep_cpu_2026-10-05.md)
for a completed run.

For a larger sweep on a GPU:

```bash
python scripts/compare_defect_vs_normalization.py --device cuda \
  --state-sizes 64 128 256 --widths 16 \
  --lengths 128 512 --batch-sizes 4 16 --warmup 10 --repeats 50
```

Widths may be one shared value or one value per state size. Both cells receive
identical inputs and use `scan` by default; use `--l2n-mode conv` to compare
the old model's convolution engine against defect's scan. Device operations
are synchronized around each repetition. Warmup is excluded and compilation
is not enabled. Add `--train-steps 80` for the separate learning experiment.

The tests compare the optimized formulas against independent dense symmetric
square roots, check the complete one-step energy inequality, measure a finite
horizon operator norm, compare loop/scan trajectories and all parameter
gradients, run finite-difference gradient checks, exercise checkpoints/caches
and chunked inference, train a certified DeepSSM, and check full-graph
compilation with `aot_eager`.

The comparison script measures uncached matrix setup, cached inference, full
forward/backward, parameter storage, retained matrices, and a shared synthetic
learning task over several seeds. It supports `--device cuda` and measures CUDA
peak training allocation when available. CPU storage counts are not estimates
of peak activation memory. Both models use the same task and optimizer, but
have different parametrizations and initial input-output maps; this is a smoke
comparison, not a tuned accuracy ranking. Timings use eager execution.

See the [local CPU measurements](benchmarks/defect_cpu_2026-10-02.md) for the
initial results, including the cases where the existing normalization model
is faster or learns better.

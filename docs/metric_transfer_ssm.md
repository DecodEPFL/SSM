# Metric-transfer Deep SSM

`MetricDeepSSM` is a separate architecture with a prescribed whole-network
incremental l2 gain. It supports both `defect` and exact `l2n` cores. The original
`DeepSSM` remains available for existing experiments and checkpoints.

```python
import torch
from neural_ssm import MetricDeepSSM, MetricSSMConfig

config = MetricSSMConfig(
    d_model=16, d_state=64, n_layers=4, param="defect",
    gamma=1.0, activation="tanh", ff="residual", d_hidden=32,
    state_metric="full", max_radius=1.0, factor_margin=0.0,
)
model = MetricDeepSSM(3, 2, config=config)
u = torch.randn(8, 256, 3)
y = model(u, return_state=False)  # (8, 256, 2)
y.square().mean().backward()
```

Use `param="l2n"` to change the LTI parametrization. `state_metric="full"`
works for either core and allows a final scalar for odd state sizes; identity
L2N requires an even size. Defect's default identity storage retains the original
thin completion. Its full storage supports block sizes up to four;
`block_size=2` also accepts an odd state dimension with a final scalar pole.
Otherwise, `block_size` must divide `d_state`. With identity storage,
`block_size=d_state` allows a generic dense contractive state matrix at the cost
of dense recurrent matrix computations.

## Which LTI blocks fit?

The required property is a **bounded-real contraction**, not just stable poles.
A core exposes `normalized_block_matrices()` returning `(A_blocks, B, C, D)` with

$$
\bar K=\begin{bmatrix}\operatorname{blkdiag}(A_{\rm blocks})&B\\ C&D\end{bmatrix},
\qquad \bar K^\top\operatorname{diag}(P,I)\bar K
\preceq\operatorname{diag}(P,I),\quad P\succ0.
$$

`A_blocks` has shape `(number_of_blocks, block_size, block_size)`. A dense
`A` of shape `(n,n)` is also accepted and treated as one block. The other
shapes are `(n,m)`, `(m,n)`, and `(m,m)`, where `m=d_model`.

The original cores use `P=I`. A core with a different storage matrix also
exposes `state_energy(x)` returning `x.T @ P @ x` for each state. This lets the
stack report the correct initial and terminal energies without changing the
coordinates used by scan.

One way to obtain the identity-storage form from a bounded-real certificate
`P=F_P.T @ F_P` and gain `g` is to transform the matrices to

$$
\bar A=F_P A F_P^{-1},\quad \bar B=F_P B/g,\quad
\bar C=C F_P^{-1},\quad \bar D=D/g.
$$

An arbitrary stable LRU does not provide this certificate by itself. Whitening
with a dense `P` generally destroys small-block structure. The full-metric defect
core instead solves `P-A.T @ P @ A=R.T @ R` for its block `A`, and completes
`B,C,D` directly in those coordinates. Thus it keeps block scan while allowing
a dense certificate. Its LTI core covers generic strict gain-bounded transfer
maps with the open limits above; exceptional non-diagonalizable systems may need
larger blocks. See [the construction](defect_parametrization.md). A fixed gain
`g` is an input scaling in this interface, not an extra accumulated layer bound.

Full-metric L2N instead measures the weighted norm using `L A L^-1`, with
`P=L.T @ L`, then divides its original free block matrices by a scalar. Its
recurrence keeps the original blocks; it has the same generic strict LTI
expressivity as full-metric defect. The original L2N's post-normalization `S`
is omitted in this stack and is unrelated to the new certificate factor `L`.

Use `core_factory=lambda n, ni, no: YourCertifiedCore(n, ni, no)` to add a core.
It must return an `nn.Module` exposing `d_state`, `d_input`, `d_output`, and the
method above, plus `state_energy` when its storage is not the identity. The
optional `scan_state_energy` hook takes precedence when a core uses another
state basis outside this protocol, as standalone identity L2N does with `S`. The
factory takes precedence over `config.param`. The contraction
is an algebraic contract the core must enforce throughout training; the stack
does not estimate it from sampled inputs or run a large SVD to validate it on
every forward. The current protocol is LTI, not an input-dependent selective
core protocol.

## What a layer does

The architecture is

`contractive encoder -> [LTI core -> activation -> weighted residual FF] x L -> contractive decoder`.

The LTI core uses the pre-update state to read out its output:

$$x_{t+1}=A x_t+B u_t,\qquad z_t=Cx_t+Du_t,\qquad y_t=\sigma(z_t).$$

The direct term `D` supplies a fast path through the temporal part. The residual
feedforward adds its own identity path. There is no extra unrestricted temporal
identity skip or arbitrary GLU branch.

For each intermediate feature interface, learn diagonal positive `V` and a
lower triangular `L` with positive diagonal. Set

$$
S=LL^\top,\qquad Q=2V^{1/2}S(I+S)^{-1}V^{1/2},\qquad
W=\tfrac12V^{1/2}(I+S)V^{1/2}.
$$

Then `0<Q<2V` and `W=V(2V-Q)^{-1}V`. This allows `Q>V` in some directions;
it does not impose the smaller `Q<=V` range. Small positive diagonal floors
keep the finite parametrization away from singular boundaries.

For any componentwise activation with slope in `[0,1]`, the sector inequality
`dy.T @ V @ (dz-dy) >= 0` implies

$$\|\delta y\|_Q^2\leq\|\delta z\|_W^2.$$

Writing `Q_prev=F_prev.T @ F_prev`, `W=F_W.T @ F_W`, the prepared core is

$$A=\bar A,\quad B=\bar B F_{\rm prev},\quad
C=F_W^{-1}\bar C,\quad D=F_W^{-1}\bar D F_{\rm prev}.$$

Combining the core contraction and sector inequality gives

$$
\|\delta x_{t+1}\|_P^2+\|\delta y_t\|_Q^2
\leq\|\delta x_t\|_P^2+\|\delta u_t\|_{Q_{\rm prev}}^2.
$$

### Certified residual feedforward

The optional FF is `f(u)=u+O sigma(Hu)`. Its free parameters are a rectangular
contraction `R` and a positive diagonal multiplier `Lambda`. With
`Q=F_Q.T @ F_Q`, set

$$
H=\sqrt2\Lambda^{-1/2}RF_Q,\qquad
O=-\sqrt2 F_Q^{-1}R^\top\Lambda^{1/2}.
$$

These coupled weights satisfy `QO=-H.T @ Lambda` and
`O.T @ Q @ O <= 2 Lambda`. Expanding the difference energy gives

$$
\|\delta f\|_Q^2-\|\delta u\|_Q^2
\leq-2\delta a^\top\Lambda(H\delta u-\delta a)\leq0,
\quad \delta a=\sigma(Hu)-\sigma(H\tilde u).
$$

Thus the **complete residual map** is nonexpansive in `Q`. The FF has no
independent scalar gain budget. Set `ff="none"` to omit it.

### Whole-network guarantee

The feature metric leaving a layer is exactly the metric entering the next.
Set `Q_0=gamma² I` and `Q_L=I`; sum the layer inequalities over depth and time.
Intermediate feature energies cancel. Contractive encoder/decoder maps preserve
the endpoint inequalities, yielding

$$
\sum_{t=0}^{T-1}\|y_t-\tilde y_t\|^2+
\sum_{\ell=1}^L\|x_T^\ell-\tilde x_T^\ell\|_{P_\ell}^2
\leq\gamma^2\sum_{t=0}^{T-1}\|u_t-\tilde u_t\|^2+
\sum_{\ell=1}^L\|x_0^\ell-\tilde x_0^\ell\|_{P_\ell}^2.
$$

Equal initial states give the prescribed incremental gain `gamma`. Zero initial
states, zero biases, and `sigma(0)=0` also give a zero-state induced-gain bound.
The supported activations are `tanh`, `relu`, `arctan`, and `identity`.

Unlike the existing scalar composition in `DeepSSM`, this certificate learns
full feature metrics and does not multiply independently assigned scalar layer
bounds. This can reduce conservatism; it does not guarantee better prediction
accuracy or unrestricted representational completeness. The terminal activation
uses `V=W=Q=I`, which avoids redundant terminal metric parameters.

## Computation and caching

All new architecture code is in `ssm/metric_layers.py`, in reading order:
configuration, feature metric, residual FF, layer, core construction, stack.
Existing cell files supply the unit-contraction protocol.

* Use `mode="scan"` for the existing shared-power parallel LTI scan. Dense `A`
  remains valid, but requires dense block products; small blocks are cheaper.
  `mode="loop"` is the sequential reference. FFT convolution is not currently
  exposed by this stack, including when its core is l2n.
* Build feature metrics and FF weights once per training forward. Cholesky and
  triangular solves replace explicit inverses. The transforms are folded into
  `B,C,D`, so there are no extra feature-metric products at every time step.
* With `model.eval()` and gradients disabled, cache the complete prepared stack.
  Any parameter/buffer update invalidates it, including a shared metric changing
  the next layer's input projection. Checkpoint loads and dtype/device changes
  are covered by the existing cache lifecycle. Training always rebuilds the graph.
* When trajectories are not requested, avoid retaining them across layers.
  Inference and detached streaming states own only `(batch,n)` storage, so a
  final-state view does not keep an entire `(batch,T+1,n)` allocation alive.
* Defect uses thin contraction factors and never assembles dense state defects
  or performs a state-sized SVD. l2n retains its exact full-K spectral
  normalization but omits its unused dense state-basis parameter in this stack.
  Full state storage adds `n(n+1)/2` parameters to either construction. Defect
  needs block-pair Lyapunov solves and one dense Cholesky; L2N uses dense
  triangular transformations and a full system spectral norm. Full preparation
  costs `O(n³)` time and `O(n²)` memory for narrow ports. Full/full uses the same
  real small-block scan after preparation. Runtime ordering needs measurement.
* Feature-metric factorization costs `O(m³)` per layer, independent of `n=d_state`.
  Folding dense feature transforms into rectangular projections costs `O(n m²)`.
  The recurrent/sequence work is shared with the existing block scan. Large
  feature width can therefore still be expensive.
* Parameters support float32/float64. Under AMP, certificate construction stays
  at parameter precision while sequence projections can use autocast. Algebraic
  bounds are mathematical guarantees; floating-point/AMP arithmetic introduces
  the usual numerical error. Full-graph training is tested with `aot_eager`;
  this does not measure CUDA/Inductor performance.

`P` describes **state storage** and may be identity or full. The learned `Q`
between layers describes **feature energy**. They serve different purposes;
learning a full state certificate does not remove the restrictions of the
nonlinear activation or residual architecture. Both metrics stay fixed over
time for a prepared LTI layer.

## Streaming, tutorials, and benchmarks

The default return is `(y, trajectories)`, with one `(batch,T+1,n)` trajectory
per layer, including the initial state. To stream using final states:

```python
model.eval()
with torch.no_grad():
    y1, state = model(u[:, :128], return_state=False, return_last=True)
    y2, state = model(u[:, 128:], state=state,
                     return_state=False, return_last=True)
```

Alternatively, use `reset_state=False` to continue from stored internal states.
For truncated backpropagation use `detach_state=True` on the call producing the
state. `model.reset()` clears internal state.

```bash
python Test_files/Tutorial_MetricDeepSSM.py
python Test_files/run_benchmarks.py --benchmarks Cascaded_Tanks \
  --models defect l2n metric_defect metric_l2n --gamma 1 --ff MBLIP \
  --l2n-state-metric full --defect-state-metric full --defect-max-radius 1 --defect-factor-margin 0 \
  --metric-activation tanh --metric-ff residual --param-budget none
python Test_files/benchmark_ui.py
```

The UI has `metric_defect` and `metric_l2n` choices and a shared metric-stack
panel. Existing defect/l2n initialization controls apply to the corresponding
new core. Each core's **state storage** selector controls its ordinary and metric
architectures. CLI equivalents are `--defect-state-metric` and `--l2n-state-metric`.
`--ff` still selects the original DeepSSM branch; `--metric-ff` selects
the new weighted residual. `gamma` must be positive or `auto`; `none` is rejected
for metric stacks. Benchmark gain diagnostics refer to normalized data units.

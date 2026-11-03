# Output prediction with a learned linear path

This readout predicts in output units, adds a learned bias-free linear path,
and enforces one energy budget on the **combined returned output**:

```text
z_t = core_theta(u_0:t, context_0:t)
r_t = z_t + D u_t
a_t = gamma² ||u_t||²
B_t = E_t + a_t
y_t = r_t                         if ||r_t||² <= B_t
      sqrt(B_t) r_t / ||r_t||      otherwise
E_{t+1} = max(B_t - ||r_t||², 0)
E_0 = beta
```

At zero available energy the returned output is zero, including a zero
proposal. Norms sum over features only. Each batch row has a separate scalar
reservoir; all output channels share that row's budget. Context feeds the core
only. The learned linear map uses external energy-input features, not context.
The core and linear path are independent of reservoir energy and filtered output.

In real arithmetic,

```text
E_{T+1} + sum(t=0..T) ||y_t||² = beta + gamma² sum(t=0..T) ||u_t||².
```

With beta=0 this gives zero-state induced l2 gain at most gamma for the full
returned output, including the learned linear path. Neither D nor the neural
core needs a separate gain constraint. Finite beta gives an additive allowance.
The guarantee does not imply hidden-state stability, incremental gain,
contraction, successful optimization, or universal finite-network expressivity.
For l2 external input and finite beta, bounded persistent context cannot break
output square summability because it never supplies energy. Context alone
produces zero output when external input and beta are zero.

## Use in the benchmark UI

Restart an already-open UI, then select **direct_spending** in Models. Under
**Model options**, choose **Readout → Output prediction + linear**, then choose
any existing neural core: MLP, GRU, LSTM, TCN or Transformer. The original
**Spending fractions** option retains its behavior and remains the default for
old settings. No duplicate model entries were added.

**Linear path start** is visible only for the new readout:

- `zero` (default): D starts at zero and is learned. Backbone initialization
  matches the corresponding spending model for the same seed/configuration.
- `identity`: initializes the diagonal of the rectangular D matrix to one and
  all other entries to zero. D remains fully learnable.

The path is native `nn.Linear(n_input,n_output,bias=False)`. It adds
`n_input * n_output` trainable parameters. **Count parameters** uses the real
factory, including this matrix and any width adjustment from parameter matching.
Save as defaults and saved UI configuration files include the readout and
linear initialization; existing saved defaults are not rewritten by installation.

```bash
python Test_files/run_benchmarks.py --models direct_spending \
  --spending-readout projected --spending-linear-init zero --spending-core lstm \
  --d-model 16 --n-layers 2 --spending-gamma 10 --spending-beta-mode zero \
  --param-budget off --benchmarks Cascaded_Tanks \
  --out direct_spending/output_projection/validation/my_run
```

The existing fixed/learned gamma controls both work. Learned gamma remains one
bounded positive global scalar. Reports distinguish its fitted value from its
initial value and configured ceiling. Hold weights/gamma fixed through each
streamed trajectory; reset states after optimizer updates.

## Python and streaming

```python
from direct_spending import DirectSpendingModel, SpendingConfig

cfg = SpendingConfig(core="lstm", readout="projected", linear_init="zero",
                     width=16, layers=2, gamma=2., context_features=3)
model = DirectSpendingModel(1, 1, cfg)
y, state = model(u, context=c)  # u: (batch,time,1), c: (batch,time,3)
```

Use `model(next_chunk, state, context=next_context)` or
`model.step(state, u_t, context=c_t)`. Preserve core state and energy across
chunks. `initial_features` remains a separate once-per-trajectory beta context.
There is no silent detachment. The existing explicit detach/device/dtype
helpers support the unchanged `SpendingState` structure.

Padding emits zero, pauses both neural and energy state, and adds no input
energy or padding-token gradients. Masked NaN padding is removed before the
linear operation and energy arithmetic. The scan uses the identity pair on
padding, preserving incoming-state gradients even for an empty tank.

The D matrix is available as `model.proposal_readout.linear.weight`. The neural
residual is the existing final core readout. Projection always follows their
sum. With a zero neural residual, D=I, beta=0 and gamma=1, signed identity
outputs are preserved exactly by the normal arithmetic path.

## Scan, precision and differentiation

The online reference implements the recurrence above. The sequence path reuses
the repository's work-efficient associative scan for pairs representing
`F(E)=max(E+a,b)`. Pair composition is

```text
(a1,b1) star (a2,b2) = (a1+a2, max(b1+a2,b2)).
```

Scan `(a_t-q_t,0)` and recover `E_after=max(E_init+A_prefix,B_prefix)`.
Padding uses `(0,-infinity)`. The sequence path has no Python loop over time.

To avoid overflow and cancellation from enormous raw proposals, use
`q_t=min(||r_t||²,C_t)`, where `C_t=E_init+sum(k<=t) a_k` bounds every reachable
available budget. This changes no real-arithmetic reachable transition: when
the cap is active, both the uncapped and capped recurrence exhaust the tank.
Ordinary proposal energies use direct native squares/sums, preserving exact
under-budget comparisons. Very large proposals use scaled norms without
forming overflowing squares. Every evaluated autograd branch is safe at zero.
This max-plus scan avoids subtracting two large cumulative sums.

Automatic energy precision is FP64 on CPU/CUDA and FP32 on MPS; outputs retain
at least FP32. MPS FP64 is explicitly unsupported. Under-budget proposal values
are unchanged. Clipped outputs have a small conservative rounding guard, which
can discard energy rather than refill the reservoir. Consequently the identity
is exact in real arithmetic and audited to dtype-appropriate tolerances against
actual returned outputs in floating point. State and output tolerances are
separate when FP32 outputs accompany FP64 energy arithmetic.

The scan audits its local recurrence and prefix energy balance. A failed audit
warns and uses the sequential reference, reporting `reference_roundoff`.
Normal scan diagnostics report `scan_max_plus`; online steps report `reference`.
Tests exercise the fallback explicitly and verify no fallback during the stated
long saturation/refill stresses. Nonfinite input/proposal/energy arithmetic is
reported; the certificate does not prevent neural overflow or make unrepresentable
input-energy totals representable.

Autograd passes through the carried energy, D, core, beta and learned gamma.
At exhaustion the positive-part state update chooses zero subgradient; equality
in the output budget selects the unchanged proposal. These are finite boundary
conventions for a genuinely nonsmooth map. Finite-difference checks away from
switching boundaries and finite gradients at zero/boundaries are tested.

Projection work beyond forming the proposal is linear in batch/time/features,
with O(batch*time) scalar scan storage and O(batch) persistent energy. The D
path has ordinary batched-linear cost O(batch*time*n_input*n_output). Core costs
remain architecture-dependent. Transformer caches are real KV caches, not
full-prefix recomputation. No new custom kernel or compilation requirement.

## Results and checkpoints

Diagnostics include clipping fraction, mean fraction of available energy spent,
energy traces, combined raw-proposal norms and independently normalized prefix
energy/identity errors. No traces are retained by default. The UI summarizes
trace lengths in run details; full traces remain in the optional report.
Clipping here refers to the **combined** proposal exceeding the current budget,
not the original spending head's unit-ball saturation.

New checkpoints use `energy_projection_v1` / family `energy_projection` and
store the readout configuration, D, core, gain/beta parameters, dtype, normalizer,
history and actual certificate. The common `load_checkpoint` loader accepts
this tag and old `direct_spending_v1` checkpoints, verifies the readout agrees
with the tag, and loads strictly. Original `energy_bounded/` formats remain
separate. Run folders append `--projected-linear` to avoid overwriting the
spending-fraction artifacts for the same core/beta choice.

The benchmark certificate applies in its selected coordinates. Standardized
coordinates include centering and do not establish an absolute physical-output
zero-state bound after adding the output mean back. Choose physical coordinates
when that is the intended signal guarantee. The native split/warm-up and target
initial-energy audit remain unchanged; no gain or beta is adjusted after seeing
validation to hide an infeasible target.

## Reproduce validation and the small comparison

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest direct_spending.output_projection.test_projection -v
BENCHMARK_UI_TK_TESTS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest direct_spending.output_projection.test_projection -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s direct_spending/tests -t . -v
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python -m direct_spending.output_projection.comparison \
  --output direct_spending/output_projection/validation/my_comparison --epochs 50
```

The comparison uses the existing trainer, identical initial core weights, D=0,
fixed gamma, beta=0, native trajectory splits and one seed. It records prediction
errors, energy diagnostics, parameter counts, training time, warmed CPU core-only
and full-model inference/backward timings, step latency and persistent cache
storage at increasing contexts. It saves/reloads both checkpoint types. Timing
is CPU-only, compilation is off, and warmup is excluded from steady-state samples.
Single-seed accuracy and short timing samples are exploratory.

See [REPORT.md](REPORT.md) for measured results. Visual UI verification remains
unavailable as requested; native Tk behavior is checked programmatically.

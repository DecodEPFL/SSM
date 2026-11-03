# Cascaded Tanks Transformer training diagnostic — 2026-11-03

The latest saved suite report records **learning rate 3000**, gradient clip 5,
and a width-16, three-layer direct-spending Transformer (2 heads, dropout 0,
gamma 10, learned scalar initial energy starting at 0.1). It failed with
`Learned softplus beta underflowed to zero`. The saved UI defaults instead
contain `3e-3` (0.003), so the defaults were preserved. The saved report may not
describe another run still open in a UI.

An isolated reproduction at rate 3000 failed with the same error. A 40-epoch
pilot at 0.003 remained finite, followed by this small 80-epoch comparison:

| Rate | Training MSE, first → last | Training loss increases | Best validation MSE | Validation RMSE |
| --- | --- | --- | --- | --- |
| 0.003 | 97.1255 → 12.6221 | 21 / 79 transitions | 15.3762 | 8.4900 |
| 0.0003 | 97.1255 → 87.5144 | 0 / 79 transitions | 90.0290 | 20.5436 |

MSE is in standardized coordinates; RMSE is in the dataset's physical units,
with the official 50-sample validation initialization window excluded. The
smaller rate smoothed training but learned much more slowly. Validation was
not monotonic at either rate (23 and 13 increases respectively). Both fits are
poor after this short schedule. These are single-seed diagnostics, not an
architecture ranking or a selected optimal learning rate.

The full native train and validation trajectories each have 1024 samples.
There is one training trajectory: batch size 32 still gives one optimizer
update per epoch. Seed 0 and identical initialization were used; AdamW,
weight decay 0, gradient clip 5, and the existing trainer's 80-epoch cosine
schedule were fixed across the two runs. The compressed schedule differs
from the reported 2000-epoch job. No dropout, plotting, compilation, or AMP.
Apple M1 Max, CPU with two threads, Python 3.13.9, PyTorch 2.9.0; FP32 neural
computation and automatic FP64 storage. Successful training/evaluation runs
took about 3–4 seconds each; these wall times are not controlled throughput
benchmarks and exclude the separate reference check. There are 6754 parameters.

Gamma and initialization were unchanged. The necessary target-prefix audit
requires zero additional initial energy at gamma 10 in these coordinates,
for both native trajectories; it does not prove neural representability or
optimization feasibility. Learned beta follows the existing additive-initial-
energy certificate, rather than a zero-initial-energy certificate.

At initialization, the full training trajectory produced identical outputs,
MSE, and parameter gradients with the scan and sequential implementations
(maximum output difference 0, relative gradient difference 0). Gradients were
finite. Initial head saturation was 4.004%; gradient norm before clipping was
21.3453. The worst normalized energy violation was 1.36e-8 and identity error
1.61e-8, within the existing FP32 audit tolerance of 2e-6. The two selected
checkpoints had zero head saturation and no positive measured prefix energy
violation. This check gives no evidence of a scan error at this initialization;
it is not a proof covering every optimization iterate or backend.

The certificate controls returned-output energy, not optimizer stability.
Spending changes current predictions and future available storage, while the
unit-ball head introduces switching boundaries. Those make long-trajectory
optimization sensitive; the experiments establish learning-rate sensitivity,
not which individual mechanism accounts for every oscillation. Smoother loss
alone is an insufficient selection criterion. A useful next optimization
experiment is to measure gradient norms and storage sensitivity while testing
an explicit rate schedule or head initialization, keeping gamma and beta policy
fixed and comparing accuracy under the same training budget.

The UI now previews the parsed learning rate as a decimal and repeats it in
the launch log. Malformed shared and direct-spending numeric entries name the
field and prevent launch/save. Values are never silently repaired or clamped.
Restart an already-open UI to load the change. No model equations, optimizer
defaults, saved UI defaults, or user benchmark outputs were modified.

Changed source files: `Test_files/benchmark_ui.py`, `direct_spending/ui.py`,
`direct_spending/ui_layout.py`, new `direct_spending/ui_numbers.py`,
`direct_spending/tests/test_ui.py`, `direct_spending/README.md`, and new
`direct_spending/validation/learning_rate_check.py`. All additions and diagnostic
artifacts are inside `direct_spending/`, apart from the existing UI wiring.

Validation: the UI/integration suite ran 20 checks, with 19 passing and the
opt-in native Tk check skipped. The full native Tk UI suite then passed all
12 checks, including the previously skipped check, decimal previews, conditional
controls, saved defaults, and narrow-window geometry. Visual UI verification
remains **unavailable**, as requested. This diagnostic was run on CPU; no new
MPS/CUDA training claims or fallback changes are made.

Reproduce from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m direct_spending.validation.learning_rate_check \
  --output /tmp/spending-rate-check.json --lrs 3000 0.003 0.0003 --epochs 80
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  direct_spending.tests.test_ui direct_spending.tests.test_integration -v
BENCHMARK_UI_TK_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  direct_spending.tests.test_ui -v
```

The diagnostic writes only the requested JSON and reuses cached dataset splits.
`learning-rate-comparison.json` contains complete losses, configuration,
reference checks, selected-checkpoint metrics, and energy diagnostics.
`learning-rate-losses.png` plots the successful runs. `observed-user-run.json`
records the relevant settings and error from the original saved report.

# Full state metric: CPU measurements, 2026-10-09

The defect cell now supports a full compatible state metric without changing
its small-block recurrence. Enable it with `state_metric="full"`,
`max_radius=1`, and `factor_margin=0` on the standalone cell. DeepSSM prefixes
these fields with `defect_`. Identity storage remains the existing default.

The implementation learns `Q=R.T @ R`, solves `P-A.T @ P @ A=Q` blockwise,
and completes the input/output matrices using the existing thin contractions.
Its exact implicit backward reuses the block-pair LU factors for the adjoint
equation. One dense Cholesky and dense products/triangular solves remain in
preparation. Full storage adds `n(n+1)/2` parameters and makes setup cubic;
the sequence scan still uses the same size-two blocks. See
[the construction and expressivity scope](../defect_parametrization.md).

## Measurement conditions

macOS 15.6.1, ARM64, PyTorch 2.11.0, eager CPU execution, float32, one thread.
Input/output width 16, batch 4, sequence length 128, state blocks of size two,
gain 1, seed 0. Each number is a median of 20 repetitions after five warmups.
The two runs execute separately; small differences in cached inference can
reflect timing variability. No concurrent test or tutorial workload ran during
these measurements. CUDA was unavailable and was not measured.

Training steps include fresh parameter construction, gradient reset, forward,
loss, backward, and an actual Adam update (`lr=0.001`, `foreach=False`).
Optimizer construction and initial moment-buffer allocation are excluded.
Cached inference uses `eval()` and `no_grad()` after priming the matrices.
Initial radii are 0.9; normalization changes its realized radius. The full
defect run uses open limits, while identity defect retains its existing
0.999 radius limit and 0.001 factor margin. Models receive identical inputs,
but their parameters and initial transfer maps differ.

## Complete training steps

Times are milliseconds.

| State size | Defect, identity P | Defect, full P | Existing L2N | Full P / L2N |
| --- | ---: | ---: | ---: | ---: |
| 128 | 4.212 | 6.708 | 4.575 | 1.47 |
| 256 | 5.926 | 14.291 | 9.553 | 1.50 |
| 512 | 9.780 | 44.907 | 26.830 | 1.67 |
| 1024 | 17.937 | 157.459 | 109.567 | 1.44 |

Full storage is substantially more expensive than identity storage. It also
trains more slowly than the existing normalization baseline in these CPU
cases. The existing L2N scan operates with identity storage in its block
coordinates; its dense `S` changes returned state coordinates rather than
removing that input/output restriction. This is a runtime reference against
a more restricted family, not a comparison of equally expressive full-metric
constructions or of time to reach a particular prediction accuracy.

## Cached sequence inference

| State size | Defect, identity P | Defect, full P | Existing L2N |
| --- | ---: | ---: | ---: |
| 128 | 0.741 | 0.740 | 0.791 |
| 256 | 1.248 | 1.282 | 1.149 |
| 512 | 2.328 | 2.303 | 2.017 |
| 1024 | 5.383 | 4.875 | 5.130 |

Once the matrices are cached, full storage does not require applying dense P
at each time step. These results show similar inference costs, not a consistent
speed advantage. Explicit state-energy diagnostics still need a dense product.

At `n=1024`, full defect has 560,688 parameters versus 35,888 for identity
defect: the difference is exactly 524,800 metric parameters. CPU byte fields
in the raw files count parameters and returned diagnostic matrices, not all
inference caches, optimizer buffers, or peak training activations.

## Verification and reproduction

The test suite ran 169 tests: 167 passed and two CUDA tests were skipped.
Checks include weighted bounded-real inequalities, a transfer map excluded by
identity block storage at the same gain, scan/loop outputs and gradients,
first and second derivatives of the Lyapunov solver, odd state sizes, streaming,
cache invalidation, AMP preparation, and full-graph `aot_eager` compilation.
The three step-by-step tutorials also run successfully. Existing default
identity-cell outputs, states, and gradients match the saved pre-change source
exactly in a seeded float64 comparison.

```bash
PYTHONPATH=src python scripts/compare_defect_vs_normalization.py \
  --defect-state-metric full --state-sizes 128 256 512 1024 --widths 16 \
  --lengths 128 --batch-sizes 4 --repeats 20 --warmup 5 --workload all \
  --include-optimizer --output /tmp/full-metric.json
```

Repeat with `--defect-state-metric identity` for the original defect family.
The raw files include construction timings, forward/backward without Adam,
individual training samples, and scan/loop errors:
[full storage JSON](defect_full_metric_cpu_2026-10-09.json),
[full storage CSV](defect_full_metric_cpu_2026-10-09.csv),
[identity storage JSON](defect_identity_metric_cpu_2026-10-09.json), and
[identity storage CSV](defect_identity_metric_cpu_2026-10-09.csv).

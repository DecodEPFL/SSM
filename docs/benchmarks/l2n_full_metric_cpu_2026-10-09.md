# Full-metric normalization versus full-metric defect

Both full modes cover generic real LTI transfer maps of the selected order with gain strictly below the prescribed bound. Each uses the same real small-block scan and a dense state certificate. Exact larger Jordan chains are excluded. Initial transfer maps and parameterizations differ, so this measures runtime rather than learning quality.

CPU, PyTorch 2.11.0, float32, one thread; state sizes 128/512/1024, input/output width 16, batch 4, length 128, seed 0. Median of 20 repetitions after five warmups, with synchronized wall times. No other tests or benchmark processes ran during these measurements. GPU performance was not measured.

Each training forward rebuilds its matrices and gradient graph; cached inference uses evaluation mode with gradients disabled. Adam uses learning rate 0.001 and foreach=False, with first-use allocation excluded. The identity L2N reference retains its historical learned returned-state basis S; full L2N has no extra S.

| States | L2N full: build+bwd | Defect full: build+bwd | L2N full: fwd+bwd | Defect full: fwd+bwd |
|---:|---:|---:|---:|---:|
| 128 | 1.98 ms | 4.08 ms | 4.97 ms | 7.42 ms |
| 512 | 22.04 ms | 36.71 ms | 32.49 ms | 47.89 ms |
| 1024 | 105.42 ms | 169.95 ms | 128.14 ms | 184.77 ms |

| States | L2N full: fwd+bwd+Adam | Defect full: fwd+bwd+Adam | L2N full: cached inference | Defect full: cached inference |
|---:|---:|---:|---:|---:|
| 128 | 5.52 ms | 7.60 ms | 0.89 ms | 0.80 ms |
| 512 | 37.53 ms | 47.20 ms | 3.44 ms | 2.75 ms |
| 1024 | 133.38 ms | 184.21 ms | 6.51 ms | 6.71 ms |

Full L2N was faster for training in this run. It learns P through its triangular factor, builds the weighted norm with a triangular solve, and uses one full system spectral norm. It avoids the defect construction’s block-pair Lyapunov solver and dense completion Cholesky. Both still have cubic preparation/backward for narrow input/output widths, so these CPU results do not establish universal hardware or library-version ordering.

All four variants, timing samples, parameters, scan/loop errors, and optimizer timings are retained in the raw results.

```bash
python scripts/compare_defect_vs_normalization.py \
  --l2n-state-metric full --defect-state-metric full \
  --state-sizes 128 512 1024 --widths 16 --batch-sizes 4 --lengths 128 \
  --workload all --include-optimizer --plot --output /tmp/full-metric-comparison.json
```

[Raw JSON](l2n_full_metric_cpu_2026-10-09.json), [CSV](l2n_full_metric_cpu_2026-10-09.csv), [forward/backward figure](l2n_full_metric_cpu_2026-10-09-forward-backward.png), [Adam figure](l2n_full_metric_cpu_2026-10-09-training-step.png).

The full library suite passed: 188 tests run, 186 passed, two CUDA-only tests skipped. Full L2N checks include an independently constructed weighted norm, bounded-real inequalities, a formerly excluded cancellation transfer, distinct real poles, odd-state scan/loop/step agreement, finite-difference gradient checks, active metric gradients, streaming, cache invalidation, checkpoints, AMP preparation, full-graph compilation, DeepSSM/MetricDeepSSM/context integration, and CLI exports. Legacy L2N outputs, states, and all gradients were verified bit-for-bit against a saved pre-change run.

# Measured pilot comparison

Hardware: {'machine_model': 'MacBookPro18,2', 'chip_type': 'Apple M1 Max', 'physical_memory': '32 GB', 'number_processors': 'proc 10:8:2'}. macOS-15.6.1-arm64-arm-64bit-Mach-O; Python 3.13.9; PyTorch 2.9.0; device cpu; model dtype float32; energy torch.float64; 1 CPU thread(s).

Fixed gamma=1.0, E0=0. Seeds [0, 1, 2]; 120 Adam steps/arm, lr=0.003, batch=8, train length=64, long held-out length=256, width=16. No tuning sweep or compilation. Identical initial weights and training batches within each raw/wrapped pair; independent trajectory splits.

The teacher has analytic zero-state gain <=.9; all generated target prefix audits pass at the fixed gain. Longer evaluation is a forced response on fresh trajectories; autonomous rollout is not applicable to this exogenous-input task.

## Prediction and actual training cost

Mean ± sample standard deviation across seeds. CPU training time excludes cloned warmup and dataset creation, and includes forward, backward, optimizer, dispatch and scalar loss logging. Equal steps/width do not imply equal parameter or compute budgets.

| Core / arm | Parameters | Test MSE | Long MSE | Clip fraction | Worst prefix violation | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gru / raw | 1,714 | 0.00055615 ± 0.00017315 | 0.00054451 ± 0.0001454 | — | 0.0202 | 0.6944 ± 0.03952 |
| gru / wrapped | 1,714 | 0.00049858 ± 0.00010824 | 0.00052949 ± 0.00012097 | 0.04688 ± 0 | 0 | 0.7638 ± 0.04757 |
| tcn / raw | 2,434 | 0.0016583 ± 0.0005348 | 0.0015351 ± 0.00027396 | — | 0.0182 | 0.1977 ± 0.003375 |
| tcn / wrapped | 2,434 | 0.0015564 ± 0.00045901 | 0.0014696 ± 0.0002221 | 0.04688 ± 0 | 0 | 0.2546 ± 0.00825 |
| transformer / raw | 4,562 | 0.0095724 ± 0.00027033 | 0.044186 ± 0.010829 | — | 0.00498 | 0.401 ± 0.01469 |
| transformer / wrapped | 4,562 | 0.066985 ± 0.09934 | 0.11427 ± 0.1173 | 0.3604 ± 0.543 | 0 | 0.5007 ± 0.0656 |

Normalized prefix violation is `max(0, max_{sample,prefix} (spent - supplied)/(1 + supplied))`, audited in FP64 from actual outputs. Raw clipping is undefined; its hypothetical reservoir clipping is recorded separately in JSON.

Across ALL wrapped validation/test/long evaluations, worst normalized prefix violation: **0**. Largest normalized absolute energy-identity error: **9.45731e-07**. Observed evaluation backends: ['scan'].

### Seed variation

| Core | Seed | Raw test MSE | Wrapped test MSE | Wrapped long MSE |
| --- | ---: | ---: | ---: | ---: |
| gru | 0 | 0.000420854 | 0.000393336 | 0.000427746 |
| gru | 1 | 0.000496301 | 0.000492819 | 0.000497478 |
| gru | 2 | 0.000751286 | 0.000609579 | 0.000663246 |
| tcn | 0 | 0.00123216 | 0.00121118 | 0.00130767 |
| tcn | 1 | 0.0022584 | 0.00207726 | 0.0017228 |
| tcn | 2 | 0.0014842 | 0.00138062 | 0.00137834 |
| transformer | 0 | 0.0096046 | 0.009426 | 0.0601928 |
| transformer | 1 | 0.00982525 | 0.00983698 | 0.0337551 |
| transformer | 2 | 0.00928747 | 0.181693 | 0.248851 |

These are exploratory runs of small fixed configurations. Transformer wrapped seed 2 has 19.3× the best Transformer wrapped error while preserving its certificate. gru has the lowest mean wrapped test error and tcn the lowest mean wrapped training time in this pilot. The seed table and complete loss curves expose variation. These observations do not establish a universal architecture winner.

## Isolated core and wrapper timing

Medians of 7 synchronized repetitions after 2 warmup calls per workload. These timing pairs use the SAME trained wrapped-core weights and the same inputs, isolating wrapper overhead; timing uses the first seed only. Test processes were finished before the final timing run. Forward/backward timing excludes optimizer updates. Warmup durations and every timing sample are saved in JSON.

| Core | B×T | Path | Forward ms | Forward tokens/s | Fwd+bwd ms | Fwd+bwd tokens/s | State bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| gru | 8×64 | core | 0.7638 | 670,341 | 5.1850 | 98,746 | 512 |
| gru | 8×64 | core+wrapper | 1.0178 | 503,050 | 5.3072 | 96,473 | 576 |
| gru | 8×1024 | core | 12.5390 | 653,322 | 85.4446 | 95,875 | 512 |
| gru | 8×1024 | core+wrapper | 13.0428 | 628,084 | 83.7915 | 97,766 | 576 |
| tcn | 8×64 | core | 0.6648 | 770,118 | 1.3870 | 369,153 | 7,168 |
| tcn | 8×64 | core+wrapper | 0.7830 | 653,929 | 1.6852 | 303,828 | 7,232 |
| tcn | 8×1024 | core | 3.8040 | 2,153,523 | 10.6003 | 772,809 | 7,168 |
| tcn | 8×1024 | core+wrapper | 4.3983 | 1,862,541 | 11.8050 | 693,943 | 7,232 |
| transformer | 8×64 | core | 1.0150 | 504,413 | 2.7371 | 187,058 | 131,648 |
| transformer | 8×64 | core+wrapper | 1.3568 | 377,349 | 2.7950 | 183,187 | 131,712 |
| transformer | 8×1024 | core | 82.2165 | 99,639 | 153.0345 | 53,530 | 2,105,408 |
| transformer | 8×1024 | core+wrapper | 88.1702 | 92,911 | 153.8118 | 53,260 | 2,105,472 |

### Cached single-step latency and context growth

Batch 1; prefill excluded; each repetition branches from the same immutable incoming cache. Persistent bytes count logical tensor contents, including padding masks and position offsets, not allocator overhead or autograd graphs. CPU peak activation memory was not measured; CUDA allocation counters are only reported when CUDA is selected.

| Core | Context | Core µs | Wrapped µs | Core state bytes | Wrapped state bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| gru | 16 | 56.08 | 206.25 | 64 | 72 |
| gru | 128 | 48.46 | 184.17 | 64 | 72 |
| gru | 512 | 55.83 | 196.25 | 64 | 72 |
| gru | 1024 | 51.71 | 191.92 | 64 | 72 |
| tcn | 16 | 115.75 | 286.13 | 896 | 904 |
| tcn | 128 | 116.00 | 306.92 | 896 | 904 |
| tcn | 512 | 118.83 | 265.79 | 896 | 904 |
| tcn | 1024 | 105.79 | 254.21 | 896 | 904 |
| transformer | 16 | 227.71 | 371.58 | 4,120 | 4,128 |
| transformer | 128 | 225.96 | 378.83 | 32,904 | 32,912 |
| transformer | 512 | 320.29 | 401.13 | 131,592 | 131,600 |
| transformer | 1024 | 243.96 | 433.17 | 263,176 | 263,184 |

### Reservoir alone

| B×T | Requested path | Actual path | Forward ms | Fwd+bwd ms |
| --- | --- | --- | ---: | ---: |
| 8×64 | scan | scan | 0.2008 | 0.5111 |
| 8×64 | reference | reference | 3.0352 | 9.3357 |
| 8×1024 | scan | scan | 0.8393 | 1.9330 |
| 8×1024 | reference | reference | 59.9817 | 180.2269 |

## Validation

- Focused suite: 30 tests run, 29 passed, 1 accelerator check skipped; includes real SystemIDTrainer training/validation for all six arms.
- Existing repository suite: 188 tests run, 186 passed, 2 CUDA checks skipped (76.590 seconds).
- Source and results: All Python sources parse; comparison source hashes match delivered code.
- Repository preservation: Pre-existing tracked diff and every status entry outside energy_bounded/ match the recorded baseline.
- Numerical stress: 32,768-token scan; 8,192-token saturation/refill; 1e100 proposals use a reported sequential fallback; FP64 gradient checks and finite zero/switch/diagnostic gradients pass.

## Limits and next experiment

CUDA and MPS were unavailable in this environment; their optional checks skip explicitly. CPU FP32/FP64, low-precision reservoir inputs, and CPU BF16 autocast forward/backward were tested. No backend-specific speed claims follow from CPU results. The FP64 scan is used in ordinary pilot workloads; adversarial cancellation warns and uses the sequential path. Prototype associative-scan APIs and compilation are not used. TCN memory is limited to 15 valid tokens. Transformer inference caches are real but grow without a window and copy on append.

The measured scan already removes substantial sequential reservoir dispatch cost. For short online steps, the wrapper's small tensor operations, validation checks and scalar validation checks are a material fraction of latency. The next performance experiment should profile those operations and compare a fused step projection while retaining the same energy audits and safe gradients. For Transformers, profile KV append allocations before adding a preallocated inference-only cache. Separately, investigate initialization and projection saturation in the poorly trained seed under a declared tuning budget and unchanged gamma; do not hide it by choosing initial energy or gamma after evaluation.

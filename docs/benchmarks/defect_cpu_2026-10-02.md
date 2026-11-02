# Defect versus normalization: local CPU check

Raw measurements: [defect_cpu_2026-10-02.json](defect_cpu_2026-10-02.json).

Environment: macOS 15.6.1 arm64, PyTorch 2.11.0, float32, one CPU thread.
No CUDA or MPS device was available. Eager execution, three warmups and median
of ten timing repetitions. These small local timings are not GPU predictions.

Reproduce with:

```bash
python scripts/compare_defect_vs_normalization.py \
  --threads 1 --repeats 10 --train-steps 400 --seeds 3 \
  --output /tmp/defect-results.json
```

## Timing

Batch size 4, sequence length 128, equal input/output widths. Times in ms.
The baseline is the existing `Block2x2DenseL2SSM` (`l2n`) with exact spectral
normalization; the new cell uses default 2-by-2 blocks.

| State / width | Setup l2n | Setup defect | Cached scan l2n | Cached scan defect | Forward+backward l2n | Forward+backward defect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 / 8 | 0.102 | 0.374 | 0.533 | 0.414 | 2.077 | 3.166 |
| 128 / 16 | 0.453 | 0.437 | 0.958 | 0.807 | 4.455 | 4.543 |
| 256 / 32 | 2.378 | 0.492 | 1.579 | 1.319 | 10.017 | 6.738 |

At the largest size, matrix construction was about 4.8x faster and the complete
forward/backward about 1.5x faster. At the smallest size, the new construction
and backward were slower: several small orthogonal-factor solves have overhead.
Earlier repetitions also varied in backward timing, so avoid treating these
ratios as universal speedups.

The comparison includes the existing baseline's learned dense coordinate
transform and state conversion. It compares the cells as implemented, not just
the algebraic cost of normalization versus defect completion.

## Storage

| State / width | Parameters l2n | Parameters defect | Retained matrix bytes l2n | Retained matrix bytes defect |
| --- | ---: | ---: | ---: | ---: |
| 32 / 8 | 1,632 | 856 | 10,496 | 2,560 |
| 128 / 16 | 20,864 | 5,424 | 148,480 | 18,432 |
| 256 / 32 | 83,200 | 21,088 | 593,920 | 71,680 |

Parameter counts include the old dense coordinate transform. Retained matrices
include the old dense A and identity P; the new forward retains only A blocks
and B,C,D. These are storage counts, **not peak training memory**. The script
can additionally measure peak allocated training memory on CUDA.

## Learning smoke test

Both cells fit the same stable linear target `x_next=.7*x+.3*u`,
`y=.2*u+.2*x` on the same random training/validation sequences, with gain bound
1, Adam learning rate .01, 400 steps, and seeds 0,1,2. State size 16, width 4,
batch size 8, sequence length 48. Both receive nominal initialization radius .9
and scale .05, but their realized matrices and optimization geometry differ.

| Cell | Mean validation MSE after 400 steps |
| --- | ---: |
| Existing l2n | 7.88e-7 |
| New defect | 3.49e-6 |

All runs had finite gradients and ended with normalized system norm below one.
Both learned the target. The old parametrization converged faster and achieved
lower error here; a preliminary 80-step run showed a larger gap. There is no
accuracy improvement claim, and no task-specific initialization or learning-rate
tuning was performed. These results do not establish behavior on nonlinear
benchmarks or long sequences.

## Correctness checks

The complete repository suite passed: **118 tests**, including 18 new tests
for the defect cell. Coverage includes independent dense-root comparisons,
rectangular dimensions, arbitrary blocks, saturation, gradient checks, energy
inequalities, finite-horizon induced gain, scan/loop equivalence, streaming,
checkpoints, evaluation caching, DeepSSM training, and full-graph `aot_eager`
compilation. CUDA kernels and reduced-precision arithmetic were not tested.

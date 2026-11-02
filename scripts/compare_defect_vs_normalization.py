"""Compare normalization and defect, each with identity and full state storage.

Run from the repository root, e.g.:
    python scripts/compare_defect_vs_normalization.py --output /tmp/defect-results.json
    python scripts/compare_defect_vs_normalization.py --device cuda \
        --state-sizes 64 128 256 --widths 16 --lengths 128 512 --batch-sizes 4 16
    python scripts/compare_defect_vs_normalization.py --state-sweep \
        --output /tmp/defect-state-sweep.json
    python scripts/compare_defect_vs_normalization.py --state-sweep --include-optimizer \
        --output /tmp/defect-training-step.json
    python scripts/compare_defect_vs_normalization.py --state-sizes 512 --widths 16 \
        --workload all --include-optimizer --profile --output /tmp/ssm-profile.json

All four variants run by default. --l2n-state-metric and --defect-state-metric
select identity/full/both independently. Full/full compares equally expressive
generic small-block LTI families, with different initial input-output maps.
Cells use the same state/input sizes, gain, requested initial radius, optimizer,
training data, and seeds. Their parameterizations and initial input-output maps
differ (normalization also changes the realized radius): the learning experiment
is a smoke comparison, not a tuned ranking.
Identity l2n has unit storage in scan coordinates; its dense S changes returned
state coordinates. Full l2n uses its metric inside normalization and general
real blocks, retaining the real block scan used by defect. Full/full timings
compare the same generic transfer family, rather than matched initial weights.
--l2n-state-basis identity omits the learned S and its final-state conversion;
the default learned basis retains the existing implementation. Both settings
use identical initial LTI matrices and have the same zero-state transfer map.
CPU memory figures count parameters and returned diagnostic tensors, not all
inference caches or peak activations.
On CUDA, peak allocated training memory is measured separately.
Timing-only by default. Add --train-steps 80 for the optional learning experiment.
Rows compare equal dimensions and identical inputs, not identical model weights.
CPU/GPU operations are synchronized before and after every timing repetition;
reported medians include host dispatch and exclude warmup. JSON and CSV are saved.
--state-sweep measures only forward/backward at logarithmically spaced state
sizes from 6 to 1024, with a fixed width of 16, and saves PNG/PDF plots.
Use --plot to plot other comparisons, or --workload all for all timing categories.
--include-optimizer also measures actual Adam training steps and saves a second
plot including optimizer updates. Optimizer creation and first-use allocation
are excluded; at least one complete warmup step is performed.
--profile records a separate forward/backward trace for each case and reports
the busiest operators. Profiling is outside the timing samples. Chrome traces
include CPU operators and CUDA kernels when running on CUDA; MPS traces cover
CPU dispatch only.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from itertools import product
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.neural_ssm import Block2x2DenseL2SSM, DefectL2SSM

STATE_SWEEP = [6, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
VARIANT_LABELS = {'l2n': 'l2n(I)', 'l2n_full': 'l2n(P)',
                  'defect_identity': 'defect(I)', 'defect_full': 'defect(P)'}


def variant_name(row):
    """Identify a row, including older result files without an explicit variant."""
    if row.get('variant'):
        return row['variant']
    if row['cell'] == 'l2n':
        return 'l2n_full' if row.get('l2n_state_metric') == 'full' else 'l2n'
    return 'defect_' + (row.get('defect_state_metric') or 'identity')


def make_cell(name, n, width, device, seed=0, dtype=torch.float32, block_size=2,
              state_metric='identity', l2n_state_basis='learned'):
    torch.manual_seed(seed)
    if name == 'l2n':
        cell = Block2x2DenseL2SSM(n, width, width, gamma=1., exact_norm=True,
                                state_metric=state_metric)
        cell.init_on_circle(rho=.9, max_phase=.04, offdiag_scale=.05)
        if l2n_state_basis == 'identity' and state_metric == 'identity':
            # Remove S after initialization to preserve the random draws and
            # exactly the same initial A,B,C,D. Initialization is not timed.
            cell.S = None
    else:
        cell = DefectL2SSM(
            n, width, width, gamma=1., init_rho=.9, init_scale=.05, block_size=block_size,
            state_metric=state_metric,
            max_radius=1. if state_metric == 'full' else .999,
            factor_margin=0. if state_metric == 'full' else .001,
        )
    return cell.to(device=device, dtype=dtype)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def time_samples_ms(fn, device, repeats, warmup=5):
    """Synchronized wall times, including dispatch and excluding warmup."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(1000 * (time.perf_counter() - start))
    return samples


def median_ms(fn, device, repeats, warmup=5):
    return statistics.median(time_samples_ms(fn, device, repeats, warmup))


def sample_statistics(samples):
    """Median and interquartile bounds, including the single-repetition case."""
    median = statistics.median(samples)
    if len(samples) == 1:
        return median, median, median
    p25, _, p75 = statistics.quantiles(samples, n=4, method='inclusive')
    return median, p25, p75


def profile_training(fn, device, path):
    """Profile a separate step; never include profiler overhead in timed samples."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(device)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        fn()
        synchronize(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))
    operators = profiler.key_averages()
    # GPU kernels and CPU operators have different bottlenecks. Both times are
    # recorded, but the printed ranking follows the selected execution device.
    ranking = 'self_device_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
    busiest = sorted(operators, key=lambda op: getattr(op, ranking, 0), reverse=True)[:15]
    return [dict(operator=op.key, calls=op.count, self_cpu_ms=op.self_cpu_time_total / 1000,
                 self_device_ms=getattr(op, 'self_device_time_total', 0) / 1000)
            for op in busiest]


def benchmark(name, n, width, batch, length, device, repeats, *, warmup=5,
              dtype=torch.float32, block_size=2, mode='scan', seed=0, workload='all',
              include_optimizer=False, optimizer_lr=1e-3, state_metric='identity',
              profile_path=None, l2n_state_basis='learned'):
    cell = make_cell(name, n, width, device, seed, dtype=dtype, block_size=block_size,
                     state_metric=state_metric, l2n_state_basis=l2n_state_basis)
    # Reset after model initialization so every variant receives identical data.
    torch.manual_seed(1000 + seed)
    u = torch.randn(batch, length, width, device=device, dtype=dtype)
    build = (cell.compute_z_matrices if name == 'l2n' and state_metric == 'identity'
             else cell.compute_block_matrices)
    def measure(fn):
        return median_ms(fn, device, repeats, warmup)

    setup = setup_with_backward = uncached_forward = cached = retained_bytes = error = None
    if workload == 'all':
        # Training mode disables the parameter cache; no-grad isolates setup work.
        with torch.no_grad():
            setup = measure(build)
            matrices = build()
            uncached_forward = measure(
                lambda: cell(u, mode=mode, return_state=False, detach_state=True)
            )
        retained_bytes = sum(x.numel() * x.element_size() for x in matrices)

        def setup_backward():
            cell.zero_grad(set_to_none=True)
            # Ignore diagnostic P; the first four tensors are the constructed model.
            loss = sum(matrix.square().mean() for matrix in build()[:4])
            loss.backward()
        setup_with_backward = measure(setup_backward)

        cell.eval()
        with torch.no_grad():
            # Prime the cache even if the requested warmup count is zero.
            build()
            cached = measure(lambda: cell(u, mode=mode, return_state=False, detach_state=True))
            reference = cell(u, mode='loop', return_state=False)
            scanned = cell(u, mode=mode, return_state=False)
            error = (reference - scanned).abs().max().item()
            tolerance = 1e-10 if dtype == torch.float64 else 1e-4
            if not torch.allclose(reference, scanned, atol=tolerance, rtol=tolerance):
                raise RuntimeError(f'{name}: {mode} and loop disagree (max error {error:g}).')

    cell.train()
    def forward_backward():
        cell.zero_grad(set_to_none=True)
        outputs = cell(u, mode=mode, return_state=False, detach_state=True)
        outputs.square().mean().backward()
    samples = time_samples_ms(forward_backward, device, repeats, warmup)
    train, p25, p75 = sample_statistics(samples)
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in cell.parameters()):
        raise RuntimeError(f'{name}: nonfinite gradient in timing workload.')
    peak_bytes = None
    if device.type == 'cuda':
        cell.zero_grad(set_to_none=True)
        cell.state = None
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        forward_backward()
        synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device) - baseline

    # Use the initial weights and the same training function as the timings.
    # Adam updates below evolve a fresh cell and are not part of this trace.
    profile_ops = profile_training(forward_backward, device, profile_path) if profile_path else None

    step_samples = step_time = step_p25 = step_p75 = None
    if include_optimizer:
        # Start from the same initialization and inputs as forward/backward.
        # Actual updates evolve the weights; this measures steady-state steps,
        # not convergence or time to a given accuracy.
        cell = make_cell(name, n, width, device, seed, dtype=dtype, block_size=block_size,
                         state_metric=state_metric, l2n_state_basis=l2n_state_basis)
        cell.train()
        optimizer = torch.optim.Adam(cell.parameters(), lr=optimizer_lr, foreach=False)

        def training_step():
            optimizer.zero_grad(set_to_none=True)
            outputs = cell(u, mode=mode, return_state=False, detach_state=True)
            outputs.square().mean().backward()
            optimizer.step()

        # Even with --warmup 0, exclude Adam's lazy moment-buffer allocation.
        step_samples = time_samples_ms(training_step, device, repeats, max(1, warmup))
        step_time, step_p25, step_p75 = sample_statistics(step_samples)
        if any(not torch.isfinite(p).all() or
               (p.grad is not None and not torch.isfinite(p.grad).all())
               for p in cell.parameters()):
            raise RuntimeError(f'{name}: nonfinite parameter/gradient after Adam updates.')
    return dict(
        cell=name, n=n, width=width, batch=batch, length=length,
        variant=('l2n' if state_metric == 'identity' else 'l2n_full') if name == 'l2n'
                else f'defect_{state_metric}',
        mode=mode, dtype=str(dtype).removeprefix('torch.'), defect_block_size=block_size,
        sequence_engine=('complex diagonal scan' if name == 'l2n' and state_metric == 'identity' and mode == 'scan'
                         else 'real block scan' if mode == 'scan' else mode),
        defect_state_metric=state_metric if name == 'defect' else None,
        l2n_state_metric=state_metric if name == 'l2n' else None,
        l2n_state_basis=(l2n_state_basis if state_metric == 'identity' else 'scan') if name == 'l2n' else None,
        setup_ms=setup, setup_backward_ms=setup_with_backward,
        uncached_forward_ms=uncached_forward, cached_forward_ms=cached,
        cached_scan_ms=cached,  # historical field; the selected mode is recorded separately
        forward_backward_ms=train,
        forward_backward_p25_ms=p25, forward_backward_p75_ms=p75,
        forward_backward_samples_ms=samples,
        training_step_ms=step_time,
        training_step_p25_ms=step_p25, training_step_p75_ms=step_p75,
        training_step_samples_ms=step_samples,
        parameters=sum(p.numel() for p in cell.parameters()),
        parameters_with_grad=sum(p.numel() for p in cell.parameters() if p.grad is not None),
        parameter_bytes=sum(p.numel() * p.element_size() for p in cell.parameters()),
        retained_matrix_bytes=retained_bytes, cuda_training_peak_extra_bytes=peak_bytes,
        scan_loop_max_abs_error=error,
        profile_trace=str(profile_path) if profile_path else None,
        profile_operators=profile_ops,
    )


def print_comparison(rows):
    """A ratio > 1 means defect was faster for that measured workload."""
    metrics = [
        ('setup_ms', 'Build'),
        ('setup_backward_ms', 'Build+bwd'),
        ('uncached_forward_ms', 'Uncached fwd'),
        ('cached_forward_ms', 'Cached fwd'),
        ('forward_backward_ms', 'Fwd+bwd'),
        ('training_step_ms', 'Fwd+bwd+Adam'),
    ]
    variants = [name for name in VARIANT_LABELS if any(variant_name(r) == name for r in rows)]
    defects = [name for name in variants if name.startswith('defect_')]
    baselines = {name: 'l2n_full' if name == 'defect_full' and 'l2n_full' in variants else 'l2n'
                 for name in defects}
    groups = {}
    for row in rows:
        key = (row['n'], row['width'], row['batch'], row['length'])
        groups.setdefault(key, {})[variant_name(row)] = row
    print('\nI = identity state storage; P = full state storage.')
    print('Median time (ms); ratios compare matching storage when present; > 1 means defect was faster.')
    print(f"{'N':>5} {'Width':>5} {'Batch':>5} {'Length':>6} {'Workload':<13} "
          + ''.join(f'{VARIANT_LABELS[v]:>12}' for v in variants)
          + ''.join(f'{VARIANT_LABELS[baselines[v]] + "/" + VARIANT_LABELS[v]:>20}' for v in defects))
    for dimensions, records in groups.items():
        for key, label in metrics:
            if all(record[key] is None for record in records.values()):
                continue
            values = {v: records[v][key] if v in records else None for v in variants}
            timings = ''.join(f'{values[v]:12.3f}' if values[v] is not None else f'{"—":>12}'
                              for v in variants)
            ratios = ''.join(
                f'{values[baselines[v]] / values[v]:19.2f}x'
                if values.get(baselines[v]) is not None and values[v] else f'{"—":>20}'
                for v in defects
            )
            n, width, batch, length = dimensions
            print(f'{n:5d} {width:5d} {batch:5d} {length:6d} {label:<13} {timings}{ratios}')


def plot_forward_backward(results, output, *, include_optimizer=False):
    """Plot matched dimensions; separate panels avoid confounding width/batch/length."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import ScalarFormatter

    metric = 'training_step' if include_optimizer else 'forward_backward'
    groups = {}
    for row in results['timings']:
        key = (row['width'], row['batch'], row['length'])
        groups.setdefault(key, []).append(row)
    columns = min(2, len(groups))
    rows = (len(groups) + columns - 1) // columns
    figure = Figure(figsize=(9 * columns, 5 * rows + 1))
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    styles = [('l2n', 'l2n (normalization, identity P)', '#0072B2', 'o'),
              ('l2n_full', 'l2n (normalization, full P)', '#009E73', 'D'),
              ('defect_identity', 'defect (identity P)', '#D55E00', 's'),
              ('defect_full', 'defect (full P)', '#CC79A7', '^')]
    for axis, ((width, batch, length), records) in zip(axes, sorted(groups.items())):
        for name, label, color, marker in styles:
            series = sorted((r for r in records if variant_name(r) == name), key=lambda r: r['n'])
            if not series:
                continue
            states = [r['n'] for r in series]
            times = [r[metric + '_ms'] for r in series]
            lower = [r[metric + '_p25_ms'] for r in series]
            upper = [r[metric + '_p75_ms'] for r in series]
            axis.plot(states, times, label=label, color=color, marker=marker, linewidth=2)
            axis.fill_between(states, lower, upper, color=color, alpha=.15)
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set_xticks(sorted({r['n'] for r in records}))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.tick_params(axis='x', labelrotation=35)
        axis.set_xlabel('Internal state dimension n')
        axis.set_ylabel(f'{"Training step" if include_optimizer else "Forward + backward"} time (ms, log scale)')
        axis.set_title(f'Input/output width = {width}   |   Batch = {batch}   |   Length = {length}')
        axis.grid(True, which='major', alpha=.25)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False)
    for axis in axes[len(groups):]:
        axis.set_visible(False)
    title = 'Forward + backward + Adam' if include_optimizer else 'Forward + backward'
    figure.suptitle(f'{title} cost versus state dimension', fontsize=16, y=.98)
    warmups = results['optimizer_warmup'] if include_optimizer else results['warmup']
    scope = ('Includes parameter construction, gradient reset, forward, loss, backward, '
             'and Adam updates; excludes optimizer initialization.' if include_optimizer else
             'Includes parameter construction, gradient reset, forward, loss, and backward; '
             'excludes optimizer updates.')
    optimizer_note = f" · Adam lr={results['optimizer_lr']:g}" if include_optimizer else ''
    figure.text(
        .5, .025,
        f"{results['device'].upper()} · {results['dtype']} · {results['threads']} CPU thread(s) · "
        f"{results['execution']} · l2n: {results['l2n_mode']}, "
        f"l2n(I) basis: {results.get('l2n_state_basis', 'learned')} · defect: {results['defect_mode']}{optimizer_note}\n"
        f"Median of {results['repeats']} repetitions; band = 25th–75th percentile; "
        f"{warmups} warmups.\n{scope}\n"
        'Equal dimensions and inputs; full/full shares the generic LTI family; initial transfer maps differ.',
        ha='center', va='bottom', fontsize=9,
    )
    figure.tight_layout(rect=(0, .13 / rows, 1, .94))
    suffix = 'training-step' if include_optimizer else 'forward-backward'
    paths = [output.with_name(output.stem + '-' + suffix + '.' + ext) for ext in ('png', 'pdf')]
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches='tight')
    return paths


def target_response(inputs):
    """A stable linear teacher representable by both families, with gain <= .4."""
    state = torch.zeros_like(inputs[:, 0])
    outputs = []
    for u in inputs.unbind(1):
        outputs.append(.2 * u + .2 * state)
        state = .7 * state + .3 * u
    return torch.stack(outputs, dim=1)


def learning_smoke(name, device, steps, seed, state_metric='identity', l2n_state_basis='learned'):
    torch.manual_seed(1000 + seed)
    inputs = torch.randn(8, 48, 4, device=device)
    validation = torch.randn(8, 48, 4, device=device)
    target, val_target = target_response(inputs), target_response(validation)
    cell = make_cell(name, 16, 4, device, seed, state_metric=state_metric,
                     l2n_state_basis=l2n_state_basis)
    optimizer = torch.optim.Adam(cell.parameters(), lr=.01)
    losses = []
    synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = cell(inputs, mode='scan', return_state=False, detach_state=True)
        loss = (output - target).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in cell.parameters()
        ):
            raise RuntimeError(f'Nonfinite loss/gradient for {name}.')
        optimizer.step()
        losses.append(loss.item())
    synchronize(device)
    elapsed = time.perf_counter() - start
    with torch.no_grad():
        prediction = cell(validation, mode='scan', return_state=False)
        val_loss = (prediction - val_target).square().mean().item()
        a, b, c, d, p = cell.compute_z_matrices()
        # Diagnostics use energy coordinates; full-metric recurrence does not.
        if state_metric == 'full':
            s = torch.linalg.cholesky(p).T
            a = torch.linalg.solve_triangular(s.T, (s @ a).T, upper=False).T
            b = s @ b
            c = torch.linalg.solve_triangular(s.T, c.T, upper=False).T
        k = torch.cat((torch.cat((a, b / cell.gamma), 1), torch.cat((c, d / cell.gamma), 1)), 0)
        norm = torch.linalg.matrix_norm(k, 2).item()
    return dict(cell=name, variant=('l2n' if state_metric == 'identity' else 'l2n_full')
                if name == 'l2n' else f'defect_{state_metric}',
                defect_state_metric=state_metric if name == 'defect' else None,
                l2n_state_metric=state_metric if name == 'l2n' else None,
                l2n_state_basis=(l2n_state_basis if state_metric == 'identity' else 'scan') if name == 'l2n' else None,
                seed=seed, steps=steps, initial_mse=losses[0],
                last_train_mse=losses[-1], validation_mse=val_loss,
                elapsed_seconds=elapsed, final_contraction_norm=norm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='cpu, cuda, cuda:0, or mps')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--state-sweep', action='store_true',
                        help='Plot forward/backward for n=6..1024 at fixed width 16')
    parser.add_argument('--plot', action='store_true', help='Save forward/backward PNG and PDF plots')
    parser.add_argument('--profile', action='store_true',
                        help='Save separate CPU/CUDA forward-backward traces and operator breakdowns')
    parser.add_argument('--include-optimizer', action='store_true',
                        help='Also time complete Adam steps and, with --plot or --state-sweep, plot them')
    parser.add_argument('--optimizer-lr', type=float, default=1e-3,
                        help='Adam learning rate for timed complete steps (default: 0.001)')
    parser.add_argument('--workload', choices=['all', 'forward-backward'], default=None,
                        help='Default: forward-backward for --state-sweep, otherwise all')
    parser.add_argument('--state-sizes', type=int, nargs='+', default=None)
    parser.add_argument('--widths', type=int, nargs='+', default=None,
                        help='One shared width (default: 16), or one width per state size')
    parser.add_argument('--lengths', type=int, nargs='+', default=[128])
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[4])
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--defect-block-size', type=int, default=2)
    parser.add_argument('--defect-state-metric', choices=['identity', 'full', 'both'], default='both',
                        help='Default: compare both. Full uses open limits (radius=1, factor margin=0).')
    parser.add_argument('--l2n-mode', choices=['scan', 'conv', 'loop'], default='scan')
    parser.add_argument('--l2n-state-metric', choices=['identity', 'full', 'both'], default='both',
                        help='Default: both. Full uses weighted normalization and general real blocks.')
    parser.add_argument('--l2n-state-basis', choices=['learned', 'identity'], default='learned',
                        help='Keep the existing learned S, or omit its coordinate conversions')
    parser.add_argument('--defect-mode', choices=['scan', 'loop'], default='scan')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-steps', type=int, default=0,
                        help='Optional learning smoke test; 0 runs timings only')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('/tmp/defect-results.json'))
    args = parser.parse_args()
    if args.state_sizes is None:
        args.state_sizes = STATE_SWEEP if args.state_sweep else [32, 128, 1024]
    if args.widths is None:
        args.widths = [16]
    if args.workload is None:
        args.workload = 'forward-backward' if args.state_sweep else 'all'
    args.plot = args.plot or args.state_sweep
    if min(args.threads, args.repeats, args.seeds, args.defect_block_size) <= 0:
        parser.error('threads, repeats, seeds, and block size must be positive')
    if min(args.warmup, args.train_steps) < 0:
        parser.error('warmup and train-steps must be nonnegative')
    if not 0 < args.optimizer_lr < float('inf'):
        parser.error('optimizer-lr must be finite and positive')
    if min(args.state_sizes + args.widths + args.lengths + args.batch_sizes) <= 0:
        parser.error('state sizes, widths, sequence lengths, and batch sizes must be positive')
    if args.l2n_state_metric != 'full' and any(n % 2 for n in args.state_sizes):
        parser.error('identity l2n requires even state sizes; select --l2n-state-metric full for odd sizes')
    if any(n % args.defect_block_size for n in args.state_sizes):
        if not (args.defect_state_metric == 'full' and args.defect_block_size == 2):
            parser.error('state sizes must divide the defect block size, except full storage with size 2')
    if args.l2n_mode == 'conv' and args.l2n_state_metric != 'identity':
        parser.error('full l2n uses real blocks; choose --l2n-mode scan or --l2n-state-metric identity')
    if args.defect_state_metric != 'identity' and args.defect_block_size > 4:
        parser.error('full state metric requires defect block size <= 4')
    if len(args.widths) == 1:
        args.widths = args.widths * len(args.state_sizes)
    if len(args.widths) != len(args.state_sizes):
        parser.error('provide one width, or one width per state size')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available in this environment')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available in this environment')
    if device.type == 'mps' and args.dtype == 'float64':
        parser.error('MPS does not support float64; choose float32 or a different device')
    dtype = getattr(torch, args.dtype)
    storage_choices = ('identity', 'full') if args.defect_state_metric == 'both' else (
        args.defect_state_metric,
    )
    l2n_storage_choices = ('identity', 'full') if args.l2n_state_metric == 'both' else (
        args.l2n_state_metric,
    )
    variants = [('l2n', args.l2n_mode, storage) for storage in l2n_storage_choices]
    variants += [('defect', args.defect_mode, storage) for storage in storage_choices]
    results = dict(torch=torch.__version__, platform=platform.platform(), device=str(device),
                   dtype=args.dtype, threads=args.threads, warmup=args.warmup,
                   repeats=args.repeats, seed=args.seed,
                   workload=args.workload, l2n_mode=args.l2n_mode, defect_mode=args.defect_mode,
                   l2n_state_basis=args.l2n_state_basis,
                   l2n_state_metric=args.l2n_state_metric,
                   defect_state_metric=args.defect_state_metric,
                   variants=[('l2n' if storage == 'identity' else 'l2n_full')
                             for storage in l2n_storage_choices] + [f'defect_{storage}' for storage in storage_choices],
                   comparison_scope='Equal dimensions and inputs; full/full shares the generic LTI family; initial maps differ.',
                   optimizer='Adam' if args.include_optimizer else None,
                   optimizer_lr=args.optimizer_lr if args.include_optimizer else None,
                   optimizer_foreach=False if args.include_optimizer else None,
                   optimizer_warmup=max(1, args.warmup) if args.include_optimizer else None,
                   execution='eager', profiling=args.profile, timings=[], learning=[])
    print(f"Device: {device}; dtype: {args.dtype}; threads: {args.threads}; "
          f"warmup: {args.warmup}; repetitions: {args.repeats}", flush=True)
    print(f'l2n mode: {args.l2n_mode}; defect mode: {args.defect_mode}', flush=True)
    if args.l2n_mode == args.defect_mode == 'scan':
        print('Scan kernels: l2n(I) uses complex scalar modes; l2n(P) and defect use real matrix blocks.', flush=True)
    print(f'Defect state metric: {args.defect_state_metric}', flush=True)
    print(f'L2N state metric: {args.l2n_state_metric}', flush=True)
    print('l2n(I): identity scan storage and optional returned-state S; l2n(P): dense scan storage inside normalization.',
          flush=True)
    if 'identity' in l2n_storage_choices:
        print(f'L2N(I) returned-state basis: {args.l2n_state_basis}', flush=True)
    if 'full' in storage_choices and 'full' in l2n_storage_choices:
        print('Full/full compares the same generic small-block LTI family at equal dimensions.',
              flush=True)
    for (n, width), batch, length in product(
        zip(args.state_sizes, args.widths), args.batch_sizes, args.lengths
    ):
        for name, mode, storage in variants:
            variant = ('l2n' if storage == 'identity' else 'l2n_full') if name == 'l2n' else f'defect_{storage}'
            trace = args.output.with_name(
                f'{args.output.stem}-{variant}-n{n}-w{width}-b{batch}-t{length}.trace.json'
            ) if args.profile else None
            print(f'Timing {VARIANT_LABELS[variant]}: state={n}, width={width}, batch={batch}, '
                  f'length={length}', flush=True)
            result = benchmark(
                name, n, width, batch, length, device, args.repeats,
                warmup=args.warmup, dtype=dtype, block_size=args.defect_block_size,
                mode=mode, seed=args.seed, workload=args.workload,
                include_optimizer=args.include_optimizer, optimizer_lr=args.optimizer_lr,
                state_metric=storage, profile_path=trace,
                l2n_state_basis=args.l2n_state_basis,
            )
            results['timings'].append(result)
    print_comparison(results['timings'])
    if args.train_steps:
        # Keep the historical smoke experiment's fixed dimensions and float32.
        for seed in range(args.seeds):
            for name, _, storage in variants:
                result = learning_smoke(name, device, args.train_steps, seed,
                                        state_metric=storage, l2n_state_basis=args.l2n_state_basis)
                results['learning'].append(result)
                print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    csv_path = args.output.with_suffix('.csv')
    if csv_path == args.output:
        csv_path = args.output.with_name(args.output.stem + '-timings.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results['timings'][0]))
        writer.writeheader()
        writer.writerows(results['timings'])
    print(f'Saved {args.output} and {csv_path}')
    if args.profile:
        print('\nProfiler operator times are diagnostic; timed medians above exclude profiling.')
        for row in results['timings']:
            print(f"{VARIANT_LABELS[variant_name(row)]}, n={row['n']}, width={row['width']}, "
                  f"batch={row['batch']}, length={row['length']}: {row['profile_trace']}")
            for op in row['profile_operators'][:5]:
                print(f"  {op['operator']:<40} CPU {op['self_cpu_ms']:8.3f} ms, "
                      f"device {op['self_device_ms']:8.3f} ms, {op['calls']} calls")
    if args.plot:
        for path in plot_forward_backward(results, args.output):
            print(f'Saved {path}')
        if args.include_optimizer:
            for path in plot_forward_backward(results, args.output, include_optimizer=True):
                print(f'Saved {path}')


if __name__ == '__main__':
    main()

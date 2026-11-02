"""Exercise comparison exports, weighted variants, and profiling through the CLI."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts.compare_defect_vs_normalization import make_cell


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compare_defect_vs_normalization.py'


class ComparisonScriptTests(unittest.TestCase):
    def run_comparison(self, output, *options):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), '--lengths', '5', '--batch-sizes', '2',
             '--repeats', '1', '--warmup', '0', '--output', str(output), *options],
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(output.read_text()), completed.stdout

    def test_default_four_way_comparison_exports_plots_traces_and_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'comparison.json'
            plotting = importlib.util.find_spec('matplotlib') is not None
            data, stdout = self.run_comparison(
                output, '--state-sizes', '6', '8', '--widths', '2', '--include-optimizer',
                '--profile', '--train-steps', '2', '--seeds', '1',
                *(['--plot'] if plotting else []),
            )
            variants = {'l2n', 'l2n_full', 'defect_identity', 'defect_full'}
            self.assertEqual(set(data['variants']), variants)
            self.assertEqual(len(data['timings']), 8)
            for n in (6, 8):
                rows = {row['variant']: row for row in data['timings'] if row['n'] == n}
                self.assertEqual(set(rows), variants)
                self.assertEqual(rows['defect_full']['parameters']
                                 - rows['defect_identity']['parameters'], n * (n + 1) // 2)
                # Zero gradients through S's initial-state path still allocate
                # gradient tensors; distinguish them from absent gradients.
                self.assertEqual(rows['l2n']['parameters_with_grad'], rows['l2n']['parameters'])
                self.assertEqual(rows['l2n_full']['l2n_state_metric'], 'full')
                self.assertEqual(rows['l2n_full']['sequence_engine'], 'real block scan')
                for row in rows.values():
                    for metric in ('setup_ms', 'setup_backward_ms', 'cached_forward_ms',
                                   'forward_backward_ms', 'training_step_ms'):
                        self.assertTrue(math.isfinite(row[metric]) and row[metric] > 0)
                    self.assertLess(row['scan_loop_max_abs_error'], 1e-4)
                    trace = json.loads(Path(row['profile_trace']).read_text())
                    self.assertTrue(trace['traceEvents'])
                    self.assertTrue(row['profile_operators'])
            self.assertEqual({row['variant'] for row in data['learning']}, variants)
            self.assertIn('l2n(I)/defect(I)', stdout)
            self.assertIn('l2n(P)/defect(P)', stdout)
            with output.with_suffix('.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            if plotting:
                for workload in ('forward-backward', 'training-step'):
                    for extension in ('png', 'pdf'):
                        self.assertGreater(output.with_name(f'comparison-{workload}.{extension}').stat().st_size, 0)

    def test_single_metric_flags_and_default_width_still_work(self):
        with tempfile.TemporaryDirectory() as directory:
            for storage in ('identity', 'full'):
                with self.subTest(storage=storage):
                    output = Path(directory) / f'{storage}.json'
                    data, _ = self.run_comparison(
                        output, '--state-sizes', '6', '--defect-state-metric', storage,
                        '--workload', 'forward-backward', '--l2n-state-basis', 'identity',
                        '--l2n-state-metric', 'identity',
                    )
                    self.assertEqual({row['variant'] for row in data['timings']},
                                     {'l2n', f'defect_{storage}'})
                    for row in data['timings']:
                        self.assertEqual(row['width'], 16)
                        self.assertIsNone(row['setup_ms'])
                        self.assertIsNone(row['profile_trace'])
                    l2n = next(row for row in data['timings'] if row['cell'] == 'l2n')
                    self.assertEqual(l2n['l2n_state_basis'], 'identity')

    def test_state_basis_toggle_preserves_the_initial_transfer_map(self):
        device = torch.device('cpu')
        learned = make_cell('l2n', 6, 2, device, dtype=torch.float64)
        identity = make_cell('l2n', 6, 2, device, dtype=torch.float64, l2n_state_basis='identity')
        self.assertIsNone(identity.S)
        self.assertEqual(sum(p.numel() for p in learned.parameters())
                         - sum(p.numel() for p in identity.parameters()), 36)
        for a, b in zip(learned.compute_z_matrices(), identity.compute_z_matrices()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        u = torch.randn(2, 7, 2, dtype=torch.float64)
        yl = learned(u, return_state=False)
        yi = identity(u, return_state=False)
        torch.testing.assert_close(yl, yi, rtol=0, atol=0)
        yl.square().mean().backward()
        torch.testing.assert_close(learned.S.grad, torch.zeros_like(learned.S))

    def test_full_only_comparison_supports_odd_state_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            data, stdout = self.run_comparison(
                Path(directory) / 'full.json', '--state-sizes', '5', '--widths', '2',
                '--l2n-state-metric', 'full', '--defect-state-metric', 'full',
            )
            self.assertEqual(set(data['variants']), {'l2n_full', 'defect_full'})
            self.assertIn('l2n(P)/defect(P)', stdout)
            for row in data['timings']:
                self.assertEqual(row['sequence_engine'], 'real block scan')
                self.assertLess(row['scan_loop_max_abs_error'], 1e-4)


if __name__ == '__main__':
    unittest.main()

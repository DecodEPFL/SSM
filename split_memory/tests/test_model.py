import copy
from dataclasses import replace
import unittest

import torch

from split_memory import SplitMemoryConfig, SplitMemoryContractiveRNN, SplitMemoryState, detach_state, state_to
from split_memory.config import sufficient_sweeps
from split_memory.model import ContractiveWeight
from split_memory.solvers import ConvergenceError, fixed_jacobi, recurrence


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def model(self, **kwargs):
        return SplitMemoryContractiveRNN(2, 3, SplitMemoryConfig(heads=3, computation_width=5, **kwargs)).double()

    def test_row_certificate_after_optimizer_updates_and_empirical_contraction(self):
        weight = ContractiveWeight(11, .7, 1e-4, .02).double()
        opt = torch.optim.Adam(weight.parameters(), lr=.1)
        for _ in range(8):
            w = weight()
            self.assertLessEqual(float(w.abs().sum(-1).amax()), .7)
            left, right, forcing = (torch.randn(9, 11, dtype=torch.float64) for _ in range(3))
            diff = (torch.tanh(left @ w.T + forcing) - torch.tanh(right @ w.T + forcing)).abs().amax(-1)
            self.assertTrue(bool((diff <= .7 * (left - right).abs().amax(-1) + 1e-12).all()))
            opt.zero_grad()
            loss = -w.square().sum(); loss.backward(); opt.step()
        with torch.no_grad():
            weight.raw.fill_(1e300)
        self.assertTrue(bool(torch.isfinite(weight()).all()))
        self.assertLessEqual(float(weight().abs().sum(-1).amax()), .7)

    def test_fixed_recovers_sequential_outputs_states_and_all_gradients(self):
        model = self.model(fixed_sweeps=9, tolerance=1e-12)
        initial = SplitMemoryState(torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True),
                                   torch.randn(2, 5, dtype=torch.float64, requires_grad=True))
        x = torch.randn(2, 9, 2, dtype=torch.float64, requires_grad=True)
        y, state = model(x, initial)
        target, expected = model.reference(x, initial)
        torch.testing.assert_close(y, target, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(state.m, expected.m, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(state.c, expected.c, atol=1e-12, rtol=1e-12)
        variables = (x, initial.m, initial.c, *model.parameters())
        left = y.square().sum() + state.m.square().sum() + state.c.square().sum()
        right = target.square().sum() + expected.m.square().sum() + expected.c.square().sum()
        g1 = torch.autograd.grad(left, variables, retain_graph=True)
        g2 = torch.autograd.grad(right, variables)
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, atol=5e-11, rtol=5e-10)
        # Every projection, normalized raw recurrence, input and both incoming states were checked.
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in g1))

    def test_shorter_sweep_gradients_converge_to_sequential_bptt(self):
        model = self.model(require_convergence=False)
        x = torch.randn(1, 21, 2, dtype=torch.float64, requires_grad=True)
        target = model.reference(x)[0].sum()
        variables = (x, *model.parameters())
        reference = torch.autograd.grad(target, variables)
        errors = []
        for k in (2, 7, 21):
            clone = copy.deepcopy(model)
            clone.config = replace(model.config, fixed_sweeps=k)
            proposal = clone(x)[0].sum()
            gradients = torch.autograd.grad(proposal, (x, *clone.parameters()))
            errors.append(sum(float((a - b).square().sum()) for a, b in zip(gradients, reference)))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], 1e-20)

    def test_reference_full_chunks_steps_reset_and_cross_chunk_autograd(self):
        model = self.model()
        x = torch.randn(2, 19, 2, dtype=torch.float64, requires_grad=True)
        initial = SplitMemoryState(torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True),
                                   torch.randn(2, 5, dtype=torch.float64, requires_grad=True))
        target, last = model.reference(x, initial)
        chunks, state = [], initial
        for left, right in ((0, 2), (2, 7), (7, 8), (8, 19)):
            y, state = model.reference(x[:, left:right], state)
            chunks.append(y)
        torch.testing.assert_close(torch.cat(chunks, 1), target)
        steps, step_state = [], initial
        for sample in x.unbind(1):
            y, step_state = model.step(step_state, sample); steps.append(y)
            self.assertEqual(model.last_solver_info["cell_evaluations"], 2)
        torch.testing.assert_close(torch.stack(steps, 1), target)
        torch.testing.assert_close(step_state.m, last.m)
        torch.testing.assert_close(step_state.c, last.c)
        gradients = torch.autograd.grad(state.c.sum() + state.m.sum(), (x, initial.m, initial.c))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertFalse(detach_state(state).m.requires_grad)
        self.assertEqual(state_to(state, dtype=torch.float16).m.dtype, torch.float32)
        self.assertEqual(float(model.reset(2).m.abs().sum()), 0.)

    def test_adaptive_residual_and_chunk_assembled_residual_independent_audit(self):
        model = self.model(solver="adaptive", tolerance=1e-7, max_sweeps=80)
        x = torch.randn(2, 57, 2, dtype=torch.float64)
        y, last, info = model(x, diagnostics=True, return_trajectory=True)
        target, truth, exact = model.reference(x, diagnostics=True, return_trajectory=True)
        self.assertTrue(info["converged"])
        c, m = info["computation"], info["memory"]
        u = model.memory_projection(m.flatten(-2)) + model.input_projection(x) + model.bias
        predecessor = torch.cat((model.init_state(2).c[:, None], c[:, :-1]), 1)
        independent_r = (c - torch.tanh(predecessor @ model.computation().T + u)).abs().amax()
        self.assertAlmostEqual(float(independent_r), info["residual"], places=14)
        error = float((c - exact["computation"]).abs().amax())
        self.assertLessEqual(error, info["state_error_bound"] + 1e-12)
        chunks, states, state = [], [], None
        for left, right in ((0, 3), (3, 17), (17, 57)):
            part, state, diag = model(x[:, left:right], state, return_trajectory=True)
            chunks.append(part); states.append(diag["computation"])
        assembled = torch.cat(states, 1)
        predecessor = torch.cat((model.init_state(2).c[:, None], assembled[:, :-1]), 1)
        residual = (assembled - torch.tanh(predecessor @ model.computation().T + u)).abs().amax()
        self.assertLessEqual(float(residual) / (1 - model.config.rho), model.config.tolerance + 1e-12)
        # Approximate chunk paths solve the same recurrence, not necessarily the same finite graph.
        self.assertLess(float((torch.cat(chunks, 1) - target).abs().amax()), 1e-6)
        self.assertEqual(info["cell_passes"], info["sweeps"] + 1)

    def test_causality_fixed_adaptive_and_reference_including_batch_independence(self):
        model = self.model(tolerance=1e-4)
        x = torch.randn(2, 53, 2, dtype=torch.float64)
        changed = x.clone(); changed[:, 17:] = 10 * torch.randn_like(changed[:, 17:])
        changed[1] *= 100
        for solver in ("sequential", "fixed", "adaptive"):
            y = model(x, solver=solver)[0]
            other = model(changed, solver=solver)[0]
            torch.testing.assert_close(y[0, :17], other[0, :17], atol=1e-13, rtol=1e-13)

    def test_padding_empty_samples_internal_gaps_and_masked_nan_values(self):
        model = self.model(fixed_sweeps=13, tolerance=1e-10)
        initial = SplitMemoryState(torch.randn(3, 3, 2, dtype=torch.float64), torch.randn(3, 5, dtype=torch.float64))
        x = torch.randn(3, 13, 2, dtype=torch.float64)
        valid = torch.arange(13)[None] < torch.tensor([13, 5, 0])[:, None]
        x = x.masked_fill(~valid[..., None], float("nan"))
        y, state = model(x, initial, mask=valid)
        self.assertEqual(model.last_solver_info["cell_evaluations"], 3 * 13 * 14)
        self.assertEqual(model.last_solver_info["valid_cell_evaluations"], 18 * 14)
        for row, length in enumerate((13, 5, 0)):
            one = SplitMemoryState(initial.m[row:row+1], initial.c[row:row+1])
            expected, final = model.reference(x[row:row+1, :length], one)
            torch.testing.assert_close(y[row:row+1, :length], expected)
            torch.testing.assert_close(state.c[row:row+1], final.c)
            torch.testing.assert_close(state.m[row:row+1], final.m)
            self.assertEqual(float(y[row, length:].abs().sum()), 0.)
        valid[0, 2] = False
        y, state, info = model(x, initial, mask=valid, diagnostics=True)
        self.assertTrue(info["mask_fallback"])
        self.assertEqual(info["solver"], "sequential_mask_fallback")
        compact, final = model.reference(x[0:1, valid[0]], SplitMemoryState(initial.m[:1], initial.c[:1]))
        torch.testing.assert_close(y[0:1, valid[0]], compact)
        torch.testing.assert_close(state.c[:1], final.c)
        empty, untouched = model(x[:, :0], initial)
        self.assertEqual(empty.shape, (3, 0, 3))
        torch.testing.assert_close(untouched.c, initial.c)

    def test_iteration_cap_failure_rho_zero_and_large_tolerance(self):
        model = self.model(fixed_sweeps=1, tolerance=1e-12, max_sweeps=1)
        x = torch.randn(2, 17, 2, dtype=torch.float64)
        for solver in ("fixed", "adaptive"):
            with self.assertRaises(ConvergenceError) as error:
                model(x, solver=solver)
            self.assertFalse(error.exception.diagnostics["converged"])
            self.assertEqual(error.exception.diagnostics["sweeps"], 1)
        model.config = replace(model.config, require_convergence=False)
        _, _, info = model(x, diagnostics=True)
        self.assertFalse(info["converged"])
        zero = self.model(rho=0, fixed_sweeps=1, tolerance=1e-12)
        torch.testing.assert_close(zero(x)[0], zero.reference(x)[0])
        zero.config = replace(zero.config, solver="adaptive", tolerance=1)
        _, _, info = zero(x, diagnostics=True)
        self.assertEqual(info["sweeps"], 0)
        self.assertEqual(info["state_error_bound"], min(info["residual"], 1))
        self.assertEqual(sufficient_sweeps(0, 1e-6), 1)
        self.assertEqual(sufficient_sweeps(.7, 1), 0)

    def test_ablation_readers_and_generic_optional_energy_adapter(self):
        for reader in ("linear", "pointwise"):
            model = self.model(reader=reader)
            x = torch.randn(2, 9, 2, dtype=torch.float64, requires_grad=True)
            y, _ = model(x)
            y.sum().backward()
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
            self.assertIsNone(model.computation)
            part, state = model(x[:, :3])
            tail, _ = model(x[:, 3:], state)
            torch.testing.assert_close(torch.cat((part, tail), 1), y)
        from energy_bounded import EnergyBoundedModel
        core = self.model(fixed_sweeps=9)
        # Existing generic wrapper remains an explicit, optional Python composition.
        wrapped = EnergyBoundedModel(core, gamma=1)
        output, _ = wrapped(torch.zeros(2, 9, 2, dtype=torch.float64))
        self.assertEqual(float(output.detach().abs().sum()), 0.)

    def test_dtypes_autocast_finite_zero_boundaries_and_validation(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            model = self.model(rho=0, fixed_sweeps=1).to(dtype=dtype)
            x = torch.zeros(2, 7, 2, dtype=dtype, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y, state = model(x)
            self.assertEqual(state.m.dtype, torch.float64 if dtype == torch.float64 else torch.float32)
            y.sum().backward()
            self.assertTrue(bool(torch.isfinite(x.grad).all()))
        for args in ({"rho": 1}, {"rho": -1}, {"memory_radius": 0}, {"initial_retention": 1},
                     {"max_sweeps": 0}, {"fixed_sweeps": True}, {"tolerance": float("nan")}, {"heads": 0}):
            with self.assertRaises(ValueError): SplitMemoryConfig(**args)
        with self.assertRaises(FloatingPointError): self.model()(torch.full((1, 2, 2), float("inf")))
        with self.assertRaises(ValueError): self.model()(torch.zeros(1, 2, 2), mask=torch.ones(1, 2))

    def test_fp64_small_solver_weight_and_incoming_state_gradcheck(self):
        u = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
        raw = (torch.randn(3, 3, dtype=torch.float64) + .13).requires_grad_()
        initial = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
        def solve(u, raw, initial):
            weight = .6 * raw / (.01 + raw.abs().sum(-1, keepdim=True))
            return fixed_jacobi(u, weight, initial, torch.ones(1, 4, dtype=torch.bool), 4)
        self.assertTrue(torch.autograd.gradcheck(solve, (u, raw, initial)))

    def test_compilable_fixed_kernel_outputs_and_gradients(self):
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")
        u = torch.randn(1, 7, 3, requires_grad=True)
        w = (torch.eye(3) * .6).requires_grad_()
        initial = torch.randn(1, 3, requires_grad=True)
        valid = torch.ones(1, 7, dtype=torch.bool)
        compiled = torch.compile(fixed_jacobi, backend="aot_eager", fullgraph=True)
        expected = fixed_jacobi(u, w, initial, valid, 7)
        actual = compiled(u, w, initial, valid, 7)
        torch.testing.assert_close(actual, expected)
        for left, right in zip(torch.autograd.grad(actual.sum(), (u,w,initial), retain_graph=True),
                               torch.autograd.grad(expected.sum(), (u,w,initial))):
            torch.testing.assert_close(left, right)

    def test_supported_accelerators_explicitly(self):
        devices = [name for name, ready in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ready]
        if not devices:
            self.skipTest("CUDA/MPS unavailable in this environment")
        for device in devices:
            for solver in ("fixed", "adaptive", "sequential"):
                model = self.model().float().to(device)
                x = torch.randn(2, 17, 2, device=device, requires_grad=True)
                y, state, info = model(x, solver=solver, diagnostics=True)
                y.square().mean().backward()
                self.assertTrue(bool(torch.isfinite(x.grad).all()))
                self.assertTrue(info["converged"])

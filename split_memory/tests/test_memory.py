import unittest
import torch

from split_memory.config import SplitMemoryConfig
from split_memory.memory import BoundedMemory, bounded_proposal, memory_reference, memory_scan, rotate


class MemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_scan_reference_nonpowers_initial_states_extreme_gates_and_long_sequences(self):
        for dtype, atol in ((torch.float64, 3e-12), (torch.float32, 2e-5)):
            for length in (0, 1, 3, 17, 1025, 8193):
                gate = torch.rand(2, length, 3, dtype=dtype)
                if length:
                    gate[:, ::3] = 1
                    gate[:, 1::3] = 0
                theta = torch.randn(2, length, 3, dtype=dtype)
                a = gate[..., None] * torch.stack((theta.cos(), theta.sin()), -1)
                b = (1 - gate[..., None]) * bounded_proposal(torch.randn(2, length, 3, 2, dtype=dtype), 1.3)
                initial = torch.randn(2, 3, 2, dtype=dtype) * 2
                scan, ref = memory_scan(a, b, initial), memory_reference(a, b, initial)
                torch.testing.assert_close(scan, ref, atol=atol, rtol=atol)
                if length:
                    bound = initial.norm(dim=-1).clamp_min(1.3)
                    self.assertTrue(bool((scan.norm(dim=-1) <= bound[:, None] + atol).all()))
                    chunks, state = [], initial
                    for left in range(0, length, 137):
                        part = memory_scan(a[:, left:left + 137], b[:, left:left + 137], state)
                        chunks.append(part)
                        state = part[:, -1]
                    torch.testing.assert_close(torch.cat(chunks, 1), scan, atol=atol, rtol=atol)

    def test_coefficients_are_input_driven_and_padding_is_identity(self):
        module = BoundedMemory(2, SplitMemoryConfig(heads=2)).double()
        x = torch.randn(2, 7, 2, dtype=torch.float64)
        valid = torch.tensor([[True] * 4 + [False] * 3, [False] * 7])
        a, b = module.coefficients(x, valid)
        initial = torch.randn(2, 2, 2, dtype=torch.float64)
        memory = memory_scan(a, b, initial)
        torch.testing.assert_close(memory[1], initial[1].expand(7, -1, -1))
        torch.testing.assert_close(memory[0, 4:], memory[0, 3:4].expand(3, -1, -1))
        changed = x.clone(); changed[:, 5:] += 20
        a2, b2 = module.coefficients(changed, valid)
        torch.testing.assert_close(a, a2)
        torch.testing.assert_close(b, b2)

    def test_long_isometric_retention_without_writes_reports_floating_slack(self):
        for dtype, slack in ((torch.float64, 2e-11), (torch.float32, 2e-3)):
            angle = torch.full((2, 8193, 2), .013, dtype=dtype)
            a = torch.stack((angle.cos(), angle.sin()), -1)
            b = torch.zeros_like(a)
            initial = torch.randn(2, 2, 2, dtype=dtype)
            scan, reference = memory_scan(a, b, initial), memory_reference(a, b, initial)
            torch.testing.assert_close(scan, reference, atol=slack, rtol=slack)
            drift = (scan.norm(dim=-1) - initial.norm(dim=-1)[:, None]).abs().amax()
            self.assertLess(float(drift), slack)

    def test_memory_scan_and_proposal_gradcheck_and_huge_zero_values(self):
        a = (torch.randn(1, 5, 2, 2, dtype=torch.float64) * .3).requires_grad_()
        b = torch.randn_like(a, requires_grad=True)
        initial = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(memory_scan, (a, b, initial)))
        z = torch.tensor([[0., 0.], [1e30, -1e30], [.2, -.7]], requires_grad=True)
        v = bounded_proposal(z, 2)
        self.assertTrue(bool((v.norm(dim=-1) <= 2 + 1e-6).all()))
        v.sum().backward()
        self.assertTrue(bool(torch.isfinite(z.grad).all()))
        torch.testing.assert_close(rotate(torch.tensor([0., 1.]), torch.tensor([2., 3.])), torch.tensor([-3., 2.]))

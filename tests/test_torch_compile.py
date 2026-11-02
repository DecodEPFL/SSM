"""torch.compile readiness regression tests.

Locks in the Dynamo-tracing guarantees established for the package:

* the associative scan is pure PyTorch (torch pytree, no jax, no ScriptFunction),
  so the selective/certified models capture as a SINGLE graph (zero breaks);
* the complex LTI paths compile too: LRU eigenvalues use ``torch.polar`` (no
  Python ``1j`` literal, which Inductor cannot codegen) and the FFT conv solver
  transforms along the last dim (dim-0 FFTs hit an Inductor stride assertion).
  Only ``lruz`` (zak) and ``REN.forward`` remain deliberate eager islands.

These tests use ``torch._dynamo.explain`` and the ``aot_eager`` backend, so no
C++/Triton toolchain is required (portable to CI and laptops). Full Inductor
equivalence was additionally verified manually.
"""
import unittest
import warnings

import torch
import torch._dynamo as dynamo

from src.neural_ssm import ContextualDeepSSM, DeepSSM

warnings.filterwarnings("ignore")


def _deepssm(param, ff, gamma=1.0):
    return DeepSSM(
        d_input=3, d_output=2, d_model=8, d_state=8, n_layers=2,
        param=param, ff=ff, gamma=gamma,
    ).train()


class DynamoGraphBreakTests(unittest.TestCase):
    def _assert_single_graph(self, make_model, call):
        dynamo.reset()
        torch.manual_seed(0)
        model = make_model()
        u = torch.randn(2, 32, 3)
        ex = dynamo.explain(lambda x: call(model, x))(u)
        self.assertEqual(
            ex.graph_break_count, 0,
            f"expected zero graph breaks, got {ex.graph_break_count}: "
            f"{[str(b.reason).splitlines()[0] for b in ex.break_reasons][:4]}",
        )
        self.assertEqual(ex.graph_count, 1)

    def test_tv_scan_captures_as_single_graph(self):
        self._assert_single_graph(
            lambda: _deepssm("tv", "MBLIP"),
            lambda m, x: m(x, mode="scan", reset_state=True)[0],
        )

    def test_tvc_scan_captures_as_single_graph(self):
        self._assert_single_graph(
            lambda: _deepssm("tvc", "MBLIP"),
            lambda m, x: m(x, mode="scan", reset_state=True)[0],
        )


    def test_l2n_scan_captures_as_single_graph(self):
        self._assert_single_graph(
            lambda: _deepssm("l2n", "LGLU2"),
            lambda m, x: m(x, mode="scan", reset_state=True)[0],
        )

    def test_l2n_conv_captures_as_single_graph(self):
        self._assert_single_graph(
            lambda: _deepssm("l2n", "LGLU2"),
            lambda m, x: m(x, mode="conv", reset_state=True)[0],
        )

    def test_lru_scan_captures_as_single_graph(self):
        self._assert_single_graph(
            lambda: _deepssm("lru", "GLU", gamma=None),
            lambda m, x: m(x, mode="scan", reset_state=True)[0],
        )

    def test_lru_conv_captures_as_single_graph(self):
        self._assert_single_graph(
            lambda: _deepssm("lru", "GLU", gamma=None),
            lambda m, x: m(x, mode="conv", reset_state=True)[0],
        )

    def test_contextual_all_ports_capture_as_single_graph(self):
        def make():
            return ContextualDeepSSM(
                d_input=3, d_context=2, d_output=2,
                context_modes=("input", "gate", "mixer", "select"),
                context_filter="difference", d_features=4,
                d_model=8, d_state=8, n_layers=2,
                param="tv", ff="MBLIP", gamma=1.0,
            ).train()

        def call(m, x):
            z = torch.ones(x.shape[0], x.shape[1], 2)
            return m(x, z, mode="scan", reset_state=True)[0]

        self._assert_single_graph(make, call)


class CompiledEquivalenceTests(unittest.TestCase):
    """aot_eager compile (trace + autograd, no codegen) must match eager."""

    def _check(self, param, ff, gamma, mode, fullgraph):
        dynamo.reset()
        torch.manual_seed(0)
        m1 = _deepssm(param, ff, gamma)
        torch.manual_seed(0)
        m2 = _deepssm(param, ff, gamma)
        cm = torch.compile(m2, backend="aot_eager", fullgraph=fullgraph)
        u = torch.randn(2, 32, 3)

        y1 = m1(u, mode=mode, reset_state=True)[0]
        y1.sum().backward()
        y2 = cm(u, mode=mode, reset_state=True)[0]
        y2.sum().backward()

        self.assertLess((y1 - y2).abs().max().item(), 1e-5)
        grad_diff = max(
            (p1.grad - p2.grad).abs().max().item()
            for p1, p2 in zip(m1.parameters(), m2.parameters())
            if p1.grad is not None and p2.grad is not None
        )
        self.assertLess(grad_diff, 1e-4)

    def test_tv_scan_compiled_matches_eager(self):
        self._check("tv", "MBLIP", 1.0, "scan", fullgraph=True)

    def test_lru_scan_compiled_matches_eager(self):
        self._check("lru", "GLU", None, "scan", fullgraph=True)

    def test_l2n_conv_compiled_matches_eager(self):
        self._check("l2n", "LGLU2", 1.0, "conv", fullgraph=True)


if __name__ == "__main__":
    unittest.main()

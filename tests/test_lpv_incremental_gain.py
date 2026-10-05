"""Incremental-gain tests for the context-only (LPV) selector mode.

Section 3.3 of the theory note disclaims an incremental bound: two inputs select
different per-step matrices, so a zero-state l2 certificate says nothing about
``||Q(e) - Q(e~)||``. That disclaimer is a consequence of letting the selector
see the cell input, not of the certificate itself.

With ``select_input='context'`` the per-step matrices depend only on the
exogenous context, each cell is linear time-varying for a fixed context
sequence, and the same cascade bounds the difference of two trajectories. These
tests check that the bound genuinely holds in that mode and genuinely fails
without it -- a test that only confirmed the bound in LPV mode would not show the
mode is doing anything.
"""

import unittest

import torch

from neural_ssm.ssm.contextual import ContextualDeepSSM
from neural_ssm import DeepSSM
from neural_ssm.ssm.cells.selective import RobustMambaDiagLTI, RobustMambaDiagSSM


def excite(module, scale=2.0):
    """Push a selector away from its near-degenerate initialization.

    At init the ``param_net`` output is small, so ``b`` and ``c`` are near zero
    and the cell barely responds to anything. An attack on that model measures
    the initialization, not the mode.
    """
    with torch.no_grad():
        for p in module.parameters():
            p += torch.randn_like(p) * scale
    return module


def incremental_attack(forward, d_in, gamma, steps=600, B=4, T=30, lr=0.05, seed=1):
    """Maximize ``||f(u1) - f(u2)|| / ||u1 - u2||`` by gradient ascent.

    ``forward`` takes one input tensor and returns the output at zero state under
    a context sequence it holds fixed internally.
    """
    torch.manual_seed(seed)
    delta = (torch.randn(B, T, d_in) * 0.3).requires_grad_(True)
    base = (torch.randn(B, T, d_in) * 1.0).requires_grad_(True)
    opt = torch.optim.Adam([delta, base], lr=lr)
    best = 0.0
    for _ in range(steps):
        u1, u2 = base + 0.5 * delta, base - 0.5 * delta
        y1, y2 = forward(u1), forward(u2)
        num = (y1 - y2).reshape(B, -1).norm(dim=-1)
        den = (u1 - u2).reshape(B, -1).norm(dim=-1).clamp_min(1e-9)
        ratio = num / den
        opt.zero_grad()
        (-ratio.max()).backward()
        opt.step()
        best = max(best, float(ratio.max().detach()))
    return best


class CellIncrementalGainTests(unittest.TestCase):
    """At the cell level the normalization is tight, so the split is stark."""

    def _cell(self, cls, mode, **kwargs):
        torch.manual_seed(0)
        cell = cls(
            d_model=6, d_state=8, d_out=6, gamma=1.0, train_gamma=False,
            param_net="mlp", hidden=64, context_dim=4, select_input=mode, **kwargs
        ).eval()
        return excite(cell.param_net) and cell

    def _attack(self, cell):
        torch.manual_seed(1)
        context = torch.randn(4, 30, 4) * 2.0

        def forward(u):
            out, _ = cell(u, select_context=context, mode="loop", reset_state=True)
            return out

        return incremental_attack(forward, 6, float(cell.gamma))

    def test_input_dependent_selector_has_no_incremental_bound(self):
        """The disclaimed case: gamma bounds the zero-state gain and nothing else."""
        for cls in (RobustMambaDiagSSM, RobustMambaDiagLTI):
            cell = self._cell(cls, "both")
            worst = self._attack(cell)
            self.assertGreater(
                worst, 10.0 * float(cell.gamma),
                f"{cls.__name__}: expected the attack to blow past gamma, got {worst}",
            )

    def test_context_only_selector_respects_gamma_incrementally(self):
        for cls in (RobustMambaDiagSSM, RobustMambaDiagLTI):
            cell = self._cell(cls, "context")
            worst = self._attack(cell)
            self.assertLessEqual(
                worst, float(cell.gamma) * 1.01,
                f"{cls.__name__}: incremental gain {worst} exceeds gamma",
            )

    def test_the_incremental_bound_is_nearly_tight_in_lpv_mode(self):
        """Guards against a vacuous pass from a cell that barely responds."""
        cell = self._cell(RobustMambaDiagSSM, "context")
        worst = self._attack(cell)
        self.assertGreater(worst, 0.5 * float(cell.gamma), f"attack too weak: {worst}")

    def test_cell_is_linear_in_its_input_under_a_fixed_context(self):
        """The property the mode buys: superposition at the cell."""
        cell = self._cell(RobustMambaDiagSSM, "context")
        torch.manual_seed(2)
        context = torch.randn(4, 20, 4) * 2.0
        u1, u2 = torch.randn(4, 20, 6), torch.randn(4, 20, 6)

        def run(u):
            out, _ = cell(u, select_context=context, mode="loop", reset_state=True)
            return out

        with torch.no_grad():
            additive = run(u1 + u2) - (run(u1) + run(u2))
            homogeneous = run(2.5 * u1) - 2.5 * run(u1)
            reference = run(u1).abs().max()
        self.assertLess(float(additive.abs().max()) / float(reference), 1e-4)
        self.assertLess(float(homogeneous.abs().max()) / float(reference), 1e-4)

    def test_cell_is_not_linear_when_the_selector_sees_the_input(self):
        cell = self._cell(RobustMambaDiagSSM, "both")
        torch.manual_seed(2)
        context = torch.randn(4, 20, 4) * 2.0
        u1 = torch.randn(4, 20, 6)

        def run(u):
            out, _ = cell(u, select_context=context, mode="loop", reset_state=True)
            return out

        with torch.no_grad():
            homogeneous = run(2.5 * u1) - 2.5 * run(u1)
            reference = run(u1).abs().max()
        self.assertGreater(float(homogeneous.abs().max()) / float(reference), 1e-2)


class StackIncrementalGainTests(unittest.TestCase):
    def _model(self, mode, **overrides):
        torch.manual_seed(0)
        kwargs = dict(
            context_modes=("select",), d_model=16, d_state=12, n_layers=3,
            gamma=0.9, param="tv", ff="LGLU2", dropout=0.0, learn_x0=False,
            select_input=mode,
        )
        kwargs.update(overrides)
        model = ContextualDeepSSM(3, 4, 2, **kwargs)
        model.eval()
        for block in model.core.blocks:
            excite(block.lru.param_net, scale=1.0)
        return model

    def test_is_lpv_reflects_the_selector_mode(self):
        self.assertTrue(self._model("context").is_lpv)
        self.assertFalse(self._model("both").is_lpv)

    def test_incremental_bound_is_reported_only_in_lpv_mode(self):
        lpv = self._model("context")
        self.assertAlmostEqual(
            float(lpv.incremental_gain_bound()),
            float(lpv.certified_gain_bound()),
            places=6,
        )
        self.assertTrue(torch.isinf(self._model("both").incremental_gain_bound()))

    def test_stack_respects_its_incremental_bound_under_attack(self):
        model = self._model("context")
        bound = float(model.incremental_gain_bound())
        torch.manual_seed(3)
        context = torch.randn(4, 30, 4) * 2.0

        def forward(u):
            out, _ = model(u, context, mode="loop", reset_state=True)
            return out

        worst = incremental_attack(forward, 3, bound, steps=400)
        self.assertLessEqual(worst, bound * 1.01)

    def test_an_input_dependent_gate_disqualifies_lpv(self):
        """A gate computed from the input makes the block nonlinear again."""
        model = self._model(
            "context", context_modes=("select", "gate"), gate_include_disturbance=True
        )
        self.assertFalse(model.is_lpv)
        self.assertTrue(torch.isinf(model.incremental_gain_bound()))

    def test_a_context_only_gate_keeps_lpv(self):
        model = self._model(
            "context", context_modes=("select", "gate"), gate_include_disturbance=False
        )
        self.assertTrue(model.is_lpv)

    def test_an_input_dependent_mixer_disqualifies_lpv(self):
        model = self._model(
            "context", context_modes=("select", "mixer"), mixer_include_disturbance=True
        )
        self.assertFalse(model.is_lpv)

    def test_additive_context_disqualifies_lpv(self):
        """The input port adds context to the signal instead of scheduling it."""
        model = self._model(
            "context", context_modes=("select", "input"), horizon=16
        )
        self.assertFalse(model.is_lpv)

    def test_lpv_mode_keeps_the_zero_state_certificate(self):
        model = self._model("context")
        bound = float(model.certified_gain_bound())
        self.assertLessEqual(bound, 0.9 + 1e-6)
        torch.manual_seed(4)
        u = torch.randn(4, 40, 3) * 2.0
        with torch.no_grad():
            y, _ = model(u, torch.randn(4, 40, 4) * 3.0, mode="loop", reset_state=True)
        self.assertLessEqual(
            float(y.reshape(4, -1).norm(dim=-1).max()),
            bound * float(u.reshape(4, -1).norm(dim=-1).max()) + 1e-5,
        )


class ConfigValidationTests(unittest.TestCase):
    def test_context_mode_requires_a_context_width(self):
        with self.assertRaises(ValueError):
            DeepSSM(3, 2, d_model=8, d_state=6, n_layers=2, gamma=0.9,
                    param="tv", ff="LGLU2", learn_x0=False,
                    select_input="context", select_context_dim=0)

    def test_context_mode_requires_a_selective_parametrization(self):
        with self.assertRaises(ValueError):
            DeepSSM(3, 2, d_model=8, d_state=6, n_layers=2, gamma=0.9,
                    param="l2ru", ff="LGLU2", learn_x0=False,
                    select_input="context", select_context_dim=4)

    def test_unknown_select_input_is_rejected(self):
        with self.assertRaises(ValueError):
            DeepSSM(3, 2, d_model=8, d_state=6, n_layers=2, gamma=0.9,
                    param="tv", ff="LGLU2", learn_x0=False,
                    select_input="sometimes", select_context_dim=4)

    def test_default_stays_input_selective(self):
        model = DeepSSM(3, 2, d_model=8, d_state=6, n_layers=2, gamma=0.9,
                        param="tv", ff="LGLU2", learn_x0=False, select_context_dim=4)
        self.assertFalse(model.is_lpv)
        self.assertEqual(model.config.select_input, "both")


if __name__ == "__main__":
    unittest.main()

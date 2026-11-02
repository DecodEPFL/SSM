"""Complex diagonal LTI baseline (param="lru")."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....utils.runtime import (
    normalize_to_3d,
    reset_runtime_state as _reset_runtime_state,
    resolve_runtime_state as _resolve_runtime_state,
)
from ..common import (
    _conv_diag_linear,
    _scan_diag_linear,
    complex_diagonal_to_real,
    lru_forward_loop,
)


class LRU(nn.Module):
    """Linear Recurrent Unit with loop or parallel-scan simulation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        state_features: int,
        internal_state_init=None,
        rmin: float = 0.9,
        rmax: float = 1.0,
        max_phase: float = 6.283,
        learn_x0: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

        # Pre-compute scalars
        self._sqrt_in_features = math.sqrt(in_features)
        self._sqrt_2_in_features = math.sqrt(2 * in_features)
        self._sqrt_state_features = math.sqrt(state_features)
        self._rmin_rmax_diff = rmax - rmin
        self._rmin_rmax_sum = rmax + rmin
        self._rmin_squared = rmin**2

        # Real output projection
        self.D = nn.Parameter(torch.randn(out_features, in_features) / self._sqrt_in_features)

        # Complex SSM params (diagonal A via magnitudes+phases)
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(
                -0.5
                * torch.log(u1 * self._rmin_rmax_sum * self._rmin_rmax_diff + self._rmin_squared)
            )
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1.0 - lambda_abs.square())))

        B_complex = torch.complex(
            torch.randn(state_features, in_features) / self._sqrt_2_in_features,
            torch.randn(state_features, in_features) / self._sqrt_2_in_features,
        )
        self.Bp = nn.Parameter(B_complex)  # (N, U)

        C_complex = torch.complex(
            torch.randn(out_features, state_features) / self._sqrt_state_features,
            torch.randn(out_features, state_features) / self._sqrt_state_features,
        )
        self.C = nn.Parameter(C_complex)  # (H, N)

        # Runtime state
        self.state: torch.Tensor | None = None

        # Learnable initial condition (complex, shape (1, N))
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, state_features, dtype=torch.complex64))
        else:
            self.register_buffer("x0_param", None)

        # Small cache for complex->real transform 2x2 blocks
        self._T_cache: dict[str, torch.Tensor] = {}

        self.set_param()  # initialize SSM params

    def set_param(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        # torch.polar(r, theta) == r * exp(1j*theta) with no Python complex
        # literal in the graph — Inductor cannot codegen the `1j` constant
        # ("'complex' object has no attribute 'get_name'"); polar compiles.
        self.lambdas = torch.polar(lambda_abs, lambda_phase)  # (N,) complex
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)  # (N,1) real
        self.B = gammas * self.Bp  # (N,U) complex
        return self.lambdas, self.B, self.C, self.D

    def ss_real_matrices(self, to_numpy: bool = True):
        """Export the equivalent real state-space matrices."""
        lambdas, B, C, D = self.set_param()
        return complex_diagonal_to_real(lambdas, B, C, D, self._T_cache, to_numpy=to_numpy)

    def forward_loop(self, input: torch.Tensor, state: torch.Tensor, detach_state: bool = True):
        self.set_param()
        output, states = lru_forward_loop(input, state, self.lambdas, self.B, self.C, self.D)
        self.state = states[:, -1].detach() if detach_state else states[:, -1]
        return output, states

    def forward_scan(self, input: torch.Tensor, state=None, detach_state: bool = True):
        return self._forward_parallel(input, state, detach_state, _scan_diag_linear)

    def forward_conv(self, input: torch.Tensor, state=None, detach_state: bool = True):
        """Same recurrence/readout as scan, with an FFT state solver."""
        return self._forward_parallel(input, state, detach_state, _conv_diag_linear)

    def _forward_parallel(self, input, state, detach_state, solver):
        lambdas, B, C, D = self.set_param()
        drives = input.to(B.dtype) @ B.mT
        states = solver(lambdas, drives, state.to(B.dtype))
        self.state = states[:, -1].detach() if detach_state else states[:, -1]
        output = (states[:, :-1] @ C.mT).real + input @ D.T
        return output, states

    def forward(
        self,
        input: torch.Tensor,
        gamma: float | None = None,
        state: torch.Tensor | None = None,
        mode: str = "loop",
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        input = normalize_to_3d(input)
        self.state = _resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=input.shape[0],
            n_state=self.state_features,
            device=input.device,
            dtype=self.B.dtype,
            x0=self.x0_param,
        )
        if mode == "scan":
            return self.forward_scan(input, self.state, detach_state=detach_state)
        if mode == "conv":
            return self.forward_conv(input, self.state, detach_state=detach_state)
        if mode in ("loop", "loop_efficient"):
            return self.forward_loop(input, self.state, detach_state=detach_state)
        raise ValueError(
            f"Unknown mode: {mode}. Expected 'scan', 'conv', 'loop', or 'loop_efficient'."
        )

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)

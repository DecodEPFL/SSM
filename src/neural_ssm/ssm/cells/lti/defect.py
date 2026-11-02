"""Block-structured LTI systems certified by defect completion.

The identity metric keeps the original inexpensive energy-coordinate model.
A full metric couples the state blocks while retaining the same block scan and
thin input/output factors. Neither path uses numerical SVDs or matrix roots.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ....utils.lyapunov import solve_block_lyapunov, weighted_defect_completion

from ....utils.runtime import (
    EvalCacheMixin,
    normalize_to_3d,
    reset_runtime_state,
    resolve_runtime_state,
    sequence_output,
)
from ....utils.scan import compute_block_lti_states
from ..common import positive_upper_factor


def cwy_orthonormal(raw: torch.Tensor) -> torch.Tensor:
    """First r columns of a product of r Householder reflections, for (..., d, r).

    Compact WY uses one r-by-r triangular solve, not a d-by-d factorization.
    A zero column denotes an identity reflection. This convention is finite
    but discontinuous at zero: initialize columns randomly, never all zero.
    Adding epsilon to nonzero column norms would break exact orthogonality.
    """
    rows, cols = raw.shape[-2:]
    if not 0 < cols <= rows:
        raise ValueError("CWY requires 0 < columns <= rows.")
    norms = torch.linalg.vector_norm(raw, dim=-2, keepdim=True)
    vectors = raw / torch.where(norms > 0, norms, torch.ones_like(norms))
    identity = torch.eye(cols, device=raw.device, dtype=raw.dtype)
    upper = 0.5 * identity + torch.triu(vectors.mT @ vectors, diagonal=1)
    coefficients = torch.linalg.solve_triangular(upper, vectors[..., :cols, :].mT, upper=True)
    return torch.eye(rows, cols, device=raw.device, dtype=raw.dtype) - vectors @ coefficients


def signed_values_and_defects(raw: torch.Tensor, limit: float):
    """s = limit*tanh(raw), d = sqrt(1-s**2), without subtractive cancellation."""
    if limit == 1:
        # tanh can round to one, making the Lyapunov equation singular. This
        # guard is at machine precision, not a fixed modelling margin, and
        # keeps the analytic sech defect consistent with the bounded value.
        bound = math.atanh(1 - torch.finfo(raw.dtype).eps)
        raw = raw.clamp(min=-bound, max=bound)
    exp_neg_abs = torch.exp(-raw.abs())
    sech = 2 * exp_neg_abs / (1 + exp_neg_abs.square())
    values = limit * torch.tanh(raw)
    # At limit=1, squaring tiny sech values would underflow before sqrt and
    # create an infinite sqrt derivative. Return sech directly in that case.
    defects = sech if limit == 1 else torch.sqrt((1 - limit**2) + limit**2 * sech.square())
    return values, defects


def apply_thin_defect(basis: torch.Tensor, defects: torch.Tensor, value: torch.Tensor):
    """Apply [I + U diag(d-1) U.T] to value, retaining the identity complement."""
    return value + basis @ ((defects - 1).unsqueeze(-1) * (basis.mT @ value))


def _rotations(angles: torch.Tensor) -> torch.Tensor:
    c, s = angles.cos(), angles.sin()
    return torch.stack((c, -s, s, c), dim=-1).reshape(-1, 2, 2)


class ThinContraction(nn.Module):
    """A rectangular contraction represented by two thin orthonormal factors."""

    def __init__(self, rows: int, cols: int, limit: float, init_scale: float):
        super().__init__()
        rank = min(rows, cols)
        self.limit = limit
        self.left_raw = nn.Parameter(torch.randn(rows, rank))
        self.right_raw = nn.Parameter(torch.randn(cols, rank))
        self.values_raw = nn.Parameter(init_scale * torch.randn(rank))

    def forward(self):
        values, defects = signed_values_and_defects(self.values_raw, self.limit)
        return cwy_orthonormal(self.left_raw), values, cwy_orthonormal(self.right_raw), defects


class DefectL2SSM(EvalCacheMixin, nn.Module):
    r"""Block LTI cell with x_next.T P x_next + ||y||² <= x.T P x + gamma²||u||².

    With the identity metric, B/gamma = Delta_(A.T) X, C = Y Delta_A,
    D/gamma = -Y A.T X + Delta_(Y.T) Z Delta_X, with contractive A,X,Y,Z.
    Delta_M denotes sqrt(I - M.T M). No posterior norm normalization is used.

    ``block_size=2`` uses two rotation angles per state block. Larger blocks
    use batched CWY factors; ``block_size=d_state`` allows an unrestricted
    dense A with the identity metric. All sizes use the real shared-power scan.
    ``state_metric='full'`` learns Q=R.T R and solves P-A.T P A=Q, preserving
    generic gain-bounded LTI input/output expressivity with small blocks. It
    supports block sizes up to four and a final scalar for odd n with b=2.
    The identity metric needs compatible block storage. Fixed finite margins
    restrict either family; use max_radius=1 and factor_margin=0 for open limits.

    ``max_radius`` bounds singular values of A, hence also its contraction
    rate. ``factor_margin`` bounds X,Y,Z away from singular value one. ``gamma``
    is the certified gain; when trainable, it is the *current* learned bound.
    With nonzero initial state, include x_0.T P x_0 as initial energy; use
    state_energy(x_0) to evaluate it in the returned state coordinates.
    """

    def __init__(
        self,
        d_state: int,
        d_input: int,
        d_output: int,
        *,
        gamma: float = 1.0,
        train_gamma: bool = False,
        block_size: int = 2,
        max_radius: float = 0.999,
        factor_margin: float = 1e-3,
        init_rho: float = 0.9,
        max_phase: float = 0.04,
        phase_center: float = 0.0,
        random_phase: bool = True,
        init_scale: float = 0.05,
        learn_x0: bool = False,
        state_metric: str = "identity",
    ):
        super().__init__()
        if any(not isinstance(d, int) or d <= 0 for d in (d_state, d_input, d_output, block_size)):
            raise ValueError("State, input, output, and block sizes must be positive integers.")
        if state_metric not in ("identity", "full"):
            raise ValueError("state_metric must be 'identity' or 'full'.")
        if state_metric == "full" and block_size > 4:
            raise ValueError("The full state metric uses small blocks (block_size <= 4).")
        partial_block = state_metric == "full" and block_size == 2 and d_state % 2
        if d_state % block_size and not partial_block:
            raise ValueError("d_state must be divisible by block_size.")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive.")
        if not 0 < max_radius <= 1 or not 0 <= factor_margin < 1:
            raise ValueError("max_radius must be in (0, 1], factor_margin in [0, 1).")
        if not 0 <= init_rho < max_radius:
            raise ValueError("init_rho must satisfy 0 <= init_rho < max_radius.")
        if not math.isfinite(init_scale) or init_scale < 0:
            raise ValueError("init_scale must be finite and nonnegative.")
        if not math.isfinite(max_phase) or max_phase < 0 or not math.isfinite(phase_center):
            raise ValueError(
                "max_phase must be finite and nonnegative; phase_center must be finite."
            )

        self.d_state, self.d_input, self.d_output = d_state, d_input, d_output
        self.block_size, self.max_radius = block_size, max_radius
        self.state_metric = state_metric
        blocks = (d_state + block_size - 1) // block_size
        self.scan_state_size = blocks * block_size
        raw_shape = (d_state,) if partial_block else (blocks, block_size)
        self.a_raw = nn.Parameter(
            torch.full(raw_shape, math.atanh(init_rho / max_radius))
        )
        if block_size == 2:
            theta = torch.full((d_state // 2,), float(phase_center))
            if random_phase:
                theta.uniform_(phase_center - max_phase, phase_center + max_phase)
            self.theta = nn.Parameter(theta)
            self.phi = nn.Parameter(torch.zeros_like(theta))
        else:
            # Equal initial singular values give radius init_rho; orientations
            # remain free, including reflections via signed singular values.
            self.a_left_raw = nn.Parameter(torch.randn(blocks, block_size, block_size))
            self.a_right_raw = nn.Parameter(torch.randn(blocks, block_size, block_size))
        limit = 1 - factor_margin
        self.X = ThinContraction(d_state, d_input, limit, init_scale)
        self.Y = ThinContraction(d_output, d_state, limit, init_scale)
        self.Z = ThinContraction(d_output, d_input, limit, init_scale)
        if state_metric == "full":
            # Packed triangular Q factor: exactly n*(n+1)/2 metric parameters.
            # Q=(1-rho²)I starts P at I for the initial scaled-orthogonal A.
            self.register_buffer(
                "metric_indices", torch.triu_indices(d_state, d_state, offset=1),
                persistent=False,
            )
            self.metric_upper_raw = nn.Parameter(torch.zeros(d_state * (d_state - 1) // 2))
            diagonal = math.sqrt(1 - init_rho**2)
            self.metric_diag_raw = nn.Parameter(
                torch.full((d_state,), math.log(math.expm1(diagonal)))
            )
        log_gamma = torch.tensor(math.log(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(log_gamma)
        else:
            self.register_buffer("log_gamma", log_gamma)
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, d_state))
        else:
            self.register_buffer("x0_param", None)
        self.state: torch.Tensor | None = None

    @property
    def gamma(self):
        return self.log_gamma.exp()

    def contraction_rate_bound(self):
        """Euclidean contraction bound for A in the scan coordinates."""
        return self.max_radius * self.a_raw.tanh().abs().amax()

    def _a_factors(self):
        if self.block_size == 2:
            left, right = _rotations(self.theta), _rotations(self.phi)
            if self.scan_state_size != self.d_state:
                # The last real pole is a scalar, padded by a zero dummy mode.
                identity = torch.eye(2, device=left.device, dtype=left.dtype).unsqueeze(0)
                left, right = torch.cat((left, identity)), torch.cat((right, identity))
        else:
            left = cwy_orthonormal(self.a_left_raw)
            right = cwy_orthonormal(self.a_right_raw)
        raw = self.a_raw
        if self.scan_state_size != self.d_state:
            raw = F.pad(raw, (0, 1)).reshape(-1, 2)
        values, defects = signed_values_and_defects(raw, self.max_radius)
        return left, values, right, defects

    def _left_multiply_blocks(self, blocks, matrix):
        return (blocks @ matrix.reshape(-1, self.block_size, matrix.shape[-1])).reshape(
            self.d_state, matrix.shape[-1]
        )

    def _compute_normalized_block_matrices(self):
        if self.state_metric == "full":
            return self._eval_cached("full_normalized_blocks", self._compute_full_metric_blocks)
        ua, sa, va, da = self._a_factors()
        a = (ua * sa.unsqueeze(-2)) @ va.mT
        x, y, residual = self._completion_factors()

        # Absorb A's inner orthogonal factors into the free X and Y factors.
        b = self._left_multiply_blocks(ua * da.unsqueeze(-2), x)
        c = self._left_multiply_blocks(va * da.unsqueeze(-2), y.mT).mT
        d = -(y * sa.flatten()) @ x + residual
        return a, b, c, d

    def _completion_factors(self):
        """Shared thin X,Y and the residual term Delta_Y Z Delta_X."""
        ux, sx, vx, dx = self.X()
        uy, sy, vy, dy = self.Y()
        uz, sz, vz, _ = self.Z()
        x = (ux * sx) @ vx.mT
        y = (uy * sy) @ vy.mT

        # Apply both remaining defects to the thin factors of Z. The identity
        # complement is essential, especially when inputs/outputs exceed n.
        z_left = apply_thin_defect(uy, dy, uz)
        z_right = apply_thin_defect(vx, dx, vz)
        return x, y, (z_left * sz) @ z_right.mT

    def _compute_state_metric(self):
        if self.a_raw.dtype not in (torch.float32, torch.float64):
            raise TypeError("Full state metrics require float32 or float64 parameters.")
        with torch.autocast(device_type=self.a_raw.device.type, enabled=False):
            ua, sa, va, _ = self._a_factors()
            a = (ua * sa.unsqueeze(-2)) @ va.mT
            r = positive_upper_factor(self.metric_diag_raw, self.metric_upper_raw, self.metric_indices)
            p = solve_block_lyapunov(a, r.mT @ r)
            return a, r, p

    def _compute_full_metric_blocks(self):
        # Keep certificate construction at parameter precision, also under AMP.
        with torch.autocast(device_type=self.a_raw.device.type, enabled=False):
            a, r, p = self._eval_cached("state_metric", self._compute_state_metric)
            x, y, residual = self._completion_factors()
            b, c, d = weighted_defect_completion(a, r, p, x, y, residual)
            return a, b, c, d

    def storage_matrix(self):
        """Return P in the same coordinates as the returned/streamed states."""
        if self.state_metric == "full":
            return self._eval_cached("state_metric", self._compute_state_metric)[2]
        return torch.eye(self.d_state, device=self.a_raw.device, dtype=self.a_raw.dtype)

    def state_energy(self, state):
        """Evaluate x.T P x; no coordinate transform is needed in the recurrence."""
        if state is None:
            return None
        if self.state_metric == "identity":
            return state.square().sum(dim=-1)
        p = self.storage_matrix()
        state = state.to(device=p.device, dtype=p.dtype)
        return ((state @ p) * state).sum(dim=-1)

    def _compute_block_matrices(self):
        a, b, c, d = self._compute_normalized_block_matrices()
        return a, self.gamma * b, c, self.gamma * d

    def compute_block_matrices(self):
        """Return (A_blocks, B, C, D); cache only during gradient-free evaluation."""
        return self._eval_cached("blocks", self._compute_block_matrices)

    def normalized_block_matrices(self):
        """Unit-gain blocks, with the metric returned by storage_matrix()."""
        return self._eval_cached("normalized_blocks", self._compute_normalized_block_matrices)

    def compute_z_matrices(self):
        """Dense (A,B,C,D,P) in scan coordinates, for diagnostics/export."""
        blocks, b, c, d = self.compute_block_matrices()
        a = torch.block_diag(*blocks.unbind(0))[:self.d_state, :self.d_state]
        return a, b, c, d, self.storage_matrix()

    def compute_dense_matrices(self):
        return self.compute_z_matrices()

    def step(self, state: torch.Tensor, u: torch.Tensor):
        a, b, c, d = self.compute_block_matrices()
        padding = self.scan_state_size - self.d_state
        padded = F.pad(state, (0, padding)) if padding else state
        next_state = (a @ padded.reshape(-1, a.shape[0], self.block_size, 1)).flatten(1)
        next_state = next_state[:, :self.d_state]
        return next_state + u @ b.mT, state @ c.mT + u @ d.mT

    def forward(
        self,
        u: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        time_first: bool = False,
        return_state: bool = True,
        mode: str = "scan",
        return_last: bool = False,
        reset_state: bool = True,
        detach_state: bool = False,
    ):
        """Return outputs and x_0..x_T, using y_t = C x_t + D u_t.

        Inputs may be (features,), (time,features), or (batch,time,features).
        ``time_first=True`` swaps the batch/time axes for 3D inputs and outputs.
        Only loop and scan are supported: complex-scalar convolution does not
        apply to general real blocks. Stored state follows the existing cell API.
        """
        if mode not in ("scan", "loop"):
            raise ValueError("DefectL2SSM supports mode='scan' or 'loop', not complex convolution.")
        inputs = normalize_to_3d(u, time_first=time_first)
        if inputs.shape[-1] != self.d_input:
            raise ValueError(f"Expected d_input={self.d_input}, got {inputs.shape[-1]}.")
        batch, length, _ = inputs.shape
        x0 = resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=batch,
            n_state=self.d_state,
            device=inputs.device,
            dtype=inputs.dtype,
            x0=self.x0_param,
        )
        a, b, c, d = self.compute_block_matrices()
        drives = inputs @ b.mT
        if mode == "scan":
            states = compute_block_lti_states(a, drives.transpose(0, 1), x0).transpose(0, 1)
        else:
            padding = self.scan_state_size - self.d_state
            padded_x0 = F.pad(x0, (0, padding)) if padding else x0
            padded_drives = F.pad(drives, (0, padding)) if padding else drives
            current = padded_x0.reshape(batch, -1, self.block_size)
            trajectory = [x0]
            for t in range(length):
                current = (a @ current.unsqueeze(-1)).squeeze(-1)
                current = current + padded_drives[:, t].reshape_as(current)
                trajectory.append(current.flatten(1)[:, :self.d_state])
            states = torch.stack(trajectory, dim=1)
        outputs = states[:, :-1] @ c.mT + inputs @ d.mT
        last = states[:, -1]
        self.state = last.detach() if detach_state else last
        if time_first:
            outputs, states = outputs.transpose(0, 1), states.transpose(0, 1)
        return sequence_output(
            outputs, states, last, return_state=return_state, return_last=return_last
        )

    def reset(self):
        self.state = reset_runtime_state(self.state, x0=self.x0_param)

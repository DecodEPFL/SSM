"""Shared cell algebra; parametrizations remain in their individual cell files.

The first section handles fixed complex-diagonal recurrences and coordinate
changes. The second handles small selective contractions and selector inputs.
Sequence-parallel kernels themselves live in utils/scan.py.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...utils.runtime import normalize_to_3d as _normalize_to_3d
from ...utils.scan import (
    _conv_diag_complex,
    associative_scan,
    binary_operator_diag,
    diag_affine_scan,
)

# Fixed LTI recurrences and coordinate changes

_CONTRACTION_EPS = 0.002


def positive_upper_factor(diagonal, upper, indices):
    """Packed upper-triangular factor with positive diagonal, without a dense raw parameter."""
    factor = torch.diag(F.softplus(diagonal))
    return factor.index_put((indices[0], indices[1]), upper)


def block_diagonal(blocks, width):
    """Assemble small real blocks for diagnostics; crop a padded final scalar."""
    count, size, _ = blocks.shape
    indices = torch.arange(count, device=blocks.device)
    pairs = blocks.new_zeros(count, count, size, size)
    pairs = pairs.index_put((indices, indices), blocks)
    return pairs.permute(0, 2, 1, 3).reshape(count * size, count * size)[:width, :width]


def rotation_block_diagonal(rho: Tensor, theta: Tensor) -> Tensor:
    """Assemble independent rho_i R(theta_i) blocks without a Python loop."""
    pairs = rho.shape[0]
    result = rho.new_zeros(2 * pairs, 2 * pairs)
    first = 2 * torch.arange(pairs, device=rho.device)
    second = first + 1
    real, imag = rho * theta.cos(), rho * theta.sin()
    result[first, first] = real
    result[first, second] = -imag
    result[second, first] = imag
    result[second, second] = real
    return result


def bounded_real_matrix(A, B, C, D, P, gamma):
    """Discrete bounded-real LMI, for diagnostics/export rather than forward."""
    gamma = torch.as_tensor(gamma, device=A.device, dtype=A.dtype)
    top_left = A.T @ P @ A - P + C.T @ C
    top_right = A.T @ P @ B + C.T @ D
    bottom_left = B.T @ P @ A + D.T @ C
    identity = torch.eye(B.shape[1], device=A.device, dtype=A.dtype)
    bottom_right = B.T @ P @ B + D.T @ D - gamma.square() * identity
    return torch.cat(
        (torch.cat((top_left, top_right), dim=1), torch.cat((bottom_left, bottom_right), dim=1)),
        dim=0,
    )


def complex_diagonal_to_real(lambdas, B, C, D, cache, *, to_numpy=True):
    """Real realization of y_t = Re(C x_t) + D u_t for a complex diagonal A."""
    n = lambdas.shape[0]
    A_full = torch.diag(torch.stack((lambdas, lambdas.conj()), dim=1).flatten())
    B_full = torch.stack((B, B.conj()), dim=1).reshape(2 * n, B.shape[1])
    C_half = 0.5 * C
    C_full = torch.stack((C_half, C_half.conj()), dim=2).reshape(C.shape[0], 2 * n)
    transform, inverse = _complex_real_transform_blocks(n, lambdas.dtype, lambdas.device, cache)
    matrices = (
        (transform @ A_full @ inverse).real,
        (transform @ B_full).real,
        (C_full @ inverse).real,
        D,
    )
    if to_numpy:
        return tuple(matrix.detach().resolve_conj().cpu().numpy() for matrix in matrices)
    return matrices


def _scan_diag_linear(lambdas: torch.Tensor, Bu: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    Bsz, L, N = Bu.shape

    # ensure x0 is (B,N)
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0).expand(Bsz, -1)
    elif x0.dim() == 2 and x0.size(0) == 1 and Bsz > 1:
        x0 = x0.expand(Bsz, -1)
    if x0.shape != (Bsz, N):
        raise ValueError(f"x0 has shape {tuple(x0.shape)}, expected {(Bsz, N)}")

    if L == 0:
        return x0.unsqueeze(1)  # (B,1,N) = [x0]

    Bu = Bu.clone()
    Bu[:, 0, :] = Bu[:, 0, :] + lambdas * x0  # fold x0 into first step

    lam_seq = lambdas.expand(L, -1)  # (L,N)

    def _scan_fn(bu_seq):
        return associative_scan(binary_operator_diag, (lam_seq, bu_seq))[1]

    x_next = torch.vmap(_scan_fn)(Bu)  # (B,L,N) = x_1..x_L

    states = torch.empty(Bsz, L + 1, N, device=Bu.device, dtype=Bu.dtype)
    states[:, 0] = x0
    states[:, 1:] = x_next
    return states


def _conv_diag_linear(lambdas: torch.Tensor, Bu: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Batch-major FFT-convolution twin of :func:`_scan_diag_linear`.

    Returns the same state trajectory ``states`` (B, L+1, N) — i.e.
    ``states[:, s] = lambdas**s * x0 + sum_{k<s} lambdas**(s-1-k) Bu[:, k]`` — but
    computes it with two FFTs instead of the associative scan. Maths are identical
    up to floating-point rounding (see :func:`scan_utils._conv_diag_complex`).

    lambdas: (N,) complex ; Bu: (B, L, N) complex ; x0: (B, N) complex.
    """
    Bsz, L, N = Bu.shape

    # ensure x0 is (B,N) — mirror _scan_diag_linear's shape handling
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0).expand(Bsz, -1)
    elif x0.dim() == 2 and x0.size(0) == 1 and Bsz > 1:
        x0 = x0.expand(Bsz, -1)
    if x0.shape != (Bsz, N):
        raise ValueError(f"x0 has shape {tuple(x0.shape)}, expected {(Bsz, N)}")

    if L == 0:
        return x0.unsqueeze(1)  # (B,1,N) = [x0]

    Bu_tb = Bu.transpose(0, 1).contiguous()  # (L, B, N)
    states_tb = _conv_diag_complex(lambdas, Bu_tb, x0)  # (L+1, B, N)
    return states_tb.transpose(0, 1).contiguous()  # (B, L+1, N)


@torch.jit.script
def lru_forward_loop(
    input: Tensor,  # (B,L,H) real (usually)
    state: Tensor,  # (B,N) complex for LRU
    A: Tensor,  # (N,) or (N,N) complex
    B: Tensor,  # (N,H) complex
    C: Tensor,  # (H_out,N) complex
    D: Tensor,  # (H_out,H) real
) -> tuple[Tensor, Tensor]:
    BATCH, SEQ, H = input.shape
    N = state.size(1)

    # compute recurrence in the state dtype (complex)
    sdtype = state.dtype
    device = input.device

    x = state.to(device=device, dtype=sdtype)
    A = A.to(device=device, dtype=sdtype)
    B = B.to(device=device, dtype=sdtype)
    C = C.to(device=device, dtype=sdtype)

    # D stays in input dtype (real), so y is real like in scan
    D = D.to(device=device, dtype=input.dtype)

    BT = B.mT  # (H,N) complex
    CT = C.mT  # (N,H_out) complex
    DT = D.mT  # (H,H_out) real

    states = torch.empty((BATCH, SEQ + 1, N), device=device, dtype=sdtype)
    states[:, 0] = x

    if A.dim() == 1:
        lambdas = A  # (N,) complex
        for t in range(SEQ):
            u_t_c = input[:, t, :].to(dtype=sdtype)  # complex for state update
            x = x * lambdas + u_t_c @ BT
            states[:, t + 1] = x
    elif A.dim() == 2:
        A_T = A.mT
        for t in range(SEQ):
            u_t_c = input[:, t, :].to(dtype=sdtype)
            x = x @ A_T + u_t_c @ BT
            states[:, t + 1] = x
    else:
        raise RuntimeError("Unsupported A.dim(), expected 1 or 2")

    pre_states = states[:, :-1, :]  # x_0..x_{L-1}

    # output is real (match scan): Re(C x_t) + D u_t
    y_lin = (pre_states @ CT).real.to(dtype=input.dtype)  # (B,L,H_out)
    y_dir = input @ DT  # (B,L,H_out)
    output = y_lin + y_dir

    return output, states


def _complex_real_transform_blocks(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Cache 2x2 block and its inverse per dtype/device to avoid reallocation
    key = f"{str(dtype)}@{device.type}:{device.index}"
    T_key, Ti_key = f"T_{key}", f"Tinv_{key}"
    if T_key not in cache or Ti_key not in cache:
        T = torch.tensor([[1, 1], [1j, -1j]], device=device, dtype=dtype)
        cache[T_key] = T
        cache[Ti_key] = torch.linalg.inv(T)
    Tblk = torch.block_diag(*([cache[T_key]] * n))
    Tiblk = torch.block_diag(*([cache[Ti_key]] * n))
    return Tblk, Tiblk


# Selective normalization, recurrence, and inputs


def build_selector(d_input, context_dim, out_dim, *, select_input, kind, hidden):
    """Build the shared TV/TVC selector; its weights keep their existing keys."""
    if context_dim < 0:
        raise ValueError("context_dim must be non-negative.")
    if select_input not in ("both", "context"):
        raise ValueError(f"select_input must be 'both' or 'context', got {select_input!r}.")
    if select_input == "context" and context_dim == 0:
        raise ValueError("select_input='context' requires context_dim > 0.")
    width = context_dim if select_input == "context" else d_input + context_dim
    if kind == "linear":
        return nn.Linear(width, out_dim)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    raise ValueError(f"param_net must be 'linear' or 'mlp', got {kind!r}.")


@torch.no_grad()
def initialize_selector_decay(alpha_log, delta_bias, param_net, *, rho, delta0, scale):
    """Initialize decay and selector weights identically for both selective cells."""
    rho = min(max(float(rho), 1e-4), 1.0 - 1e-6)
    delta0 = max(float(delta0), 1e-4)
    alpha0 = -math.log(rho) / delta0
    alpha_log.fill_(math.log(math.expm1(alpha0)) if alpha0 > 1e-6 else math.log(alpha0 + 1e-6))
    delta_bias.fill_(math.log(math.expm1(delta0)) if delta0 > 1e-6 else math.log(delta0 + 1e-6))
    for layer in param_net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def spectral_norm_2x2_a_b_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Exact spectral norm of K = [[a, b],
                                [c, 0]].
    a, b, c broadcastable (e.g. (B,T,N)). Returns tensor of same shape.
    """
    p = a * a + c * c
    r = b * b
    q = a * b
    disc = (p - r) * (p - r) + 4.0 * q * q
    lam_max = 0.5 * (p + r + torch.sqrt(disc + eps))
    return torch.sqrt(lam_max + eps)


def spectral_norm_2x2_abcd(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Exact || [[a,b],[c,d]] ||_2 for broadcastable real tensors.
    """
    tau = a * a + b * b + c * c + d * d
    det2 = (a * d - b * c) ** 2
    inner = (tau * tau - 4.0 * det2).clamp_min(0.0)
    lam_max = 0.5 * (tau + torch.sqrt(inner + eps))
    return torch.sqrt(lam_max + eps)


def _diag_scan(
    a_tb: torch.Tensor, bu_tb: torch.Tensor, z0: torch.Tensor, scan_impl=None
) -> torch.Tensor:
    """
    Parallel prefix scan for the diagonal real recurrence:
        z_{t+1} = a_t ⊙ z_t + bu_t

    Uses the same diagonal operator as LRU with a work-efficient O(T) scan.

    a_tb, bu_tb : (T, B, N)  — time-major
    z0          : (B, N)     — initial state
    scan_impl   : optional callable(a, b) -> inclusive scan (T,B,N). Defaults to
                  the eager :func:`diag_affine_scan`. A :class:`GraphedDiagScan`
                  instance can be passed to replay a captured CUDA graph instead
                  (same maths, far less launch overhead).
    Returns states : (T+1, B, N)  — [z_0, z_1, ..., z_T]
    """
    T, B, N = a_tb.shape
    if T == 0:
        return z0.unsqueeze(0)

    # Fold z0 into the first time step so that after the inclusive prefix scan
    # b_prefix[t] == z_{t+1} directly (mirrors _scan_diag_complex in scan_utils).
    bu_tb = bu_tb.clone()
    bu_tb[0] = bu_tb[0] + a_tb[0] * z0

    scan = scan_impl if scan_impl is not None else diag_affine_scan
    z_next = scan(a_tb, bu_tb)  # (T,B,N)

    states = torch.empty(T + 1, B, N, device=a_tb.device, dtype=a_tb.dtype)
    states[0] = z0
    states[1:] = z_next
    return states


def _selector_input(
    context_dim: int,
    u_bt: torch.Tensor,
    ctx_bt: torch.Tensor | None,
    select_input: str = "both",
) -> torch.Tensor:
    """Build the ``param_net`` input from the cell input and optional context.

    The cell renormalizes its per-step transition to spectral norm <= 1 for ANY
    ``param_net`` output, so conditioning the selector on context preserves the
    gamma certificate either way -- context needs no l2 projection and may be
    non-l2 or endogenous.

    ``select_input`` decides what the selector is allowed to see, and that choice
    changes which guarantee holds:

    ``"both"``
        ``param_net([u_t, ctx_t])``. Mamba-style selectivity: the per-step
        matrices depend on the input, so the map is nonlinear and the certificate
        is a *zero-state* l2 bound only. Two different inputs select different
        matrices, so no incremental bound follows.

    ``"context"``
        ``param_net(ctx_t)``. The matrices depend only on the exogenous context,
        so for a fixed context sequence the map from input to output is linear
        time-varying. The same normalization then certifies the *difference* of
        two trajectories as well, which upgrades the guarantee to an incremental
        l2 bound and makes tracking about a nonzero equilibrium well posed.
        See :meth:`DeepSSM.incremental_gain_bound`.
    """
    if select_input not in ("both", "context"):
        raise ValueError(f"select_input must be 'both' or 'context', got {select_input!r}.")
    if select_input == "context" and context_dim == 0:
        raise ValueError(
            "select_input='context' requires context_dim > 0: the selector would "
            "otherwise have no input at all."
        )
    if context_dim == 0:
        return u_bt
    if ctx_bt is None:
        raise ValueError("select_context is required when context_dim > 0.")
    ctx = _normalize_to_3d(ctx_bt)
    if ctx.shape[-1] != context_dim:
        raise ValueError(f"select_context last dim must be {context_dim}, got {ctx.shape[-1]}.")
    if ctx.shape[:2] != u_bt.shape[:2]:
        raise ValueError("select_context must match (batch, time) of the cell input.")
    if select_input == "context":
        return ctx
    return torch.cat([u_bt, ctx], dim=-1)

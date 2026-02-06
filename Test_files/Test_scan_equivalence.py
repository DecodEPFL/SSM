"""
Check equivalence of the real 2x2-block scan vs the complex-scan implementation.

Run:
    python Test_files/Test_scan_equivalence.py
"""

from __future__ import annotations

import torch

from src.neural_ssm.ssm.scan_utils import (
    compute_linear_recurrence_parallel_block2x2,
    compute_linear_recurrence_parallel_block2x2_complex,
    compute_linear_recurrence_sequential,
)


def build_block2x2_A(n_blocks: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Construct a stable block-diagonal A with 2x2 rotation+scaling blocks:
        [[a, -b],
         [b,  a]]
    """
    # radii < 1 for stability, random angles
    rho = torch.empty(n_blocks, device=device, dtype=dtype).uniform_(0.6, 0.98)
    theta = torch.empty(n_blocks, device=device, dtype=dtype).uniform_(-3.14, 3.14)
    a = rho * torch.cos(theta)
    b = rho * torch.sin(theta)

    blocks = torch.zeros(n_blocks, 2, 2, device=device, dtype=dtype)
    blocks[:, 0, 0] = a
    blocks[:, 0, 1] = -b
    blocks[:, 1, 0] = b
    blocks[:, 1, 1] = a

    return torch.block_diag(*[blocks[i] for i in range(n_blocks)])


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Shapes
    T = 200
    B = 4
    d_state = 16  # must be even
    d_in = 3
    n_blocks = d_state // 2

    # Build A, B, u, x0
    A = build_block2x2_A(n_blocks, device, dtype)
    Bm = torch.randn(d_state, d_in, device=device, dtype=dtype) * 0.1
    u = torch.randn(T, B, d_in, device=device, dtype=dtype)
    x0 = torch.randn(B, d_state, device=device, dtype=dtype)

    # Real block-scan (legacy)
    states_real = compute_linear_recurrence_parallel_block2x2(
        A, Bm, u, x0, use_complex_scan=False
    )

    # Complex-scan
    states_cplx = compute_linear_recurrence_parallel_block2x2_complex(A, Bm, u, x0)

    # Standard loop recursion
    states_loop = compute_linear_recurrence_sequential(A, Bm, u, x0)  # (T, B, D)
    # Sequential returns x_1..x_T, so prepend x0 to match (T+1, B, D)
    states_loop = torch.cat([x0.unsqueeze(0), states_loop], dim=0)

    # Compare
    max_abs = (states_real - states_cplx).abs().max().item()
    mean_abs = (states_real - states_cplx).abs().mean().item()
    print(f"max |real - complex| = {max_abs:.6e}")
    print(f"mean |real - complex| = {mean_abs:.6e}")

    max_abs_loop = (states_real - states_loop).abs().max().item()
    mean_abs_loop = (states_real - states_loop).abs().mean().item()
    print(f"max |real - loop|    = {max_abs_loop:.6e}")
    print(f"mean |real - loop|   = {mean_abs_loop:.6e}")

    # Stronger check (tolerances may need adjustment on GPU)
    if torch.allclose(states_real, states_cplx, atol=1e-5, rtol=1e-4):
        print("✅ Outputs are numerically close.")
    else:
        print("⚠️  Outputs differ beyond tolerance (still may be acceptable due to FP order).")

    if torch.allclose(states_real, states_loop, atol=1e-5, rtol=1e-4):
        print("✅ Loop recursion matches real scan.")
    else:
        print("⚠️  Loop recursion differs beyond tolerance.")


if __name__ == "__main__":
    main()

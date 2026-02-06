"""
Test whether the lag terms A^k B of the L2N LTI system form an almost-orthogonal dictionary.

We build a single LTI system using the L2N parametrization (Block2x2DenseL2SSM),
extract (A, B) in x-coordinates, and compute the Gram matrix of the vectors:
    vec(A^k B),  k = 0..K-1

If the dictionary is close to orthogonal, the Gram matrix should be close to identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    # Preferred package API (after pip install neural-ssm)
    from neural_ssm.ssm.lru import Block2x2DenseL2SSM
except ImportError:
    # Fallback for running directly from this repository without installing
    from src.neural_ssm.ssm.lru import Block2x2DenseL2SSM


@dataclass
class DictTestConfig:
    d_state: int = 110    # must be even
    d_input: int = 1
    d_output: int = 1
    gamma: float  = .1
    K: int = 79             # sequence length for dictionary
    seed: int = 7

    # Initialization of the L2N system
    # Match the default L2N initialization used in the main model code.
    rho: float = .99
    max_phase_b: float = 2 * np.pi
    phase_center: float = 0.0
    random_phase: bool = True
    offdiag_scale: float = .005

    # Initialization of the LRU system (complex diagonal)
    r_min: float = 0.8
    r_max: float = 0.95
    max_phase_lru: float = 2 * np.pi

    # Plotting
    show_plot: bool = True


def main() -> None:
    cfg = DictTestConfig()
    torch.manual_seed(cfg.seed)

    if cfg.d_input != 1:
        raise ValueError("This script is configured for scalar input (d_input=1).")

    # ------------------------------------------------------------------
    # 1) Build a single LTI system using L2N parametrization
    # ------------------------------------------------------------------
    cell = Block2x2DenseL2SSM(
        d_state=cfg.d_state,
        d_input=cfg.d_input,
        d_output=cfg.d_output,
        gamma=cfg.gamma,
        train_gamma=False,
    )
    # Initialize A using the same procedure as the L2N parametrization
    # (see SSL -> Block2x2DenseL2SSM.init_on_circle in lru.py).
    cell.init_on_circle(
        rho=cfg.rho,
        max_phase=cfg.max_phase_b,
        phase_center=cfg.phase_center,
        random_phase=cfg.random_phase,
        offdiag_scale=cfg.offdiag_scale
    )

    # Extract dense (A, B) in x-coordinates
    A, B, _, _, _ = cell.compute_dense_matrices()
    A = A.detach()
    B = B.detach()

    # ------------------------------------------------------------------
    # 2) Build dictionary D = [vec(A^0 B), vec(A^1 B), ..., vec(A^{K-1} B)]
    # ------------------------------------------------------------------
    d_state, d_in = B.shape
    K = cfg.K

    # Precompute A^k B
    AkB = []
    Ak = torch.eye(d_state, device=A.device, dtype=A.dtype)
    for _ in range(K):
        AkB.append((Ak @ B).reshape(-1))  # vec(A^k B)
        Ak = Ak @ A

    D = torch.stack(AkB, dim=1)  # shape: (d_state * d_in, K)

    # Normalize columns
    D_norm = D / (D.norm(dim=0, keepdim=True) + 1e-12)

    # ------------------------------------------------------------------
    # 3) Compute Gram matrix and orthogonality metrics
    # ------------------------------------------------------------------
    G = D_norm.T @ D_norm  # (K, K)
    G_np = G.cpu().numpy()

    # Off-diagonal stats
    off_diag = G_np - np.eye(K)
    max_off_diag = np.max(np.abs(off_diag))
    mean_off_diag = np.mean(np.abs(off_diag))
    frob_off_diag = np.linalg.norm(off_diag, ord="fro")

    print("=== Dictionary Orthogonality Test (L2N) ===")
    print(f"d_state={cfg.d_state}, d_input={cfg.d_input}, K={K}")
    print(f"max |off-diagonal| = {max_off_diag:.4e}")
    print(f"mean |off-diagonal| = {mean_off_diag:.4e}")
    print(f"frobenius ||G - I|| = {frob_off_diag:.4e}")

    # Optionally print a small slice of G for inspection
    print("\nTop-left corner of Gram matrix (5x5):")
    with np.printoptions(precision=3, suppress=True):
        print(G_np[:5, :5])

    # ------------------------------------------------------------------
    # 4) Heatmap of the Gram matrix to visualize orthogonality
    # ------------------------------------------------------------------
    if cfg.show_plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(G_np, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        plt.colorbar(label="Inner product")
        plt.title("Gram matrix of normalized lag terms (A^k B)")
        plt.xlabel("k")
        plt.ylabel("k")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 5) Repeat the same test for the LRU parametrization
    # ------------------------------------------------------------------
    torch.manual_seed(cfg.seed)
    n_pairs = cfg.d_state // 2

    # Sample complex eigenvalues on an annulus with random phase
    r = torch.empty(n_pairs).uniform_(cfg.r_min, cfg.r_max)
    theta = torch.empty(n_pairs).uniform_(0.0, cfg.max_phase_lru)
    lambdas = torch.complex(r * torch.cos(theta), r * torch.sin(theta))

    # Convert to real 2x2 block-diagonal A
    A_blocks = []
    for lam in lambdas:
        a = lam.real.item()
        b = lam.imag.item()
        A_blocks.append(torch.tensor([[a, -b], [b, a]], dtype=A.dtype))
    A_lru = torch.block_diag(*A_blocks).to(A.device, A.dtype)

    # Random B for LRU case (same dimensions)
    B_lru = torch.randn(cfg.d_state, cfg.d_input, device=A.device, dtype=A.dtype) * 0.1

    # Build dictionary for LRU
    AkB_lru = []
    Ak = torch.eye(cfg.d_state, device=A.device, dtype=A.dtype)
    for _ in range(K):
        AkB_lru.append((Ak @ B_lru).reshape(-1))
        Ak = Ak @ A_lru

    D_lru = torch.stack(AkB_lru, dim=1)
    D_lru_norm = D_lru / (D_lru.norm(dim=0, keepdim=True) + 1e-12)
    G_lru = D_lru_norm.T @ D_lru_norm
    G_lru_np = G_lru.cpu().numpy()

    off_diag_lru = G_lru_np - np.eye(K)
    max_off_diag_lru = np.max(np.abs(off_diag_lru))
    mean_off_diag_lru = np.mean(np.abs(off_diag_lru))
    frob_off_diag_lru = np.linalg.norm(off_diag_lru, ord="fro")

    print("\n=== Dictionary Orthogonality Test (LRU) ===")
    print(f"d_state={cfg.d_state}, d_input={cfg.d_input}, K={K}")
    print(f"max |off-diagonal| = {max_off_diag_lru:.4e}")
    print(f"mean |off-diagonal| = {mean_off_diag_lru:.4e}")
    print(f"frobenius ||G - I|| = {frob_off_diag_lru:.4e}")

    print("\nTop-left corner of Gram matrix (5x5) for LRU:")
    with np.printoptions(precision=3, suppress=True):
        print(G_lru_np[:5, :5])

    if cfg.show_plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(G_lru_np, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        plt.colorbar(label="Inner product")
        plt.title("Gram matrix of normalized lag terms (A^k B) - LRU")
        plt.xlabel("k")
        plt.ylabel("k")
        plt.tight_layout()
        plt.show()

        cfg


if __name__ == "__main__":
    main()

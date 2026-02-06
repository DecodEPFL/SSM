"""
Toy training example that learns ONLY the L2N LTI system on a nonlinear
benchmark dataset, then checks whether the lag terms A^k B remain close
to an orthogonal dictionary before and after training.

This is intentionally "hard" for a pure LTI: the benchmark is nonlinear.
The goal is not perfect fit, but to see how training affects orthogonality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import nonlinear_benchmarks  # type: ignore
except Exception:  # pragma: no cover - optional dependency in CI
    nonlinear_benchmarks = None


from src.neural_ssm.ssm.lru import Block2x2DenseL2SSM, LRU


@dataclass
class TrainDictConfig:
    # Dimensions
    d_state: int = 1298   # must be even
    d_input: int = 1
    d_output: int = 1

    # Model choice: "l2n" or "lru"
    model: str = "l2n"

    # L2 gain
    gamma: float = 99
    train_gamma: bool = True

    # L2N initialization (student only)
    rho_student: float = 0.95
    max_phase_b: float = 2 * np.pi
    phase_center: float = 0.0
    random_phase: bool = True
    offdiag_scale: float = 0.005

    # LRU initialization (complex diagonal)
    lru_rmin: float = 0.8
    lru_rmax: float = 0.95
    lru_max_phase: float = 2 * np.pi

    # Dataset selection
    dataset_name: str = "Cascaded_Tanks()"  # fallback to WienerHammer if unavailable
    dataset_fallback: str = "WienerHammer"

    # Training
    n_epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    log_every: int = 10

    # Orthogonality test
    K: int = 900

    # Misc
    seed: int = 7
    mode: str = "scan"  # "scan" or "loop"
    show_plot: bool = True


def to_bln(arr: np.ndarray) -> torch.Tensor:
    """
    Convert input/output arrays to (B, L, N) tensors:
    - (L,) -> (1, L, 1)
    - (L, N) -> (1, L, N)
    - (B, L, N) -> unchanged
    """
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 1D/2D/3D array, got shape {arr.shape}")
    return torch.from_numpy(arr).float()


def resolve_benchmark_fn(
    name: str,
    fallback: str,
) -> Tuple[Callable, str]:
    """
    Resolve a benchmark function from nonlinear_benchmarks by name.
    Accepts short names (e.g., "Silverbox") or full names ("SilverboxBenchMark").
    Falls back to the provided fallback name if not found.
    """
    if nonlinear_benchmarks is None:
        raise ImportError(
            "nonlinear_benchmarks is not installed. "
            "Install it or change dataset settings."
        )

    name = name.strip()
    if name.endswith("()"):
        name = name[:-2]

    def candidate_names(base: str) -> list[str]:
        if base.endswith("BenchMark") or base.endswith("Benchmark"):
            return [base]
        return [
            base,
            f"{base}BenchMark",
            f"{base}Benchmark",
            f"{base.capitalize()}BenchMark",
            f"{base.capitalize()}Benchmark",
        ]

    for cand in candidate_names(name):
        if hasattr(nonlinear_benchmarks, cand):
            return getattr(nonlinear_benchmarks, cand), cand

    # Case-insensitive search
    name_lower = name.lower()
    for attr in dir(nonlinear_benchmarks):
        if attr.lower() == name_lower:
            return getattr(nonlinear_benchmarks, attr), attr
        if attr.lower().startswith(name_lower) and attr.lower().endswith("benchmark"):
            return getattr(nonlinear_benchmarks, attr), attr

    # Fallback
    for cand in candidate_names(fallback):
        if hasattr(nonlinear_benchmarks, cand):
            return getattr(nonlinear_benchmarks, cand), cand

    raise ValueError(
        f"Could not resolve dataset '{name}'. "
        f"Also tried fallback '{fallback}'."
    )


def build_l2n_cell(
    *,
    d_state: int,
    d_input: int,
    d_output: int,
    gamma: float,
    train_gamma: bool,
    rho: float,
    max_phase_b: float,
    phase_center: float,
    random_phase: bool,
    offdiag_scale: float,
) -> Block2x2DenseL2SSM:
    cell = Block2x2DenseL2SSM(
        d_state=d_state,
        d_input=d_input,
        d_output=d_output,
        gamma=gamma,
        train_gamma=train_gamma,
    )
    cell.init_on_circle(
        rho=rho,
        max_phase=max_phase_b,
        phase_center=phase_center,
        random_phase=random_phase,
        offdiag_scale=offdiag_scale,
    )
    return cell


def build_lru_cell(
    *,
    d_state: int,
    d_input: int,
    d_output: int,
    rmin: float,
    rmax: float,
    max_phase: float,
) -> LRU:
    # LRU uses complex diagonal dynamics with state_features = d_state
    return LRU(
        in_features=d_input,
        out_features=d_output,
        state_features=d_state,
        rmin=rmin,
        rmax=rmax,
        max_phase=max_phase,
    )


def compute_dictionary_stats(A: torch.Tensor, B: torch.Tensor, K: int):
    """
    Build dictionary D = [vec(A^0 B), ..., vec(A^{K-1} B)], normalize columns,
    and return the Gram matrix and orthogonality stats.
    """
    d_state, _ = B.shape

    Ak = torch.eye(d_state, device=A.device, dtype=A.dtype)
    cols = []
    for _ in range(K):
        cols.append((Ak @ B).reshape(-1))
        Ak = Ak @ A

    D = torch.stack(cols, dim=1)  # (d_state * d_in, K)
    D_norm = D / (D.norm(dim=0, keepdim=True) + 1e-12)
    G = D_norm.T @ D_norm

    off = G - torch.eye(K, device=G.device, dtype=G.dtype)
    max_off = off.abs().max().item()
    mean_off = off.abs().mean().item()
    frob_off = torch.linalg.norm(off, ord="fro").item()

    return G, max_off, mean_off, frob_off


def forward_model(
    model: nn.Module,
    u: torch.Tensor,
    *,
    model_type: str,
    mode: str,
) -> torch.Tensor:
    if model_type == "l2n":
        return model(u, return_state=False, mode=mode)
    if model_type == "lru":
        y, _ = model(u, mode=mode)
        return y
    raise ValueError(f"Unknown model type: {model_type}")


def extract_dense_matrices(
    model: nn.Module,
    *,
    model_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model_type == "l2n":
        A, B, _, _, _ = model.compute_dense_matrices()
        return A, B
    if model_type == "lru":
        A, B, _, _ = model.ss_real_matrices(to_numpy=False)
        return A, B
    raise ValueError(f"Unknown model type: {model_type}")


def plot_gram(G: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(G, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Inner product")
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("k")
    plt.tight_layout()
    plt.show()


def main() -> None:
    cfg = TrainDictConfig()
    if cfg.d_input != 1 or cfg.d_output != 1:
        raise ValueError("This script assumes scalar input/output (d_input=d_output=1).")

    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1) Load a nonlinear benchmark (more challenging than synthetic LTI)
    # ------------------------------------------------------------------
    bench_fn, bench_name = resolve_benchmark_fn(cfg.dataset_name, cfg.dataset_fallback)
    train_split, test_split = bench_fn()
    u_train_np, y_train_np = train_split
    u_test_np, y_test_np = test_split

    # Some benchmarks provide a warm-up window for evaluation.
    n_init = getattr(test_split, "state_initialization_window_length", 0)

    u_train = to_bln(u_train_np).to(device)
    y_train = to_bln(y_train_np).to(device)
    u_test = to_bln(u_test_np).to(device)
    y_test = to_bln(y_test_np).to(device)

    print(f"Using dataset: {bench_name}")
    print(f"Model: {cfg.model} (d_state={cfg.d_state})")
    if cfg.model == "lru":
        print("Note: LRU uses a complex state; real A,B have size 2*d_state.")
    print(f"Train shape: {tuple(u_train.shape)} -> {tuple(y_train.shape)}")
    print(f"Test  shape: {tuple(u_test.shape)} -> {tuple(y_test.shape)}")

    # ------------------------------------------------------------------
    # 2) Build the student LTI system (L2N or LRU)
    # ------------------------------------------------------------------
    if cfg.model == "l2n":
        student = build_l2n_cell(
            d_state=cfg.d_state,
            d_input=cfg.d_input,
            d_output=cfg.d_output,
            gamma=cfg.gamma,
            train_gamma=cfg.train_gamma,
            rho=cfg.rho_student,
            max_phase_b=cfg.max_phase_b,
            phase_center=cfg.phase_center,
            random_phase=cfg.random_phase,
            offdiag_scale=cfg.offdiag_scale,
        ).to(device)
    elif cfg.model == "lru":
        student = build_lru_cell(
            d_state=cfg.d_state,
            d_input=cfg.d_input,
            d_output=cfg.d_output,
            rmin=cfg.lru_rmin,
            rmax=cfg.lru_rmax,
            max_phase=cfg.lru_max_phase,
        ).to(device)
    else:
        raise ValueError(f"Unknown model '{cfg.model}'. Use 'l2n' or 'lru'.")

    # ------------------------------------------------------------------
    # 3) Orthogonality before training
    # ------------------------------------------------------------------
    with torch.no_grad():
        A0, B0 = extract_dense_matrices(student, model_type=cfg.model)
        G0, max0, mean0, frob0 = compute_dictionary_stats(A0, B0, cfg.K)

    print("=== Orthogonality BEFORE training ===")
    print(f"max |off-diagonal| = {max0:.4e}")
    print(f"mean |off-diagonal| = {mean0:.4e}")
    print(f"frobenius ||G - I|| = {frob0:.4e}")

    # ------------------------------------------------------------------
    # 4) Train the student LTI system on the benchmark data
    # ------------------------------------------------------------------
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(u_train, y_train)
    batch_size = min(cfg.batch_size, u_train.size(0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, cfg.n_epochs + 1):
        running = 0.0
        for u_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = forward_model(student, u_batch, model_type=cfg.model, mode=cfg.mode)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * u_batch.size(0)

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.n_epochs:
            avg = running / len(loader.dataset)
            print(f"Epoch {epoch:03d} | MSE = {avg:.6f}")

    # Quick test MSE (ignoring warm-up window if provided)
    student.eval()
    with torch.no_grad():
        y_test_pred = forward_model(student, u_test, model_type=cfg.model, mode=cfg.mode)
        if n_init > 0:
            y_test_eval = y_test[:, n_init:, :]
            y_pred_eval = y_test_pred[:, n_init:, :]
        else:
            y_test_eval = y_test
            y_pred_eval = y_test_pred
        test_mse = loss_fn(y_pred_eval, y_test_eval).item()
        print(f"Test MSE (after warm-up) = {test_mse:.6f}")

    # ------------------------------------------------------------------
    # 5) Orthogonality after training
    # ------------------------------------------------------------------
    with torch.no_grad():
        A1, B1 = extract_dense_matrices(student, model_type=cfg.model)
        G1, max1, mean1, frob1 = compute_dictionary_stats(A1, B1, cfg.K)

    print("\n=== Orthogonality AFTER training ===")
    print(f"max |off-diagonal| = {max1:.4e}")
    print(f"mean |off-diagonal| = {mean1:.4e}")
    print(f"frobenius ||G - I|| = {frob1:.4e}")

    # ------------------------------------------------------------------
    # 6) Visualize Gram matrices (before/after)
    # ------------------------------------------------------------------
    if cfg.show_plot:
        plot_gram(G0.detach().cpu().numpy(), "Gram matrix of lag terms (before training)")
        plot_gram(G1.detach().cpu().numpy(), "Gram matrix of lag terms (after training)")


if __name__ == "__main__":
    main()

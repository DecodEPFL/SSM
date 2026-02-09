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
from src.neural_ssm.ssm.mamba import RobustMambaDiagSSM


@dataclass
class TrainDictConfig:
    # Dimensions
    d_state: int = 1298   # must be even
    d_input: int = 1
    d_output: int = 1

    # Model choice: "l2n", "lru", or "tv"
    model: str = "tv"

    # L2 gain
    gamma: float = 99
    train_gamma: bool = True

    # L2N initialization (student only)
    rho_student: float = 0.99
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
    n_epochs: int = 2000
    batch_size: int = 16
    lr: float = 1e-3
    log_every: int = 10

    # Orthogonality test
    K: int = 500
    tv_ref_steps: int = 512
    tv_ref_batches: int = 1
    track_orthogonality: bool = True
    orthogonality_every: int = 10
    use_p_metric: bool = True
    use_output_dictionary: bool = True
    override_output_identity: bool = False
    override_output_identity_model: bool = False

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


def build_tv_cell(
    *,
    d_state: int,
    d_input: int,
    d_output: int,
    gamma: float,
    train_gamma: bool,
) -> RobustMambaDiagSSM:
    return RobustMambaDiagSSM(
        d_state=d_state,
        d_model=d_input,
        d_out=d_output,
        gamma=gamma,
        train_gamma=train_gamma,
    )


def compute_tv_effective_matrices(
    model: RobustMambaDiagSSM,
    u_ref: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        a_bt, b_bt, c_bt, _ = model._compute_params(u_ref)
        a_mean = a_bt.mean(dim=(0, 1))
        b_mean = b_bt.mean(dim=(0, 1))
        c_mean = c_bt.mean(dim=(0, 1))

        A = torch.diag(a_mean)

        W_raw = model.in_proj.W_raw
        sigma = torch.linalg.matrix_norm(W_raw, ord=2)
        scale = torch.clamp(sigma / model.in_proj.bound, min=1.0)
        W_norm = W_raw / scale
        B = torch.diag(b_mean) @ (model.gamma.to(W_norm) * W_norm)

        W_out_raw = model.out_proj.W_raw
        sigma_out = torch.linalg.matrix_norm(W_out_raw, ord=2)
        scale_out = torch.clamp(sigma_out / model.out_proj.bound, min=1.0)
        W_out_norm = W_out_raw / scale_out
        C = W_out_norm @ torch.diag(c_mean)
        D = torch.zeros(model.D_out, model.D, device=A.device, dtype=A.dtype)

    return A, B, C, D


def compute_dictionary_stats(
    A: torch.Tensor,
    B: torch.Tensor,
    K: int,
    *,
    C: torch.Tensor | None = None,
    P: torch.Tensor | None = None,
    use_output: bool = False,
    use_p_metric: bool = False,
):
    """
    Build dictionary D = [vec(A^0 B), ..., vec(A^{K-1} B)], normalize columns,
    and return the Gram matrix and orthogonality stats.
    """
    d_state, _ = B.shape

    Ak = torch.eye(d_state, device=A.device, dtype=A.dtype)
    cols = []
    for _ in range(K):
        if use_output:
            if C is None:
                raise ValueError("C must be provided when use_output=True.")
            cols.append((C @ Ak @ B).reshape(-1))
        else:
            cols.append((Ak @ B).reshape(-1))
        Ak = Ak @ A

    D = torch.stack(cols, dim=1)  # (d_state * d_in, K)
    if use_p_metric:
        if P is None:
            raise ValueError("P must be provided when use_p_metric=True.")
        if use_output:
            raise ValueError("P-metric is only defined for state dictionaries.")
        D_state = D.view(d_state, -1, K)
        D_state = D_state[:, 0, :]
        norms = torch.sqrt((D_state * (P @ D_state)).sum(dim=0, keepdim=True)) + 1e-12
        D_norm = D_state / norms
        G = D_norm.T @ (P @ D_norm)
    else:
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
    override_output_identity: bool,
) -> torch.Tensor:
    if model_type == "l2n":
        if override_output_identity:
            _, x_seq = model(u, return_state=True, mode=mode)
            return x_seq[:, :-1, :]
        return model(u, return_state=False, mode=mode)
    if model_type == "lru":
        y, states = model(u, mode=mode)
        if override_output_identity:
            return states[:, :-1, :].real
        return y
    if model_type == "tv":
        y, z_seq = model(u, mode=mode)
        if override_output_identity:
            return z_seq[:, :-1, :]
        return y
    raise ValueError(f"Unknown model type: {model_type}")


def extract_dense_matrices(
    model: nn.Module,
    *,
    model_type: str,
    u_ref: torch.Tensor | None = None,
    override_output_identity: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if model_type == "l2n":
        A, B, C, D, P = model.compute_dense_matrices()
        if override_output_identity:
            C = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            D = torch.zeros(A.size(0), B.size(1), device=A.device, dtype=A.dtype)
        return A, B, C, D, P
    if model_type == "lru":
        A, B, C, D = model.ss_real_matrices(to_numpy=False)
        if override_output_identity:
            C = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            D = torch.zeros(A.size(0), B.size(1), device=A.device, dtype=A.dtype)
        return A, B, C, D, None
    if model_type == "tv":
        if u_ref is None:
            raise ValueError("u_ref must be provided for tv parametrization.")
        A, B, C, D = compute_tv_effective_matrices(model, u_ref)
        if override_output_identity:
            C = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            D = torch.zeros(A.size(0), B.size(1), device=A.device, dtype=A.dtype)
        return A, B, C, D, None
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


def plot_orthogonality(frob_history: list[float], title: str) -> None:
    plt.figure(figsize=(6, 4))
    epochs = np.arange(1, len(frob_history) + 1)
    plt.plot(epochs, frob_history, marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Frobenius ||G - I||")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def main() -> None:
    cfg = TrainDictConfig()
    if cfg.d_input != 1 or cfg.d_output != 1:
        raise ValueError("This script assumes scalar input/output (d_input=d_output=1).")
    if cfg.override_output_identity_model and cfg.d_output != cfg.d_state:
        raise ValueError(
            "override_output_identity_model requires d_output == d_state so the "
            "identity output is well-defined."
        )
    if cfg.override_output_identity_model and cfg.model == "lru":
        raise ValueError(
            "override_output_identity_model is not supported for LRU models "
            "because the internal state is complex-valued."
        )

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
    elif cfg.model == "tv":
        student = build_tv_cell(
            d_state=cfg.d_state,
            d_input=cfg.d_input,
            d_output=cfg.d_output,
            gamma=cfg.gamma,
            train_gamma=cfg.train_gamma,
        ).to(device)
    else:
        raise ValueError(f"Unknown model '{cfg.model}'. Use 'l2n', 'lru', or 'tv'.")

    if cfg.model == "tv":
        u_ref = u_train[: cfg.tv_ref_batches, : cfg.tv_ref_steps, :]
    else:
        u_ref = None

    # ------------------------------------------------------------------
    # 3) Orthogonality before training
    # ------------------------------------------------------------------
    with torch.no_grad():
        A0, B0, C0, _, P0 = extract_dense_matrices(
            student,
            model_type=cfg.model,
            u_ref=u_ref,
            override_output_identity=cfg.override_output_identity_model,
        )
        G0, max0, mean0, frob0 = compute_dictionary_stats(A0, B0, cfg.K)

        if cfg.use_p_metric and P0 is not None:
            _, max0_p, mean0_p, frob0_p = compute_dictionary_stats(
                A0,
                B0,
                cfg.K,
                P=P0,
                use_p_metric=True,
            )
        else:
            max0_p = mean0_p = frob0_p = None

        if cfg.use_output_dictionary:
            if cfg.override_output_identity or cfg.override_output_identity_model:
                C0_use = torch.eye(A0.size(0), device=A0.device, dtype=A0.dtype)
            else:
                C0_use = C0
            if C0_use is None:
                raise ValueError("C must be available for output dictionary tracking.")
            _, max0_out, mean0_out, frob0_out = compute_dictionary_stats(
                A0,
                B0,
                cfg.K,
                C=C0_use,
                use_output=True,
            )
        else:
            max0_out = mean0_out = frob0_out = None

    print("=== Orthogonality BEFORE training ===")
    print(f"max |off-diagonal| = {max0:.4e}")
    print(f"mean |off-diagonal| = {mean0:.4e}")
    print(f"frobenius ||G - I|| = {frob0:.4e}")
    if frob0_p is not None:
        print(f"max |off-diagonal| (P-metric) = {max0_p:.4e}")
        print(f"mean |off-diagonal| (P-metric) = {mean0_p:.4e}")
        print(f"frobenius ||G - I|| (P-metric) = {frob0_p:.4e}")
    if frob0_out is not None:
        print(f"max |off-diagonal| (output) = {max0_out:.4e}")
        print(f"mean |off-diagonal| (output) = {mean0_out:.4e}")
        print(f"frobenius ||G - I|| (output) = {frob0_out:.4e}")

    # ------------------------------------------------------------------
    # 4) Train the student LTI system on the benchmark data
    # ------------------------------------------------------------------
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(u_train, y_train)
    batch_size = min(cfg.batch_size, u_train.size(0))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    frob_history: list[float] = []
    frob_history_p: list[float] = []
    frob_history_out: list[float] = []

    for epoch in range(1, cfg.n_epochs + 1):
        running = 0.0
        for u_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = forward_model(
                student,
                u_batch,
                model_type=cfg.model,
                mode=cfg.mode,
                override_output_identity=cfg.override_output_identity_model,
            )
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * u_batch.size(0)

        frob_epoch = None
        frob_epoch_p = None
        frob_epoch_out = None
        if cfg.track_orthogonality and (epoch % cfg.orthogonality_every == 0 or epoch == cfg.n_epochs):
            with torch.no_grad():
                A_epoch, B_epoch, C_epoch, _, P_epoch = extract_dense_matrices(
                    student,
                    model_type=cfg.model,
                    u_ref=u_ref,
                    override_output_identity=cfg.override_output_identity_model,
                )
                _, _, _, frob_epoch = compute_dictionary_stats(A_epoch, B_epoch, cfg.K)
                frob_history.append(frob_epoch)
                if cfg.use_p_metric and P_epoch is not None:
                    _, _, _, frob_epoch_p = compute_dictionary_stats(
                        A_epoch,
                        B_epoch,
                        cfg.K,
                        P=P_epoch,
                        use_p_metric=True,
                    )
                    frob_history_p.append(frob_epoch_p)
                if cfg.use_output_dictionary:
                    if cfg.override_output_identity or cfg.override_output_identity_model:
                        C_use = torch.eye(A_epoch.size(0), device=A_epoch.device, dtype=A_epoch.dtype)
                    else:
                        C_use = C_epoch
                    if C_use is None:
                        raise ValueError("C must be available for output dictionary tracking.")
                    _, _, _, frob_epoch_out = compute_dictionary_stats(
                        A_epoch,
                        B_epoch,
                        cfg.K,
                        C=C_use,
                        use_output=True,
                    )
                    frob_history_out.append(frob_epoch_out)

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.n_epochs:
            avg = running / len(loader.dataset)
            if frob_epoch is None:
                print(f"Epoch {epoch:03d} | MSE = {avg:.6f}")
            else:
                msg = f"Epoch {epoch:03d} | MSE = {avg:.6f} | Frobenius ||G-I|| = {frob_epoch:.4e}"
                if frob_epoch_p is not None:
                    msg += f" | (P) {frob_epoch_p:.4e}"
                if frob_epoch_out is not None:
                    msg += f" | (out) {frob_epoch_out:.4e}"
                print(msg)

    # Quick test MSE (ignoring warm-up window if provided)
    student.eval()
    with torch.no_grad():
        y_test_pred = forward_model(
            student,
            u_test,
            model_type=cfg.model,
            mode=cfg.mode,
            override_output_identity=cfg.override_output_identity_model,
        )
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
        A1, B1, C1, _, P1 = extract_dense_matrices(
            student,
            model_type=cfg.model,
            u_ref=u_ref,
            override_output_identity=cfg.override_output_identity_model,
        )
        G1, max1, mean1, frob1 = compute_dictionary_stats(A1, B1, cfg.K)
        if cfg.use_p_metric and P1 is not None:
            _, max1_p, mean1_p, frob1_p = compute_dictionary_stats(
                A1,
                B1,
                cfg.K,
                P=P1,
                use_p_metric=True,
            )
        else:
            max1_p = mean1_p = frob1_p = None

        if cfg.use_output_dictionary:
            if cfg.override_output_identity or cfg.override_output_identity_model:
                C1_use = torch.eye(A1.size(0), device=A1.device, dtype=A1.dtype)
            else:
                C1_use = C1
            if C1_use is None:
                raise ValueError("C must be available for output dictionary tracking.")
            _, max1_out, mean1_out, frob1_out = compute_dictionary_stats(
                A1,
                B1,
                cfg.K,
                C=C1_use,
                use_output=True,
            )
        else:
            max1_out = mean1_out = frob1_out = None

    print("\n=== Orthogonality AFTER training ===")
    print(f"max |off-diagonal| = {max1:.4e}")
    print(f"mean |off-diagonal| = {mean1:.4e}")
    print(f"frobenius ||G - I|| = {frob1:.4e}")
    if frob1_p is not None:
        print(f"max |off-diagonal| (P-metric) = {max1_p:.4e}")
        print(f"mean |off-diagonal| (P-metric) = {mean1_p:.4e}")
        print(f"frobenius ||G - I|| (P-metric) = {frob1_p:.4e}")
    if frob1_out is not None:
        print(f"max |off-diagonal| (output) = {max1_out:.4e}")
        print(f"mean |off-diagonal| (output) = {mean1_out:.4e}")
        print(f"frobenius ||G - I|| (output) = {frob1_out:.4e}")

    # ------------------------------------------------------------------
    # 6) Visualize Gram matrices (before/after)
    # ------------------------------------------------------------------
    if cfg.show_plot:
        plot_gram(G0.detach().cpu().numpy(), "Gram matrix of lag terms (before training)")
        plot_gram(G1.detach().cpu().numpy(), "Gram matrix of lag terms (after training)")
        if cfg.track_orthogonality and frob_history:
            plot_orthogonality(frob_history, "Orthogonality drift (state, Euclidean)")
        if cfg.track_orthogonality and frob_history_p:
            plot_orthogonality(frob_history_p, "Orthogonality drift (state, P-metric)")
        if cfg.track_orthogonality and frob_history_out:
            plot_orthogonality(frob_history_out, "Orthogonality drift (output)")


if __name__ == "__main__":
    main()

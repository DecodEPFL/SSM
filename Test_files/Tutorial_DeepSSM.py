"""
Tutorial: train DeepSSM on a nonlinear benchmark dataset.

This tutorial is intentionally simple and uses only the core DeepSSM API.
What we do:
1) load a nonlinear benchmark dataset
2) build DeepSSM directly from constructor arguments
3) train with a small, standard PyTorch loop
4) evaluate with full-sequence and chunked stateful inference

The goal is to make each step easy to read and reuse in your own scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nonlinear_benchmarks
import numpy as np
import torch
import torch.nn as nn

try:
    # Preferred package API (after pip install neural-ssm)
    from neural_ssm import DeepSSM
except ImportError:
    # Fallback for running directly from this repository without installing
    from src.neural_ssm import DeepSSM


@dataclass
class TutorialConfig:
    # Reproducibility / runtime
    seed: int = 9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    epochs: int = 200
    learning_rate: float = 1e-2
    log_every: int = 50

    # set up the DeepSSM architecture
    d_model: int = 8
    d_state: int = 8
    n_layers: int = 4
    param: str = "tv"  # "lru" | "l2n" | "tv" | ...
    ff: str = "LGLU"  # "GLU" | "MLP" | "LMLP" | "LGLU" | "TLIP"
    gamma: float | None = 2.0  # set to None if you want gamma to be trainable

    # Forward execution mode
    train_mode: str = "scan"  # "scan" or "loop"
    eval_mode: str = "scan"

    # Optional plotting
    save_plot: bool = True
    out_dir: Path = Path(__file__).resolve().parent / "checkpoints" / "tutorial"

    # Optional stateful inference demo
    stream_chunk_len: int = 200


def set_seed(seed: int) -> None:
    # Reproducibility helper: keeps runs comparable across executions.
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def to_bln(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert tensors to the canonical DeepSSM shape (B, L, N):
      B = batch size, L = sequence length, N = feature dimension.
    """
    x_t = torch.as_tensor(x, dtype=torch.float32)
    if x_t.ndim == 1:  # (L,) -> (1, L, 1)
        return x_t[None, :, None]
    if x_t.ndim == 2:  # (L, N) -> (1, L, N)
        return x_t[None, :, :]
    if x_t.ndim == 3:  # already (B, L, N)
        return x_t
    raise ValueError(f"Unsupported tensor rank {x_t.ndim}; expected 1, 2 or 3.")


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


@torch.no_grad()
def predict_streaming(
    model: DeepSSM,
    u: torch.Tensor,
    chunk_len: int,
    mode: str = "scan",
) -> torch.Tensor:
    """
    Run stateful inference by feeding chunks sequentially.

    We keep and pass `state` between chunks, which is how you do
    streaming/online usage without resetting the recurrent memory.
    """
    model.eval()
    outputs = []
    state = None
    for start in range(0, u.size(1), chunk_len):
        u_chunk = u[:, start : start + chunk_len, :]
        y_chunk, state = model(u_chunk, state=state, mode=mode)
        outputs.append(y_chunk)
    return torch.cat(outputs, dim=1)


def main() -> None:
    cfg = TutorialConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ------------------------------------------------------------------
    # 1) Load benchmark dataset
    # ------------------------------------------------------------------
    # The Wiener-Hammerstein benchmark returns two splits:
    # - training identification data
    # - test data with a warm-up window definition
    train_split, test_split = nonlinear_benchmarks.WienerHammerBenchMark()
    u_train_np, y_train_np = train_split
    u_test_np, y_test_np = test_split

    # Number of initial points commonly ignored during evaluation.
    # The model uses this prefix to initialize internal dynamics.
    n_init = test_split.state_initialization_window_length

    # Convert to canonical DeepSSM shape: (B, L, N).
    u_train = to_bln(u_train_np).to(device)
    y_train = to_bln(y_train_np).to(device)
    u_test = to_bln(u_test_np).to(device)
    y_test = to_bln(y_test_np).to(device)

    # ------------------------------------------------------------------
    # 2) Build DeepSSM (direct constructor style)
    # ------------------------------------------------------------------
    # This is the direct API. We do not use SSMConfig here on purpose.
    # DeepSSM returns (y, state):
    # - y: output sequence, shape (B, L, d_output)
    # - state: list of last states (one per SSL block)
    model = DeepSSM(
        d_input=u_train.size(-1),
        d_output=y_train.size(-1),
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        n_layers=cfg.n_layers,
        param=cfg.param,
        ff=cfg.ff,
        gamma=cfg.gamma,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    print("Starting training...")
    print(f"Device: {device}")
    print(f"Train input shape: {tuple(u_train.shape)}")
    print(f"Train output shape: {tuple(y_train.shape)}")
    print(
        "Model settings: "
        f"param={cfg.param}, ff={cfg.ff}, d_model={cfg.d_model}, "
        f"d_state={cfg.d_state}, n_layers={cfg.n_layers}, gamma={cfg.gamma}"
    )

    train_losses: list[float] = []
    test_losses: list[float] = []

    # ------------------------------------------------------------------
    # 3) Simple training loop
    # ------------------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        # ---- training pass ----
        model.train()
        optimizer.zero_grad()

        # Forward pass with selected execution mode ("scan" or "loop").
        # We do not pass a state here -> zero initialization at each call.
        y_train_pred, _ = model(u_train, mode=cfg.train_mode)
        train_loss = criterion(y_train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        # ---- validation pass ----
        model.eval()
        with torch.no_grad():
            y_test_pred, _ = model(u_test, mode=cfg.eval_mode)
            # Ignore warm-up prefix for metric computation.
            test_loss = criterion(y_test_pred[:, n_init:, :], y_test[:, n_init:, :])

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            test_rmse = rmse(y_test[:, n_init:, :], y_test_pred[:, n_init:, :])
            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"train_loss={train_loss.item():.4e} | "
                f"test_loss={test_loss.item():.4e} | "
                f"test_rmse={test_rmse:.4e}"
            )

    # ------------------------------------------------------------------
    # 4) Full vs stateful inference (same model, chunked evaluation)
    # ------------------------------------------------------------------
    with torch.no_grad():
        # Standard full-sequence inference.
        y_full, _ = model(u_test, state=None, mode=cfg.eval_mode)
        # Chunked inference where the state is passed from one chunk to the next.
        y_stream = predict_streaming(
            model,
            u_test,
            chunk_len=cfg.stream_chunk_len,
            mode=cfg.eval_mode,
        )
    max_diff = (y_full - y_stream).abs().max().item()
    final_rmse = rmse(y_test[:, n_init:, :], y_full[:, n_init:, :])

    print("\nTraining complete.")
    print(f"Final test RMSE (after warm-up): {final_rmse:.6e}")
    print(f"Max |full - streaming| difference: {max_diff:.6e}")

    # ------------------------------------------------------------------
    # 5) Optional plot
    # ------------------------------------------------------------------
    if cfg.save_plot:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        fig_path = cfg.out_dir / "tutorial_prediction.png"

        y_true_np = y_test[0, :, 0].detach().cpu().numpy()
        y_pred_np = y_full[0, :, 0].detach().cpu().numpy()

        plt.figure(figsize=(8, 3))
        plt.plot(y_true_np, label="True", linewidth=1.2)
        plt.plot(y_pred_np, label="DeepSSM", linewidth=1.2, linestyle="--")
        plt.axvline(n_init, color="gray", linestyle=":", linewidth=1, label="Warm-up end")
        plt.xlabel("Time step")
        plt.ylabel("Output")
        plt.title("DeepSSM Tutorial - Wiener Hammerstein benchmark")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
        print(f"Saved plot to: {fig_path}")


if __name__ == "__main__":
    main()

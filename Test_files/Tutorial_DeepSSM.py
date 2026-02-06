"""
Tutorial: train DeepSSM on a nonlinear benchmark dataset.

This tutorial is intentionally simple and uses only the core DeepSSM API.
What we do:
1) load a nonlinear benchmark dataset
2) build DeepSSM
3) train with a small, standard PyTorch loop
4) evaluate with full-sequence and chunked stateful inference

The goal is to make each step easy to read and reuse in your own scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import nonlinear_benchmarks
from nonlinear_benchmarks.error_metrics import RMSE
import numpy as np
import torch
import torch.nn as nn
from nonlinear_benchmarks.error_metrics import RMSE

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
    epochs: int = 9000
    learning_rate: float = 1.6568e-02
    log_every: int = 50

    # set up the DeepSSM architecture
    d_model: int = 2
    d_state: int = 600
    n_layers: int = 1
    param: str = "l2n"  # "lru" | "l2n" | "tv" | ...
    ff: str = "LGLU"  # "GLU" | "MLP" | "LMLP" | "LGLU" | "TLIP"
    gamma: float | None = None  # set to None if you want gamma to be trainable
    max_phase_b: float = 2 * np.pi

    # Forward execution mode
    train_mode: str = "scan"  # "scan" or "loop"
    eval_mode: str = "scan"

    # Plotting
    show_plots: bool = True

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
    # - state: list of states (one per SSL block)
    model = DeepSSM(
        d_input=u_train.size(-1),
        d_output=y_train.size(-1),
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        n_layers=cfg.n_layers,
        param=cfg.param,
        ff=cfg.ff,
        gamma=cfg.gamma,
        max_phase_b=cfg.max_phase_b
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
    best_val_rmse = float("inf")

    # ------------------------------------------------------------------
    # 3) Simple training loop
    # ------------------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        # ---- training pass ----
        model.train()
        optimizer.zero_grad()

        # Forward pass with selected execution mode ("scan" or "loop").
        # We do not pass a state here -> zero initialization at each call since we feed the whole input sequence at once.
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
            test_rmse = 1000*RMSE(
                y_test[0, n_init:, 0].detach().cpu().numpy(),
                y_test_pred[0, n_init:, 0].detach().cpu().numpy(),
            )
            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"train_loss={train_loss.item():.4e} | "
                f"test_loss={test_loss.item():.4e} | "
                f"test_rmse={test_rmse:.4e}"
            )
        best_val_rmse = min(
            best_val_rmse,
            1000*RMSE(
                y_test[0, n_init:, 0].detach().cpu().numpy(),
                y_test_pred[0, n_init:, 0].detach().cpu().numpy(),
            ),
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
    final_rmse = 1000*RMSE(
        y_test[0, n_init:, 0].detach().cpu().numpy(),
        y_full[0, n_init:, 0].detach().cpu().numpy(),
    )

    print("\nTraining complete.")
    print(f"Final test RMSE (after warm-up): {final_rmse:.6e}")
    print(f"Best validation RMSE (after warm-up): {best_val_rmse:.6e}")
    print(f"Max |full - streaming| difference: {max_diff:.6e}")

    # ------------------------------------------------------------------
    # 5) Plot results at the end
    # ------------------------------------------------------------------
    if cfg.show_plots:
        y_true_np = y_test[0, :, 0].detach().cpu().numpy()
        y_pred_np = y_full[0, :, 0].detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

        # Prediction plot
        ax1.plot(y_true_np, label="True", linewidth=1.2)
        ax1.plot(y_pred_np, label="DeepSSM", linewidth=1.2, linestyle="--")
        ax1.axvline(n_init, color="gray", linestyle=":", linewidth=1, label="Warm-up end")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Output")
        ax1.set_title("DeepSSM Tutorial - Prediction")
        ax1.legend()
        ax1.grid(alpha=0.25)

        # Log-loss plot
        epochs = np.arange(1, len(train_losses) + 1)
        ax2.plot(epochs, train_losses, label="Train loss", linewidth=1.2)
        ax2.plot(epochs, test_losses, label="Test loss", linewidth=1.2)
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE loss (log scale)")
        ax2.set_title("Training curves")
        ax2.legend()
        ax2.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

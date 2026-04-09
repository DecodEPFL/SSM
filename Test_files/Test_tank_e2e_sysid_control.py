"""
End-to-end tank system identification + certified closed-loop control with DeepSSM.

Workflow:
1. Identify a tank plant from open-loop data with a certified DeepSSM.
2. Freeze the identified plant.
3. Train a second certified DeepSSM as a feedback controller for setpoint tracking.
4. Enforce the small-gain condition by choosing the controller certificate so that
   ||G_plant||_2 * ||G_ctrl||_2 < 1.

Why this version uses `scan` for identification and `loop` for control:
- Open-loop identification is naturally sequence-to-sequence, so `mode="scan"` is fast.
- Closed-loop control is implemented with the sampled-data convention
      y_k -> controller -> u_k -> plant -> y_{k+1}
  which avoids an algebraic loop and the per-iteration fixed-point solve.

If you later want same-sample feedback with direct feedthrough, that is the moment to
bring back the trajectory-level fixed-point iterations from `Test_files/Control.py`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import nonlinear_benchmarks
except ImportError:
    nonlinear_benchmarks = None

from src.neural_ssm.ssm import DeepSSM, SSMConfig


TensorState = Optional[Sequence[Optional[torch.Tensor]]]


@dataclass
class ChannelwiseStandardizer:
    """Per-channel affine normalization built on the training split only."""

    u_mean: torch.Tensor
    u_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    @staticmethod
    def _reduce_dims(x: torch.Tensor) -> tuple[int, ...]:
        return tuple(range(x.ndim - 1))

    @staticmethod
    def _match(x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        out = stats
        while out.ndim > x.ndim and out.shape[0] == 1:
            out = out.squeeze(0)
        while out.ndim < x.ndim:
            out = out.unsqueeze(0)
        return out.to(device=x.device, dtype=x.dtype)

    @classmethod
    def fit(
        cls,
        u_train: torch.Tensor,
        y_train: torch.Tensor,
        eps: float = 1e-6,
    ) -> "ChannelwiseStandardizer":
        reduce_u = cls._reduce_dims(u_train)
        reduce_y = cls._reduce_dims(y_train)
        return cls(
            u_mean=u_train.mean(dim=reduce_u, keepdim=True),
            u_std=u_train.std(dim=reduce_u, keepdim=True, unbiased=False).clamp_min(eps),
            y_mean=y_train.mean(dim=reduce_y, keepdim=True),
            y_std=y_train.std(dim=reduce_y, keepdim=True, unbiased=False).clamp_min(eps),
        )

    def transform_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self._match(u, self.u_mean)) / self._match(u, self.u_std)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self._match(y, self.y_mean)) / self._match(y, self.y_std)

    def inverse_transform_u(self, u: torch.Tensor) -> torch.Tensor:
        return u * self._match(u, self.u_std) + self._match(u, self.u_mean)

    def inverse_transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._match(y, self.y_std) + self._match(y, self.y_mean)


@dataclass
class ExperimentConfig:
    seed: int = 9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_data: bool = True

    # Plant identification
    plant_d_model: int = 16
    plant_d_state: int = 24
    plant_n_layers: int = 2
    plant_d_hidden: int = 24
    plant_nl_layers: int = 2
    plant_gamma: float = 1.8
    plant_epochs: int = 2500
    plant_lr: float = 2e-3
    plant_log_every: int = 100

    # Controller
    controller_d_model: int = 8
    controller_d_state: int = 12
    controller_n_layers: int = 2
    controller_d_hidden: int = 16
    controller_nl_layers: int = 2
    small_gain_margin: float = 0.85
    controller_epochs: int = 2500
    controller_lr: float = 1e-3
    controller_batch_size: int = 64
    controller_horizon: int = 120
    controller_warmup_steps: int = 80
    controller_log_every: int = 100
    controller_grad_clip: float = 1.0

    # Closed-loop training domain
    ref_low_quantile: float = 0.15
    ref_high_quantile: float = 0.85
    u_low_quantile: float = 0.10
    u_high_quantile: float = 0.90
    u_clip_quantile: float = 0.98

    # Cost
    tracking_weight: float = 1.0
    terminal_weight: float = 2.0
    effort_weight: float = 1e-3
    delta_u_weight: float = 5e-3
    settle_fraction: float = 0.4

    # Evaluation
    eval_horizon: int = 180
    eval_cases: int = 4
    show_plots: bool = True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def to_bln(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    x_t = torch.as_tensor(x, dtype=torch.float32)
    if x_t.ndim == 1:
        return x_t[None, :, None]
    if x_t.ndim == 2:
        return x_t[None, :, :]
    if x_t.ndim == 3:
        return x_t
    raise ValueError(f"Unsupported tensor rank {x_t.ndim}; expected 1, 2 or 3.")


def load_tank_data(cfg: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, str]:
    """
    Load exactly the same nonlinear benchmark used in `Benchmark.py`.
    """
    del cfg
    if nonlinear_benchmarks is None:
        raise RuntimeError(
            "This script is configured to use only `nonlinear_benchmarks.Cascaded_Tanks()`, "
            "matching `Test_files/Benchmark.py`, but `nonlinear_benchmarks` is not installed."
        )

    train_split, test_split = nonlinear_benchmarks.Cascaded_Tanks()
    u_train_np, y_train_np = train_split
    u_val_np, y_val_np = test_split
    init_window = int(test_split.state_initialization_window_length)
    return (
        to_bln(u_train_np),
        to_bln(y_train_np),
        to_bln(u_val_np),
        to_bln(y_val_np),
        init_window,
        "nonlinear_benchmarks.Cascaded_Tanks",
    )


def build_certified_ssm(
    d_input: int,
    d_output: int,
    *,
    d_model: int,
    d_state: int,
    n_layers: int,
    d_hidden: int,
    nl_layers: int,
    gamma: float,
    learn_x0: bool,
) -> DeepSSM:
    config = SSMConfig(
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        ff="LGLU2",
        d_hidden=d_hidden,
        nl_layers=nl_layers,
        param="tv",
        gamma=gamma,
        train_gamma=True,
        init="rand",
        rho=0.92,
        max_phase_b=2*np.pi,
        phase_center=0.0,
        random_phase=True,
        learn_x0=learn_x0,
    )
    return DeepSSM(d_input=d_input, d_output=d_output, config=config)


def certified_gain(model: DeepSSM) -> float:
    gamma_t = getattr(model, "gamma_t", None)
    if gamma_t is not None:
        return float(gamma_t.abs().item())
    if getattr(model.config, "gamma", None) is not None:
        return float(abs(model.config.gamma))
    return float(model.conservative_gamma_product().item())


def raw_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())


def channel_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    return torch.quantile(flat, q, dim=0)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def train_identified_plant(
    model: DeepSSM,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    u_val: torch.Tensor,
    y_val: torch.Tensor,
    init_window: int,
    normalizer: ChannelwiseStandardizer,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, list[float] | float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.plant_lr)
    criterion = nn.MSELoss()

    u_train_n = normalizer.transform_u(u_train).to(device)
    y_train_n = normalizer.transform_y(y_train).to(device)
    u_val_n = normalizer.transform_u(u_val).to(device)
    y_val_n = normalizer.transform_y(y_val).to(device)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.plant_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        y_train_pred_n, _ = model(u_train_n, mode="scan")
        train_loss = criterion(y_train_pred_n, y_train_n)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred_n, _ = model(u_val_n, mode="scan")
            val_loss = criterion(y_val_pred_n[:, init_window:, :], y_val_n[:, init_window:, :])

        train_value = float(train_loss.item())
        val_value = float(val_loss.item())
        history["train_loss"].append(train_value)
        history["val_loss"].append(val_value)

        if val_value < best_val:
            best_val = val_value
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % cfg.plant_log_every == 0 or epoch == cfg.plant_epochs:
            print(
                f"[plant] epoch {epoch:4d}/{cfg.plant_epochs} "
                f"| train={train_value:.4e} | val={val_value:.4e}"
            )

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    return history


class ClampedSSMController(nn.Module):
    """
    Output clamp is 1-Lipschitz, so it does not invalidate the DeepSSM gain certificate.
    """

    def __init__(self, ssm: DeepSSM, u_min: torch.Tensor, u_max: torch.Tensor):
        super().__init__()
        self.ssm = ssm
        self.register_buffer("u_min", u_min.reshape(1, 1, -1))
        self.register_buffer("u_max", u_max.reshape(1, 1, -1))

    @property
    def certificate_gamma(self) -> float:
        return certified_gain(self.ssm)

    def forward(
        self,
        error: torch.Tensor,
        state: TensorState = None,
        *,
        reset_state: bool,
        detach_state: bool,
    ) -> tuple[torch.Tensor, TensorState]:
        u, state = self.ssm(
            error,
            state=state,
            mode="loop",
            reset_state=reset_state,
            detach_state=detach_state,
        )
        u = torch.maximum(torch.minimum(u, self.u_max), self.u_min)
        return u, state


@torch.no_grad()
def warmup_plant(
    plant: DeepSSM,
    u_init_n: torch.Tensor,
    warmup_steps: int,
) -> tuple[torch.Tensor, TensorState]:
    warmup_u = u_init_n.unsqueeze(1).expand(-1, warmup_steps, -1)
    y_hist, plant_state = plant(warmup_u, mode="scan", reset_state=True, detach_state=True)
    return y_hist[:, -1:, :], plant_state


def rollout_closed_loop(
    plant: DeepSSM,
    controller: ClampedSSMController,
    ref_n: torch.Tensor,
    u_init_n: torch.Tensor,
    warmup_steps: int,
    *,
    detach_state: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, horizon, _ = ref_n.shape
    del batch_size

    y_prev, plant_state = warmup_plant(plant, u_init_n, warmup_steps)
    controller_state: TensorState = None

    ys = []
    us = []
    errors = []

    for step in range(horizon):
        e_t = ref_n[:, step : step + 1, :] - y_prev
        u_t, controller_state = controller(
            e_t,
            state=controller_state,
            reset_state=(step == 0),
            detach_state=detach_state,
        )
        y_t, plant_state = plant(
            u_t,
            state=plant_state,
            mode="loop",
            reset_state=False,
            detach_state=detach_state,
        )
        ys.append(y_t)
        us.append(u_t)
        errors.append(e_t)
        y_prev = y_t

    return torch.cat(ys, dim=1), torch.cat(us, dim=1), torch.cat(errors, dim=1)


def closed_loop_tracking_loss(
    y_n: torch.Tensor,
    u_n: torch.Tensor,
    ref_n: torch.Tensor,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    settle_start = min(int(cfg.settle_fraction * y_n.shape[1]), y_n.shape[1] - 1)
    tracking = (y_n[:, settle_start:, :] - ref_n[:, settle_start:, :]).pow(2).mean()
    terminal = (y_n[:, -1:, :] - ref_n[:, -1:, :]).pow(2).mean()
    effort = u_n.pow(2).mean()
    delta_u = (u_n[:, 1:, :] - u_n[:, :-1, :]).pow(2).mean() if u_n.shape[1] > 1 else torch.zeros_like(effort)
    return (
        cfg.tracking_weight * tracking
        + cfg.terminal_weight * terminal
        + cfg.effort_weight * effort
        + cfg.delta_u_weight * delta_u
    )


def sample_closed_loop_batch(
    batch_size: int,
    horizon: int,
    *,
    ref_low_raw: torch.Tensor,
    ref_high_raw: torch.Tensor,
    u_low_raw: torch.Tensor,
    u_high_raw: torch.Tensor,
    normalizer: ChannelwiseStandardizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref_span = (ref_high_raw - ref_low_raw).to(device)
    u_span = (u_high_raw - u_low_raw).to(device)
    ref_base = ref_low_raw.to(device)
    u_base = u_low_raw.to(device)

    ref_level_raw = ref_base + torch.rand(batch_size, ref_base.numel(), device=device) * ref_span
    u_init_raw = u_base + torch.rand(batch_size, u_base.numel(), device=device) * u_span

    ref_raw = ref_level_raw.unsqueeze(1).expand(-1, horizon, -1)
    ref_n = normalizer.transform_y(ref_raw)
    u_init_n = normalizer.transform_u(u_init_raw)
    return ref_n, u_init_n


def train_controller(
    plant: DeepSSM,
    controller: ClampedSSMController,
    normalizer: ChannelwiseStandardizer,
    u_train_raw: torch.Tensor,
    y_train_raw: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.Adam(controller.parameters(), lr=cfg.controller_lr)
    loss_history: list[float] = []

    ref_low_raw = channel_quantile(y_train_raw, cfg.ref_low_quantile)
    ref_high_raw = channel_quantile(y_train_raw, cfg.ref_high_quantile)
    u_low_raw = channel_quantile(u_train_raw, cfg.u_low_quantile)
    u_high_raw = channel_quantile(u_train_raw, cfg.u_high_quantile)

    best_loss = float("inf")
    best_state = copy.deepcopy(controller.state_dict())

    for epoch in range(1, cfg.controller_epochs + 1):
        controller.train()
        ref_n, u_init_n = sample_closed_loop_batch(
            cfg.controller_batch_size,
            cfg.controller_horizon,
            ref_low_raw=ref_low_raw,
            ref_high_raw=ref_high_raw,
            u_low_raw=u_low_raw,
            u_high_raw=u_high_raw,
            normalizer=normalizer,
            device=device,
        )

        y_n, u_n, _ = rollout_closed_loop(
            plant,
            controller,
            ref_n,
            u_init_n,
            cfg.controller_warmup_steps,
            detach_state=False,
        )
        loss = closed_loop_tracking_loss(y_n, u_n, ref_n, cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.controller_grad_clip)
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(controller.state_dict())

        if epoch == 1 or epoch % cfg.controller_log_every == 0 or epoch == cfg.controller_epochs:
            with torch.no_grad():
                y_raw = normalizer.inverse_transform_y(y_n)
                ref_raw = normalizer.inverse_transform_y(ref_n)
                track_rmse = raw_rmse(
                    y_raw[:, -max(10, cfg.controller_horizon // 2) :, :],
                    ref_raw[:, -max(10, cfg.controller_horizon // 2) :, :],
                )
            print(
                f"[ctrl ] epoch {epoch:4d}/{cfg.controller_epochs} "
                f"| loss={loss_value:.4e} | tail_rmse={track_rmse:.4e}"
            )

    controller.load_state_dict(best_state)
    return loss_history


@torch.no_grad()
def evaluate_closed_loop(
    plant: DeepSSM,
    controller: ClampedSSMController,
    normalizer: ChannelwiseStandardizer,
    u_train_raw: torch.Tensor,
    y_train_raw: torch.Tensor,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_levels = torch.linspace(
        channel_quantile(y_train_raw, cfg.ref_low_quantile).item(),
        channel_quantile(y_train_raw, cfg.ref_high_quantile).item(),
        cfg.eval_cases,
    )
    u_levels = torch.linspace(
        channel_quantile(u_train_raw, cfg.u_low_quantile).item(),
        channel_quantile(u_train_raw, cfg.u_high_quantile).item(),
        cfg.eval_cases,
    ).flip(0)

    ref_raw = ref_levels.view(-1, 1, 1).expand(-1, cfg.eval_horizon, 1).to(device)
    ref_n = normalizer.transform_y(ref_raw)
    u_init_raw = u_levels.view(-1, 1).to(device)
    u_init_n = normalizer.transform_u(u_init_raw)

    y_n, u_n, _ = rollout_closed_loop(
        plant,
        controller,
        ref_n,
        u_init_n,
        cfg.controller_warmup_steps,
        detach_state=True,
    )
    return (
        normalizer.inverse_transform_y(ref_n),
        normalizer.inverse_transform_y(y_n),
        normalizer.inverse_transform_u(u_n),
    )


def plot_results(
    plant_history: dict[str, list[float] | float],
    controller_history: list[float],
    ref_raw: torch.Tensor,
    y_raw: torch.Tensor,
    u_raw: torch.Tensor,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

    axes[0].plot(plant_history["train_loss"], label="plant train")
    axes[0].plot(plant_history["val_loss"], label="plant val")
    axes[0].set_yscale("log")
    axes[0].set_title("Open-loop identification losses")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(controller_history)
    axes[1].set_yscale("log")
    axes[1].set_title("Closed-loop controller loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].grid(True)

    t = np.arange(y_raw.shape[1])
    for case_idx in range(y_raw.shape[0]):
        axes[2].plot(t, y_raw[case_idx, :, 0].cpu().numpy(), label=f"y case {case_idx + 1}")
        axes[2].plot(
            t,
            ref_raw[case_idx, :, 0].cpu().numpy(),
            linestyle="--",
            alpha=0.8,
            label=f"r case {case_idx + 1}",
        )
    axes[2].set_title("Closed-loop level tracking")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("level")
    axes[2].grid(True)
    axes[2].legend(ncol=2)

    fig.tight_layout()

    plt.figure(figsize=(9, 4))
    for case_idx in range(u_raw.shape[0]):
        plt.plot(u_raw[case_idx, :, 0].cpu().numpy(), label=f"u case {case_idx + 1}")
    plt.title("Closed-loop control actions")
    plt.xlabel("k")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    cfg = ExperimentConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    u_train_raw, y_train_raw, u_val_raw, y_val_raw, init_window, data_source = load_tank_data(cfg)
    print(f"Loaded data from: {data_source}")
    print(f"train input shape:  {tuple(u_train_raw.shape)}")
    print(f"train output shape: {tuple(y_train_raw.shape)}")
    print(f"val input shape:    {tuple(u_val_raw.shape)}")
    print(f"val output shape:   {tuple(y_val_raw.shape)}")
    print(f"init window:        {init_window}")

    if not cfg.normalize_data:
        raise RuntimeError("This demo currently assumes normalized signals for gain tuning and controller training.")

    normalizer = ChannelwiseStandardizer.fit(u_train_raw, y_train_raw)

    plant = build_certified_ssm(
        d_input=u_train_raw.shape[-1],
        d_output=y_train_raw.shape[-1],
        d_model=cfg.plant_d_model,
        d_state=cfg.plant_d_state,
        n_layers=cfg.plant_n_layers,
        d_hidden=cfg.plant_d_hidden,
        nl_layers=cfg.plant_nl_layers,
        gamma=cfg.plant_gamma,
        learn_x0=True,
    ).to(device)

    plant_history = train_identified_plant(
        plant,
        u_train_raw,
        y_train_raw,
        u_val_raw,
        y_val_raw,
        init_window,
        normalizer,
        cfg,
        device,
    )

    plant.eval()
    with torch.no_grad():
        y_val_pred_n, _ = plant(normalizer.transform_u(u_val_raw).to(device), mode="scan")
        y_val_pred_raw = normalizer.inverse_transform_y(y_val_pred_n).cpu()
    plant_val_rmse = raw_rmse(y_val_raw[:, init_window:, :], y_val_pred_raw[:, init_window:, :])

    freeze_module(plant)
    plant_gain = certified_gain(plant)

    controller_gamma = cfg.small_gain_margin / max(plant_gain, 1e-8)
    u_clip = channel_quantile(normalizer.transform_u(u_train_raw), cfg.u_clip_quantile).abs().clamp_min(1.0)
    controller_ssm = build_certified_ssm(
        d_input=y_train_raw.shape[-1],
        d_output=u_train_raw.shape[-1],
        d_model=cfg.controller_d_model,
        d_state=cfg.controller_d_state,
        n_layers=cfg.controller_n_layers,
        d_hidden=cfg.controller_d_hidden,
        nl_layers=cfg.controller_nl_layers,
        gamma=controller_gamma,
        learn_x0=False,
    ).to(device)
    controller = ClampedSSMController(controller_ssm, u_min=-u_clip.to(device), u_max=u_clip.to(device)).to(device)

    small_gain_product = plant_gain * controller.certificate_gamma
    if not small_gain_product < 1.0:
        raise RuntimeError(
            "The configured certificates do not satisfy the small-gain condition. "
            f"Got ||G_p||*||G_c|| = {small_gain_product:.4f}."
        )
    print(f"identified plant RMSE (raw, post-warmup): {plant_val_rmse:.4e}")
    print(f"plant certificate gamma:                  {plant_gain:.4f}")
    print(f"controller certificate gamma:             {controller.certificate_gamma:.4f}")
    print(f"small-gain product:                       {small_gain_product:.4f}")

    controller_history = train_controller(
        plant,
        controller,
        normalizer,
        u_train_raw,
        y_train_raw,
        cfg,
        device,
    )

    ref_raw, y_raw, u_raw = evaluate_closed_loop(
        
        plant,
        controller,
        normalizer,
        u_train_raw,
        y_train_raw,
        cfg,
        device,
    )
    tail_start = cfg.eval_horizon // 2
    eval_tail_rmse = raw_rmse(y_raw[:, tail_start:, :], ref_raw[:, tail_start:, :])
    print(f"closed-loop tail RMSE (raw):             {eval_tail_rmse:.4e}")

    if cfg.show_plots:
        plot_results(plant_history, controller_history, ref_raw, y_raw, u_raw)


if __name__ == "__main__":
    main()

# train_optimal_feedback.py
import os
from typing import Callable, Optional, Tuple, Dict, Any
import math
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------
# Generic interfaces / helpers
# ---------------------------

class DynamicsModule(nn.Module):
    """
    Base class for a discrete-time dynamics module.
    Users should implement forward(self, x, u) -> x_next.
    - x: (batch, state_dim)
    - u: (batch, control_dim)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement forward(x,u) -> x_next")


class ControllerModule(nn.Module):
    """
    Base class for a feedback controller module.
    Users should implement forward(self, x) -> u
    - x: (batch, state_dim)
    Returns:
    - u: (batch, control_dim)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Implement forward(x) -> u")


def simulate_trajectory(
    dynamics: DynamicsModule,
    controller: ControllerModule,
    x0: torch.Tensor,
    horizon: int,
    device: torch.device,
    open_loop_u: Optional[torch.Tensor] = None,
    return_all: bool = True,
    detach_every: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate batched trajectories with gradient flow through both controller and dynamics.
    - x0: (batch, state_dim)
    - horizon: int
    - open_loop_u: optional (horizon, batch, control_dim) tensor for forced inputs (overrides controller)
    Returns:
    - states: (horizon+1, batch, state_dim)
    - actions: (horizon, batch, control_dim)
    """
    batch = x0.shape[0]
    device = x0.device
    state_dim = x0.shape[1]

    xs = [x0]
    us = []

    x = x0
    for t in range(horizon):
        if open_loop_u is not None:
            u = open_loop_u[t]
        else:
            u = controller(x)  # (batch, control_dim)

        x_next = dynamics(x, u)

        # Optionally detach to limit backprop through time
        if detach_every is not None and (t + 1) % detach_every == 0:
            x_next = x_next.detach()  # cut gradient but keep parameters differentiable earlier
            x = x_next.clone().requires_grad_(True)
        else:
            x = x_next

        us.append(u)
        xs.append(x)

    states = torch.stack(xs, dim=0)  # (T+1, batch, state_dim)
    actions = torch.stack(us, dim=0)  # (T, batch, control_dim)
    if return_all:
        return states, actions
    else:
        return states[-1], actions


# ---------------------------
# Trainer
# ---------------------------

class MBTrainer:
    """
    Model-Based Trainer for differentiable dynamics + controller.
    - dynamics: DynamicsModule
    - controller: ControllerModule (to be optimized)
    - criterion: Callable(states, actions, targets, info) -> scalar loss
    """

    def __init__(
        self,
        dynamics: DynamicsModule,
        controller: ControllerModule,
        criterion: Callable[..., torch.Tensor],
        controller_optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        val_criterion: Optional[Callable[..., torch.Tensor]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.dynamics = dynamics.to(device) if device is not None else dynamics
        self.controller = controller.to(device) if device is not None else controller
        self.device = device or torch.device("cpu")
        self.criterion = criterion
        self.val_criterion = val_criterion or criterion
        self.opt = controller_optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        default_ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
        self.ckpt_dir = str(default_ckpt_dir if checkpoint_dir is None else checkpoint_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(
        self,
        train_init_sampler: Callable[[int], torch.Tensor],
        horizon: int,
        n_epochs: int,
        batch_size: int = 64,
        val_init_sampler: Optional[Callable[[int], torch.Tensor]] = None,
        val_every: int = 1,
        verbose: bool = True,
        save_best: bool = True,
        max_grad_norm: Optional[float] = None,
        detach_every: Optional[int] = None,
        **criterion_kwargs,
    ) -> Dict[str, Any]:
        best_val = float("inf")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            # Training step(s) per epoch - here we do one batch per epoch for simplicity,
            # but you can loop multiple batches per epoch externally.
            self.controller.train()
            x0 = train_init_sampler(batch_size).to(self.device)
            states, actions = simulate_trajectory(
                self.dynamics, self.controller, x0, horizon, self.device, detach_every=detach_every
            )
            loss = self.criterion(states, actions, **criterion_kwargs)
            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.grad_clip)
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            train_loss = float(loss.detach().cpu())
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = None
            if (val_init_sampler is not None) and (epoch % val_every == 0):
                self.controller.eval()
                with torch.no_grad():
                    x0_val = val_init_sampler(batch_size).to(self.device)
                    val_states, val_actions = simulate_trajectory(
                        self.dynamics, self.controller, x0_val, horizon, self.device, return_all=True
                    )
                    val_loss_tensor = self.val_criterion(val_states, val_actions, **criterion_kwargs)
                    val_loss = float(val_loss_tensor.cpu())
                    history["val_loss"].append(val_loss)

                if verbose:
                    print(
                        f"Epoch {epoch}/{n_epochs}  train_loss={train_loss:.6e}  val_loss={val_loss:.6e}  dt={time.time()-t0:.2f}s"
                    )

                if save_best and val_loss < best_val:
                    best_val = val_loss
                    self._save_checkpoint("best.pth")
            else:
                if verbose:
                    print(f"Epoch {epoch}/{n_epochs}  train_loss={train_loss:.6e}  dt={time.time()-t0:.2f}s")

        return {"history": history, "best_val": best_val}

    def _save_checkpoint(self, name: str):
        path = os.path.join(self.ckpt_dir, name)
        state = {
            "controller_state_dict": self.controller.state_dict(),
            "dynamics_state_dict": self.dynamics.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
        }
        torch.save(state, path)
        print(f"[Trainer] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.controller.load_state_dict(state["controller_state_dict"])
        self.dynamics.load_state_dict(state["dynamics_state_dict"])
        self.opt.load_state_dict(state["optimizer_state_dict"])


# ---------------------------
# Example: Linear dynamics & MLP controller
# ---------------------------

class LinearDynamics(DynamicsModule):
    """
    Discrete-time linear dynamics: x_{t+1} = A x_t + B u_t + w_t (optional)
    A, B are torch parameters or buffers; we implement them as buffers to allow learning if desired.
    """
    def __init__(self, A: torch.Tensor, B: torch.Tensor, process_noise_std: float = 0.0):
        super().__init__()
        # store as buffers (not learnable by default)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.process_noise_std = process_noise_std

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_next = x @ self.A.T + u @ self.B.T
        if self.process_noise_std > 0.0 and self.training:
            x_next = x_next + torch.randn_like(x_next) * self.process_noise_std
        return x_next


class MLPController(ControllerModule):
    """
    Simple MLP feedback controller: u = scale * tanh( MLP(x) )
    """
    def __init__(self, state_dim: int, control_dim: int, hidden: Tuple[int, ...] = (64, 64), scale: float = 1.0):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, control_dim))
        self.net = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(x))


# ---------------------------
# Example criterion (generic)
# ---------------------------

def quadratic_tracking_loss(states: torch.Tensor, actions: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Generic quadratic loss over trajectory.
    - states: (T+1, batch, state_dim)
    - actions: (T, batch, control_dim)
    - Q: (state_dim, state_dim) positive semidef
    - R: (control_dim, control_dim) positive semidef
    - target: optional (T+1, batch, state_dim) or (state_dim,) or None (defaults to zeros)
    Returns scalar loss (mean over batch).
    """
    T_plus_1, batch, state_dim = states.shape
    T = T_plus_1 - 1
    device = states.device

    if target is None:
        target = torch.zeros((T_plus_1, batch, state_dim), device=device)
    elif target.dim() == 1:
        target = target.view(1, 1, -1).expand(T_plus_1, batch, -1)
    elif target.dim() == 2:  # (state_dim, ) or (batch, state_dim)
        target = target.view(1, batch, -1).expand(T_plus_1, batch, -1)

    state_err = states - target  # (T+1, batch, state_dim)
    # state cost (exclude initial or not? we include all steps here)
    Q = Q.to(device)
    R = R.to(device)

    # correct matmul order
    state_cost = (state_err.unsqueeze(-2) @ Q @ state_err.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    state_cost = state_cost.sum(dim=0)  # (batch,)
    action_cost = (actions.unsqueeze(-1) @ R @ actions.unsqueeze(-2)).squeeze(-1).squeeze(-1).sum(dim=0)  # (batch,)
    total = state_cost + action_cost
    return total.mean()


# ---------------------------
# Demo training run
# ---------------------------

def demo():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Problem dimensions
    state_dim = 2
    control_dim = 1
    horizon = 50

    # Define dynamics (stable linear system)
    A = torch.tensor([[0.95, 0.02], [-0.03, 0.9]], dtype=torch.float32)
    B = torch.tensor([[0.1], [0.05]], dtype=torch.float32)
    dynamics = LinearDynamics(A, B, process_noise_std=0.0).to(device)

    # Controller
    controller = MLPController(state_dim, control_dim, hidden=(64, 64), scale=1.0).to(device)

    # Criterion hyperparams
    Q = torch.eye(state_dim) * 1.0
    R = torch.eye(control_dim) * 0.1
    target = torch.zeros(state_dim)  # regulate to zero

    # criterion function wrapper matching trainer signature
    def criterion_wrapper(states, actions, **kwargs):
        return quadratic_tracking_loss(states, actions, Q=Q, R=R, target=target)

    # initial state sampler for training/validation
    def train_init_sampler(batch_size: int) -> torch.Tensor:
        # sample initial states uniformly in a box
        return (torch.rand((batch_size, state_dim)) - 0.5) * 4.0  # in [-2,2]

    def val_init_sampler(batch_size: int) -> torch.Tensor:
        return (torch.rand((batch_size, state_dim)) - 0.5) * 4.0

    # optimizer
    opt = optim.Adam(controller.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

    trainer = MBTrainer(
        dynamics=dynamics,
        controller=controller,
        criterion=criterion_wrapper,
        controller_optimizer=opt,
        device=device,
        val_criterion=criterion_wrapper,
        scheduler=scheduler,
        grad_clip=1.0,
        checkpoint_dir=str(Path(__file__).resolve().parent / "checkpoints_demo"),
    )

    results = trainer.train(
        train_init_sampler=train_init_sampler,
        val_init_sampler=val_init_sampler,
        horizon=horizon,
        n_epochs=600,
        batch_size=128,
        val_every=10,
        verbose=True,
        save_best=True,
        detach_every=None,  # set to an integer to truncate BPTT
    )

    print("Training finished. Best val:", results["best_val"])

    # Quick evaluation of a trajectory
    controller.eval()
    with torch.no_grad():
        x0 = torch.tensor([[1.5, -1.0]], device=device)  # single initial state
        states, actions = simulate_trajectory(dynamics, controller, x0, horizon, device)
        states = states.cpu().numpy()
        actions = actions.cpu().numpy()
        print("Initial state:", x0.cpu().numpy())
        print("First 5 states:\n", states[:6, 0, :])
        print("First 5 actions:\n", actions[:5, 0, :])


if __name__ == "__main__":
    demo()

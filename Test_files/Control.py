# closed_loop_fixed_point_with_hinf_and_plots.py
# ------------------------------------------------------------
# - DT LTI plant with D != 0 (algebraic loop possible)
# - Plant is dense but built as A = Q diag(lam) Q^T so we can
#   scan the diagonal recurrence in modal coords
# - Nonlinear SSM controller with signature: u, x_last = model(y, mode='scan')
# - Whole-horizon closed-loop solved by damped fixed-point iterations
# - Computes plant l2-gain (= H-infinity norm) using python-control
# - Produces nice plots at the end
#
# Requirements:
#   pip install control matplotlib
#   (optional for accurate hinfnorm: pip install slycot)
# ------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.neural_ssm.ssm.lru import DeepSSM, SSMConfig
from src.neural_ssm.ssm.scan_utils import binary_operator_diag, associative_scan

import control  # python-control

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    n_u: int = 1
    n_y: int = 1
    d_model: int = 16
    d_state: int = 11
    n_layers: int = 1
    ff: str = "LMLP"  # GLU | MLP | LMLP
    max_phase: float = math.pi / 60
    r_min: float = 0.7
    r_max: float = 0.98
    d_amp: int = 8
    param: str = 'l2ru'
    d_hidden: int = 8
    nl_layers: int = 3
    gamma: Optional[float] = 2
    init: str = 'rand'
    rho: float = 0.9
    max_phase_b: float = 0.5  # small spread
    phase_center: float = 0  # center angle ≈ 17°
    random_phase = True

    def to_ssm_config(self) -> SSMConfig:
        """Convert to SSMConfig object."""
        return SSMConfig(
            d_model=self.d_model,
            d_state=self.d_state,
            n_layers=self.n_layers,
            ff=self.ff,
            rmin=self.r_min,
            rmax=self.r_max,
            max_phase=self.max_phase,
            d_hidden = self.d_hidden,
            nl_layers = self.nl_layers,
            dim_amp=self.d_amp,
            param=self.param,
            gamma=self.gamma,
            init=self.init,
            rho = self.rho,
            max_phase_b = self.max_phase_b,
            phase_center = self.phase_center,
            random_phase=self.random_phase,
        )

# ============================================================
# 0) YOUR diagonal scan hook
#    Replace diag_recurrence_scan(...) with your optimized scan.
# ============================================================
def diag_recurrence_scan(
        lambdas: torch.Tensor,  # (N,)
        Bu: torch.Tensor,  # (B, L, N) = B u_t already
        x0: torch.Tensor,  # (B, N)
) -> torch.Tensor:
    """
    Diagonal linear recurrence via parallel scan:

        x_{t+1} = lambdas * x_t + Bu[:, t]

    Args:
        lambdas: (N,)
        Bu:      (B, L, N)  precomputed B @ u_t
        x0:      (B, N)     initial state x_0

    Returns:
        states:  (B, L+1, N) with
                 states[:, 0]   = x_0
                 states[:, t+1] = x_{t+1} for t = 0..L-1
    """
    Bsz, L, N = Bu.shape
    Bu = Bu.clone()
    x0 = x0.squeeze(1)
    # fold x0 into the first step
    Bu[:, 0, :] += lambdas * x0

    lam_seq = lambdas.expand(L, -1)  # (L, N)

    def _scan_fn(bu_seq):
        # returns sequence x_1..x_L, shape (L, N)
        return associative_scan(binary_operator_diag, (lam_seq, bu_seq))[1]

    x_next = torch.vmap(_scan_fn)(Bu)  # (B, L, N): x_1..x_L

    # assemble full trajectory [x_0, ..., x_L]
    states = torch.empty(Bsz, L + 1, N, device=Bu.device, dtype=Bu.dtype)
    states[:, 0] = x0
    states[:, 1:] = x_next
    return states


# ============================================================
# 1) Dense diagonalizable DT plant with D != 0
# ============================================================

@dataclass
class PlantCfg:
    n_x: int = 16
    n_u: int = 2
    n_y: int = 2
    dt: float = 1.0
    lam_min: float = 0.90
    lam_max: float = 0.995
    b_scale: float = 0.25
    d_scale: float = 0.25  # keep moderate; too large can worsen well-posedness


class DiagonalizablePlant(nn.Module):
    """
    Construct a dense stable A with an orthogonal diagonalization:
        A = Q diag(lam) Q^T
    Modal coordinates z = Q^T x (row-form: z = x Q):
        z_{k+1} = lam ⊙ z_k + (Q^T B) u_k
        y_k     = (C Q) z_k + D u_k
    """
    def __init__(self, cfg: PlantCfg, device: str, dtype=torch.float32):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        M = torch.randn(cfg.n_x, cfg.n_x, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(M)
        lam = torch.empty(cfg.n_x, device=device, dtype=dtype).uniform_(cfg.lam_min, cfg.lam_max)

        A = Q @ torch.diag(lam) @ Q.t()
        B = cfg.b_scale * torch.randn(cfg.n_x, cfg.n_u, device=device, dtype=dtype)

        # output: read x[0], x[1] as "position"
        C = torch.zeros(cfg.n_y, cfg.n_x, device=device, dtype=dtype)
        C[0, 0] = 1.0
        C[1, 1] = 1.0

        D = cfg.d_scale * torch.randn(cfg.n_y, cfg.n_u, device=device, dtype=dtype)

        self.register_buffer("Q", Q)
        self.register_buffer("lam", lam)          # diag in modal coordinates
        self.register_buffer("A_dense", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        self.register_buffer("D", D)

        self.register_buffer("Bz", Q.t() @ B)     # (n_x,n_u)
        self.register_buffer("Cz", C @ Q)         # (n_y,n_x)

    def x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Q  # row-form

    def rollout_y_from_u(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x0: (B,n_x) original coords
        u:  (B,T,n_u)
        y:  (B,T,n_y)
        """
        z0 = self.x_to_z(x0)             # (B,n_x)
        v = u @ self.Bz.t()              # (B,T,n_x)
        z = diag_recurrence_scan(self.lam, v, z0)  # (B,T+1,n_x)
        y = (z[:, :-1] @ self.Cz.t()) + (u @ self.D.t())
        return y

    def as_control_statespace(self):
        """Return a python-control discrete-time StateSpace model for gain computation."""
        A = self.A_dense.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        C = self.C.detach().cpu().numpy()
        D = self.D.detach().cpu().numpy()
        return control.ss(A, B, C, D, self.cfg.dt)


# ============================================================
# 2) l2-gain of the plant via python-control (H∞ norm)
# ============================================================

def plant_l2_gain_control(sys: control.StateSpace, n_grid: int = 4000):
    """
    For stable DT LTI, induced l2 gain = H∞ norm.
    Tries:
      1) control.hinfnorm(sys) (accurate; may need slycot)
      2) control.norm(sys, 'inf')
      3) fallback: frequency grid + max singular value

    Returns: (gamma, w_peak, method_str)
      w_peak in rad/sample (for dt=1), or None if unknown
    """
    # 1) hinfnorm
    try:
        gamma, w_peak = control.hinfnorm(sys)
        return float(gamma), float(w_peak), "control.hinfnorm"
    except Exception:
        pass

    # 2) norm(..., 'inf')
    try:
        gamma = control.norm(sys, 'inf')
        return float(gamma), None, "control.norm(sys, 'inf')"
    except Exception:
        pass

    # 3) grid approximation
    dt = sys.dt if sys.dt is not None else 1.0
    w = np.linspace(0.0, math.pi / dt, n_grid)

    # try different APIs depending on control version
    try:
        resp = control.freqresp(sys, w)[0]  # (ny,nu,len(w))
    except Exception:
        fr = control.frequency_response(sys, w)
        resp = fr.fresp  # (ny,nu,len(w))

    sigmax = np.empty_like(w)
    for i in range(len(w)):
        G = resp[:, :, i]
        sigmax[i] = np.linalg.svd(G, compute_uv=False)[0]
    idx = int(np.argmax(sigmax))
    return float(sigmax[idx]), float(w[idx]), f"grid svd ({n_grid} pts)"


# ============================================================
# 3) Nonlinear objective: complicated “landscape navigation”
# ============================================================

@dataclass
class ObjCfg:
    # goal
    target: Tuple[float, float] = (1.5, -1.0)
    eps_goal: float = 0.15                 # "reach" radius (soft)
    softmin_beta: float = 25.0             # larger -> closer to true min over time

    # obstacles
    obstacles: Optional[List[Tuple[Tuple[float, float], float]]] = None
    softplus_temp: float = 0.25            # barrier smoothness

    # "stay there" (last L steps)
    stay_last_L: int = 30

    # weights
    w_obs: float = 16.0
    w_reach: float = 2.0
    w_stay: float = 2.0
    w_u2: float = 0.02


def objective(y: torch.Tensor, u: torch.Tensor, cfg: ObjCfg) -> torch.Tensor:
    """
    y: (B,T,2), u: (B,T,n_u)
    Keeps only:
      - obstacle avoidance (soft barrier)
      - reach target (eventually)
      - stay near target (last L steps)
      (+ optional control effort)
    """
    device, dtype = y.device, y.dtype
    B, T, _ = y.shape
    target = torch.tensor(cfg.target, device=device, dtype=dtype)

    # -------------------
    # obstacle avoidance
    # -------------------
    obs_pen = 0.0
    if cfg.obstacles is not None:
        for (cx, cy), r in cfg.obstacles:
            c = torch.tensor([cx, cy], device=device, dtype=dtype)
            d2 = ((y - c) ** 2).sum(dim=-1)  # (B,T)
            # penalty if inside: softplus(r^2 - d^2)
            obs_pen = obs_pen + F.softplus((r * r - d2) / cfg.softplus_temp).pow(2)  # (B,T)

    # -------------------
    # reach target (eventually)
    # -------------------
    d2_goal = ((y - target) ** 2).sum(dim=-1)  # (B,T)
    softmin = -(1.0 / cfg.softmin_beta) * torch.logsumexp(-cfg.softmin_beta * d2_goal, dim=1)  # (B,)
    reach_pen = F.softplus((softmin - cfg.eps_goal ** 2) / 0.02)  # (B,)

    # -------------------
    # stay near target at the end
    # -------------------
    L = min(cfg.stay_last_L, T)
    stay_pen = d2_goal[:, T - L :].mean(dim=1)  # (B,)

    # -------------------
    # control effort
    # -------------------
    u_pen = (u * u).sum(dim=-1).mean(dim=1)  # (B,)

    # total
    loss = (
        cfg.w_obs * obs_pen.mean()
        + cfg.w_reach * reach_pen.mean()
        + cfg.w_stay * stay_pen.mean()
        + cfg.w_u2 * u_pen.mean()
    )
    return loss


# ============================================================
# 4) Controller adapter: u, x_last = SSM(y, mode='scan')
# ============================================================

class ControllerWrapper(nn.Module):
    def __init__(self, ssm_model: nn.Module):
        super().__init__()
        self.ssm = ssm_model

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        u, _ = self.ssm(y, mode="scan")
        return u


# ============================================================
# 5) Closed-loop fixed point (trajectory-level)
# ============================================================

@dataclass
class FPcfg:
    n_iter: int = 10
    relax: float = 0.7


def solve_closed_loop(
    plant: DiagonalizablePlant,
    controller: nn.Module,
    x0: torch.Tensor,
    T: int,
    cfg: FPcfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed point on trajectories:
      y = G(u)
      u = Π(y)
    Damped Picard iteration:
      u <- (1-ω)u + ω Π(G(u))
    """
    B = x0.shape[0]
    n_u = plant.cfg.n_u
    u = torch.zeros(B, T, n_u, device=x0.device, dtype=x0.dtype)

    for _ in range(cfg.n_iter):
        y = plant.rollout_y_from_u(x0, u)
        u_hat = controller(y)
        u = (1.0 - cfg.relax) * u + cfg.relax * u_hat

    y = plant.rollout_y_from_u(x0, u)
    return y, u


# ============================================================
# 6) Build your DeepSSM controller (edit imports to match your repo)
# ============================================================

def build_deepssm_controller(n_y: int, n_u: int, device: str) -> nn.Module:
    """
    Controller is DeepSSM mapping y -> u.

    Edit the imports below to match your project structure.
    """


    # Your init pattern (adapted: controller input dim=n_y, output dim=n_u)
    model_config = ModelConfig(
        n_u=n_y, n_y=n_u, param="l2n",
        d_model=8, d_state=8, gamma=None,
        ff="GLU", init="eye",
        n_layers=7, d_amp=3, rho=0.9,
        phase_center=0.0, max_phase_b=0.04,
        d_hidden=12, nl_layers=3
    )
    ssm_config = model_config.to_ssm_config()
    model = DeepSSM(d_input=n_y, d_output=n_u, config=ssm_config).to(device)
    return model


# ============================================================
# 7) Plot helpers
# ============================================================

def plot_landscape_and_trajectory(y_traj: np.ndarray, obj_cfg: ObjCfg, title: str = ""):
    """
    Visualizes a 2D "cost field" that matches the updated objective:
      - obstacle barrier (softplus)
      - goal well (quadratic), plus a ring at eps_goal
    and overlays the trajectory.

    y_traj: (T,2)
    """
    # grid
    lo, hi = -3.2, 3.2
    gx = np.linspace(lo, hi, 240)
    gy = np.linspace(lo, hi, 240)
    X, Y = np.meshgrid(gx, gy)

    tx, ty = obj_cfg.target
    d2_goal = (X - tx) ** 2 + (Y - ty) ** 2

    # "goal well": quadratic distance (for plotting only; training uses temporal reach+stay)
    V = obj_cfg.w_stay * d2_goal

    # obstacle barriers (match objective's softplus barrier shape)
    if obj_cfg.obstacles is not None:
        for (cx, cy), r in obj_cfg.obstacles:
            d2 = (X - cx) ** 2 + (Y - cy) ** 2
            V = V + obj_cfg.w_obs * (np.log1p(np.exp((r * r - d2) / obj_cfg.softplus_temp)) ** 2)

    # plot
    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, V, levels=60)
    plt.colorbar()

    # trajectory
    plt.plot(y_traj[:, 0], y_traj[:, 1], marker="o", markersize=2, linewidth=1)

    ax = plt.gca()

    # obstacles (hard boundary)
    if obj_cfg.obstacles is not None:
        for (cx, cy), r in obj_cfg.obstacles:
            ax.add_patch(plt.Circle((cx, cy), r, fill=False))

    # target marker + reach radius
    ax.scatter([tx], [ty], marker="x")
    ax.add_patch(plt.Circle((tx, ty), obj_cfg.eps_goal, fill=False, linestyle="--"))

    # highlight "stay window" points (last L steps)
    T = y_traj.shape[0]
    L = min(getattr(obj_cfg, "stay_last_L", 0), T)
    if L > 0:
        plt.scatter(y_traj[-L:, 0], y_traj[-L:, 1], s=10)

    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.title(title or "Obstacle avoidance + reach target + stay there")
    plt.axis("equal")
    plt.grid(True)


# ============================================================
# 8) Main: gain computation + training + plots
# ============================================================

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # ---- Plant ----
    plant_cfg = PlantCfg(n_x=16, n_u=2, n_y=2, dt=1.0, d_scale=0.25)
    plant = DiagonalizablePlant(plant_cfg, device=device, dtype=dtype)

    # Compute l2-gain (= H∞ norm) of the plant
    sys = plant.as_control_statespace()
    gamma, w_peak, method = plant_l2_gain_control(sys)
    print(f"Plant induced l2 gain (H∞): {gamma:.4f}  | method: {method}"
          + (f" | w_peak≈{w_peak:.4f} rad/sample" if w_peak is not None else ""))

    # ---- Controller (your DeepSSM) ----
    ssm=DeepSSM(d_input=plant_cfg.n_y, d_output=plant_cfg.n_u, param='l2n', gamma=1/(gamma+0.001), ff='LGLU', n_layers=2, d_model=8, d_state=8).to(device)
    controller = ControllerWrapper(ssm).to(device)

    # ---- Objective ----
    obj_cfg = ObjCfg(
        target=(1.5, -1.0),
        obstacles=[((0.0, 0.5), 0.6), ((1.0, -0.2), 0.5), ((-1.2, -1.2), 0.55)],
    )

    # ---- Fixed point solver cfg ----
    fp_cfg = FPcfg(n_iter=10, relax=0.7)

    # ---- Training ----
    B = 16
    T = 160
    steps = 1300
    opt = torch.optim.Adam(controller.parameters(), lr=2e-3)

    loss_hist = []

    for it in range(steps):
        # batch of initial states (no reference, no noise)
        x0 = torch.zeros(B, plant_cfg.n_x, device=device, dtype=dtype)
        x0[:, 0:2] = 3.0 * (2.0 * torch.rand(B, 2, device=device, dtype=dtype) - 1.0)

        y, u = solve_closed_loop(plant, controller, x0, T=T, cfg=fp_cfg)
        loss = objective(y, u, obj_cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        opt.step()

        loss_hist.append(float(loss.item()))
        if it % 50 == 0:
            print(f"iter {it:04d} | loss {loss.item():.4f}")

    # ---- Make plots with a single rollout ----
    with torch.no_grad():
        x0 = torch.zeros(1, plant_cfg.n_x, device=device, dtype=dtype)
        x0[:, 0:2] = torch.tensor([[2.5, 2.0]], device=device, dtype=dtype)
        y, u = solve_closed_loop(plant, controller, x0, T=T, cfg=FPcfg(n_iter=12, relax=0.7))

        y_np = y[0].cpu().numpy()
        u_np = u[0].cpu().numpy()

    # Loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(loss_hist)
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.grid(True)

    # Landscape + trajectory
    plot_landscape_and_trajectory(y_np, obj_cfg, title="Closed-loop trajectory on nonlinear landscape")

    # Signals over time
    t = np.arange(T)
    plt.figure(figsize=(7, 4))
    plt.plot(t, y_np[:, 0], label="y1")
    plt.plot(t, y_np[:, 1], label="y2")
    plt.xlabel("k")
    plt.ylabel("y")
    plt.title("Outputs over time")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(7, 4))
    plt.plot(t, u_np[:, 0], label="u1")
    plt.plot(t, u_np[:, 1], label="u2")
    plt.xlabel("k")
    plt.ylabel("u")
    plt.title("Inputs over time")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

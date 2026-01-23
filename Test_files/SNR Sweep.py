import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, Dict, List
from src.neural_ssm.ssm.lru import DeepSSM, SSMConfig
import math
from argparse import Namespace

seed = 79
torch.manual_seed(seed)
np.random.seed(seed)


# ---------------------------
# Torch Simulate Wrapper (Modified for Torch inputs/gradients)
# ---------------------------
def make_torch_simulate_fn(model: torch.nn.Module, device: torch.device = torch.device('cpu'), mode: str = "scan") -> \
Callable[[torch.Tensor], torch.Tensor]:
    """ Returns a simulate_fn(u_torch) that takes/returns torch tensors, preserves gradients.
    Assumes model.forward accepts input shape (B, L, H) and returns (outputs, states). """
    model = model.to(device)
    model.eval()  # Eval mode, but gradients can still flow

    def simulate_fn(u_torch: torch.Tensor) -> torch.Tensor:
        # u_torch: (T, m) -> add batch (1, T, m)
        if u_torch.ndim == 2:
            u_torch = u_torch.unsqueeze(0)  # (1, T, m)
        out, _states = model.forward(u_torch, mode=mode)  # (1, T, output_dim)
        return out.squeeze(0)

    return simulate_fn


# ---------------------------
# Adversarial Noise Attack for L2 Gain Estimation
# ---------------------------
def adversarial_noise_attack(
        simulate_fn: Callable[[torch.Tensor], torch.Tensor],
        u_clean: torch.Tensor,
        epsilon: float,
        steps: int = 100,
        lr: float = 0.01,
        proj_norm: str = 'l2',  # 'l2' or 'linf'
        device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """ Crafts adversarial noise with fixed budget epsilon to maximize ||y_noisy - y_clean|| / epsilon.
    This approximates the local L2 gain around u_clean. """
    with torch.no_grad():
        y_clean = simulate_fn(u_clean.to(device))

    # Init noise: small random
    noise = torch.randn_like(u_clean) * (epsilon / 10)
    noise = noise.to(device)
    noise.requires_grad_(True)

    for step in range(steps):
        u_noisy = u_clean + noise
        y_noisy = simulate_fn(u_noisy)
        loss = torch.norm(y_noisy - y_clean, p=2)
        loss.backward()

        grad = noise.grad.detach()

        if proj_norm == 'l2':
            grad_norm = torch.norm(grad) + 1e-10
            step_dir = grad / grad_norm
        elif proj_norm == 'linf':
            step_dir = grad.sign()
        else:
            raise ValueError("Unsupported proj_norm")

        noise.data += lr * step_dir

        # Project back to norm ball
        if proj_norm == 'l2':
            norm = torch.norm(noise)
            if norm > epsilon:
                noise.data *= (epsilon / norm)
        elif proj_norm == 'linf':
            noise.data = torch.clamp(noise.data, -epsilon, epsilon)

        noise.grad.zero_()

    # Final metrics
    with torch.no_grad():
        u_noisy = u_clean + noise
        y_noisy = simulate_fn(u_noisy)
        error = y_noisy - y_clean
        adv_gain = torch.norm(error, p=2).item() / epsilon if epsilon > 0 else 0
    return {
        "adv_gain": adv_gain,
        "final_noise": noise.detach().cpu().numpy(),
        "final_error": error.detach().cpu().numpy(),
        "final_loss": torch.norm(error, p=2).item()
    }


def estimate_l2_gain(
        simulate_fns: List[Callable],
        labels: List[str],
        u_clean: torch.Tensor,
        epsilons: List[float],
        trials: int = 10,  # More trials for better supremum estimate
        steps: int = 200,  # More steps for better optimization
        lr: float = 0.005,
        proj_norm: str = 'l2'
):
    """ Estimates the L2 gain by finding the max ||error|| / ||noise|| over adversarial attacks at various epsilon.
    Returns mean/std/max gains per epsilon, and overall estimated L2 gain (max over all). """
    results = {label: {"mean_gains": [], "std_gains": [], "max_gains": [], "overall_max_gain": 0.0} for label in labels}
    device = torch.device('cpu')  # Change to 'cuda' if available

    for eps in epsilons:
        for i, simulate_fn in enumerate(simulate_fns):
            label = labels[i]
            gains = []
            for _ in range(trials):
                res = adversarial_noise_attack(simulate_fn, u_clean, eps, steps, lr, proj_norm, device)
                gains.append(res["adv_gain"])
                results[label]["overall_max_gain"] = max(results[label]["overall_max_gain"], res["adv_gain"])
            results[label]["mean_gains"].append(np.mean(gains))
            results[label]["std_gains"].append(np.std(gains))
            results[label]["max_gains"].append(np.max(gains))

    # Convert to arrays
    for label in labels:
        for k in ["mean_gains", "std_gains", "max_gains"]:
            results[label][k] = np.array(results[label][k])

    return results


def plot_l2_gain_estimation(results: Dict[str, Dict], epsilons: np.ndarray, gamma: float = None,
                            title: str = "Estimated L2 Gain via Adversarial Noise"):
    """ Plots mean/std adversarial gains vs. epsilon for multiple models. """
    plt.figure(figsize=(10, 6))
    for label, res in results.items():
        mean = res["mean_gains"]
        std = res["std_gains"]
        plt.plot(epsilons, mean, '-o', label=f'Mean Est. Gain - {label}')
        plt.fill_between(epsilons, mean - std, mean + std, alpha=0.25)
    if gamma is not None:
        plt.axhline(gamma, color='r', ls='--', label='Prescribed Gamma')
    plt.xlabel("Noise Budget (Epsilon)")
    plt.ylabel("Estimated L2 Gain ||error|| / ||noise||")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()


# ---------------------------
# Main Execution: L2 Gain Estimation
# ---------------------------
if __name__ == "__main__":
    # Build clean input: T x m
    T = 256
    m = 3
    t = np.arange(T)
    u_clean_np = np.zeros((T, m), dtype=np.float64)
    u_clean_np[:, 0] = np.sin(2 * np.pi * 5 * t / T)  # Sinusoidal
    u_clean = torch.tensor(u_clean_np, dtype=torch.float32)  # Torch version

    # Configs
    cfg_robust = {
        "n_u": 3,
        "n_y": 3,
        "d_model": 4,
        "d_state": 12,
        "n_layers": 1,
        "ff": "GLU",  # GLU | MLP | LMLP
        "max_phase": math.pi / 50,
        "r_min": 0.7,
        "r_max": 0.98,
        "param": 'lru',
        "gamma": 2,
        "init": 'eye'
    }
    cfg_robust = Namespace(**cfg_robust)

    cfg_vanilla = {
        "n_u": 3,
        "n_y": 3,
        "d_model": 5,
        "d_state": 6,
        "n_layers": 1,
        "ff": "LGLU",  # GLU | MLP | LMLP
        "max_phase": math.pi / 50,
        "r_min": 0.7,
        "r_max": 0.98,
        "param": 'l2ru',
        "gamma": 2,
        "init": 'eye'
    }
    cfg_vanilla = Namespace(**cfg_vanilla)

    # Build models
    config_robust = SSMConfig(d_model=cfg_robust.d_model, d_state=cfg_robust.d_state, n_layers=cfg_robust.n_layers,
                              ff="GLU",
                              rmin=cfg_robust.r_min, rmax=cfg_robust.r_max, max_phase=cfg_robust.max_phase,
                              param=cfg_robust.param, gamma=cfg_robust.gamma, init=cfg_robust.init)
    model_robust = DeepSSM(d_input=cfg_robust.n_u, d_output= cfg_robust.n_y, config=config_robust)

    config_vanilla = SSMConfig(d_model=cfg_vanilla.d_model, d_state=cfg_vanilla.d_state, n_layers=cfg_vanilla.n_layers,
                               ff=cfg_vanilla.ff,
                               rmin=cfg_vanilla.r_min, rmax=cfg_vanilla.r_max, max_phase=cfg_vanilla.max_phase,
                               param=cfg_vanilla.param, gamma=cfg_vanilla.gamma)
    model_vanilla = DeepSSM(d_input=cfg_vanilla.n_u, d_output=cfg_vanilla.n_y, config=config_vanilla)

    u = torch.randn(5, 2, 3)
    a, b = model_robust(u = u, mode = 'loop')
    a2, b2 = model_robust(u=u, mode='scan')

    # Simulate functions (torch in/out for grads)
    simulate_fn_robust = make_torch_simulate_fn(model_robust, device=torch.device('cpu'), mode='scan')
    simulate_fn_vanilla = make_torch_simulate_fn(model_vanilla, device=torch.device('cpu'), mode='scan')

    # Choose epsilons: small to large, scaled to signal norm for relevance
    u_norm = torch.norm(u_clean).item()
    epsilons = np.logspace(-3, 0, num=10) * u_norm  # From 0.001*u_norm to u_norm

    l2_results = estimate_l2_gain(
        [simulate_fn_robust, simulate_fn_vanilla],
        ["Robust SSM", "Vanilla SSM"],
        u_clean,
        epsilons,
        trials=10,
        steps=200,
        lr=0.005,
        proj_norm='l2'
    )

    print("\nEstimated L2 Gains:")
    for label, res in l2_results.items():
        print(f"{label} overall estimated L2 gain (max over trials/eps): {res['overall_max_gain']:.2f}")
        print(f"{label} mean gains per epsilon: {res['mean_gains']}")
        print(f"{label} max gains per epsilon: {res['max_gains']}")

    plot_l2_gain_estimation(l2_results, epsilons, gamma=cfg_robust.gamma)

    SSM_sim

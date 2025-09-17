import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict, List
from SSM.ssm import DeepSSM, SSMConfig  # Assuming this is your module; adjust if needed
import math
from argparse import Namespace

seed = 79
torch.manual_seed(seed)
np.random.seed(seed)


# ---------------------------
# Original Functions (SNR Sweep, etc.) - Kept for completeness
# ---------------------------

def add_noise_for_snr(u: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to achieve target input SNR."""
    sig_power = np.mean(u ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = rng.normal(scale=np.sqrt(noise_power), size=u.shape)
    return u + noise, noise


def snr_sweep(simulate_fn, u_clean, snr_db_list, trials=5, seed=42, noisy_inputs=None):
    rng = np.random.default_rng(seed)
    y_clean = simulate_fn(u_clean)
    results = {
        "snr_db": snr_db_list,
        "out_snr_mean": [], "out_snr_std": [],
        "amp_mean": [], "amp_std": [],
        "noise_amp_mean": [], "noise_amp_std": [],  # New: noise amplification
        "mse_mean": [], "mse_std": [],  # New: MSE for error vs SNR
        "worst_trials": {},
        "y_clean": y_clean,
        "u_clean": u_clean,
        "empirical_max_gain": 0.0  # Track max noise amp across all
    }
    for snr_idx, snr in enumerate(snr_db_list):
        out_snrs, amps, noise_amps, mses = [], [], [], []
        worst_err = -np.inf
        worst_trial_data = None
        for t in range(trials):
            if noisy_inputs is None:
                u_noisy, noise = add_noise_for_snr(u_clean, snr, rng)
            else:
                u_noisy = noisy_inputs[snr_idx][t]
                noise = u_noisy - u_clean  # Compute noise if not generated here
            y = simulate_fn(u_noisy)
            error = y - y_clean
            noise_power = np.mean(error ** 2)
            sig_power = np.mean(y_clean ** 2)
            out_snr = 10 * np.log10(sig_power / noise_power) if noise_power > 0 else np.inf
            amp = np.linalg.norm(y) / np.linalg.norm(u_noisy)
            noise_amp = np.linalg.norm(error) / np.linalg.norm(noise) if np.linalg.norm(noise) > 0 else 0
            mse = np.mean(error ** 2)  # MSE for error degradation
            out_snrs.append(out_snr)
            amps.append(amp)
            noise_amps.append(noise_amp)
            mses.append(mse)
            # Update empirical max gain
            results["empirical_max_gain"] = max(results["empirical_max_gain"], noise_amp)
            # Track worst-case trial (based on error norm)
            err_norm = np.linalg.norm(error)
            if err_norm > worst_err:
                worst_err = err_norm
                worst_trial_data = {"u_noisy": u_noisy, "y_noisy": y, "noise": noise}
        # Store stats
        results["out_snr_mean"].append(np.mean(out_snrs))
        results["out_snr_std"].append(np.std(out_snrs))
        results["amp_mean"].append(np.mean(amps))
        results["amp_std"].append(np.std(amps))
        results["noise_amp_mean"].append(np.mean(noise_amps))
        results["noise_amp_std"].append(np.std(noise_amps))
        results["mse_mean"].append(np.mean(mses))
        results["mse_std"].append(np.std(mses))
        results["worst_trials"][snr] = worst_trial_data
    # Convert lists to numpy
    for k in results.keys():
        if isinstance(results[k], list):
            results[k] = np.array(results[k])
    return results


def plot_snr_sweep(results_list: List[Dict[str, np.ndarray]], labels: List[str], title: str = "SNR Sweep Comparison",
                   snr_focus: float = 0.0, channel: int = 0, plot_mse: bool = False):
    """Plots sweep + worst-case trials for multiple models."""
    if not results_list or not labels or len(results_list) != len(labels):
        raise ValueError("Results and labels must match in length.")
    s = results_list[0]["snr_db"]  # Assume same SNRs
    num_subplots = 4 if plot_mse else 3  # Extra for MSE if requested
    plt.figure(figsize=(12, 10 + (2 if plot_mse else 0)))

    # Output SNR plot
    plt.subplot(num_subplots, 1, 1)
    for i, results in enumerate(results_list):
        mean = results["out_snr_mean"]
        std = results["out_snr_std"]
        plt.plot(s, mean, '-o', label=f'Output SNR (mean) - {labels[i]}')
        plt.fill_between(s, mean - std, mean + std, alpha=0.25)
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Output SNR (dB)")
    plt.grid(True)
    plt.legend()

    # Amplification ratio plot
    plt.subplot(num_subplots, 1, 2)
    for i, results in enumerate(results_list):
        mean_amp = results["amp_mean"]
        std_amp = results["amp_std"]
        plt.plot(s, mean_amp, '-o', label=f'Amplification (mean) ||y||/||u|| - {labels[i]}')
        plt.fill_between(s, mean_amp - std_amp, mean_amp + std_amp, alpha=0.25)
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Amplification ratio")
    plt.grid(True)
    plt.legend()

    # Noise amplification plot (new)
    plt.subplot(num_subplots, 1, 3)
    for i, results in enumerate(results_list):
        mean_noise_amp = results["noise_amp_mean"]
        std_noise_amp = results["noise_amp_std"]
        plt.plot(s, mean_noise_amp, '-o', label=f'Noise Amp (mean) ||error||/||noise|| - {labels[i]}')
        plt.fill_between(s, mean_noise_amp - std_noise_amp, mean_noise_amp + std_noise_amp, alpha=0.25)
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Noise amplification ratio")
    plt.grid(True)
    plt.legend()

    # Optional MSE plot
    if plot_mse:
        plt.subplot(num_subplots, 1, 4)
        for i, results in enumerate(results_list):
            mean_mse = results["mse_mean"]
            std_mse = results["mse_std"]
            plt.plot(s, mean_mse, '-o', label=f'MSE (mean) - {labels[i]}')
            plt.fill_between(s, mean_mse - std_mse, mean_mse + std_mse, alpha=0.25)
        plt.xlabel("Input SNR (dB)")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.legend()

    # Worst-case trials (separate figure for clarity)
    plt.figure(figsize=(10, 6))
    for i, results in enumerate(results_list):
        if snr_focus in results["worst_trials"]:
            y_clean = results["y_clean"][:, channel]
            worst_data = results["worst_trials"][snr_focus]
            y_noisy = worst_data["y_noisy"][:, channel]
            error = y_noisy - y_clean
            plt.plot(y_clean, label=f"Clean output - {labels[i]}")
            plt.plot(y_noisy, label=f"Noisy output (worst @ {snr_focus} dB) - {labels[i]}")
            plt.plot(error, label=f"Error - {labels[i]}", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Output value")
    plt.legend()
    plt.grid(True)
    plt.title(f"Worst-Case Trials at {snr_focus} dB SNR")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title)
    plt.show()


def generate_noisy_trials(u_clean, snr_db_list, trials, seed=42):
    rng = np.random.default_rng(seed)
    noisy_inputs = []
    for snr in snr_db_list:
        trials_list = []
        for _ in range(trials):
            u_noisy, _ = add_noise_for_snr(u_clean, snr, rng)
            trials_list.append(u_noisy)
        noisy_inputs.append(trials_list)
    return noisy_inputs


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
# New: Adversarial Noise Attack Benchmark (Fixed to avoid RuntimeError)
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
    """ Crafts adversarial noise with fixed budget epsilon to maximize ||y_noisy - y_clean||.
    Returns empirical gain = max_error / epsilon, and other stats.
    Uses manual PGD to avoid graph errors. """
    with torch.no_grad():
        y_clean = simulate_fn(u_clean.to(device))

    # Init noise: random within budget
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
            # Normalize grad for L2 step
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


def run_adversarial_benchmark(
        simulate_fns: List[Callable],
        labels: List[str],
        u_clean: torch.Tensor,
        epsilons: List[float],
        trials: int = 5,  # Multiple attacks from different inits
        steps: int = 100,
        lr: float = 0.01,
        proj_norm: str = 'l2'
):
    """ Runs adversarial attacks over multiple epsilons and trials, computes mean/std gains. """
    results = {label: {"mean_gains": [], "std_gains": [], "max_gains": []} for label in labels}
    device = torch.device('cpu')  # Change to 'cuda' if available

    for eps in epsilons:
        for i, simulate_fn in enumerate(simulate_fns):
            label = labels[i]
            gains = []
            for _ in range(trials):
                # Each trial with different random init (handled inside attack)
                res = adversarial_noise_attack(simulate_fn, u_clean, eps, steps, lr, proj_norm, device)
                gains.append(res["adv_gain"])
            results[label]["mean_gains"].append(np.mean(gains))
            results[label]["std_gains"].append(np.std(gains))
            results[label]["max_gains"].append(np.max(gains))

    # Convert to arrays
    for label in labels:
        for k in results[label].keys():
            results[label][k] = np.array(results[label][k])

    return results


def plot_adversarial_results(results: Dict[str, Dict], epsilons: np.ndarray, gamma: float = None,
                             title: str = "Adversarial Noise Amplification"):
    """ Plots mean/std adversarial gains vs. epsilon for multiple models. """
    plt.figure(figsize=(10, 6))
    for label, res in results.items():
        mean = res["mean_gains"]
        std = res["std_gains"]
        plt.plot(epsilons, mean, '-o', label=f'Mean Adv Gain - {label}')
        plt.fill_between(epsilons, mean - std, mean + std, alpha=0.25)
    if gamma is not None:
        plt.axhline(gamma, color='r', ls='--', label='Prescribed Gamma')
    plt.xlabel("Noise Budget (Epsilon)")
    plt.ylabel("Adversarial Gain ||error|| / ||noise||")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Build clean input: T x m
    T = 256
    m = 3
    t = np.arange(T)
    u_clean_np = np.zeros((T, m), dtype=np.float64)
    u_clean_np[:, 0] = np.sin(2 * np.pi * 5 * t / T)  # Sinusoidal
    u_clean = torch.tensor(u_clean_np, dtype=torch.float32)  # Torch version for adv

    # Configs
    cfg_robust = {
        "n_u": 3, "n_y": 3, "d_model": 5, "d_state": 6, "n_layers": 1,
        "ff": "LMLP", "max_phase": math.pi / 50, "r_min": 0.7, "r_max": 0.98,
        "robust": True, "gamma": 60
    }
    cfg_robust = Namespace(**cfg_robust)

    cfg_vanilla = {
        "n_u": 3, "n_y": 3, "d_model": 5, "d_state": 6, "n_layers": 1,
        "ff": "LMLP", "max_phase": math.pi / 50, "r_min": 0.7, "r_max": 0.98,
        "robust": False, "gamma": None
    }
    cfg_vanilla = Namespace(**cfg_vanilla)

    # Build models
    config_robust = SSMConfig(d_model=cfg_robust.d_model, d_state=cfg_robust.d_state, n_layers=cfg_robust.n_layers,
                              ff=cfg_robust.ff,
                              rmin=cfg_robust.r_min, rmax=cfg_robust.r_max, max_phase=cfg_robust.max_phase,
                              robust=cfg_robust.robust, gamma=cfg_robust.gamma)
    model_robust = DeepSSM(cfg_robust.n_u, cfg_robust.n_y, config_robust)

    config_vanilla = SSMConfig(d_model=cfg_vanilla.d_model, d_state=cfg_vanilla.d_state, n_layers=cfg_vanilla.n_layers,
                               ff=cfg_vanilla.ff,
                               rmin=cfg_vanilla.r_min, rmax=cfg_vanilla.r_max, max_phase=cfg_vanilla.max_phase,
                               robust=cfg_vanilla.robust, gamma=cfg_vanilla.gamma)
    model_vanilla = DeepSSM(cfg_vanilla.n_u, cfg_vanilla.n_y, config_vanilla)

    # Simulate functions (torch in/out for grads)
    simulate_fn_robust = make_torch_simulate_fn(model_robust, device=torch.device('cpu'), mode='scan')
    simulate_fn_vanilla = make_torch_simulate_fn(model_vanilla, device=torch.device('cpu'), mode='scan')


    # For original SNR sweep (numpy-based; convert as needed)
    def np_wrapper(sim_fn):
        def wrapped(u_np):
            u_t = torch.tensor(u_np, dtype=torch.float32)
            y_t = sim_fn(u_t)
            return y_t.detach().cpu().numpy()

        return wrapped


    snr_db_list = np.arange(-10, 21, 5)
    trials = 400
    noisy_inputs = generate_noisy_trials(u_clean_np, snr_db_list, trials)

    res_robust = snr_sweep(np_wrapper(simulate_fn_robust), u_clean_np, snr_db_list, trials, noisy_inputs=noisy_inputs)
    res_vanilla = snr_sweep(np_wrapper(simulate_fn_vanilla), u_clean_np, snr_db_list, trials, noisy_inputs=noisy_inputs)

    print(
        f"Robust SSM empirical max L2 gain (random noise): {res_robust['empirical_max_gain']:.2f} (prescribed gamma={cfg_robust.gamma})")
    print(f"Vanilla SSM empirical max L2 gain (random noise): {res_vanilla['empirical_max_gain']:.2f} (unknown)")

    # plot_snr_sweep([res_robust, res_vanilla], ["Robust SSM", "Vanilla SSM"], snr_focus=5, plot_mse=True)  # Uncomment to plot SNR

    # New: Adversarial Benchmark
    # Choose epsilons: e.g., corresponding to noise norms at certain SNRs
    # For demo, fixed values; adjust based on ||u_clean||
    u_norm = torch.norm(u_clean).item()
    epsilons = np.array([0.01 * u_norm, 0.05 * u_norm, 0.1 * u_norm, 0.2 * u_norm])  # Scale to signal norm

    adv_results = run_adversarial_benchmark(
        [simulate_fn_robust, simulate_fn_vanilla],
        ["Robust SSM", "Vanilla SSM"],
        u_clean,
        epsilons,
        trials=400,  # Multiple inits for robustness
        steps=100,
        lr=0.005,  # Tune lr if convergence issues
        proj_norm='l2'
    )

    print("\nAdversarial Gains:")
    for label, res in adv_results.items():
        print(f"{label} mean gains: {res['mean_gains']}")
        print(f"{label} max gains: {res['max_gains']}")

    plot_adversarial_results(adv_results, epsilons, gamma=cfg_robust.gamma)

    SSM_sim

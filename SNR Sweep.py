import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict
from SSM.ssm import DeepSSM, SSMConfig
import math
from argparse import Namespace


seed = 9
torch.manual_seed(seed)

def add_noise_for_snr(u: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to achieve target input SNR."""
    sig_power = np.mean(u**2)
    snr_linear = 10**(snr_db / 10)
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
        "worst_trials": {},
        "y_clean": y_clean,
        "u_clean": u_clean,
    }

    for snr_idx, snr in enumerate(snr_db_list):
        out_snrs, amps = [], []
        worst_err = -np.inf
        worst_trial_data = None

        for t in range(trials):
            if noisy_inputs is None:
                u_noisy, _ = add_noise_for_snr(u_clean, snr, rng)
            else:
                u_noisy = noisy_inputs[snr_idx][t]

            y = simulate_fn(u_noisy)

            noise_power = np.mean((y - y_clean) ** 2)
            sig_power = np.mean(y_clean ** 2)
            out_snr = 10 * np.log10(sig_power / noise_power)
            amp = np.linalg.norm(y) / np.linalg.norm(u_noisy)

            out_snrs.append(out_snr)
            amps.append(amp)

            # track worst-case trial
            err_norm = np.linalg.norm(y - y_clean)
            if err_norm > worst_err:
                worst_err = err_norm
                worst_trial_data = {"u_noisy": u_noisy, "y_noisy": y}

        # store stats
        results["out_snr_mean"].append(np.mean(out_snrs))
        results["out_snr_std"].append(np.std(out_snrs))
        results["amp_mean"].append(np.mean(amps))
        results["amp_std"].append(np.std(amps))
        results["worst_trials"][snr] = worst_trial_data



    # convert lists to numpy
    for k in results.keys():
        if isinstance(results[k], list):
            results[k] = np.array(results[k])

    return results




def plot_snr_sweep(results: Dict[str, np.ndarray], title: str = "SNR Sweep",
                   snr_focus: float = 0.0, channel: int = 0):
    """Plots sweep + worst-case trial at selected SNR."""
    s = results["snr_db"]
    plt.figure(figsize=(10, 9))

    # Output SNR plot
    plt.subplot(3, 1, 1)
    mean = results["out_snr_mean"]
    std = results["out_snr_std"]
    plt.plot(s, mean, '-o', label='Output SNR (mean)')
    plt.fill_between(s, mean - std, mean + std, alpha=0.25)
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Output SNR (dB)")
    plt.grid(True)
    plt.legend()

    # Amplification ratio plot
    plt.subplot(3, 1, 2)
    mean_amp = results["amp_mean"]
    std_amp = results["amp_std"]
    plt.plot(s, mean_amp, '-o', label='Amplification (mean) ||y||/||u||')
    plt.fill_between(s, mean_amp - std_amp, mean_amp + std_amp, alpha=0.25)
    plt.xlabel("Input SNR (dB)")
    plt.ylabel("Amplification ratio")
    plt.grid(True)
    plt.legend()

    # Worst-case trial at chosen SNR
    plt.subplot(3, 1, 3)
    if snr_focus in results["worst_trials"]:
        y_clean = results["y_clean"][:, channel]
        y_noisy = results["worst_trials"][snr_focus]["y_noisy"][:, channel]
        error = y_noisy - y_clean

        plt.plot(y_clean, label="Clean output")
        plt.plot(y_noisy, label=f"Noisy output (worst trial @ {snr_focus} dB)")
        plt.plot(error, label="Error", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Output value")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, f"No trial stored for {snr_focus} dB",
                 ha='center', va='center', transform=plt.gca().transAxes)

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
# Example: wrapper for a torch model (LRU_Robust-like)
# ---------------------------
def make_torch_simulate_fn(model: torch.nn.Module, device: torch.device = torch.device('cpu'),
                           mode: str = "scan") -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a simulate_fn(u_np) wrapper that calls model.forward.
    Assumes model.forward accepts input shape (B, L, H) and returns (outputs, states).
    Adjust if your model's API differs.
    """
    model = model.to(device)
    model.eval()

    def simulate_fn(u_np: np.ndarray) -> np.ndarray:
        # u_np: (T, input_dim)
        T, m = u_np.shape
        # model expects (B, L, H); we use batch=1
        inp = torch.tensor(u_np, dtype=next(model.parameters()).dtype, device=device).unsqueeze(0)  # (1, T, m)
        with torch.no_grad():
            out, _states = model.forward(inp, mode=mode)  # adjust if your API differs
        # out: (B, T, output_dim)
        out_np = out.squeeze(0).cpu().numpy()
        return out_np

    return simulate_fn

# ---------------------------
# Minimal runnable example usage
# ---------------------------
if __name__ == "__main__":
    # Build a simple clean input (example): T x m
    T = 256
    m = 3
    # example clean signal: sinusoidal on channel 0, zeros elsewhere
    t = np.arange(T)
    u_clean = np.zeros((T, m), dtype=np.float64)
    u_clean[:, 0] = np.sin(2 * np.pi * 5 * t / T)  # 5 cycles across the sequence

    cfg = {
        "n_u": 3,
        "n_y": 3,
        "d_model": 5,
        "d_state": 6,
        "n_layers": 2,
        "ff": "LMLP",  # GLU | MLP | LMLP
        "max_phase": math.pi / 50,
        "r_min": 0.7,
        "r_max": 0.98,
        "robust": True,
        "gamma": 33
    }
    cfg = Namespace(**cfg)

    cfg2 = {
        "n_u": 3,
        "n_y": 3,
        "d_model": 5,
        "d_state": 6,
        "n_layers": 2,
        "ff": "LMLP",  # GLU | MLP | LMLP
        "max_phase": math.pi / 50,
        "r_min": 0.7,
        "r_max": 0.98,
        "robust": False,
        "gamma": None
    }
    cfg2 = Namespace(**cfg2)


    # Build models
    config = SSMConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                       rmax=cfg.r_max, max_phase=cfg.max_phase, robust=cfg.robust, gamma=cfg.gamma)
    model = DeepSSM(cfg.n_u, cfg.n_y, config)

    SSM_sim=make_torch_simulate_fn(model, device=torch.device('cpu'), mode='scan')

    config2 = SSMConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                       rmax=cfg.r_max, max_phase=cfg2.max_phase, robust=cfg2.robust, gamma=cfg2.gamma)
    model2 = DeepSSM(cfg.n_u, cfg.n_y, config2)

    SSM_sim2=make_torch_simulate_fn(model2, device=torch.device('cpu'), mode='scan')

    # Replace simulate_fn with your actual simulator or model wrapper:
    # e.g. simulate_fn = make_torch_simulate_fn(my_model, device=torch.device('cuda'), mode='scan')
    # For demo we use a simple linear toy simulate_fn (replace this)


    simulate_fn = SSM_sim
    simulate_fn2 = SSM_sim2
    # replace with real simulate function

    snr_db_list = np.arange(-10, 21, 5)
    trials = 400
    noisy_inputs = generate_noisy_trials(u_clean, snr_db_list, trials)

    res1 = snr_sweep(simulate_fn, u_clean, snr_db_list, trials, noisy_inputs=noisy_inputs)
    res2 = snr_sweep(simulate_fn2, u_clean, snr_db_list, trials, noisy_inputs=noisy_inputs)

    plot_snr_sweep(res1, snr_focus=5)
    plot_snr_sweep(res2, snr_focus=5)

    SSM_sim

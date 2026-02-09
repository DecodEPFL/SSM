"""
train_and_track_lag_orthogonality_FIXED.py

Robust training + orthogonality tracking for Block2x2DenseL2SSM on nonlinear_benchmarks.

Fixes "loss always zero" by:
  - guaranteeing non-empty window dataset,
  - ensuring DataLoader yields >= 1 batch (drop_last=False + batch_size<=len(ds)),
  - ensuring warmup < win_len so the loss slice is non-empty,
  - printing diagnostics to pinpoint remaining issues fast.

Tracks orthogonality of the z-basis state lag dictionary: {A_z^k B_z}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

# --- optional nonlinear_benchmarks ---
try:
    import nonlinear_benchmarks  # type: ignore
except Exception:
    nonlinear_benchmarks = None

# --- your model (required) ---
try:
    from src.neural_ssm.ssm.lru import Block2x2DenseL2SSM  # type: ignore
except Exception as e:
    Block2x2DenseL2SSM = None  # type: ignore
    _IMPORT_ERR = e


# ======================================================================================
# Config
# ======================================================================================
@dataclass
class Cfg:
    # dataset
    use_nonlinear_benchmarks: bool = True
    benchmark_name: str = "Cascaded_Tanks()"     # change if you want
    benchmark_fallback: str = "WienerHammer"

    # model dims (supports du/dy >= 1, but your model must match)
    d_state: int = 2298   # must be even
    d_in: int = 1
    d_out: int = 1

    # gain/certification
    gamma: float = 99.0
    train_gamma: bool = True

    # init for L2N
    rho_init: float = 0.99
    full_phase: bool = True          # if True: theta ~ U(-pi, pi)
    max_phase: float = 0.25          # if full_phase=False: theta in [-max_phase, +max_phase]
    offdiag_scale: float = 0.01

    # training on windows
    win_len: int = 2048
    stride: int = 256
    warmup: int = 64                 # ignored in loss
    max_windows_per_seq: int = 2000

    epochs: int = 900
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0

    # orthogonality tracking
    K: int = 400
    ortho_every: int = 20
    min_lag_norm: float = 1e-10

    # misc
    seed: int = 7
    normalize_data: bool = True
    detach_sigma_in_spectral_norm: bool = True
    show_plots: bool = True


# ======================================================================================
# Helpers: robust conversions and benchmark resolving
# ======================================================================================
def as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def to_bln(arr: Any) -> torch.Tensor:
    """
    Convert to (B,L,d) float tensor:
      (L,)      -> (1,L,1)
      (L,d)     -> (1,L,d)
      (B,L,d)   -> unchanged
    """
    a = as_numpy(arr)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise ValueError(f"Expected 1D/2D/3D, got {a.shape}")
    return torch.from_numpy(a).float()

def resolve_benchmark_fn(name: str, fallback: str) -> Tuple[Callable, str]:
    if nonlinear_benchmarks is None:
        raise ImportError("nonlinear_benchmarks is not installed.")
    name = name.strip()
    if name.endswith("()"):
        name = name[:-2]

    def candidates(base: str) -> List[str]:
        if base.lower().endswith("benchmark"):
            return [base]
        return [
            base,
            base + "BenchMark",
            base + "Benchmark",
            base.capitalize() + "BenchMark",
            base.capitalize() + "Benchmark",
        ]

    for cand in candidates(name):
        if hasattr(nonlinear_benchmarks, cand):
            return getattr(nonlinear_benchmarks, cand), cand

    low = name.lower()
    for attr in dir(nonlinear_benchmarks):
        if attr.lower() == low:
            return getattr(nonlinear_benchmarks, attr), attr
        if attr.lower().startswith(low) and attr.lower().endswith("benchmark"):
            return getattr(nonlinear_benchmarks, attr), attr

    for cand in candidates(fallback):
        if hasattr(nonlinear_benchmarks, cand):
            return getattr(nonlinear_benchmarks, cand), cand

    raise ValueError(f"Could not resolve '{name}' (fallback '{fallback}').")


# ======================================================================================
# Built-in fallback dataset (only if nonlinear_benchmarks missing)
# ======================================================================================
def _stable_matrix(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    M = rng.standard_normal((n, n)).astype(np.float32)
    v = rng.standard_normal((n, 1)).astype(np.float32)
    for _ in range(30):
        v = M @ v
        v /= (np.linalg.norm(v) + 1e-12)
    s = float(np.linalg.norm(M @ v))
    return (rho / max(s, 1e-6)) * M

def make_builtin_dataset(seed: int, length: int = 80000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n1, n2 = 6, 6
    A1 = _stable_matrix(n1, 0.95, rng)
    A2 = _stable_matrix(n2, 0.90, rng)
    b1 = rng.standard_normal((n1, 1)).astype(np.float32) * 0.6
    c1 = rng.standard_normal((n1, 1)).astype(np.float32) * 0.6
    b2 = rng.standard_normal((n2, 1)).astype(np.float32) * 0.8
    c2 = rng.standard_normal((n2, 1)).astype(np.float32) * 0.8

    def sim(u: np.ndarray) -> np.ndarray:
        z = np.zeros((n1, 1), dtype=np.float32)
        x = np.zeros((n2, 1), dtype=np.float32)
        y = np.zeros((u.shape[0], 1), dtype=np.float32)
        for t in range(u.shape[0]):
            z = A1 @ z + b1 * u[t]
            w = np.tanh(float((c1.T @ z)[0, 0]))
            x = A2 @ x + b2 * w
            y[t, 0] = float((c2.T @ x)[0, 0])
        return y

    def gen(B: int) -> Tuple[np.ndarray, np.ndarray]:
        u = rng.standard_normal((B, length, 1)).astype(np.float32)
        t = np.linspace(0, 60 * np.pi, length, dtype=np.float32)[None, :, None]
        u += 0.25 * np.sin(t)
        y = np.stack([sim(u[i, :, 0:1]) for i in range(B)], axis=0)
        return u, y

    u_tr, y_tr = gen(1)
    u_te, y_te = gen(1)
    return u_tr, y_tr, u_te, y_te, 64


def load_data(cfg: Cfg, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if cfg.use_nonlinear_benchmarks and nonlinear_benchmarks is not None:
        bench_fn, bench_name = resolve_benchmark_fn(cfg.benchmark_name, cfg.benchmark_fallback)
        train_split, test_split = bench_fn()
        u_tr, y_tr = train_split
        u_te, y_te = test_split
        warmup_hint = int(getattr(test_split, "state_initialization_window_length", cfg.warmup))

        u_train = to_bln(u_tr).to(device)
        y_train = to_bln(y_tr).to(device)
        u_test  = to_bln(u_te).to(device)
        y_test  = to_bln(y_te).to(device)

        print(f"Using nonlinear_benchmarks: {bench_name}")
        return u_train, y_train, u_test, y_test, warmup_hint

    # fallback
    u_tr, y_tr, u_te, y_te, warmup_hint = make_builtin_dataset(cfg.seed)
    u_train = torch.from_numpy(u_tr).to(device)
    y_train = torch.from_numpy(y_tr).to(device)
    u_test  = torch.from_numpy(u_te).to(device)
    y_test  = torch.from_numpy(y_te).to(device)

    print("Using built-in nonlinear dataset fallback.")
    return u_train, y_train, u_test, y_test, warmup_hint


# ======================================================================================
# Windowing (guaranteed non-empty)
# ======================================================================================
def make_windows(u: torch.Tensor, y: torch.Tensor, win_len: int, stride: int, max_windows_per_seq: int) -> TensorDataset:
    """
    u,y: (B,L,d). Returns (N,win_len,d) windows. Always N>=1.
    """
    assert u.ndim == 3 and y.ndim == 3
    B, L, du = u.shape
    By, Ly, dy = y.shape
    assert (B, L) == (By, Ly)

    win_len = min(int(win_len), int(L))
    if win_len < 2:
        raise ValueError(f"win_len too small: win_len={win_len}, L={L}. Check shapes.")
    stride = int(stride) if stride > 0 else max(1, win_len // 4)

    U_list: List[torch.Tensor] = []
    Y_list: List[torch.Tensor] = []

    for b in range(B):
        last = L - win_len
        starts = list(range(0, last + 1, stride)) if last >= 0 else [0]
        if len(starts) == 0:
            starts = [0]
        if len(starts) > max_windows_per_seq:
            starts = starts[:max_windows_per_seq]
        for s in starts:
            U_list.append(u[b, s:s + win_len, :])
            Y_list.append(y[b, s:s + win_len, :])

    if len(U_list) == 0:
        # absolute fallback
        U_list = [u[0, :win_len, :]]
        Y_list = [y[0, :win_len, :]]

    U = torch.stack(U_list, dim=0)
    Y = torch.stack(Y_list, dim=0)
    return TensorDataset(U, Y)


# ======================================================================================
# Model and orthogonality
# ======================================================================================
def build_model(cfg: Cfg, device: torch.device) -> nn.Module:
    if Block2x2DenseL2SSM is None:
        raise ImportError(f"Could not import Block2x2DenseL2SSM.\nOriginal error:\n{_IMPORT_ERR}")
    assert cfg.d_state % 2 == 0

    m = Block2x2DenseL2SSM(
        d_state=cfg.d_state,
        d_input=cfg.d_in,
        d_output=cfg.d_out,
        gamma=cfg.gamma,
        train_gamma=cfg.train_gamma,
        exact_norm=True,
    ).to(device)

    max_phase = None if cfg.full_phase else cfg.max_phase
    m.init_on_circle(
        rho=cfg.rho_init,
        max_phase=max_phase,
        phase_center=0.0,
        random_phase=True,
        offdiag_scale=cfg.offdiag_scale,
    )

    # Stabilize gradients through spectral normalization (keeps forward scaling/certificate)
    if cfg.detach_sigma_in_spectral_norm and hasattr(m, "_spectral_normalize"):
        import types
        def _spectral_normalize_detached(self, M: torch.Tensor) -> torch.Tensor:
            sigma = torch.linalg.matrix_norm(M, ord=2).clamp(min=1e-6)
            scale = sigma.detach().clamp(min=1.0)
            return M / (scale + 0.002)
        m._spectral_normalize = types.MethodType(_spectral_normalize_detached, m)

    return m

def forward_model(model: nn.Module, u: torch.Tensor) -> torch.Tensor:
    # expected: (B,L,dy)
    y = model(u, return_state=False, mode="scan")
    if not torch.is_tensor(y):
        raise RuntimeError(f"Model forward returned {type(y)}; expected Tensor.")
    return y

@torch.no_grad()
def lag_mats_block2x2(Az: torch.Tensor, Bz: torch.Tensor, K: int, min_norm: float) -> List[torch.Tensor]:
    """
    Fast recursion for block2x2 Az with blocks [[a,-b],[b,a]].
    """
    d_state, d_in = Bz.shape
    n_blocks = d_state // 2
    ar = Az[0::2, 0::2].diagonal()
    ai = Az[1::2, 0::2].diagonal()

    Mr = Bz[0::2, :].clone()
    Mi = Bz[1::2, :].clone()

    mats: List[torch.Tensor] = []
    for _ in range(K):
        M = torch.empty((d_state, d_in), device=Bz.device, dtype=Bz.dtype)
        M[0::2, :] = Mr
        M[1::2, :] = Mi
        mats.append(M)
        if float(torch.linalg.norm(M).item()) < min_norm:
            break
        Mr_next = ar[:, None] * Mr - ai[:, None] * Mi
        Mi_next = ai[:, None] * Mr + ar[:, None] * Mi
        Mr, Mi = Mr_next, Mi_next
    return mats

@torch.no_grad()
def gram_stats(mats: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    K = len(mats)
    D = torch.stack([m.reshape(-1) for m in mats], dim=1)
    Dn = D / torch.linalg.norm(D, dim=0).clamp_min(1e-12)
    G = Dn.T @ Dn
    off = G - torch.eye(K, device=G.device, dtype=G.dtype)
    mask = ~torch.eye(K, dtype=torch.bool, device=G.device)
    max_off = float(off.abs()[mask].max().item()) if K > 1 else 0.0
    mean_off = float(off.abs()[mask].mean().item()) if K > 1 else 0.0
    frob = float(torch.linalg.norm(off, ord="fro").item())
    return G, {"K_used": float(K), "max_off": max_off, "mean_off": mean_off, "frob": frob}

def plot_heatmap(G: torch.Tensor, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(G.detach().cpu().numpy(), vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("k")
    plt.tight_layout()
    plt.show()


# ======================================================================================
# Main
# ======================================================================================
def main():
    cfg = Cfg()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- data ---
    u_train, y_train, u_test, y_test, warmup_hint = load_data(cfg, device)
    print("Raw shapes:",
          "u_train", tuple(u_train.shape), "y_train", tuple(y_train.shape),
          "u_test", tuple(u_test.shape), "y_test", tuple(y_test.shape))

    # Sanity dims
    assert u_train.shape[-1] == cfg.d_in, f"d_in mismatch: data {u_train.shape[-1]} vs cfg {cfg.d_in}"
    assert y_train.shape[-1] == cfg.d_out, f"d_out mismatch: data {y_train.shape[-1]} vs cfg {cfg.d_out}"

    # --- normalize (strongly recommended) ---
    if cfg.normalize_data:
        u_mu = u_train.mean(dim=(0, 1), keepdim=True)
        u_sd = u_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        y_mu = y_train.mean(dim=(0, 1), keepdim=True)
        y_sd = y_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        u_train = (u_train - u_mu) / u_sd
        y_train = (y_train - y_mu) / y_sd
        u_test = (u_test - u_mu) / u_sd
        y_test = (y_test - y_mu) / y_sd

    print(f"Train stats: u std={float(u_train.std()):.3e}, y std={float(y_train.std()):.3e}")
    if float(y_train.std().item()) == 0.0:
        raise RuntimeError("y_train has zero variance -> your dataset extraction is wrong (or the dataset is constant).")

    # --- windows ---
    train_ds = make_windows(u_train, y_train, cfg.win_len, cfg.stride, cfg.max_windows_per_seq)
    N = len(train_ds)
    bs = min(cfg.batch_size, N)          # IMPORTANT: avoid bs > N
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)  # IMPORTANT: drop_last=False

    print(f"Windowing: N_windows={N}, batch_size={bs}, n_batches/epoch={len(loader)}")
    if len(loader) == 0:
        raise RuntimeError("DataLoader has 0 batches. This was causing 'loss always zero' before.")

    # --- warmup for loss: ensure non-empty slice ---
    win_len_eff = train_ds.tensors[0].shape[1]
    warmup_loss = min(int(cfg.warmup), int(win_len_eff - 1))
    warmup_loss = max(warmup_loss, 0)
    print(f"Loss warmup: warmup_loss={warmup_loss}, win_len_eff={win_len_eff}")

    # --- model ---
    model = build_model(cfg, device=device)

    # --- orthogonality before ---
    model.eval()
    with torch.no_grad():
        Az, Bz, Cz, Dz, _ = model.compute_z_matrices()
        mats0 = lag_mats_block2x2(Az, Bz, cfg.K, cfg.min_lag_norm)
        G0, st0 = gram_stats(mats0)
    print("\nOrthogonality BEFORE (z):", st0)

    # --- sanity: first batch loss must be > 0 generally ---
    u0, y0 = next(iter(loader))
    model.train()
    with torch.no_grad():
        yhat0 = forward_model(model, u0)
        err0 = (yhat0[:, warmup_loss:, :] - y0[:, warmup_loss:, :])
        print(f"[Sanity] first-batch: yhat std={float(yhat0.std()):.3e}, "
              f"err max={float(err0.abs().max()):.3e}, "
              f"err mean={float(err0.abs().mean()):.3e}")

    # If error is exactly zero here, you're literally predicting the targets already,
    # which is extremely unlikely. Catch it explicitly:
    if float(err0.abs().max().item()) == 0.0:
        raise RuntimeError(
            "First-batch prediction error is EXACTLY zero. "
            "This usually means y0 and yhat0 are the same tensor/data by mistake."
        )

    # --- training ---
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    loss_hist: List[float] = []
    frob_hist: List[float] = []
    ticks: List[int] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for u_win, y_win in loader:
            opt.zero_grad(set_to_none=True)
            y_pred = forward_model(model, u_win)

            # guaranteed non-empty slice
            yp = y_pred[:, warmup_loss:, :]
            yt = y_win[:, warmup_loss:, :]
            if yp.numel() == 0 or yt.numel() == 0:
                raise RuntimeError("Loss slice is empty. Increase win_len or reduce warmup.")

            loss = loss_fn(yp, yt)
            loss.backward()
            opt.step()

            running += float(loss.item()) * u_win.size(0)
            n_seen += u_win.size(0)

        avg = running / max(n_seen, 1)
        loss_hist.append(avg)

        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:04d} | loss={avg:.6e} | gamma={float(model.gamma.detach().cpu()):.3f}")

        if epoch % cfg.ortho_every == 0 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                Az, Bz, Cz, Dz, _ = model.compute_z_matrices()
                mats = lag_mats_block2x2(Az, Bz, cfg.K, cfg.min_lag_norm)
                _, st = gram_stats(mats)
            frob_hist.append(st["frob"])
            ticks.append(epoch)

    # --- test ---
    model.eval()
    with torch.no_grad():
        y_test_pred = forward_model(model, u_test)
        n_init = min(int(max(0, warmup_hint)), int(y_test_pred.shape[1] - 1))
        test_mse = float(loss_fn(y_test_pred[:, n_init:, :], y_test[:, n_init:, :]).item())
    print(f"\nTest MSE (after warmup={n_init}) = {test_mse:.6e}")

    # --- orthogonality after ---
    with torch.no_grad():
        Az, Bz, Cz, Dz, _ = model.compute_z_matrices()
        mats1 = lag_mats_block2x2(Az, Bz, cfg.K, cfg.min_lag_norm)
        G1, st1 = gram_stats(mats1)
    print("Orthogonality AFTER (z):", st1)

    # --- plots ---
    if cfg.show_plots:
        plot_heatmap(G0, "Gram of {A_z^k B_z} (before)")
        plot_heatmap(G1, "Gram of {A_z^k B_z} (after)")

        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(loss_hist) + 1), loss_hist, linewidth=1.5)
        plt.title("Training loss")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        if frob_hist:
            plt.figure(figsize=(6, 4))
            plt.plot(ticks, frob_hist, marker="o", linewidth=1.5)
            plt.title("Orthogonality drift: ||G - I||_F")
            plt.xlabel(f"epoch (every {cfg.ortho_every})")
            plt.ylabel("Frobenius")
            plt.yscale("log")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()

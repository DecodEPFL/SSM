"""Compare mode/state behavior for LRU and l2n models.

This script checks:
1) `loop` vs `scan` on full-sequence inference.
2) Chunked inference with explicit state vs internal state.
3) `reset_state=True` vs `reset()` behavior.

Usage:
    python scripts/check_state_and_mode_equivalence.py
    python scripts/check_state_and_mode_equivalence.py --seq-len 96 --chunks 24,24,24,24
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neural_ssm.ssm.lru import LRU, Block2x2DenseL2SSM


def parse_chunks(spec: str | None, seq_len: int) -> list[int]:
    if spec is None:
        a = max(1, seq_len // 3)
        b = max(1, (seq_len - a) // 2)
        c = seq_len - a - b
        if c <= 0:
            return [seq_len]
        return [a, b, c]

    parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not parts:
        raise ValueError("`--chunks` cannot be empty.")
    if any(x <= 0 for x in parts):
        raise ValueError("All chunk sizes must be > 0.")
    if sum(parts) != seq_len:
        raise ValueError(f"Chunk sizes must sum to seq-len ({seq_len}), got {sum(parts)}.")
    return parts


def iter_chunks(x: torch.Tensor, chunk_sizes: Iterable[int]) -> Iterable[torch.Tensor]:
    start = 0
    for size in chunk_sizes:
        yield x[:, start:start + size, :]
        start += size


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return (a - b).abs().max().item()


def check_close(name: str, a: torch.Tensor, b: torch.Tensor, *, rtol: float, atol: float) -> bool:
    diff = max_abs_diff(a, b)
    try:
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        print(f"[PASS] {name:<52} max_abs_diff={diff:.3e}")
        return True
    except AssertionError as exc:
        print(f"[FAIL] {name:<52} max_abs_diff={diff:.3e}")
        print(f"       {str(exc).splitlines()[0]}")
        return False


def run_full(model: torch.nn.Module, u: torch.Tensor, mode: str):
    y, states = model(u, mode=mode, reset_state=True)
    return y.detach(), states.detach(), model.state.detach()


def run_chunked_explicit(model: torch.nn.Module, u: torch.Tensor, mode: str, chunk_sizes: list[int]):
    state = None
    outs = []
    for chunk in iter_chunks(u, chunk_sizes):
        y, states = model(chunk, state=state, mode=mode, reset_state=False)
        outs.append(y.detach())
        state = states[:, -1, :].detach()
    return torch.cat(outs, dim=1), state


def run_chunked_internal(model: torch.nn.Module, u: torch.Tensor, mode: str, chunk_sizes: list[int]):
    model.reset()
    outs = []
    for chunk in iter_chunks(u, chunk_sizes):
        y, _ = model(chunk, mode=mode, reset_state=False)
        outs.append(y.detach())
    return torch.cat(outs, dim=1), model.state.detach()


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def evaluate_model(
    name: str,
    model_builder: Callable[[], torch.nn.Module],
    reference_state: dict[str, torch.Tensor],
    u: torch.Tensor,
    chunk_sizes: list[int],
    *,
    rtol: float,
    atol: float,
) -> bool:
    ok = True
    print(f"\n=== {name} ===")

    def fresh_model() -> torch.nn.Module:
        m = model_builder().to(u.device)
        m.load_state_dict(reference_state, strict=True)
        return m

    m_loop = fresh_model()
    y_loop, _, s_loop = run_full(m_loop, u, "loop")

    m_scan = fresh_model()
    y_scan, _, s_scan = run_full(m_scan, u, "scan")

    ok &= check_close(f"{name}:full loop_vs_scan output", y_loop, y_scan, rtol=rtol, atol=atol)
    ok &= check_close(f"{name}:full loop_vs_scan final_state", s_loop, s_scan, rtol=rtol, atol=atol)

    for mode in ("loop", "scan"):
        print(f"\n-- mode={mode} --")

        m_full = fresh_model()
        y_full, st_full, s_full = run_full(m_full, u, mode)

        m_exp = fresh_model()
        y_exp, s_exp = run_chunked_explicit(m_exp, u, mode, chunk_sizes)

        m_int = fresh_model()
        y_int, s_int = run_chunked_internal(m_int, u, mode, chunk_sizes)

        ok &= check_close(f"{name}:{mode}:state_buffer_matches_returned", s_full, st_full[:, -1, :], rtol=rtol, atol=atol)
        ok &= check_close(f"{name}:{mode}:full_vs_chunked_explicit output", y_full, y_exp, rtol=rtol, atol=atol)
        ok &= check_close(f"{name}:{mode}:full_vs_chunked_explicit final_state", s_full, s_exp, rtol=rtol, atol=atol)
        ok &= check_close(f"{name}:{mode}:explicit_vs_internal_chunked output", y_exp, y_int, rtol=rtol, atol=atol)
        ok &= check_close(f"{name}:{mode}:explicit_vs_internal_chunked final_state", s_exp, s_int, rtol=rtol, atol=atol)

        m_reset_flag = fresh_model()
        _ = m_reset_flag(u, mode=mode)  # move state away from zero
        y_flag, st_flag = m_reset_flag(u, mode=mode, reset_state=True)

        m_reset_method = fresh_model()
        _ = m_reset_method(u, mode=mode)  # move state away from zero
        m_reset_method.reset()
        y_method, st_method = m_reset_method(u, mode=mode)

        ok &= check_close(f"{name}:{mode}:reset_flag_vs_reset_method output", y_flag, y_method, rtol=rtol, atol=atol)
        ok &= check_close(
            f"{name}:{mode}:reset_flag_vs_reset_method final_state",
            st_flag[:, -1, :],
            st_method[:, -1, :],
            rtol=rtol,
            atol=atol,
        )

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check mode/state equivalence for LRU and l2n.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--chunks", type=str, default=None, help="Comma-separated chunk sizes summing to seq-len.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--atol", type=float, default=5e-5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if args.d_state % 2 != 0:
        raise ValueError("d-state must be even for l2n (Block2x2DenseL2SSM).")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    chunk_sizes = parse_chunks(args.chunks, args.seq_len)
    u = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)

    def build_lru() -> torch.nn.Module:
        return LRU(
            in_features=args.d_model,
            out_features=args.d_model,
            state_features=args.d_state,
        )

    def build_l2n() -> torch.nn.Module:
        return Block2x2DenseL2SSM(
            d_state=args.d_state,
            d_input=args.d_model,
            d_output=args.d_model,
        )

    lru_ref = build_lru().to(device)
    lru_state = clone_state_dict(lru_ref)

    l2n_ref = build_l2n().to(device)
    l2n_ref.init_on_circle(rho=0.9, max_phase=0.1, phase_center=0.0, random_phase=True)
    l2n_state = clone_state_dict(l2n_ref)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Shape: B={args.batch_size}, T={args.seq_len}, D={args.d_model}, N={args.d_state}")
    print(f"Chunks: {chunk_sizes}")
    print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")

    ok_lru = evaluate_model("LRU", build_lru, lru_state, u, chunk_sizes, rtol=args.rtol, atol=args.atol)
    ok_l2n = evaluate_model("l2n", build_l2n, l2n_state, u, chunk_sizes, rtol=args.rtol, atol=args.atol)
    all_ok = ok_lru and ok_l2n

    print("\n=== Summary ===")
    print("All checks passed." if all_ok else "Some checks failed.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

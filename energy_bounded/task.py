"""Reproducible nonlinear forced-response task with a known zero-state gain <= .9."""
from __future__ import annotations

import torch


def teacher(u):
    """h_t=.65 h_{t-1}+.35 tanh(R u_t); y_t=.8 tanh(h_t)+.1 tanh(u_t).

    R is orthogonal, h_-1=0. The causal convolution's l1 norm is <=1;
    tanh is pointwise norm nonexpansive. Triangle inequality gives prefix l2
    gain <=.9. There is no unobserved initial-condition free response.
    """
    rotation = u.new_tensor([[.8, -.6], [.6, .8]])
    h, outputs = torch.zeros_like(u[:, 0]), []
    for t in range(u.shape[1]):
        h = .65 * h + .35 * torch.tanh(u[:, t] @ rotation)
        outputs.append(.8 * torch.tanh(h) + .1 * torch.tanh(u[:, t]))
    return torch.stack(outputs, 1)


def trajectories(ids, length, *, device="cpu", dtype=torch.float32):
    """Each ID is a complete independently seeded trajectory; never split in time."""
    rows = []
    for trajectory_id in ids:
        generator = torch.Generator().manual_seed(123_000 + int(trajectory_id))
        u = torch.randn(length, 2, generator=generator)
        # Bursts, an initial zero segment, and an unforced tail stress memory/energy.
        amplitudes = .15 + .85 * torch.rand((length + 7) // 8, 1, generator=generator)
        u = u * amplitudes.repeat_interleave(8, 0)[:length]
        u[:min(3, length)] = 0
        u[-max(1, length // 8):] = 0
        rows.append(u)
    u = torch.stack(rows).to(device=device, dtype=dtype)
    return u, teacher(u)


def certificate_metrics(u, y, gamma):
    """Audit the ACTUAL returned tensor in FP64 on CPU, independently per prefix."""
    supplied = (gamma * u.detach().cpu().double()).square().sum(-1).cumsum(1)
    spent = y.detach().cpu().double().square().sum(-1).cumsum(1)
    normalized = (spent - supplied) / (1 + supplied)
    return {"worst_normalized_prefix_violation": max(0., normalized.max().item()),
            "max_signed_normalized_prefix_residual": normalized.max().item()}

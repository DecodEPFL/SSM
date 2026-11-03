"""Pre-norm causal Transformer with per-layer, differentiable KV caches."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .state import valid_mask


@dataclass
class KVState:
    layers: tuple  # each entry is (keys, values): (B,H,physical_context,head_dim)
    valid: Tensor  # (B,physical_context), includes padding slots
    position: Tensor  # (B,), counts VALID tokens only


class _Block(nn.Module):
    def __init__(self, width, heads, expansion, dropout):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, width // heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attn_out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, expansion * width), nn.GELU(),
                                nn.Linear(expansion * width, width))

    def forward(self, x, cache, attention_mask):
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads,
                                                 self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        k = torch.cat((cache[0], k), 2)
        v = torch.cat((cache[1], v), 2)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + F.dropout(self.attn_out(z.transpose(1, 2).reshape(b, t, w)),
                           self.dropout, self.training)
        x = x + F.dropout(self.ff(self.norm2(x)), self.dropout, self.training)
        return x, (k, v)


class TransformerCore(nn.Module):
    """No context truncation. Caches grow linearly; dense attention is quadratic.

    Cached steps compute QKV only for new tokens. Concatenation copies O(context)
    cache elements per call; this is real KV caching, not prefix recomputation.
    Padding slots remain in the physical cache but never become visible keys.
    """

    def __init__(self, d_input, d_output, *, width=24, layers=2, heads=2,
                 expansion=2, dropout=0.0):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or layers < 1 or width % 2:
            raise ValueError("Require positive layers, even width divisible by positive heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        self.d_input, self.d_output = d_input, d_output
        self.width, self.heads, self.layer_count = width, heads, layers
        self.in_proj = nn.Linear(d_input, width)
        self.blocks = nn.ModuleList(_Block(width, heads, expansion, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, d_output)

    def init_state(self, batch_size, *, device=None, dtype=None):
        p = next(self.parameters())
        device, dtype = device or p.device, dtype or p.dtype
        caches = tuple((torch.empty(batch_size, self.heads, 0, self.width // self.heads,
                                    device=device, dtype=dtype),
                        torch.empty(batch_size, self.heads, 0, self.width // self.heads,
                                    device=device, dtype=dtype)) for _ in self.blocks)
        return KVState(caches, torch.empty(batch_size, 0, device=device, dtype=torch.bool),
                       torch.zeros(batch_size, device=device, dtype=torch.long))

    def forward(self, u, state=None, *, mask=None):
        valid = valid_mask(u, mask)
        if u.shape[-1] != self.d_input:
            raise ValueError("Wrong input feature dimension")
        state = self.init_state(u.shape[0], device=u.device, dtype=u.dtype) if state is None else state
        b, t, _ = u.shape
        if not t:
            return u.new_empty(b, 0, self.d_output), state
        old = state.valid.shape[1]
        if state.valid.shape[0] != b or len(state.layers) != self.layer_count:
            raise ValueError("Incompatible Transformer cache")
        positions = (state.position[:, None] + valid.long().cumsum(1) - 1).clamp(min=0)
        # Positional arithmetic in FP32 minimum, then cast to activation dtype.
        pos_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
        frequency = torch.exp(torch.arange(0, self.width, 2, device=u.device, dtype=pos_dtype)
                              * (-math.log(10000.0) / self.width))
        phase = positions.to(pos_dtype)[..., None] * frequency
        encoding = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        x = self.in_proj(torch.where(valid[..., None], u, 0))
        x = x + encoding.to(x.dtype)
        keys_valid = torch.cat((state.valid, valid), 1)
        queries = old + torch.arange(t, device=u.device)
        keys = torch.arange(old + t, device=u.device)
        allowed = (keys[None, :] <= queries[:, None])[None] & keys_valid[:, None, :]
        # Invalid queries are ignored. Give them their own key so every SDPA row
        # has a finite softmax on older supported backends as well.
        own_key = keys[None, :] == queries[:, None]
        allowed = allowed | (~valid[..., None] & own_key[None])
        caches = []
        for block, cache in zip(self.blocks, state.layers):
            x, cache = block(x, cache, allowed[:, None])
            caches.append(cache)
        y = self.out_proj(self.norm(x))
        y = torch.where(valid[..., None], y, 0)
        return y, KVState(tuple(caches), keys_valid, state.position + valid.sum(1))

    def step(self, state, u, *, mask=None):
        if u.ndim != 2:
            raise ValueError("step expects (batch,features), with explicit state first")
        y, state = self(u[:, None], state, mask=None if mask is None else mask[:, None])
        return y[:, 0], state

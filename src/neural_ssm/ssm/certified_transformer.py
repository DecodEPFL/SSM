# python
# file: src/neural_ssm/ssm/certified_transformer.py
r"""Certified Softmax Transformer — an ℓ2-gain-bounded transformer block.

Keeps genuine multi-head softmax self-attention (Q,Kᵀ, softmax, V-mixing, heads,
residuals, FFN, optional causal mask, arbitrary length) but modifies the internals
so the block has a *length-uniform* incremental ℓ2 (Lipschitz) certificate:

  * bounded Q,K,V (``tanh`` radii) — bounded scores make softmax Lipschitz;
  * column-budgeted attention ``P̃`` with ``‖P̃‖₂ ≤ √c_max`` (Schur bound);
  * spectrally-controlled projections (``L2BoundedLinearExact``);
  * gain-budgeted residuals (:func:`split_gain_budget`) and a certified FFN.

Per head, with ``A(X) = Concat_h(P̃_h V_h) W_O``, the Frobenius Lipschitz constant
splits into a value path and a pattern path::

    γ_mha ≤ σ_O · ( √c_max · σ_V                                   # value path (rigorous)
                    + √H · (√2 · R_v / τ) · √(K_max²σ_Q² + c_max·Q_max²σ_K²) )  # pattern path

with ``Q_max=R_q√d_head``, ``K_max=R_k√d_head``, ``σ_•=‖W_•‖₂`` (capped), softmax
Jacobian norm ≤ ½. The **value path is rigorous and length-uniform** (the column
budget gives ``‖P̃‖₂ ≤ √c_max`` regardless of T — without it the operator norm is
``√T``). The **pattern path is a conservative length-uniform estimate**: it uses
the softmax variance-centering ``(J_sV)_j = P_{sj}(v_j-v̄_s)`` (removes ``‖V‖_F~√T``)
and the column-sum cap (removes the key-perturbation ``√T``); the constant is an
over-estimate (the realized gain is far smaller — see tests, which verify it
empirically and that it does *not* grow with T). ``gamma_mha`` may also be passed
explicitly to override the derived value.

This is a *Certified Softmax Transformer*, not vanilla attention: bounded Q,K,V,
column-budgeted ``P``, controlled residuals/FFN. It is also not an SSM memory block.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lti_cells import _normalize_to_3d
from .stable_recurrent_transformer import split_gain_budget
from ..static_layers.lipschitz_mlps import L2BoundedLinearExact

_ACT_LIP = {"relu": 1.0, "tanh": 1.0, "gelu": 1.13, "silu": 1.10}


class SpectralLinear(nn.Module):
    """Bias-free linear map with spectral norm capped at ``gamma`` (declared gain)."""

    def __init__(self, d_in: int, d_out: int, gamma: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)
        self.lin = L2BoundedLinearExact(d_in, d_out, bound=self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class BoundedQKVProjection(nn.Module):
    """Produce bounded multi-head Q, K, V of shape ``(B, H, T, d_head)``."""

    def __init__(self, d_model: int, n_heads: int, d_head: Optional[int] = None,
                 qk_mode: str = "tanh", v_mode: str = "tanh",
                 q_radius: float = 1.0, k_radius: float = 1.0, v_radius: float = 1.0,
                 bound_q: float = 1.0, bound_k: float = 1.0, bound_v: float = 1.0,
                 eps: float = 1e-6):
        super().__init__()
        self.n_heads = int(n_heads)
        self.d_head = int(d_head) if d_head else max(1, d_model // n_heads)
        self.inner = self.n_heads * self.d_head
        self.qk_mode, self.v_mode = qk_mode, v_mode
        self.q_radius, self.k_radius, self.v_radius = float(q_radius), float(k_radius), float(v_radius)
        self.bound_q, self.bound_k, self.bound_v = float(bound_q), float(bound_k), float(bound_v)
        self.eps = float(eps)
        self.W_q = L2BoundedLinearExact(d_model, self.inner, bound=self.bound_q)
        self.W_k = L2BoundedLinearExact(d_model, self.inner, bound=self.bound_k)
        self.W_v = L2BoundedLinearExact(d_model, self.inner, bound=self.bound_v)

    def _bound(self, t: torch.Tensor, mode: str, R: float) -> torch.Tensor:
        if mode == "tanh":
            return R * torch.tanh(t / R)
        if mode == "l2norm":
            return R * t / (t.norm(dim=-1, keepdim=True) + self.eps)
        raise ValueError(f"Unknown bound mode {mode!r} (use 'tanh' or 'l2norm').")

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape

        def proj(W, mode, R):
            t = W(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,dh)
            return self._bound(t, mode, R)

        return (proj(self.W_q, self.qk_mode, self.q_radius),
                proj(self.W_k, self.qk_mode, self.k_radius),
                proj(self.W_v, self.v_mode, self.v_radius))


class ColumnBudgetedSoftmaxAttention(nn.Module):
    """Real softmax attention with a differentiable column budget.

    ``P = softmax(qkᵀ/(τ√d_head) + mask)``; ``P̃ = P / max(1, colsum/c_max)``; ``out = P̃ V``.
    After budgeting: row-sums ≤ 1, column-sums ≤ c_max, ``‖P̃‖₂ ≤ √c_max``.
    """

    def __init__(self, d_head: int, tau: float = 1.0, c_max: float = 1.0, causal: bool = False):
        super().__init__()
        self.d_head = int(d_head)
        self.tau = float(tau)
        self.c_max = float(c_max)
        self.causal = bool(causal)

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        T = q.shape[-2]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.tau * math.sqrt(self.d_head))
        if self.causal:
            cm = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(cm, float("-inf"))
        if attn_mask is not None:
            scores = scores + attn_mask
        P = torch.softmax(scores, dim=-1)
        if self.causal:
            # Causal column budget: query t's per-key scale uses only queries s<=t
            # (a cumulative column sum), otherwise a future token would change the
            # normalization of past outputs and break causality.
            cc = P.cumsum(dim=-2)                               # (B,H,T_query,T_key)
            P_tilde = P / torch.clamp(cc / self.c_max, min=1.0)
        else:
            col = P.sum(dim=-2, keepdim=True)                   # (B,H,1,T_key)
            P_tilde = P / torch.clamp(col / self.c_max, min=1.0)
        out = torch.matmul(P_tilde, v)
        if return_attn:
            return out, {"P": P, "P_tilde": P_tilde}
        return out


class CertifiedMHA(nn.Module):
    """Multi-head attention with bounded QKV, column budget, spectral W_O, declared gain."""

    def __init__(self, d_model: int, n_heads: int, d_head: Optional[int] = None,
                 tau: float = 1.0, c_max: float = 1.0, causal: bool = False,
                 q_radius: float = 1.0, k_radius: float = 1.0, v_radius: float = 1.0,
                 bound_q: float = 1.0, bound_k: float = 1.0, bound_v: float = 1.0,
                 bound_o: float = 1.0, gamma_mha: Optional[float] = None,
                 qk_mode: str = "tanh", v_mode: str = "tanh", max_len: int = 4096,
                 eps: float = 1e-6):
        super().__init__()
        self.qkv = BoundedQKVProjection(
            d_model, n_heads, d_head, qk_mode, v_mode,
            q_radius, k_radius, v_radius, bound_q, bound_k, bound_v, eps)
        self.attn = ColumnBudgetedSoftmaxAttention(self.qkv.d_head, tau, c_max, causal)
        self.W_o = L2BoundedLinearExact(self.qkv.inner, d_model, bound=bound_o)
        self.tau, self.c_max = float(tau), float(c_max)
        self.causal, self.max_len = bool(causal), int(max_len)
        self.bound_o = float(bound_o)
        self._gamma_mha = None if gamma_mha is None else float(gamma_mha)

    def forward(self, x, attn_mask=None, return_attn=False):
        B, T, _ = x.shape
        q, k, v = self.qkv(x)
        out = self.attn(q, k, v, attn_mask, return_attn)
        info = None
        if return_attn:
            out, info = out
        cat = out.transpose(1, 2).reshape(B, T, self.qkv.inner)
        a = self.W_o(cat)
        return (a, info) if return_attn else a

    def declared_gain(self) -> float:
        if self._gamma_mha is not None:
            return self._gamma_mha
        qkv, dh, H = self.qkv, self.qkv.d_head, self.qkv.n_heads
        q_max, k_max = qkv.q_radius * math.sqrt(dh), qkv.k_radius * math.sqrt(dh)
        l_smax = 0.5  # ||diag(p) - p pᵀ||₂ <= 1/2
        # Causal mode uses a cumulative column budget, whose max column sum is
        # c_max*(1+ln(T/c_max)); bound it for sequences up to max_len.
        c_eff = self.c_max
        if self.causal:
            c_eff = self.c_max * (1.0 + max(0.0, math.log(max(self.max_len, 1) / self.c_max)))
        value = math.sqrt(c_eff) * qkv.bound_v
        pattern = (math.sqrt(H) * (math.sqrt(2.0) * qkv.v_radius * l_smax / self.tau)
                   * math.sqrt(k_max ** 2 * qkv.bound_q ** 2
                               + c_eff * q_max ** 2 * qkv.bound_k ** 2))
        return self.bound_o * (value + pattern)


class CertifiedFFN(nn.Module):
    """Transformer FFN ``x -> W1 -> act -> W2`` with controlled Lipschitz gain."""

    def __init__(self, d_model: int, d_ff: int, activation: str = "relu",
                 bound_1: float = 1.0, bound_2: float = 1.0):
        super().__init__()
        if activation not in _ACT_LIP:
            raise ValueError(f"activation must be one of {sorted(_ACT_LIP)}, got {activation!r}.")
        self.W1 = L2BoundedLinearExact(d_model, d_ff, bound=bound_1)
        self.W2 = L2BoundedLinearExact(d_ff, d_model, bound=bound_2)
        self.activation = activation
        self.l_act = _ACT_LIP[activation]
        self.bound_1, self.bound_2 = float(bound_1), float(bound_2)
        self._act = {"relu": F.relu, "tanh": torch.tanh, "gelu": F.gelu, "silu": F.silu}[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self._act(self.W1(x)))

    def declared_gain(self) -> float:
        return self.bound_2 * self.l_act * self.bound_1


class CertifiedTransformerBlock(nn.Module):
    r"""Pre-norm-free certified block: ``x1 = ρ₁x + η₁·MHA(x)``; ``y = ρ₂x1 + η₂·FFN(x1)``.

    Gain budgeting gives ``|ρ₁|+|η₁|γ_mha ≤ γ_att_block`` and ``|ρ₂|+|η₂|γ_ffn ≤
    γ_ffn_block``, so the block's incremental ℓ2 gain ≤ ``γ_att_block·γ_ffn_block``.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 gamma_att_block: float = 1.0, gamma_ffn_block: float = 1.0,
                 gamma_mha: Optional[float] = None, gamma_ffn: Optional[float] = None,
                 q_radius: float = 1.0, k_radius: float = 1.0, v_radius: float = 1.0,
                 c_max: float = 1.0, tau: float = 1.0, causal: bool = False,
                 activation: str = "relu", use_layernorm: bool = False,
                 max_len: int = 4096, eps: float = 1e-6):
        super().__init__()
        if use_layernorm:
            import warnings
            warnings.warn(
                "use_layernorm=True is ignored: LayerNorm is not globally Lipschitz "
                "and would void the certificate. The gain budget controls stability.",
                stacklevel=2,
            )
        self.gamma_att_block = float(gamma_att_block)
        self.gamma_ffn_block = float(gamma_ffn_block)
        self.mha = CertifiedMHA(
            d_model, n_heads, tau=tau, c_max=c_max, causal=causal,
            q_radius=q_radius, k_radius=k_radius, v_radius=v_radius,
            gamma_mha=gamma_mha, max_len=max_len, eps=eps)
        self.ffn = CertifiedFFN(d_model, d_ff, activation=activation)
        self.gamma_mha = self.mha.declared_gain()
        self.gamma_ffn = float(gamma_ffn) if gamma_ffn is not None else self.ffn.declared_gain()
        # Initialize the residual gates ACTIVE: raw_eta=0 => tanh(0)=0 => eta=0 would
        # leave the attention/FFN branches off (no gradient to their weights), so the
        # block would train as a near-identity residual. tanh(1.5)≈0.9 turns them on.
        # (The gain budget |rho|+|eta|*gamma_sub <= gamma_block holds for any raw values.)
        self.raw_alpha_att = nn.Parameter(torch.tensor(0.0))
        self.raw_eta_att = nn.Parameter(torch.tensor(1.5))
        self.raw_alpha_ffn = nn.Parameter(torch.tensor(0.0))
        self.raw_eta_ffn = nn.Parameter(torch.tensor(1.5))
        self.eps = float(eps)

    def forward(self, x, attn_mask=None, return_attn=False):
        a = self.mha(x, attn_mask, return_attn)
        info = None
        if return_attn:
            a, info = a
        rho1, eta1 = split_gain_budget(self.gamma_att_block, self.gamma_mha,
                                       self.raw_alpha_att, self.raw_eta_att, self.eps)
        x1 = rho1.to(x.dtype) * x + eta1.to(x.dtype) * a
        m = self.ffn(x1)
        rho2, eta2 = split_gain_budget(self.gamma_ffn_block, self.gamma_ffn,
                                       self.raw_alpha_ffn, self.raw_eta_ffn, self.eps)
        y = rho2.to(x1.dtype) * x1 + eta2.to(x1.dtype) * m
        return (y, info) if return_attn else y

    def declared_gain(self) -> float:
        return self.gamma_att_block * self.gamma_ffn_block


class CertifiedTransformer(nn.Module):
    r"""Encoder → stack of :class:`CertifiedTransformerBlock` → decoder.

    Spectral-norm≤1 encoder/decoder and per-sub-block budgets ``γ_total^(1/(2L))``
    give an end-to-end incremental ℓ2 gain ≤ ``γ_total``.
    """

    def __init__(self, d_input: int, d_model: int, d_output: int, n_layers: int = 2,
                 n_heads: int = 4, d_ff: Optional[int] = None, gamma_total: float = 1.0,
                 c_max: float = 1.0, tau: float = 1.0, causal: bool = False,
                 q_radius: float = 1.0, k_radius: float = 1.0, v_radius: float = 1.0,
                 activation: str = "relu", max_len: int = 4096, eps: float = 1e-6):
        super().__init__()
        n_layers = max(1, int(n_layers))
        d_ff = int(d_ff) if d_ff else 2 * d_model
        self.gamma_total = float(gamma_total)
        self.encoder = L2BoundedLinearExact(int(d_input), int(d_model), bound=1.0)
        self.decoder = L2BoundedLinearExact(int(d_model), int(d_output), bound=1.0)
        sub = self.gamma_total ** (1.0 / (2 * n_layers))   # per att/ffn sub-block budget
        self.blocks = nn.ModuleList([
            CertifiedTransformerBlock(
                d_model, n_heads, d_ff, gamma_att_block=sub, gamma_ffn_block=sub,
                q_radius=q_radius, k_radius=k_radius, v_radius=v_radius,
                c_max=c_max, tau=tau, causal=causal, activation=activation,
                max_len=max_len, eps=eps)
            for _ in range(n_layers)
        ])

    def forward(self, u, attn_mask=None, return_attn=False):
        u = _normalize_to_3d(u)
        z = self.encoder(u)
        infos: List[dict] = []
        for b in self.blocks:
            z = b(z, attn_mask, return_attn)
            if return_attn:
                z, info = z
                infos.append(info)
        y = self.decoder(z)
        return (y, infos) if return_attn else y

    def declared_gain(self) -> float:
        g = 1.0
        for b in self.blocks:
            g *= b.declared_gain()
        return g  # encoder/decoder spectral norm <= 1

    @torch.no_grad()
    def diagnostics(self) -> dict:
        return {"global_gamma": self.gamma_total,
                "certified_gain_bound": self.declared_gain(),
                "gamma_mha": [float(b.gamma_mha) for b in self.blocks],
                "gamma_ffn": [float(b.gamma_ffn) for b in self.blocks]}

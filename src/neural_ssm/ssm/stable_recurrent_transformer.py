# python
# file: src/neural_ssm/ssm/stable_recurrent_transformer.py
"""Transformerized L2RU — a stable recurrent transformer block.

A transformer-*inspired* sequence block with a **length-uniform certified l2-gain
bound by construction**. It avoids full self-attention / a growing KV cache:

    m   = L2SSM_{gamma_mem}(x)             # certified L2-bounded memory (fixed state)
    a   = LipAtt_{gamma_att}(x, m)         # static, per-token Lipschitz readout
    x^+ = rho * x + eta * a               # residual with a gain budget

For two input sequences the block is contractive/bounded:

    ||X^+ - Y^+||_2 <= ( |rho| + |eta| * gamma_att * sqrt(1 + gamma_mem^2) ) ||X - Y||_2,

and ``rho, eta`` are parametrized so this is ``<= gamma_block`` for any raw values
(:func:`split_gain_budget`). Stacking ``L`` blocks between a spectral-norm<=1
encoder/decoder gives an end-to-end bound ``<= gamma_total`` when
``gamma_layer = gamma_total ** (1/L)``.

What kind of certificate is this?
---------------------------------
The bound above is on the **incremental** gain (the Lipschitz constant w.r.t. the
input). Because the block maps the zero sequence to zero, it *also* bounds the
zero-state induced gain. The incremental claim is rigorous **iff the memory core's
``gamma`` is an incremental bound**, which holds for the *linear* (LTI) L2 cores
(``l2ru``/``l2n``/``l2nt``: linear => incremental = zero-state = ``.gamma``). For
the *selective* cores (``tv``/``tvc``) ``.gamma`` is only a zero-state bound, so
the block then carries a zero-state certificate (still ``<= gamma_block``), not the
Lipschitz one. The default memory core is therefore an LTI core.

This is NOT vanilla self-attention: no prefix attention, no growing cache, a fixed
recurrent memory state, and a declared gain budget.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SSMConfig, _build_ssm_cell, _module_gain_bound, DeepSSM
from .lti_cells import _normalize_to_3d
from ..static_layers.lipschitz_mlps import L2BoundedLinearExact

# LTI cores certify an incremental (Lipschitz) gain; selective cores only zero-state.
_LTI_MEM_CORES = frozenset({"l2ru", "l2n", "l2nt", "zak"})


def split_gain_budget(
    gamma_block: float,
    gamma_sub: float,
    raw_rho: torch.Tensor,
    raw_eta: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rho, eta)`` with ``|rho| + |eta| * gamma_sub <= gamma_block``.

    Convex split of the budget between the residual (``rho``) and the readout
    branch (``eta``), via a learnable gate ``sigmoid(raw_rho)`` and a signed
    magnitude ``tanh(raw_eta)``. Holds for *any* values of the raw parameters.
    """
    alpha = torch.sigmoid(raw_rho)
    rho = gamma_block * alpha
    eta = gamma_block * (1.0 - alpha) * torch.tanh(raw_eta) / (gamma_sub + eps)
    return rho, eta


class LipschitzStaticAttention(nn.Module):
    r"""Per-token, multi-head, attention-*like* readout with a certified Lipschitz bound.

    For each time step (no interaction across time, hence streaming-trivial and
    cache-free)::

        q = tanh(W_Q x)      k = tanh(W_K m)      v = tanh(W_V m)     # per head
        score_h = (q_h . k_h) / sqrt(d_head)        # scalar per head
        head_h  = tanh(score_h) * v_h
        a_raw   = W_O( concat_h head_h )
        a       = (gamma_att / L_raw) * a_raw

    All projections are spectral-norm <= 1 (``L2BoundedLinearExact``) and ``q,k,v``
    are ``tanh``-bounded, which makes the unscaled map ``(x, m) -> a_raw`` Lipschitz
    with the provable upper bound

        L_raw = sqrt(H) * (1 + sqrt(2 * d_head)),

    derived from: ``Lip(q_h),Lip(k_h),Lip(v_h) <= 1``; ``||v_h|| <= sqrt(d_head)``;
    ``Lip(score_h) <= sqrt(2)``; ``Lip(head_h) <= 1 + sqrt(2 d_head)``; concatenation
    over ``H`` heads multiplies by ``sqrt(H)``; ``||W_O|| <= 1``. Scaling the output
    by ``gamma_att / L_raw`` makes the realized Lipschitz constant ``<= gamma_att``
    (w.r.t. the stacked input ``[x; m]``). ``a(0, 0) = 0``. The bound is conservative
    (realized constant is typically far smaller), which is safe for the certificate.
    """

    def __init__(self, d_model: int, d_mem: int, n_heads: int = 4,
                 gamma_att: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.d_model = int(d_model)
        self.d_mem = int(d_mem)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.gamma_att = float(gamma_att)
        self.eps = float(eps)

        # Bias-free spectral-norm<=1 projections (bias would break the (0,0)->0 property).
        self.W_q = L2BoundedLinearExact(self.d_model, self.d_model, bound=1.0)
        self.W_k = L2BoundedLinearExact(self.d_mem, self.d_model, bound=1.0)
        self.W_v = L2BoundedLinearExact(self.d_mem, self.d_model, bound=1.0)
        self.W_o = L2BoundedLinearExact(self.d_model, self.d_model, bound=1.0)

        # Provable Lipschitz upper bound of the unscaled readout, and the scale that
        # enforces the declared gamma_att (registered so it moves with .to()/dtype).
        l_raw = math.sqrt(self.n_heads) * (1.0 + math.sqrt(2.0 * self.d_head))
        self.register_buffer("_readout_scale", torch.tensor(self.gamma_att / l_raw))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """``x: (..., d_model)``, ``m: (..., d_mem)`` -> ``a: (..., d_model)``."""
        lead = x.shape[:-1]
        q = torch.tanh(self.W_q(x)).reshape(*lead, self.n_heads, self.d_head)
        k = torch.tanh(self.W_k(m)).reshape(*lead, self.n_heads, self.d_head)
        v = torch.tanh(self.W_v(m)).reshape(*lead, self.n_heads, self.d_head)

        score = (q * k).sum(dim=-1, keepdim=True) / math.sqrt(self.d_head)   # (..., H, 1)
        head = torch.tanh(score) * v                                        # (..., H, d_head)
        a_raw = self.W_o(head.reshape(*lead, self.d_model))
        return self._readout_scale.to(a_raw.dtype) * a_raw


class StableRecurrentTransformerBlock(nn.Module):
    r"""One certified stable-recurrent-transformer block (memory + Lip readout + budget).

    ``y = rho * x + eta * LipAtt(x, L2SSM(x))`` with
    ``|rho| + |eta| * gamma_att * sqrt(1 + gamma_mem^2) <= gamma_block``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_heads: int = 4,
        gamma_block: float = 1.0,
        gamma_mem: float = 1.0,
        gamma_att: float = 1.0,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        scan: bool = True,
        mem_param: str = "l2ru",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.gamma_block = float(gamma_block)
        self.gamma_att = float(gamma_att)
        self.scan = bool(scan)
        self.eps = float(eps)
        self.mem_param = str(mem_param)
        if use_layernorm:
            # LayerNorm is not globally Lipschitz (division by small variance), so it
            # is intentionally excluded from the certificate-critical path.
            import warnings
            warnings.warn(
                "use_layernorm=True is ignored: normalization would void the global "
                "Lipschitz certificate. The gain budget already controls stability.",
                stacklevel=2,
            )

        # Certified L2-bounded memory core (fixed gamma). Reuses the repo registry.
        cfg = SSMConfig(
            d_model=self.d_model, d_state=int(d_state), n_layers=1,
            param=self.mem_param, gamma=float(gamma_mem), train_gamma=False, learn_x0=False,
        )
        self.memory = _build_ssm_cell(cfg, float(gamma_mem))
        # The certified memory gain actually realized by the core (an upper bound).
        gamma_mem_eff = float(_module_gain_bound(
            self.memory, device=torch.device("cpu"), dtype=torch.float32))
        self.gamma_mem = gamma_mem_eff
        self.is_lti_memory = self.mem_param in _LTI_MEM_CORES

        d_mem = self.d_model  # the L2 cores map d_model -> d_model
        self.attn = LipschitzStaticAttention(
            self.d_model, d_mem, n_heads=n_heads, gamma_att=self.gamma_att, eps=self.eps)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Static budget denominator gamma_sub = gamma_att * sqrt(1 + gamma_mem^2)
        # (the readout sees both x and m).
        self.gamma_sub = self.gamma_att * math.sqrt(1.0 + self.gamma_mem ** 2)
        # Active gate init: raw_eta=0 => eta=0 leaves the readout branch off (no
        # gradient to its weights). tanh(1.5)≈0.9 turns it on. The budget
        # |rho|+|eta|*gamma_sub <= gamma_block holds for any raw values.
        self.raw_rho = nn.Parameter(torch.tensor(0.0))
        self.raw_eta = nn.Parameter(torch.tensor(1.5))

    # -- gain budget -----------------------------------------------------------
    def gains(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return split_gain_budget(self.gamma_block, self.gamma_sub,
                                 self.raw_rho, self.raw_eta, eps=self.eps)

    def gain_bound(self) -> float:
        """Conservative certified gain of this block (``<= gamma_block``)."""
        rho, eta = self.gains()
        return float(rho.abs() + eta.abs() * self.gamma_sub)

    # -- forward / streaming ---------------------------------------------------
    def _readout(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        rho, eta = self.gains()
        a = self.dropout(self.attn(x, m))
        return rho.to(x.dtype) * x + eta.to(x.dtype) * a

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None,
                return_state: bool = False):
        """``x: (B, L, d_model)`` -> ``y: (B, L, d_model)`` (+ final state if requested)."""
        x = _normalize_to_3d(x)
        m, st = self.memory(
            x, state=state, mode=("scan" if self.scan else "loop"),
            reset_state=(state is None), detach_state=False,
        )
        y = self._readout(x, m)
        if return_state:
            return y, DeepSSM._last_runtime_state(st)
        return y

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: Optional[torch.Tensor] = None):
        """One streaming step. ``x_t: (B, d_model)`` -> ``(y_t: (B, d_model), new_state)``."""
        m_seq, st = self.memory(
            x_t.unsqueeze(1), state=state, mode="loop",
            reset_state=(state is None), detach_state=True,
        )
        y_t = self._readout(x_t, m_seq[:, 0])
        return y_t, DeepSSM._last_runtime_state(st)

    def reset(self):
        if hasattr(self.memory, "reset"):
            self.memory.reset()


class StableRecurrentTransformer(nn.Module):
    r"""A stack of :class:`StableRecurrentTransformerBlock` with a certified l2 gain.

    ``y = H ( Block_L ( ... Block_1 ( E u ) ... ) )`` where the encoder ``E`` and
    decoder ``H`` are spectral-norm <= 1, and each block has gain <= ``gamma_layer
    = gamma_total ** (1/n_layers)``, so the end-to-end (incremental, hence also
    zero-state) gain is <= ``gamma_total``.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        d_state: int,
        n_layers: int = 2,
        n_heads: int = 4,
        gamma_total: float = 1.0,
        gamma_mem: float = 1.0,
        gamma_att: float = 1.0,
        dropout: float = 0.0,
        scan: bool = True,
        mem_param: str = "l2ru",
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        self.d_input, self.d_model, self.d_output = int(d_input), int(d_model), int(d_output)
        self.gamma_total = float(gamma_total)
        gamma_layer = self.gamma_total ** (1.0 / n_layers)

        # Spectral-norm<=1 encoder/decoder (linear => Lipschitz = spectral norm).
        self.encoder = L2BoundedLinearExact(self.d_input, self.d_model, bound=1.0)
        self.decoder = L2BoundedLinearExact(self.d_model, self.d_output, bound=1.0)
        self.blocks = nn.ModuleList([
            StableRecurrentTransformerBlock(
                d_model=self.d_model, d_state=int(d_state), n_heads=n_heads,
                gamma_block=gamma_layer, gamma_mem=gamma_mem, gamma_att=gamma_att,
                dropout=dropout, scan=scan, mem_param=mem_param, eps=eps,
            )
            for _ in range(int(n_layers))
        ])

    def gain_bound(self) -> float:
        """Conservative end-to-end certified gain (``<= gamma_total``)."""
        g = 1.0
        for b in self.blocks:
            g *= b.gain_bound()
        return g  # encoder/decoder contribute spectral norm <= 1

    def forward(self, u: torch.Tensor,
                state: Optional[List[Optional[torch.Tensor]]] = None,
                return_state: bool = False):
        u = _normalize_to_3d(u)
        states = list(state) if state is not None else [None] * len(self.blocks)
        x = self.encoder(u)
        new_states: List[Optional[torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            x, st = block(x, state=states[i], return_state=True)
            new_states.append(st)
        y = self.decoder(x)
        if return_state:
            return y, new_states
        return y

    @torch.no_grad()
    def step(self, u_t: torch.Tensor, state: Optional[List[Optional[torch.Tensor]]] = None):
        """One streaming step. ``u_t: (B, d_input)`` -> ``(y_t: (B, d_output), states)``."""
        states = list(state) if state is not None else [None] * len(self.blocks)
        x = self.encoder(u_t)
        new_states: List[Optional[torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            x, st = block.step(x, state=states[i])
            new_states.append(st)
        return self.decoder(x), new_states

    def reset(self):
        for b in self.blocks:
            b.reset()

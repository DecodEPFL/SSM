# python
# file: src/neural_ssm/ssm/layers.py
"""
High-level SSM layer wrappers:
  SSMConfig, SSL, DeepSSM, PureLRUR, SimpleRNN
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TypedDict, Optional, List, Union, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lti_cells import LRU, L2RU, lruz, L2BoundedLTICell, Block2x2DenseL2SSM, _normalize_to_3d
from .selective_cells import RobustMambaDiagSSM, RobustMambaDiagLTI
from .experimental import ExpertSelectiveTimeVaryingSSM, Block2x2SelectiveBCDExpertsL2SSM
from ..static_layers.generic_layers import LayerConfig, GLU, MLP
from ..static_layers.lipschitz_mlps import (
    LMLP, L2BoundedGLU, L2BoundedGLUv2, MultiBranchLipMixer,
)

try:
    from ..static_layers.lipschitz_mlps import TLIP
except ImportError:
    TLIP = None


""" Optional data class to set up the SSM model (values here are used just to initialize all fields) """


@dataclass
class SSMConfig:
    d_model: int = 10  # input/output size of the LRU after the decoding phase (n_u = n_y)
    d_state: int = 32  # state size of the LRU (n_x)
    n_layers: int = 2  # number of SSMs blocks in cascade for deep structures
    dropout: float = 0.0  # set it different from 0 if you want to introduce dropout regularization
    bias: bool = False  # bias of MLP static_layers
    rmin: float = .8  # min. magnitude of the eigenvalues at initialization in the complex parametrization
    rmax: float = .95  # max. magnitude of the eigenvalues at initialization in the complex parametrization
    max_phase: float = 2 * math.pi  # maximum phase of the eigenvalues at initialization in the complex parametrization
    ff: str = "MLP"  # non-linear static block used in the scaffolding
    scale: float = 1  # Lipschitz constant of the Lipschitz bounded MLP (LMLP)
    dim_amp: int = 4  # controls the hidden layer's dimension of the MLP
    d_hidden: int = 4  # controls the hidden layer's dimension of the non-linear layer
    nl_layers: int = 2 # number of hidden layers of the non-linear layers (static nonlinearities)
    param: Optional[str] = None  # pick the parametrization you want to use for the LRU. Default = LRU, other options are L2RU
    # and ZAK
    gamma: Optional[float] = 1.0  # overall target l2 gain for the DeepSSM. If set, DeepSSM keeps this global gain fixed
    # through encoder/decoder certificate scaling.
    train_gamma: bool = True # controls whether the per-block / per-LTI gamma parameters are trainable. This is distinct
    # from the global target gamma above, which remains fixed whenever `gamma` is not None.
    init: str = 'eye'  # controls the initialization of the parameters when the L2RU param is chosen.
    # parameters needed for the prescribed-gain parametrization
    rho: float = 0.9
    max_phase_b: float = 0.04          # small spread
    phase_center: float = 0        # center angle
    random_phase: bool = True
    offdiag_scale: float = 0.05  # init std for K12/K21/K22 in l2n (old default was 0.005)
    learn_x0: bool = False  # if True, the initial hidden state is a learnable parameter
    zak_d_margin: float = 0.5  # ZAK-only: initialize the direct term strictly inside the feasible set
    zak_x2_margin: float = 0.5  # ZAK-only: initialize the off-diagonal coupling strictly inside the feasible set
    zak_x2_init_scale: float = 0.1  # ZAK-only: scale of the free real X2 initialization
    cert_scale_temperature: float = 0.05  # smoothness of the fixed-gamma soft cap; smaller values approach
    # the hard min without breaking the guarantee.

    # Parallel scan must be selected in the forward call of the SSM.

    # Generate TypedDict automatically


SSMConfigDict = TypedDict('SSMConfigDict',
                          {f.name: f.type for f in fields(SSMConfig)},
                          total=False)

""" SSMs blocks ----------------------------------------- """


class SSL(nn.Module):
    """State Space Layer: gated residual
    y = x + alpha * dropout(FF(LRU(x)))
    """
    def __init__(self, config: SSMConfig):
        super().__init__()
        res_logit_init = -1.0
        init_block_gamma = None
        if config.gamma is not None and config.n_layers > 0:
            # Use the exact per-block initialization requested by the user for
            # trainable LTI gammas. The global fixed gamma is now enforced as a
            # cap in DeepSSM.forward, so we do not need to inflate the branch
            # gamma to compensate for the residual gate at initialization.
            init_block_gamma = max(float(config.gamma) ** (1.0 / config.n_layers) - 1.0, 0.01)

        fixed_branch_gamma = init_block_gamma if init_block_gamma is not None else config.gamma
        if config.train_gamma:
            fixed_branch_gamma = None

        # --- SSM selection ---
        if config.param is None or config.param == "lru":
            self.lru = LRU(
                in_features=config.d_model,
                out_features=config.d_model,
                state_features=config.d_state,
                rmin=config.rmin,
                rmax=config.rmax,
                max_phase=config.max_phase,
                learn_x0=config.learn_x0,
            )
        elif config.param == "l2ru":
            self.lru = L2RU(
                state_features=config.d_model,
                gamma=fixed_branch_gamma,
                init=config.init,
                learn_x0=config.learn_x0,
            )
        elif config.param == "zak":
            self.lru = lruz(
                input_features=config.d_model,
                output_features=config.d_model,
                state_features=config.d_state,
                rmin=config.rmin,
                rmax=config.rmax,
                max_phase=config.max_phase,
                gamma=fixed_branch_gamma,
                d_margin=config.zak_d_margin,
                x2_margin=config.zak_x2_margin,
                x2_init_scale=config.zak_x2_init_scale,
                init=config.init,
                learn_x0=config.learn_x0,
            )
        elif config.param == "l2n":
            block_gamma = init_block_gamma if init_block_gamma is not None else 1.0
            self.lru = Block2x2DenseL2SSM(
                d_state=config.d_state,
                d_input=config.d_model,
                d_output=config.d_model,
                gamma=block_gamma,
                train_gamma=config.train_gamma,
                learn_x0=config.learn_x0,
            )
            self.lru.init_on_circle(
                rho=config.rho,
                max_phase=config.max_phase_b,
                phase_center=config.phase_center,
                random_phase=config.random_phase,
                offdiag_scale=config.offdiag_scale,
            )
        elif config.param == "l2nt":
            block_gamma = init_block_gamma if init_block_gamma is not None else 1.0
            self.lru = L2BoundedLTICell(
                d_state=config.d_state,
                d_input=config.d_model,
                d_output=config.d_model,
                gamma=block_gamma,
                train_gamma=config.train_gamma,
                learn_x0=config.learn_x0,
            )
        elif config.param == "tv":
            block_gamma = init_block_gamma if init_block_gamma is not None else 1.0
            self.lru = RobustMambaDiagSSM(
                d_state=config.d_state,
                d_model=config.d_model,
                d_out=config.d_model,
                gamma=block_gamma,
                train_gamma=config.train_gamma,
                learn_x0=config.learn_x0,
            )
        elif config.param == "tvc":
            block_gamma = init_block_gamma if init_block_gamma is not None else 1.0
            self.lru = RobustMambaDiagLTI(
                d_state=config.d_state,
                d_model=config.d_model,
                d_out=config.d_model,
                gamma=block_gamma,
                train_gamma=config.train_gamma,
                param_net="mlp",
                hidden=max(64, 2 * config.d_model),
                init_rho=config.rho,
                init_sign=0.995,
                init_b=0.10,
                init_c=0.10,
                init_d=0.10,
                bcd_nonlinearity="tanh",
                output_uses_post_state=False,  # keeps the exact simple certificate
                learn_x0=config.learn_x0,
            )
        else:
            raise ValueError("Invalid parametrization")

        if config.train_gamma and init_block_gamma is not None:
            with torch.no_grad():
                target_gamma = float(init_block_gamma)
                gamma_attr = getattr(self.lru, "gamma", None)
                if hasattr(self.lru, "gamma_raw") and isinstance(self.lru.gamma_raw, nn.Parameter):
                    target = torch.as_tensor(
                        target_gamma,
                        device=self.lru.gamma_raw.device,
                        dtype=self.lru.gamma_raw.dtype,
                    ).clamp_min(1e-6)
                    self.lru.gamma_raw.copy_(torch.log(torch.expm1(target)))
                elif hasattr(self.lru, "log_gamma") and isinstance(self.lru.log_gamma, nn.Parameter):
                    self.lru.log_gamma.fill_(math.log(max(target_gamma, 1e-8)))
                elif isinstance(gamma_attr, nn.Parameter):
                    gamma_attr.fill_(target_gamma)

        # --- FF selection ---
        l_config = LayerConfig()
        l_config.d_input = config.d_model
        l_config.d_output = config.d_model
        l_config.d_hidden = config.d_hidden
        l_config.n_layers = config.nl_layers
        l_config.lip = config.scale

        ff_layers = {
            "GLU": GLU,
            "MLP": MLP,
            "LGLU": L2BoundedGLU,
            "LGLU2": L2BoundedGLUv2,
            "LMLP": LMLP,
            "MBLIP": MultiBranchLipMixer,
        }
        if TLIP is not None:
            ff_layers["TLIP"] = TLIP
        if config.ff not in ff_layers:
            raise ValueError(f"Unknown feedforward type: {config.ff}")
        self.ff = ff_layers[config.ff](l_config)

        self.dropout = nn.Dropout(config.dropout)

        # Small residual gate at init: sigmoid(-1) ≈ 0.269
        self.res_logit = nn.Parameter(torch.tensor(res_logit_init))

    @property
    def res_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.res_logit)

    def forward(
        self,
        x3d: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mode: str = "loop",
        reset_state: bool = True,
        detach_state: bool = False,
    ):
        z, st = self.lru(
            x3d,
            state=state,
            mode=mode,
            reset_state=reset_state,
            detach_state=detach_state,
        )
        z = self.ff(z)
        z = self.dropout(z)
        return x3d + self.res_scale * z, st


class DeepSSM(nn.Module):
    """Deep SSM: normalized encoder -> n gated residual blocks -> normalized decoder."""
    def __init__(
        self,
        d_input: int,
        d_output: int,
        *,
        d_model: int = 10,
        d_state: int = 32,
        n_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
        rmin: float = .8,
        rmax: float = .95,
        max_phase: float = 2 * math.pi,
        ff: str = "LGLU2",
        scale: float = 1,
        dim_amp: int = 4,
        d_hidden: int = 4,
        nl_layers: int = 3,
        param: Optional[str] = "lru",
        gamma: Optional[float] = None,
        train_gamma: Optional[bool] = True,
        init: str = "eye",
        rho: float = 0.9,
        max_phase_b: float = 0.5,
        phase_center: float = 0,
        random_phase: bool = True,
        learn_x0: bool = False,
        config: Optional[SSMConfig] = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        self.config = config if config is not None else SSMConfig(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            bias=bias,
            rmin=rmin,
            rmax=rmax,
            max_phase=max_phase,
            ff=ff,
            scale=scale,
            dim_amp=dim_amp,
            d_hidden=d_hidden,
            nl_layers=nl_layers,
            param=param,
            gamma=gamma,
            train_gamma=train_gamma,
            init=init,
            rho=rho,
            max_phase_b=max_phase_b,
            phase_center=phase_center,
            random_phase=random_phase,
            learn_x0=learn_x0,
        )

        self.use_cert_scaling = (self.config.param != "lru") and (self.config.gamma is not None)

        if self.use_cert_scaling:
            self.register_buffer("gamma_t", torch.tensor(float(self.config.gamma)))

            # Balanced near-isometric init
            self.encoder_w = nn.Parameter(torch.empty(self.config.d_model, self.d_input))
            self.decoder_w = nn.Parameter(torch.empty(self.d_output, self.config.d_model))
            with torch.no_grad():
                nn.init.orthogonal_(self.encoder_w)
                nn.init.orthogonal_(self.decoder_w)

            self.ff_has_lip = False
        else:
            self.encoder = nn.Linear(d_input, self.config.d_model, bias=False)
            self.decoder = nn.Linear(self.config.d_model, d_output, bias=False)

        self.blocks = nn.ModuleList([SSL(self.config) for _ in range(self.config.n_layers)])

        if self.use_cert_scaling and len(self.blocks) > 0:
            self.ff_has_lip = hasattr(self.blocks[0].ff, "lip")

    @torch.no_grad()
    def conservative_gamma_product(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Conservative reporting-only bound:
            prod_k (1 + g_k)
        ignoring the residual gate alpha_k.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        if len(self.blocks) == 0:
            return torch.ones((), device=device, dtype=dtype)

        g = torch.stack([
            block.lru.gamma.abs().to(device=device, dtype=dtype)
            for block in self.blocks
        ])

        if self.ff_has_lip:
            ff_lips = torch.stack([
                block.ff.lip.to(device=device, dtype=dtype)
                for block in self.blocks
            ])
            g = g * ff_lips

        return torch.exp(torch.log1p(g).sum())

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]] = None,
        gamma=None,
        mode: str = "scan",
        reset_state: bool = True,
        detach_state: bool = False,
    ):
        u3d = _normalize_to_3d(u)
        if reset_state:
            self.reset()

        n_blocks = len(self.blocks)
        if state is None:
            layer_states: List[Optional[torch.Tensor]] = [None] * n_blocks
        elif isinstance(state, (list, tuple)):
            if len(state) != n_blocks:
                raise ValueError(
                    f"state must provide exactly one entry per SSL block: "
                    f"expected {n_blocks}, got {len(state)}"
                )
            layer_states = list(state)
        else:
            # Convenience path: broadcast one state tensor to every block.
            layer_states = [state] * n_blocks

        # Encode
        if self.use_cert_scaling:
            enc_norm = torch.linalg.matrix_norm(self.encoder_w, ord=2).clamp_min(1e-12)
            dec_norm = torch.linalg.matrix_norm(self.decoder_w, ord=2).clamp_min(1e-12)

            encoder_eff = self.encoder_w / enc_norm
            decoder_eff = self.decoder_w / dec_norm

            x = F.linear(u3d, encoder_eff, bias=None)
        else:
            x = self.encoder(u3d)

        # Blocks
        for i, block in enumerate(self.blocks):
            x, st = block(
                x,
                state=layer_states[i],
                mode=mode,
                reset_state=reset_state,
                detach_state=detach_state,
            )
            layer_states[i] = st[:, -1, :]

        # Decode
        if self.use_cert_scaling:
            gamma_t = (
                self.gamma_t.abs()
                if gamma is None
                else torch.as_tensor(gamma, device=x.device, dtype=x.dtype).abs()
            )

            if len(self.blocks) > 0:
                # Residual branch gain g_k
                g = torch.stack([
                    block.lru.gamma.abs().to(device=x.device, dtype=x.dtype)
                    for block in self.blocks
                ])

                if self.ff_has_lip:
                    ff_lips = torch.stack([
                        block.ff.lip.to(device=x.device, dtype=x.dtype)
                        for block in self.blocks
                    ])
                    g = g * ff_lips

                # Dropout factor during training: Lip(dropout) <= 1/(1-p)
                if self.training:
                    drop_factors = torch.stack([
                        torch.as_tensor(
                            1.0 / max(1.0 - float(block.dropout.p), 1e-12),
                            device=x.device,
                            dtype=x.dtype,
                        )
                        for block in self.blocks
                    ])
                else:
                    drop_factors = torch.ones(len(self.blocks), device=x.device, dtype=x.dtype)

                g = g * drop_factors

                # Tighter valid per-layer factor for y = x + alpha * z:
                #   ||block|| <= 1 + alpha_k * g_k
                alphas = torch.stack([
                    block.res_scale.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
                    for block in self.blocks
                ])

                g_eff = (1.0 + alphas * g).clamp_min(1e-12)
                log_gamma_prod = torch.log(g_eff).sum()
                gamma_prod = torch.exp(log_gamma_prod)
            else:
                gamma_prod = torch.ones((), device=x.device, dtype=x.dtype)

            trainable_global_gamma = (
                gamma is None
                and isinstance(getattr(self, "gamma_t", None), nn.Parameter)
                and self.gamma_t.requires_grad
            )
            if trainable_global_gamma:
                scale = gamma_t / (gamma_prod + 1e-12)
            else:
                # Treat fixed gamma as a smooth upper bound. The log-sum-exp cap is
                # always below the hard admissible scale min(1, gamma_t / gamma_prod),
                # so the guarantee is preserved while gradients stay nonzero near the boundary.
                scale = self._smooth_capped_scale(
                    gamma_t=gamma_t,
                    gamma_prod=gamma_prod,
                    temperature=self.config.cert_scale_temperature,
                )
            outputs = F.linear(x, decoder_eff * scale, bias=None)
        else:
            outputs = self.decoder(x)

        return outputs, layer_states

    @staticmethod
    def _smooth_capped_scale(
        gamma_t: torch.Tensor,
        gamma_prod: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Smooth approximation of min(1, gamma_t / gamma_prod) that preserves the
        hard guarantee by replacing max(0, log(gamma_prod / gamma_t)) with a
        log-sum-exp / softplus upper bound.
        """
        eps = 1e-12
        tau = max(float(temperature), 1e-6)
        log_ratio = torch.log(gamma_prod + eps) - torch.log(gamma_t + eps)
        smooth_log_cap = tau * F.softplus(log_ratio / tau)
        return torch.exp(-smooth_log_cap)

    def reset(self):
        for block in self.blocks:
            block.lru.reset()


# Pure LRU blocks -----------------------------------------------

# python
class PureLRUR(nn.Module):
    """Pure LRU block without scaffolding."""

    def __init__(self, n: int, gamma: float = None, param: str = "l2ru", init: str = "eye", learn_x0: bool = False):
        super().__init__()
        if param == "l2ru":
            self.lru = L2RU(state_features=n, gamma=gamma, init=init, learn_x0=learn_x0)
        elif param == "lru":
            self.lru = LRU(in_features=n, out_features=n, state_features=n, learn_x0=learn_x0)
        elif param == "zak":
            self.lru = lruz(
                input_features=n,
                output_features=n,
                state_features=n,
                gamma=gamma,
                init=init,
                learn_x0=learn_x0,
            )
        else:
            raise ValueError("Unsupported param")

    def forward(
            self,
            x: torch.Tensor,
            state: Optional[torch.Tensor] = None,
            mode: str = "scan",
            reset_state: bool = True,
            detach_state: bool = True,
    ):
        y, st = self.lru(
            _normalize_to_3d(x),
            state=state,
            mode=mode,
            reset_state=reset_state,
            detach_state=detach_state,
        )
        return y, st

    def reset(self):
        self.lru.reset()


class SimpleRNN(nn.Module):
    """
    Thin wrapper around nn.RNN that first normalizes the input to 3D
    using the same convention as DeepSSM.

    Input:
      u: (T, d_input) or (B, T, d_input)

    State:
      state (h0): (d_hidden,) or (B, d_hidden) or (1, B, d_hidden)

    Returns (batch-first):
      y: (B, T, d_output)
      h_seq (optional): (B, T+1, d_hidden) = [h0, h1, ..., hT]
      h_last (optional): (B, d_hidden)     = hT
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        *,
        num_layers: int = 1,
        nonlinearity: str = "tanh",  # "tanh" or "relu"
        bias: bool = True,
        dropout: float = 0.0,        # only applied if num_layers > 1 (PyTorch behavior)
        bidirectional: bool = False, # for simplicity, we keep return shapes as-is
        learn_x0: bool = False,
    ):
        super().__init__()
        self.d_input = int(d_input)
        self.d_hidden = int(d_hidden)
        self.d_output = int(d_output)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        self.rnn = nn.RNN(
            input_size=self.d_input,
            hidden_size=self.d_hidden,
            num_layers=self.num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )

        # Project RNN outputs to d_output
        self.out_proj = nn.Linear(self.d_hidden * self.num_directions, self.d_output, bias=bias)
        self.state: Optional[torch.Tensor] = None

        # Learnable initial hidden state: shape (num_layers * num_directions, 1, d_hidden)
        L = self.num_layers * self.num_directions
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(L, 1, self.d_hidden))
        else:
            self.register_buffer('x0_param', None)

    def _format_h0(self, h0: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        """
        nn.RNN expects h0: (num_layers * num_directions, B, d_hidden)
        Accept:
          - None
          - (d_hidden,)
          - (B, d_hidden)
          - (1, B, d_hidden)  [for convenience if user already has RNN shape]
          - (L, B, d_hidden)  [full shape]
        """
        L = self.num_layers * self.num_directions

        if h0 is None:
            return torch.zeros(L, B, self.d_hidden, device=device, dtype=dtype)

        if h0.dim() == 1:
            h0 = h0.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        elif h0.dim() == 2:
            h0 = h0.unsqueeze(0)               # (1,B,H)
        elif h0.dim() == 3:
            pass
        else:
            raise ValueError(f"h0 must have dim 1,2,3; got shape {tuple(h0.shape)}")

        # Broadcast batch if needed
        if h0.size(1) == 1 and B > 1:
            h0 = h0.expand(h0.size(0), B, h0.size(2))

        # Broadcast layers if needed
        if h0.size(0) == 1 and L > 1:
            h0 = h0.expand(L, h0.size(1), h0.size(2))

        if h0.shape != (L, B, self.d_hidden):
            raise ValueError(f"h0 has shape {tuple(h0.shape)}, expected {(L, B, self.d_hidden)}")

        return h0.to(device=device, dtype=dtype)

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,  # h0
        *,
        return_state: bool = False,
        return_last: bool = False,
        reset_state: bool = True,
        detach_state: bool = True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor],
    ]:
        u3d = _normalize_to_3d(u)  # (B,T,D)
        B, T, D = u3d.shape
        if D != self.d_input:
            raise ValueError(f"Expected input dim {self.d_input}, got {D}")

        source_state = state if state is not None else self.state
        if reset_state:
            source_state = self.x0_param  # None when learn_x0=False → zeros
        h0 = self._format_h0(source_state, B, u3d.device, u3d.dtype)

        # Run RNN
        out, hT = self.rnn(u3d, h0)  # out: (B,T,H*num_dir), hT: (L,B,H)
        self.state = hT.detach() if detach_state else hT

        # Project to output dim
        y = self.out_proj(out)       # (B,T,d_output)

        # Build full hidden trajectory if requested: [h0, h1, ..., hT]
        # nn.RNN doesn't return all hidden states per step directly,
        # but `out` *is* the last-layer hidden state at each time.
        # For multi-layer RNNs, this corresponds to the top layer only.
        h_seq = None
        if return_state:
            # top-layer h_t sequence is out; prepend the top-layer initial state
            top_layer_idx = self.num_directions * (self.num_layers - 1)
            h0_top = h0[top_layer_idx: top_layer_idx + self.num_directions]  # (num_dir,B,H)
            # If bidirectional, top "initial" is two directions; we pack them consistently
            # by taking the forward direction state as "the" h0 for the sequence.
            h0_seq = h0_top[0].transpose(0, 0)  # (B,H) no-op, explicit
            h_seq = torch.empty(B, T + 1, out.size(-1), device=u3d.device, dtype=u3d.dtype)
            h_seq[:, 0, :] = torch.cat([h0_top[d] for d in range(self.num_directions)], dim=-1) if self.num_directions > 1 else h0_top[0]
            h_seq[:, 1:, :] = out

        # last hidden (top layer)
        h_last = None
        if return_last:
            top_layer = hT[-self.num_directions:]  # (num_dir,B,H)
            h_last = torch.cat([top_layer[d] for d in range(self.num_directions)], dim=-1) if self.num_directions > 1 else top_layer[0]

        if return_state and return_last:
            return y, h_seq, h_last
        if return_state:
            return y, h_seq
        if return_last:
            return y, h_last
        return y, self.state

    def reset(self):
        from .state_utils import reset_runtime_state as _reset_runtime_state
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)

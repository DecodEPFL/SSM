# python
# file: src/neural_ssm/ssm/layers.py
"""High-level deep state-space models and lightweight recurrent baselines.

The main :class:`DeepSSM` stack uses two residual updates per block:

    x <- x + alpha_ssm * SSM(x)
    x <- x + alpha_ff  * FF(x)

Keeping the temporal and channel-mixing branches separate improves gradient
flow and lets each branch learn its own contribution. When ``gamma`` is set,
the implementation certifies the zero-state induced L2 gain using the block
bound

    (1 + alpha_ssm * gamma_ssm) * (1 + alpha_ff * lip_ff),

with the appropriate dropout factors during training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lti_cells import (
    Block2x2DenseL2SSM,
    L2BoundedLTICell,
    L2RU,
    LRU,
    _normalize_to_3d,
    lruz,
)
from .selective_cells import L2SelectiveRavenCell, RobustMambaDiagSSM, RobustMambaDiagLTI
from ..static_layers.generic_layers import GLU, MLP, LayerConfig
from ..static_layers.lipschitz_mlps import (
    BudgetedL2BoundedGLUv2,
    L2BoundedGLU,
    L2BoundedGLUv2,
    LMLP,
    MultiBranchLipMixer,
)

try:
    from ..static_layers.lipschitz_mlps import TLIP
except ImportError:
    TLIP = None


_CERTIFIED_FEEDFORWARDS = frozenset(
    {"LGLU2", "BLGLU2", "BudgetedLGLU2", "MBLIP", "TLIP"}
)


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
    gamma: Optional[float] = None  # prescribed upper bound on the zero-state l2 gain. None disables the global cap.
    train_gamma: bool = True # controls whether the per-block / per-LTI gamma parameters are trainable. This is distinct
    # from the global target gamma above, which remains fixed whenever `gamma` is not None.
    train_ff_lip: Optional[bool] = True  # if None, freeze FF Lipschitz scaling when using a fixed, non-trainable gamma
    # and leave it trainable otherwise.
    init: str = 'eye'  # controls the initialization of the parameters when the L2RU param is chosen.
    l2ru_eye_scale: float = 0.01
    l2ru_rand_scale: float = 1.0
    # L2N initialization
    rho: float = 0.9
    max_phase_b: float = 0.04          # small spread
    phase_center: float = 0        # center angle
    random_phase: bool = True
    offdiag_scale: float = 0.05  # init std for K12/K21/K22 in l2n (old default was 0.005)
    # Selective TV initialization
    tv_init_rho: float = 0.99
    tv_init_delta0: float = 1.0
    tv_init_param_scale: float = 0.02
    # Selective LTI TVC initialization
    tvc_init_rho: float = 0.9
    tvc_init_delta0: float = 1.0
    tvc_init_param_scale: float = 0.02
    tvc_init_sign: float = 0.995
    tvc_init_b: float = 0.10
    tvc_init_c: float = 0.10
    tvc_init_d: float = 0.10
    learn_x0: bool = False  # if True, the initial hidden state is a learnable parameter
    use_cuda_graph: bool = False  # tv/tvc only: replay the diagonal scan from a captured CUDA
    # graph instead of eager dispatch (same maths; removes launch overhead for fixed-shape runs)
    zak_d_margin: float = 0.5  # ZAK-only: initialize the direct term strictly inside the feasible set
    zak_x2_margin: float = 0.5  # ZAK-only: initialize the off-diagonal coupling strictly inside the feasible set
    zak_x2_init_scale: float = 0.1  # ZAK-only: scale of the free real X2 initialization
    cert_scale_temperature: float = 0.05  # smoothness of the fixed-gamma soft cap; smaller values approach
    # the hard min without breaking the guarantee.
    ssm_residual_init: float = -1.0  # logit of the SSM residual gate
    ff_residual_init: float = -1.0  # logit of the FF residual gate
    per_channel_gates: bool = False  # if True, the residual gates are per-channel (d_model)
    # vectors instead of scalars; the certificate uses the worst-channel gate (max).

    # Raven selective slot-memory cell (param="raven"). Tune these by passing an
    # SSMConfig directly; DeepSSM(...) keyword construction uses these defaults.
    raven_heads: int = 4  # number of attention heads (H)
    raven_slots: int = 16  # number of key/value memory slots (M)
    raven_key_dim: int = 16  # per-head key/query dimension (d_k)
    raven_value_dim: int = 16  # per-head value dimension (d_v)
    raven_top_k: int = 4  # router keeps the top-K slots per token (1 <= K <= M)
    raven_alpha: float = 1.0  # router normalization; larger => smaller writes / gain budget
    raven_rho_max: float = 0.999  # hard cap on the slot decay rho in (0, rho_max)
    raven_gamma_skip: float = 0.0  # gain budget reserved for the optional direct skip D
    raven_use_skip: bool = False  # include the spectrally-capped direct term D z_t

    # Parallel scan must be selected in the forward call of the SSM.

    # Generate TypedDict automatically


SSMConfigDict = TypedDict('SSMConfigDict',
                          {f.name: f.type for f in fields(SSMConfig)},
                          total=False)

"""SSM block construction and gain helpers."""


def _sigmoid_scalar(value: float) -> float:
    """Numerically stable scalar sigmoid used during module construction."""
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _initial_block_gamma(config: SSMConfig) -> Optional[float]:
    """Choose a useful recurrent-branch gain at initialization.

    The global decoder cap is responsible for the hard end-to-end guarantee.
    This initializer only tries to avoid starting with an unnecessarily large
    decoder attenuation. It accounts for the initial FF residual factor.
    """
    if config.gamma is None or config.n_layers <= 0:
        return None

    alpha_ssm = _sigmoid_scalar(config.ssm_residual_init)
    alpha_ff = _sigmoid_scalar(config.ff_residual_init)
    per_block_target = math.exp(math.log(float(config.gamma)) / config.n_layers)
    ff_factor = 1.0 + alpha_ff * float(config.scale)
    recurrent_factor = per_block_target / ff_factor
    gamma = (recurrent_factor - 1.0) / max(alpha_ssm, 1e-8)
    return max(gamma, 1e-2)


@torch.no_grad()
def _set_cell_gamma(cell: nn.Module, target_gamma: float) -> None:
    """Initialize the different positive-gamma parametrizations consistently."""
    target = max(float(target_gamma), 1e-8)
    gamma_raw = getattr(cell, "gamma_raw", None)
    log_gamma = getattr(cell, "log_gamma", None)
    gamma = getattr(cell, "gamma", None)

    if isinstance(gamma_raw, nn.Parameter):
        value = torch.as_tensor(target, device=gamma_raw.device, dtype=gamma_raw.dtype)
        value = value.clamp_min(1e-6)
        # Stable inverse softplus: x + log(1 - exp(-x)).
        gamma_raw.copy_(value + torch.log(-torch.expm1(-value)))
    elif isinstance(log_gamma, nn.Parameter):
        log_gamma.fill_(math.log(target))
    elif isinstance(gamma, nn.Parameter):
        gamma.fill_(target)


CellFactory = Callable[[SSMConfig, Optional[float]], nn.Module]


@dataclass(frozen=True)
class SSMParametrization:
    """Construction metadata for one recurrent-cell parametrization.

    ``certified`` means the produced cell exposes a finite zero-state L2-gain
    bound through either ``gain_bound()`` or the legacy ``.gamma`` attribute.
    """

    name: str
    factory: CellFactory
    certified: bool


def _fixed_gamma(config: SSMConfig, block_gamma: Optional[float]) -> Optional[float]:
    """Return the fixed per-cell gamma used by legacy fixed-gamma cells."""
    return None if config.train_gamma else block_gamma


def _gamma_init(block_gamma: Optional[float]) -> float:
    """Default positive gamma for cells that own their trainability flag."""
    return 1.0 if block_gamma is None else float(block_gamma)


def _build_lru_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    return LRU(
        in_features=config.d_model,
        out_features=config.d_model,
        state_features=config.d_state,
        rmin=config.rmin,
        rmax=config.rmax,
        max_phase=config.max_phase,
        learn_x0=config.learn_x0,
    )


def _build_l2ru_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    return L2RU(
        state_features=config.d_model,
        gamma=_fixed_gamma(config, block_gamma),
        init=config.init,
        eye_scale=config.l2ru_eye_scale,
        rand_scale=config.l2ru_rand_scale,
        learn_x0=config.learn_x0,
    )


def _build_zak_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    return lruz(
        input_features=config.d_model,
        output_features=config.d_model,
        state_features=config.d_state,
        rmin=config.rmin,
        rmax=config.rmax,
        max_phase=config.max_phase,
        gamma=_fixed_gamma(config, block_gamma),
        d_margin=config.zak_d_margin,
        x2_margin=config.zak_x2_margin,
        x2_init_scale=config.zak_x2_init_scale,
        init=config.init,
        learn_x0=config.learn_x0,
    )


def _build_l2n_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    cell = Block2x2DenseL2SSM(
        d_state=config.d_state,
        d_input=config.d_model,
        d_output=config.d_model,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        learn_x0=config.learn_x0,
    )
    cell.init_on_circle(
        rho=config.rho,
        max_phase=config.max_phase_b,
        phase_center=config.phase_center,
        random_phase=config.random_phase,
        offdiag_scale=config.offdiag_scale,
    )
    return cell


def _build_l2nt_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    return L2BoundedLTICell(
        d_state=config.d_state,
        d_input=config.d_model,
        d_output=config.d_model,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        learn_x0=config.learn_x0,
    )


def _build_tv_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    return RobustMambaDiagSSM(
        d_state=config.d_state,
        d_model=config.d_model,
        d_out=config.d_model,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        init_rho=config.tv_init_rho,
        init_delta0=config.tv_init_delta0,
        init_param_scale=config.tv_init_param_scale,
        learn_x0=config.learn_x0,
        use_cuda_graph=config.use_cuda_graph,
    )


def _build_tvc_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    return RobustMambaDiagLTI(
        d_state=config.d_state,
        d_model=config.d_model,
        d_out=config.d_model,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        param_net="mlp",
        hidden=max(64, 2 * config.d_model),
        init_rho=config.tvc_init_rho,
        init_delta0=config.tvc_init_delta0,
        init_param_scale=config.tvc_init_param_scale,
        init_sign=config.tvc_init_sign,
        init_b=config.tvc_init_b,
        init_c=config.tvc_init_c,
        init_d=config.tvc_init_d,
        bcd_nonlinearity="tanh",
        output_uses_post_state=False,
        learn_x0=config.learn_x0,
        use_cuda_graph=config.use_cuda_graph,
    )


def _build_raven_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    return L2SelectiveRavenCell(
        d_model=config.d_model,
        num_heads=config.raven_heads,
        num_slots=config.raven_slots,
        key_dim=config.raven_key_dim,
        value_dim=config.raven_value_dim,
        top_k=config.raven_top_k,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        gamma_skip=config.raven_gamma_skip,
        alpha=config.raven_alpha,
        rho_max=config.raven_rho_max,
        use_skip=config.raven_use_skip,
        learn_x0=config.learn_x0,
    )


_SSM_PARAMETRIZATIONS: dict[str, SSMParametrization] = {
    "lru": SSMParametrization("lru", _build_lru_cell, certified=False),
    "l2ru": SSMParametrization("l2ru", _build_l2ru_cell, certified=True),
    "zak": SSMParametrization("zak", _build_zak_cell, certified=True),
    "l2n": SSMParametrization("l2n", _build_l2n_cell, certified=True),
    "l2nt": SSMParametrization("l2nt", _build_l2nt_cell, certified=True),
    "tv": SSMParametrization("tv", _build_tv_cell, certified=True),
    "tvc": SSMParametrization("tvc", _build_tvc_cell, certified=True),
    "raven": SSMParametrization("raven", _build_raven_cell, certified=True),
}
_CERTIFIED_PARAMETRIZATIONS = frozenset(
    name for name, spec in _SSM_PARAMETRIZATIONS.items() if spec.certified
)


def _param_name(param: Optional[str]) -> str:
    return "lru" if param is None else str(param)


def _get_ssm_parametrization(param: Optional[str]) -> SSMParametrization:
    name = _param_name(param)
    try:
        return _SSM_PARAMETRIZATIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SSM_PARAMETRIZATIONS))
        raise ValueError(
            f"Unknown SSM parametrization: {param!r}. Available: {{{available}}}."
        ) from exc


def _has_gain_contract(module: nn.Module) -> bool:
    return callable(getattr(module, "gain_bound", None)) or hasattr(module, "gamma")


def _module_gain_bound(
    module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read a module's scalar gain contract.

    New cells can implement ``gain_bound()``. Existing cells continue to work
    through their ``.gamma`` property/parameter.
    """
    if callable(getattr(module, "gain_bound", None)):
        bound = module.gain_bound()
    else:
        bound = getattr(module, "gamma", None)
    if bound is None:
        return torch.full((), float("inf"), device=device, dtype=dtype)
    return torch.as_tensor(bound, device=device, dtype=dtype).abs()


def _module_lip_bound(
    module: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read a static branch's Lipschitz contract.

    ``lip_bound()`` is the preferred future hook; ``.lip`` keeps existing
    feedforward modules compatible.
    """
    if callable(getattr(module, "lip_bound", None)):
        bound = module.lip_bound()
    else:
        bound = getattr(module, "lip", None)
    if bound is None:
        return torch.full((), float("inf"), device=device, dtype=dtype)
    return torch.as_tensor(bound, device=device, dtype=dtype).abs()


def _build_ssm_cell(config: SSMConfig, block_gamma: Optional[float]) -> nn.Module:
    """Build one recurrent cell from the registry and initialize its gain."""
    cell = _get_ssm_parametrization(config.param).factory(config, block_gamma)
    if config.train_gamma and block_gamma is not None:
        _set_cell_gamma(cell, block_gamma)
    return cell


def _build_feedforward(config: SSMConfig) -> nn.Module:
    """Build the instantaneous channel-mixing branch."""
    layer_config = LayerConfig(
        d_input=config.d_model,
        d_output=config.d_model,
        d_hidden=config.d_hidden,
        n_layers=config.nl_layers,
        lip=config.scale,
        train_lip=(
            not (config.gamma is not None and not config.train_gamma)
            if config.train_ff_lip is None
            else bool(config.train_ff_lip)
        ),
    )
    builders = {
        "GLU": GLU,
        "MLP": MLP,
        "LGLU": L2BoundedGLU,
        "LGLU2": L2BoundedGLUv2,
        "BLGLU2": BudgetedL2BoundedGLUv2,
        "BudgetedLGLU2": BudgetedL2BoundedGLUv2,
        "LMLP": LMLP,
        "MBLIP": MultiBranchLipMixer,
    }
    if TLIP is not None:
        builders["TLIP"] = TLIP
    try:
        return builders[config.ff](layer_config)
    except KeyError as exc:
        raise ValueError(f"Unknown feedforward type: {config.ff!r}.") from exc


class SSL(nn.Module):
    """State-space block with separate temporal and feedforward residuals.

    The SSM branch first updates the representation using temporal context. The
    FF branch then mixes channels at every time step:

        x1 = x  + alpha_ssm * dropout_ssm(SSM(x))
        y  = x1 + alpha_ff  * dropout_ff(FF(x1))

    The scalar or per-channel gates are in ``(0, 1)``. In certified mode the
    block bound uses the largest gate in each branch:
    ``(1 + max(alpha_ssm)*gamma_ssm)*(1 + max(alpha_ff)*lip_ff)``.
    """

    def __init__(self, config: SSMConfig):
        super().__init__()
        block_gamma = _initial_block_gamma(config)
        self.lru = _build_ssm_cell(config, block_gamma)
        self.ff = _build_feedforward(config)

        self.ssm_dropout = nn.Dropout(config.dropout)
        self.ff_dropout = nn.Dropout(config.dropout)
        # Per-channel residual gates (vectors of size d_model) let some channels
        # emphasize memory and others feedforward; scalars recover the old behavior.
        # The certificate uses the worst-channel gate, so it stays valid either way.
        self.per_channel_gates = bool(getattr(config, "per_channel_gates", False))
        if self.per_channel_gates:
            self.ssm_res_logit = nn.Parameter(
                torch.full((config.d_model,), float(config.ssm_residual_init)))
            self.ff_res_logit = nn.Parameter(
                torch.full((config.d_model,), float(config.ff_residual_init)))
        else:
            self.ssm_res_logit = nn.Parameter(torch.tensor(float(config.ssm_residual_init)))
            self.ff_res_logit = nn.Parameter(torch.tensor(float(config.ff_residual_init)))

    @property
    def ssm_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.ssm_res_logit)

    @property
    def ff_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.ff_res_logit)

    @property
    def res_scale(self) -> torch.Tensor:
        """Compatibility alias for the former single residual gate."""
        return self.ssm_scale

    @property
    def dropout(self) -> nn.Dropout:
        """Compatibility alias used by older diagnostics."""
        return self.ff_dropout

    def gain_terms(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        training: Optional[bool] = None,
    ) -> dict[str, torch.Tensor]:
        """Return the exact factors used by the block certificate."""
        training = self.training if training is None else bool(training)
        gamma = _module_gain_bound(self.lru, device=device, dtype=dtype)
        ff_lip = _module_lip_bound(self.ff, device=device, dtype=dtype)

        ssm_drop_factor = torch.as_tensor(
            1.0 / max(1.0 - float(self.ssm_dropout.p), 1e-12) if training else 1.0,
            device=device,
            dtype=dtype,
        )
        ff_drop_factor = torch.as_tensor(
            1.0 / max(1.0 - float(self.ff_dropout.p), 1e-12) if training else 1.0,
            device=device,
            dtype=dtype,
        )
        # The certificate uses the worst-channel gate: for a per-channel residual
        # ||I + diag(alpha)*M|| <= 1 + max_c(alpha_c)*||M||. ``.max()`` is a no-op
        # for the scalar-gate case, so this stays exact there too.
        alpha_ssm = self.ssm_scale.to(device=device, dtype=dtype).max()
        alpha_ff = self.ff_scale.to(device=device, dtype=dtype).max()
        ssm_branch_gain = gamma * ssm_drop_factor
        ff_branch_gain = ff_lip * ff_drop_factor
        ssm_factor = 1.0 + alpha_ssm * ssm_branch_gain
        ff_factor = 1.0 + alpha_ff * ff_branch_gain
        return {
            "gamma": gamma,
            "ff_lip": ff_lip,
            "ssm_drop_factor": ssm_drop_factor,
            "ff_drop_factor": ff_drop_factor,
            "alpha_ssm": alpha_ssm,
            "alpha_ff": alpha_ff,
            "ssm_branch_gain": ssm_branch_gain,
            "ff_branch_gain": ff_branch_gain,
            "ssm_factor": ssm_factor,
            "ff_factor": ff_factor,
            "block_factor": ssm_factor * ff_factor,
        }

    def forward(
        self,
        x3d: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mode: str = "loop",
        reset_state: bool = True,
        detach_state: bool = False,
    ):
        ssm_out, state_trajectory = self.lru(
            x3d,
            state=state,
            mode=mode,
            reset_state=reset_state,
            detach_state=detach_state,
        )
        x = x3d + self.ssm_scale * self.ssm_dropout(ssm_out)
        x = x + self.ff_scale * self.ff_dropout(self.ff(x))
        return x, state_trajectory

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Old checkpoints used one residual logit. Initialize both new branches
        # from it so existing experiments remain loadable.
        legacy_key = prefix + "res_logit"
        if legacy_key in state_dict:
            legacy = state_dict.pop(legacy_key)
            state_dict.setdefault(prefix + "ssm_res_logit", legacy.clone())
            state_dict.setdefault(prefix + "ff_res_logit", legacy.clone())

        # A scalar-gate checkpoint has an exact behavior-preserving migration to
        # per-channel gates: repeat the scalar logit in every channel. The reverse
        # direction remains a size mismatch because reducing learned channel gates
        # to one scalar would be lossy and has no uniquely correct rule.
        for name, target in (
            ("ssm_res_logit", self.ssm_res_logit),
            ("ff_res_logit", self.ff_res_logit),
        ):
            key = prefix + name
            source = state_dict.get(key)
            if source is not None and source.numel() == 1 and target.numel() > 1:
                state_dict[key] = source.reshape(1).expand_as(target).clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class DeepSSM(nn.Module):
    """
    Deep SSM with an optional certified zero-state l2-gain upper bound.

    For certified configurations, the encoder and decoder have spectral norm at
    most one, and each block is bounded by

        (1 + alpha_ssm,k * gamma_k * lip(dropout_ssm,k))
        * (1 + alpha_ff,k * lip(ff_k) * lip(dropout_ff,k)).

    and the decoder is smoothly attenuated so the composed bound does not
    exceed the prescribed ``config.gamma``. The certificate applies to a
    full trajectory initialized at zero state. Stateful continuation remains
    valid as part of that same trajectory, but a standalone nonzero initial
    state requires an additional storage-energy term.
    """
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
        train_ff_lip: Optional[bool] = None,
        init: str = "eye",
        rho: float = 0.9,
        max_phase_b: float = 0.5,
        phase_center: float = 0,
        random_phase: bool = True,
        learn_x0: bool = False,
        ssm_residual_init: float = -1.0,
        ff_residual_init: float = -1.0,
        per_channel_gates: bool = False,
        config: Optional[SSMConfig] = None,
    ):
        super().__init__()
        if d_input <= 0 or d_output <= 0:
            raise ValueError("d_input and d_output must be positive.")
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
            train_ff_lip=train_ff_lip,
            init=init,
            rho=rho,
            max_phase_b=max_phase_b,
            phase_center=phase_center,
            random_phase=random_phase,
            learn_x0=learn_x0,
            ssm_residual_init=ssm_residual_init,
            ff_residual_init=ff_residual_init,
            per_channel_gates=per_channel_gates,
        )

        self._validate_config(self.config)
        self.use_cert_scaling = self.config.gamma is not None
        self._prescribed_gamma = (
            float(self.config.gamma) if self.config.gamma is not None else None
        )
        self.ff_has_lip = False
        # Eval-only cache of the spectrally-capped (encoder, decoder) weights.
        self._enc_dec_cache = None

        if self.use_cert_scaling:
            self.register_buffer("gamma_t", torch.tensor(float(self.config.gamma)))

            # Balanced near-isometric init
            self.encoder_w = nn.Parameter(torch.empty(self.config.d_model, self.d_input))
            self.decoder_w = nn.Parameter(torch.empty(self.d_output, self.config.d_model))
            with torch.no_grad():
                nn.init.orthogonal_(self.encoder_w)
                nn.init.orthogonal_(self.decoder_w)
        else:
            self.encoder = nn.Linear(d_input, self.config.d_model, bias=False)
            self.decoder = nn.Linear(self.config.d_model, d_output, bias=False)

        self.blocks = nn.ModuleList([SSL(self.config) for _ in range(self.config.n_layers)])

        if len(self.blocks) > 0:
            self.ff_has_lip = all(hasattr(block.ff, "lip") for block in self.blocks)

        if self.use_cert_scaling:
            missing_gamma = [
                block.lru.__class__.__name__
                for block in self.blocks
                if not _has_gain_contract(block.lru)
            ]
            if missing_gamma:
                raise RuntimeError(
                    "Certified DeepSSM blocks must expose an l2-gain bound through "
                    f"`.gamma`; missing on: {missing_gamma}."
                )
            if self.blocks and not self.ff_has_lip:
                raise RuntimeError(
                    "Certified DeepSSM feedforwards must expose a global Lipschitz "
                    "bound through `.lip`."
                )

    @staticmethod
    def _validate_config(config: SSMConfig) -> None:
        if config.d_model <= 0 or config.d_state <= 0:
            raise ValueError("d_model and d_state must be positive.")
        if config.n_layers < 0:
            raise ValueError("n_layers must be non-negative.")
        if not 0.0 <= float(config.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {config.dropout}.")
        if float(config.scale) <= 0.0 or not math.isfinite(float(config.scale)):
            raise ValueError(f"scale must be finite and positive, got {config.scale}.")
        if (
            float(config.cert_scale_temperature) <= 0.0
            or not math.isfinite(float(config.cert_scale_temperature))
        ):
            raise ValueError(
                "cert_scale_temperature must be finite and positive, got "
                f"{config.cert_scale_temperature}."
            )
        if not math.isfinite(float(config.ssm_residual_init)):
            raise ValueError(
                "ssm_residual_init must be finite, got "
                f"{config.ssm_residual_init}."
            )
        if not math.isfinite(float(config.ff_residual_init)):
            raise ValueError(
                "ff_residual_init must be finite, got "
                f"{config.ff_residual_init}."
            )

        cell_spec = _get_ssm_parametrization(config.param)

        if config.gamma is None:
            return

        gamma = float(config.gamma)
        if gamma <= 0.0 or not math.isfinite(gamma):
            raise ValueError(f"gamma must be finite and positive, got {config.gamma}.")
        if config.n_layers > 0 and not cell_spec.certified:
            supported = ", ".join(sorted(_CERTIFIED_PARAMETRIZATIONS))
            raise ValueError(
                f"param={config.param!r} does not provide a certified l2-gain bound. "
                f"Use one of {{{supported}}} or set gamma=None."
            )
        if config.n_layers > 0 and config.ff not in _CERTIFIED_FEEDFORWARDS:
            supported = ", ".join(sorted(_CERTIFIED_FEEDFORWARDS))
            hint = (
                " `LGLU` is not globally Lipschitz; use `LGLU2` instead."
                if config.ff == "LGLU"
                else ""
            )
            raise ValueError(
                f"ff={config.ff!r} cannot be used for a certified global gain bound."
                f"{hint} Use one of {{{supported}}} or set gamma=None."
            )
        if config.n_layers > 0 and config.learn_x0:
            raise ValueError(
                "learn_x0=True is incompatible with a zero-state induced l2-gain "
                "certificate because a learned initial condition can produce output "
                "with zero input. Set learn_x0=False or gamma=None."
            )

    @staticmethod
    def _spectrally_capped_weight(
        weight: torch.Tensor,
        *,
        bound: float = 1.0,
    ) -> torch.Tensor:
        """Return a weight with spectral norm <= bound without scaling small weights up."""
        if bound <= 0.0:
            raise ValueError(f"bound must be positive, got {bound}.")

        weight_for_norm = (
            weight.float() if weight.dtype in (torch.float16, torch.bfloat16) else weight
        )
        sigma = torch.linalg.matrix_norm(weight_for_norm, ord=2).to(
            device=weight.device,
            dtype=weight.dtype,
        )
        divisor = torch.clamp(sigma / float(bound), min=1.0)
        return weight / divisor

    def _block_gain_terms(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        training: Optional[bool] = None,
    ) -> List[dict[str, torch.Tensor]]:
        return [
            block.gain_terms(device=device, dtype=dtype, training=training)
            for block in self.blocks
        ]

    def _log_block_gain_product(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        terms = self._block_gain_terms(device=device, dtype=dtype)
        if not terms:
            return torch.zeros((), device=device, dtype=dtype)
        factors = torch.stack([term["block_factor"] for term in terms])
        # Configuration validation guarantees finite component bounds. Avoid a
        # tensor-to-Python finiteness check here because it synchronizes CUDA on
        # every certified forward pass.
        return torch.log(factors.clamp_min(torch.finfo(dtype).tiny)).sum()

    def _effective_gamma_cap(
        self,
        *,
        gamma,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.use_cert_scaling or self._prescribed_gamma is None:
            raise RuntimeError("No prescribed global gain is configured.")

        current = self.gamma_t.to(device=device, dtype=dtype).abs()
        candidate = current
        if gamma is not None:
            requested = torch.as_tensor(gamma, device=device, dtype=dtype)
            if requested.numel() != 1:
                raise ValueError("gamma override must be a scalar.")
            requested = requested.reshape(())
            if not bool(torch.isfinite(requested)) or not bool(requested > 0):
                raise ValueError("gamma override must be finite and positive.")
            candidate = torch.minimum(candidate, requested)

        prescribed = torch.as_tensor(
            self._prescribed_gamma,
            device=device,
            dtype=dtype,
        )
        return torch.minimum(candidate, prescribed)

    @torch.no_grad()
    def conservative_gamma_product(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Conservative reporting-only bound:
            prod_k (1 + gamma_k * lip(dropout_ssm,k))
                   * (1 + lip(ff_k) * lip(dropout_ff,k))

        This sets both residual gates to one, so it is no smaller than the block
        product used by the actual certificate.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        if len(self.blocks) == 0:
            return torch.ones((), device=device, dtype=dtype)

        terms = self._block_gain_terms(device=device, dtype=dtype)
        if any(
            not bool(torch.isfinite(term["gamma"]))
            or not bool(torch.isfinite(term["ff_lip"]))
            for term in terms
        ):
            return torch.full((), float("inf"), device=device, dtype=dtype)
        factors = torch.stack([
            (1.0 + term["ssm_branch_gain"]) * (1.0 + term["ff_branch_gain"])
            for term in terms
        ])
        return torch.exp(torch.log(factors).sum())

    @torch.no_grad()
    def certified_gain_bound(self, gamma=None) -> torch.Tensor:
        """Return the composed upper bound used for the current train/eval mode."""
        if not self.use_cert_scaling:
            raise RuntimeError("certified_gain_bound requires a prescribed gamma.")

        device = self.encoder_w.device
        dtype = self.encoder_w.dtype
        encoder_eff = self._spectrally_capped_weight(self.encoder_w)
        decoder_eff = self._spectrally_capped_weight(self.decoder_w)
        encoder_norm = torch.linalg.matrix_norm(encoder_eff.float(), ord=2).to(dtype=dtype)
        decoder_norm = torch.linalg.matrix_norm(decoder_eff.float(), ord=2).to(dtype=dtype)

        log_block_product = self._log_block_gain_product(device=device, dtype=dtype)
        gamma_cap = self._effective_gamma_cap(
            gamma=gamma,
            device=device,
            dtype=dtype,
        )
        log_scale = self._smooth_capped_log_scale_from_logs(
            gamma_t=gamma_cap,
            log_gamma_prod=log_block_product,
            temperature=self.config.cert_scale_temperature,
        )

        tiny = torch.finfo(dtype).tiny
        # Compose entirely in log space. Materializing ``scale`` first and
        # reading it back via ``log(scale.clamp_min(tiny))`` re-inflates the
        # decoder attenuation once it underflows to a subnormal float (deep
        # stacks / large learned branch gains in float32), which would report a
        # bound far above ``gamma``.
        log_bound = (
            torch.log(encoder_norm.clamp_min(tiny))
            + log_block_product
            + torch.log(decoder_norm.clamp_min(tiny))
            + log_scale
        )
        bound = torch.exp(log_bound)
        return torch.where(torch.isfinite(log_scale), bound, torch.zeros_like(bound))

    @torch.no_grad()
    def gain_diagnostics(self) -> dict[str, Any]:
        """Return certificate data using the same factors as :meth:`forward`.

        Keeping diagnostics here prevents training scripts from duplicating the
        gain formula and silently drifting away from the implementation.
        """
        try:
            reference = next(self.parameters())
        except StopIteration as exc:
            raise RuntimeError("DeepSSM has no parameters to diagnose.") from exc

        device, dtype = reference.device, reference.dtype
        terms = self._block_gain_terms(device=device, dtype=dtype)
        block_rows = []
        for index, (block, term) in enumerate(zip(self.blocks, terms)):
            raw_lip = getattr(block.ff, "raw_lip", None)
            ff_lip_trainable = isinstance(raw_lip, nn.Parameter) and raw_lip.requires_grad
            block_rows.append({
                "index": index,
                "lru_type": block.lru.__class__.__name__,
                "ff_type": block.ff.__class__.__name__,
                "core_gamma": float(term["gamma"].detach().cpu()),
                "ff_lip": float(term["ff_lip"].detach().cpu()),
                "ff_lip_trainable": ff_lip_trainable,
                "alpha_ssm": float(term["alpha_ssm"].detach().cpu()),
                "alpha_ff": float(term["alpha_ff"].detach().cpu()),
                "ssm_drop_factor": float(term["ssm_drop_factor"].detach().cpu()),
                "ff_drop_factor": float(term["ff_drop_factor"].detach().cpu()),
                "ssm_branch_gain": float(term["ssm_branch_gain"].detach().cpu()),
                "ff_branch_gain": float(term["ff_branch_gain"].detach().cpu()),
                "ssm_factor": float(term["ssm_factor"].detach().cpu()),
                "ff_factor": float(term["ff_factor"].detach().cpu()),
                "block_factor": float(term["block_factor"].detach().cpu()),
            })

        if terms:
            block_factors = torch.stack([term["block_factor"] for term in terms])
            log_gamma_prod = torch.log(block_factors).sum()
            gamma_prod = torch.exp(log_gamma_prod)
        else:
            log_gamma_prod = torch.zeros((), device=device, dtype=dtype)
            gamma_prod = torch.ones((), device=device, dtype=dtype)

        conservative = self.conservative_gamma_product(device=device, dtype=dtype)
        global_gamma = smooth_scale = hard_scale = certified_bound = None
        encoder_norm = decoder_norm = None

        if self.use_cert_scaling:
            gamma_cap = self._effective_gamma_cap(
                gamma=None,
                device=device,
                dtype=dtype,
            )
            smooth = self._smooth_capped_scale_from_logs(
                gamma_t=gamma_cap,
                log_gamma_prod=log_gamma_prod,
                temperature=self.config.cert_scale_temperature,
            )
            tiny = torch.finfo(dtype).tiny
            hard = torch.exp(
                -torch.clamp(
                    log_gamma_prod - torch.log(gamma_cap.clamp_min(tiny)),
                    min=0.0,
                )
            )
            encoder_eff = self._spectrally_capped_weight(self.encoder_w)
            decoder_eff = self._spectrally_capped_weight(self.decoder_w)
            encoder_norm_t = torch.linalg.matrix_norm(
                encoder_eff.float(), ord=2
            ).to(dtype=dtype)
            decoder_norm_t = torch.linalg.matrix_norm(
                decoder_eff.float(), ord=2
            ).to(dtype=dtype)

            global_gamma = float(gamma_cap.detach().cpu())
            smooth_scale = float(smooth.detach().cpu())
            hard_scale = float(hard.detach().cpu())
            certified_bound = float(self.certified_gain_bound().detach().cpu())
            encoder_norm = float(encoder_norm_t.detach().cpu())
            decoder_norm = float(decoder_norm_t.detach().cpu())

        return {
            "mode": "train" if self.training else "eval",
            "use_cert_scaling": self.use_cert_scaling,
            "global_gamma": global_gamma,
            "gamma_prod": float(gamma_prod.detach().cpu()),
            "conservative_gamma_prod": float(conservative.detach().cpu()),
            "smooth_scale": smooth_scale,
            "hard_scale": hard_scale,
            "certified_gain_bound": certified_bound,
            "encoder_norm": encoder_norm,
            "decoder_norm": decoder_norm,
            "n_blocks": len(block_rows),
            "blocks": block_rows,
        }

    @staticmethod
    def _last_runtime_state(state: Any) -> Any:
        """Extract a reusable final state from a returned state trajectory."""
        if state is None:
            return None
        if torch.is_tensor(state):
            return state[:, -1, :] if state.ndim >= 3 else state
        if isinstance(state, tuple):
            return tuple(DeepSSM._last_runtime_state(item) for item in state)
        if isinstance(state, list):
            return [DeepSSM._last_runtime_state(item) for item in state]
        raise TypeError(f"Unsupported state trajectory type: {type(state).__name__}.")

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
        if gamma is not None and not self.use_cert_scaling:
            raise ValueError("gamma override requires a model constructed with gamma set.")
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
            encoder_eff, decoder_eff = self._capped_encoder_decoder()
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
            layer_states[i] = self._last_runtime_state(st)

        # Decode
        if self.use_cert_scaling:
            gamma_t = self._effective_gamma_cap(
                gamma=gamma,
                device=x.device,
                dtype=x.dtype,
            )
            log_gamma_prod = self._log_block_gain_product(
                device=x.device,
                dtype=x.dtype,
            )
            scale = self._smooth_capped_scale_from_logs(
                gamma_t=gamma_t,
                log_gamma_prod=log_gamma_prod,
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
        dtype = gamma_prod.dtype
        tiny = torch.finfo(dtype).tiny
        log_gamma_prod = torch.log(gamma_prod.abs().clamp_min(tiny))
        return DeepSSM._smooth_capped_scale_from_logs(
            gamma_t=gamma_t,
            log_gamma_prod=log_gamma_prod,
            temperature=temperature,
        )

    @staticmethod
    def _smooth_capped_log_scale_from_logs(
        gamma_t: torch.Tensor,
        log_gamma_prod: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Log of the smooth decoder scale.

        Returns ``log(scale)`` where ``scale`` is no larger than
        ``min(1, gamma_t / gamma_prod)``. Keeping the result in log space lets
        callers compose the certificate without ever materializing ``scale``:
        for deep stacks ``scale`` underflows to a subnormal float, and reading
        it back through ``log(clamp_min(scale, tiny))`` would re-inflate it and
        break ``bound <= gamma``. Returns ``-inf`` for non-positive ``gamma_t``.
        """
        tau = max(float(temperature), 1e-6)
        gamma_t = gamma_t.abs()
        tiny = torch.finfo(gamma_t.dtype).tiny
        log_gamma_t = torch.log(gamma_t.clamp_min(tiny))
        log_ratio = log_gamma_prod - log_gamma_t
        smooth_log_cap = tau * F.softplus(log_ratio / tau)
        log_scale = -smooth_log_cap
        return torch.where(
            gamma_t > 0, log_scale, torch.full_like(log_scale, float("-inf"))
        )

    @staticmethod
    def _smooth_capped_scale_from_logs(
        gamma_t: torch.Tensor,
        log_gamma_prod: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Smoothly cap decoder gain without ever amplifying it.

        The result is no larger than ``min(1, gamma_t / gamma_prod)``. Working
        with ``log_gamma_prod`` avoids overflow for deep stacks or large learned
        branch bounds. ``exp(-inf) == 0`` recovers the non-positive-gamma case.
        """
        log_scale = DeepSSM._smooth_capped_log_scale_from_logs(
            gamma_t=gamma_t,
            log_gamma_prod=log_gamma_prod,
            temperature=temperature,
        )
        return torch.exp(log_scale)

    def _capped_encoder_decoder(self):
        """Spectrally-capped (encoder, decoder) weights, cached in eval.

        The cap is an SVD-based spectral-norm clip; with fixed weights (eval) the
        result is constant, so it is computed once and reused. Training always
        recomputes (weights change), and ``train()``/``eval()`` clears the cache.
        The cached tensors are exactly the recomputed ones — no math change.
        """
        if self.training:
            self._enc_dec_cache = None
            return (self._spectrally_capped_weight(self.encoder_w),
                    self._spectrally_capped_weight(self.decoder_w))
        if self._enc_dec_cache is None:
            self._enc_dec_cache = (
                self._spectrally_capped_weight(self.encoder_w).detach(),
                self._spectrally_capped_weight(self.decoder_w).detach(),
            )
        return self._enc_dec_cache

    def train(self, mode: bool = True):
        self._enc_dec_cache = None
        return super().train(mode)

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

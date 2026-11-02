"""One construction registry shared by stacks, validation, and benchmarks.

Factories run only when building a model, never inside its recurrence. Separate
family dictionaries make current and legacy choices explicit without wrappers
or extra dispatch on the forward path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..static_layers.generic_layers import GLU, MLP, LayerConfig
from ..static_layers.lipschitz_mlps import (
    LMLP,
    TLIP,
    BudgetedL2BoundedGLUv2,
    L2BoundedGLU,
    L2BoundedGLUv2,
    MultiBranchLipMixer,
)
from .cells.legacy import L2RU, L2BoundedLTICell, lruz
from .cells.lti import LRU, Block2x2DenseL2SSM, DefectL2SSM
from .cells.selective import RobustMambaDiagLTI, RobustMambaDiagSSM
from .config import SSMConfig

CellFactory = Callable[[SSMConfig, Optional[float]], nn.Module]


@dataclass(frozen=True)
class SSMParametrization:
    """Construction metadata for one recurrent-cell parametrization.

    ``certified`` means the produced cell exposes a finite zero-state L2-gain
    bound through either ``gain_bound()`` or the legacy ``.gamma`` attribute.

    ``selective`` means the cell builds its per-step transition from a
    ``param_net``, so a context signal can shape the dynamics and the cell
    accepts ``select_input``.
    """

    name: str
    factory: CellFactory
    certified: bool
    selective: bool = False
    legacy: bool = False
    family: str = "lti"
    supports_return_last: bool = False


def _sigmoid_scalar(value: float) -> float:
    """Numerically stable scalar sigmoid used during module construction."""
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _initial_block_gamma(config: SSMConfig) -> float | None:
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


def _fixed_gamma(config: SSMConfig, block_gamma: float | None) -> float | None:
    """Return the fixed per-cell gamma used by legacy fixed-gamma cells."""
    return None if config.train_gamma else block_gamma


def _gamma_init(block_gamma: float | None) -> float:
    """Default positive gamma for cells that own their trainability flag."""
    return 1.0 if block_gamma is None else float(block_gamma)


def _build_lru_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
    return LRU(
        in_features=config.d_model,
        out_features=config.d_model,
        state_features=config.d_state,
        rmin=config.rmin,
        rmax=config.rmax,
        max_phase=config.max_phase,
        learn_x0=config.learn_x0,
    )


def _build_l2n_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
    cell = Block2x2DenseL2SSM(
        d_state=config.d_state,
        d_input=config.d_model,
        d_output=config.d_model,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        learn_x0=config.learn_x0,
        state_metric=config.l2n_state_metric,
    )
    cell.init_on_circle(
        rho=config.rho,
        max_phase=config.max_phase_b,
        phase_center=config.phase_center,
        random_phase=config.random_phase,
        offdiag_scale=config.offdiag_scale,
    )
    return cell


def _build_defect_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
    return DefectL2SSM(
        d_state=config.d_state,
        d_input=config.d_model,
        d_output=config.d_model,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        block_size=config.defect_block_size,
        max_radius=config.defect_max_radius,
        factor_margin=config.defect_factor_margin,
        init_rho=config.rho,
        max_phase=config.max_phase_b,
        phase_center=config.phase_center,
        random_phase=config.random_phase,
        init_scale=config.offdiag_scale,
        learn_x0=config.learn_x0,
        state_metric=config.defect_state_metric,
    )


def _build_tv_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
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
        context_dim=config.select_context_dim,
        select_input=config.select_input,
    )


def _build_tvc_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
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
        bcd_nonlinearity=config.bcd_nonlinearity,
        output_uses_post_state=False,
        learn_x0=config.learn_x0,
        use_cuda_graph=config.use_cuda_graph,
        context_dim=config.select_context_dim,
        select_input=config.select_input,
    )


def _build_l2ru_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
    return L2RU(
        state_features=config.d_model,
        gamma=_fixed_gamma(config, block_gamma),
        init=config.init,
        eye_scale=config.l2ru_eye_scale,
        rand_scale=config.l2ru_rand_scale,
        learn_x0=config.learn_x0,
    )


def _build_zak_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
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


def _build_l2nt_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
    return L2BoundedLTICell(
        d_state=config.d_state,
        d_input=config.d_model,
        d_output=config.d_model,
        gamma=_gamma_init(block_gamma),
        train_gamma=config.train_gamma,
        learn_x0=config.learn_x0,
    )


LTI_PARAMETRIZATIONS = {
    "lru": SSMParametrization("lru", _build_lru_cell, certified=False),
    "l2n": SSMParametrization("l2n", _build_l2n_cell, certified=True, supports_return_last=True),
    "defect": SSMParametrization(
        "defect", _build_defect_cell, certified=True, supports_return_last=True
    ),
}
SELECTIVE_PARAMETRIZATIONS = {
    "tv": SSMParametrization(
        "tv",
        _build_tv_cell,
        certified=True,
        selective=True,
        family="selective",
        supports_return_last=True,
    ),
    "tvc": SSMParametrization(
        "tvc",
        _build_tvc_cell,
        certified=True,
        selective=True,
        family="selective",
        supports_return_last=True,
    ),
}
LEGACY_PARAMETRIZATIONS = {
    "l2ru": SSMParametrization("l2ru", _build_l2ru_cell, certified=True, legacy=True),
    "zak": SSMParametrization("zak", _build_zak_cell, certified=True, legacy=True),
    "l2nt": SSMParametrization("l2nt", _build_l2nt_cell, certified=True, legacy=True),
}
SSM_PARAMETRIZATIONS = {
    **LTI_PARAMETRIZATIONS,
    **SELECTIVE_PARAMETRIZATIONS,
    **LEGACY_PARAMETRIZATIONS,
}
CERTIFIED_PARAMETRIZATIONS = frozenset(
    name for name, spec in SSM_PARAMETRIZATIONS.items() if spec.certified
)
# Historical private names remain aliases, not separate registries.
_SSM_PARAMETRIZATIONS = SSM_PARAMETRIZATIONS
_CERTIFIED_PARAMETRIZATIONS = CERTIFIED_PARAMETRIZATIONS


def _param_name(param: str | None) -> str:
    return "lru" if param is None else str(param)


def _get_ssm_parametrization(param: str | None) -> SSMParametrization:
    name = _param_name(param)
    try:
        return _SSM_PARAMETRIZATIONS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SSM_PARAMETRIZATIONS))
        raise ValueError(
            f"Unknown SSM parametrization: {param!r}. Available: {{{available}}}."
        ) from exc


def _build_ssm_cell(config: SSMConfig, block_gamma: float | None) -> nn.Module:
    """Build one recurrent cell from the registry and initialize its gain."""
    cell = _get_ssm_parametrization(config.param).factory(config, block_gamma)
    if config.train_gamma and block_gamma is not None:
        _set_cell_gamma(cell, block_gamma)
    return cell


# Feedforward branches: only the listed bounded constructions certify the stack.

_CERTIFIED_FEEDFORWARDS = frozenset({"LGLU2", "BLGLU2", "BudgetedLGLU2", "MBLIP", "TLIP"})


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
        "TLIP": TLIP,
    }
    try:
        return builders[config.ff](layer_config)
    except KeyError as exc:
        raise ValueError(f"Unknown feedforward type: {config.ff!r}.") from exc

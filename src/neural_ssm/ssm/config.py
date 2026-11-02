"""Configuration for DeepSSM, its cells, and certified residual blocks.

Current and legacy fields keep their names/defaults so existing saved config
and state_dict checkpoints remain usable, apart from removed model families.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TypedDict


@dataclass
class SSMConfig:
    """param chooses the cell; ff chooses the static branch; gamma caps the stack.

    Field names, defaults, and ordering retain configuration compatibility.
    Cells use only the initialization settings relevant to their construction.
    """

    # Stack dimensions and regularization.
    d_model: int = 10  # internal feature width
    d_state: int = 32  # recurrent state dimension
    n_layers: int = 2  # number of residual SSM blocks
    dropout: float = 0.0
    bias: bool = False

    # Complex LRU pole initialization.
    rmin: float = 0.8
    rmax: float = 0.95
    max_phase: float = 2 * math.pi

    # Feedforward choice, gain budget, and hidden dimensions.
    ff: str = "MLP"
    scale: float = 1
    dim_amp: int = 4
    d_hidden: int = 4
    nl_layers: int = 2

    # Current: lru/l2n/defect/tv/tvc. Legacy: l2ru/zak/l2nt. None selects LRU.
    param: str | None = None
    gamma: float | None = None  # fixed whole-stack target; None disables the cap
    train_gamma: bool = True  # train per-cell gains, not the whole-stack target
    # None follows train_gamma when a target is prescribed; True/False is explicit.
    train_ff_lip: bool | None = True

    # Legacy L2RU/ZAK initialization.
    init: str = "eye"
    l2ru_eye_scale: float = 0.01
    l2ru_rand_scale: float = 1.0

    # Shared real-block initialization (l2n and defect).
    rho: float = 0.9
    max_phase_b: float = 0.04  # phase half-width for 2x2 blocks
    phase_center: float = 0
    random_phase: bool = True
    offdiag_scale: float = 0.05

    # Defect completion: larger blocks permit more within-block interactions.
    defect_block_size: int = 2
    defect_max_radius: float = 0.999
    defect_factor_margin: float = 1e-3

    # Selective TV initialization.
    tv_init_rho: float = 0.99
    tv_init_delta0: float = 1.0
    tv_init_param_scale: float = 0.02

    # Selective TVC initialization (includes a direct term).
    tvc_init_rho: float = 0.9
    tvc_init_delta0: float = 1.0
    tvc_init_param_scale: float = 0.02
    tvc_init_sign: float = 0.995
    tvc_init_b: float = 0.10
    tvc_init_c: float = 0.10
    tvc_init_d: float = 0.10
    bcd_nonlinearity: str = "identity"  # tvc: use 'tanh' to bound raw b/c/d

    # Initial state and optional selective context.
    learn_x0: bool = False  # a nonzero learned state needs a storage-energy term
    select_context_dim: int = 0  # 0 disables the selector's context channel
    # 'both': input-dependent selection, zero-state gain bound.
    # 'context': exogenous selection, also incremental for a fixed context.
    select_input: str = "both"
    use_cuda_graph: bool = False  # tv/tvc: replay a scan for fixed-shape GPU runs

    # Remaining legacy ZAK initialization fields; keep their historical order.
    zak_d_margin: float = 0.5
    zak_x2_margin: float = 0.5
    zak_x2_init_scale: float = 0.1

    # Whole-stack certificate and residual initialization.
    cert_scale_temperature: float = 0.05  # smoothness of the fixed-gamma decoder cap
    ssm_residual_init: float = -1.0  # residual gate logits
    ff_residual_init: float = -1.0
    # Rebalance initial gates so deep stacks do not start excessively attenuated.
    auto_residual_init: bool = False
    budget_init_utilization: float = 0.7  # in (0, 1)
    per_channel_gates: bool = False  # vector gates; certificate uses the largest entry

    # Defect state storage; full couples blocks while keeping their fast scan.
    # Choose radius=1 and factor_margin=0 to avoid fixed modelling margins.
    defect_state_metric: str = "identity"
    # L2N: full puts a dense certificate inside normalization, in scan coordinates.
    l2n_state_metric: str = "identity"


SSMConfigDict = TypedDict("SSMConfigDict", {f.name: f.type for f in fields(SSMConfig)}, total=False)


def validate_ssm_config(config: SSMConfig) -> None:
    """Validate construction-time settings, without checks in the recurrence."""
    from .registry import (
        _CERTIFIED_FEEDFORWARDS,
        _CERTIFIED_PARAMETRIZATIONS,
        _get_ssm_parametrization,
    )

    if config.d_model <= 0 or config.d_state <= 0:
        raise ValueError("d_model and d_state must be positive.")
    if config.n_layers < 0:
        raise ValueError("n_layers must be non-negative.")
    if not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}.")
    if float(config.scale) <= 0.0 or not math.isfinite(float(config.scale)):
        raise ValueError(f"scale must be finite and positive, got {config.scale}.")
    if float(config.cert_scale_temperature) <= 0.0 or not math.isfinite(
        float(config.cert_scale_temperature)
    ):
        raise ValueError(
            "cert_scale_temperature must be finite and positive, got "
            f"{config.cert_scale_temperature}."
        )
    if not math.isfinite(float(config.ssm_residual_init)):
        raise ValueError(f"ssm_residual_init must be finite, got {config.ssm_residual_init}.")
    if not math.isfinite(float(config.ff_residual_init)):
        raise ValueError(f"ff_residual_init must be finite, got {config.ff_residual_init}.")

    cell_spec = _get_ssm_parametrization(config.param)

    if config.select_input not in ("both", "context"):
        raise ValueError(f"select_input must be 'both' or 'context', got {config.select_input!r}.")
    if config.select_input == "context":
        if config.select_context_dim <= 0:
            raise ValueError(
                "select_input='context' requires select_context_dim > 0: the "
                "selector would otherwise have no input at all."
            )
        if config.n_layers > 0 and not cell_spec.selective:
            raise ValueError(
                f"select_input='context' requires a selective parametrization "
                f"whose per-step matrices the context can shape; got "
                f"param={config.param!r}. Use 'tv' or 'tvc'."
            )

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
            " `LGLU` is not globally Lipschitz; use `LGLU2` instead." if config.ff == "LGLU" else ""
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

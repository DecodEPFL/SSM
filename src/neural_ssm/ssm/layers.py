"""The SSM stack in one place: residual block, DeepSSM, and simple baselines.

Forward path: encoder -> SSL blocks -> decoder. Each SSL applies its recurrent
cell followed by its instantaneous feedforward branch, with separate residual
gates. Cell factories live in registry.py; settings live in config.py.

Reading order: SSL and DeepSSM construction/forward, then their certificates
and diagnostics, then the baselines. Shared algebra lives in common.py;
context handling lives in contextual.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.runtime import (
    EvalCacheMixin,
    last_runtime_state,
    normalize_to_3d as _normalize_to_3d,
    reset_runtime_state,
)
from .cells.legacy import L2RU, L2BoundedLTICell, lruz
from .cells.lti import LRU, Block2x2DenseL2SSM, DefectL2SSM
from .cells.selective import RobustMambaDiagLTI, RobustMambaDiagSSM
from .common import (
    has_gain_contract as _has_gain_contract,
    module_gain_bound as _module_gain_bound,
    module_lip_bound as _module_lip_bound,
    smooth_capped_log_scale_from_logs,
    smooth_capped_scale,
    smooth_capped_scale_from_logs,
    spectrally_capped_weight,
    state_energy as _state_energy,
)
from .config import SSMConfig, SSMConfigDict, validate_ssm_config
from .registry import (
    _CERTIFIED_FEEDFORWARDS as _CERTIFIED_FEEDFORWARDS,
    _CERTIFIED_PARAMETRIZATIONS as _CERTIFIED_PARAMETRIZATIONS,
    _SSM_PARAMETRIZATIONS as _SSM_PARAMETRIZATIONS,
    SSMParametrization,
    _build_feedforward,
    _build_ssm_cell,
    _get_ssm_parametrization,
    _initial_block_gamma,
)

# Residual state-space block


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
        self._supports_return_last = _get_ssm_parametrization(config.param).supports_return_last
        self.ff = _build_feedforward(config)

        self.ssm_dropout = nn.Dropout(config.dropout)
        self.ff_dropout = nn.Dropout(config.dropout)
        # Per-channel residual gates (vectors of size d_model) let some channels
        # emphasize memory and others feedforward; scalars recover the old behavior.
        # The certificate uses the worst-channel gate, so it stays valid either way.
        self.per_channel_gates = bool(getattr(config, "per_channel_gates", False))
        gate_shape = (config.d_model,) if self.per_channel_gates else ()
        self.ssm_res_logit = nn.Parameter(torch.full(gate_shape, float(config.ssm_residual_init)))
        self.ff_res_logit = nn.Parameter(torch.full(gate_shape, float(config.ff_residual_init)))

    def forward(
        self,
        x3d: torch.Tensor,
        state: torch.Tensor | None = None,
        mode: str = "loop",
        reset_state: bool = True,
        detach_state: bool = False,
        ssm_context_gate: torch.Tensor | None = None,
        ff_context_gate: torch.Tensor | None = None,
        select_context: torch.Tensor | None = None,
        return_state: bool = True,
    ):
        """Apply both residual branches and return the cell's state trajectory.

        With return_state=False, request only the final state when supported.
        This avoids full-trajectory coordinate conversion in cells such as l2n.
        """
        lru_kwargs = dict(
            state=state,
            mode=mode,
            reset_state=reset_state,
            detach_state=detach_state,
        )
        if not return_state and self._supports_return_last:
            lru_kwargs.update(return_state=False, return_last=True)
        # Only selective cells accept context in their param_net.
        if select_context is not None and getattr(self.lru, "supports_select_context", False):
            lru_kwargs["select_context"] = select_context
        ssm_out, state_trajectory = self.lru(x3d, **lru_kwargs)

        if ssm_context_gate is None:
            ssm_gate = 1.0
        else:
            ssm_gate = torch.clamp(
                ssm_context_gate.to(device=x3d.device, dtype=x3d.dtype),
                0.0,
                1.0,
            )

        x = x3d + self.ssm_scale * ssm_gate * self.ssm_dropout(ssm_out)

        if ff_context_gate is None:
            ff_gate = 1.0
        else:
            ff_gate = torch.clamp(
                ff_context_gate.to(device=x3d.device, dtype=x3d.dtype),
                0.0,
                1.0,
            )

        x = x + self.ff_scale * ff_gate * self.ff_dropout(self.ff(x))
        return x, state_trajectory if return_state else last_runtime_state(state_trajectory)

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
        training: bool | None = None,
        include_storage: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return the exact factors used by the block certificate.

        Skip c_ssm with include_storage=False when only the output gain is needed.
        """
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
        alpha_eff = alpha_ssm * ssm_drop_factor
        terms = {
            "gamma": gamma,
            "ff_lip": ff_lip,
            "ssm_drop_factor": ssm_drop_factor,
            "ff_drop_factor": ff_drop_factor,
            "alpha_ssm": alpha_ssm,
            "alpha_ff": alpha_ff,
            "alpha_eff": alpha_eff,
            "ssm_branch_gain": ssm_branch_gain,
            "ff_branch_gain": ff_branch_gain,
            "ssm_factor": ssm_factor,
            "ff_factor": ff_factor,
            "block_factor": ssm_factor * ff_factor,
        }
        if include_storage:
            # Young's inequality weights the recurrent storage by
            # alpha_eff * (alpha_eff + 1/gamma). Invalid contracts have no storage.
            terms["c_ssm"] = torch.where(
                (gamma > 0) & torch.isfinite(gamma),
                alpha_eff * (alpha_eff + 1.0 / gamma.clamp_min(torch.finfo(dtype).tiny)),
                torch.full_like(gamma, float("inf")),
            )
        return terms

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


# Deep state-space stack


class DeepSSM(EvalCacheMixin, nn.Module):
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
        rmin: float = 0.8,
        rmax: float = 0.95,
        max_phase: float = 2 * math.pi,
        ff: str = "LGLU2",
        scale: float = 1,
        dim_amp: int = 4,
        d_hidden: int = 4,
        nl_layers: int = 3,
        param: str | None = "lru",
        gamma: float | None = None,
        train_gamma: bool | None = True,
        train_ff_lip: bool | None = None,
        init: str = "eye",
        rho: float = 0.9,
        max_phase_b: float = 0.5,
        phase_center: float = 0,
        random_phase: bool = True,
        learn_x0: bool = False,
        ssm_residual_init: float = -1.0,
        ff_residual_init: float = -1.0,
        auto_residual_init: bool = False,
        budget_init_utilization: float = 0.7,
        per_channel_gates: bool = False,
        select_context_dim: int = 0,
        select_input: str = "both",
        bcd_nonlinearity: str = "identity",
        defect_state_metric: str = "identity",
        defect_block_size: int = 2,
        defect_max_radius: float = 0.999,
        defect_factor_margin: float = 1e-3,
        l2n_state_metric: str = "identity",
        config: SSMConfig | None = None,
    ):
        super().__init__()
        if d_input <= 0 or d_output <= 0:
            raise ValueError("d_input and d_output must be positive.")
        self.d_input = d_input
        self.d_output = d_output

        self.config = (
            config
            if config is not None
            else SSMConfig(
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
                auto_residual_init=auto_residual_init,
                budget_init_utilization=budget_init_utilization,
                per_channel_gates=per_channel_gates,
                select_context_dim=select_context_dim,
                select_input=select_input,
                bcd_nonlinearity=bcd_nonlinearity,
                defect_state_metric=defect_state_metric,
                defect_block_size=defect_block_size,
                defect_max_radius=defect_max_radius,
                defect_factor_margin=defect_factor_margin,
                l2n_state_metric=l2n_state_metric,
            )
        )

        self._validate_config(self.config)
        self.use_cert_scaling = self.config.gamma is not None
        self._prescribed_gamma = float(self.config.gamma) if self.config.gamma is not None else None
        self.ff_has_lip = False

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
            if self.config.auto_residual_init:
                self._rescale_residual_init()

    def forward(
        self,
        u: torch.Tensor,
        state: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
        gamma=None,
        mode: str = "scan",
        reset_state: bool = True,
        detach_state: bool = False,
        context_gates: Sequence[dict[str, torch.Tensor]] | None = None,
        select_context: torch.Tensor | None = None,
    ):
        u3d = _normalize_to_3d(u)
        if gamma is not None and not self.use_cert_scaling:
            raise ValueError("gamma override requires a model constructed with gamma set.")
        # Each cell resolves its own reset/explicit-state precedence.
        n_blocks = len(self.blocks)
        if state is None:
            layer_states: list[torch.Tensor | None] = [None] * n_blocks
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

        if context_gates is not None and len(context_gates) != n_blocks:
            raise ValueError(
                f"context_gates must provide one entry per SSL block: "
                f"expected {n_blocks}, got {len(context_gates)}"
            )

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
                ssm_context_gate=(None if context_gates is None else context_gates[i].get("ssm")),
                ff_context_gate=(None if context_gates is None else context_gates[i].get("ff")),
                select_context=select_context,
                return_state=False,
            )
            layer_states[i] = st

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

    def reset(self):
        for block in self.blocks:
            block.lru.reset()

    @torch.no_grad()
    def _rescale_residual_init(self) -> None:
        """Set the residual-gate logits so the initial block product fits the budget.

        Fixed gates can make a deep stack start with a heavily attenuated
        decoder. Bisection chooses one shared logit with block product
        ``max(gamma, 1) / budget_init_utilization``. This retains the requested
        fraction of the largest achievable decoder scale, ``min(1, gamma)``.
        The target exceeds one, so it is reachable without zeroing the gates.
        """
        if not self.blocks:
            return
        utilization = float(self.config.budget_init_utilization)
        if not 0.0 < utilization < 1.0:
            raise ValueError(f"budget_init_utilization must be in (0, 1), got {utilization}.")
        target = max(float(self._prescribed_gamma), 1.0) / utilization
        device = self.encoder_w.device
        dtype = self.encoder_w.dtype
        terms = self._block_gain_terms(device=device, dtype=dtype, training=False)
        gammas = [float(t["gamma"]) for t in terms]
        lips = [float(t["ff_lip"]) for t in terms]
        if not all(math.isfinite(g) and math.isfinite(lip) for g, lip in zip(gammas, lips)):
            return  # uncertified components: nothing meaningful to balance

        def log_product(logit: float) -> float:
            alpha = 1.0 / (1.0 + math.exp(-logit))
            return sum(
                math.log1p(alpha * g) + math.log1p(alpha * lip) for g, lip in zip(gammas, lips)
            )

        log_target = math.log(target)
        lo, hi = -30.0, 30.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if log_product(mid) < log_target:
                lo = mid
            else:
                hi = mid
        logit = 0.5 * (lo + hi)
        for block in self.blocks:
            block.ssm_res_logit.fill_(logit)
            block.ff_res_logit.fill_(logit)

    _validate_config = staticmethod(validate_ssm_config)

    _spectrally_capped_weight = staticmethod(spectrally_capped_weight)

    def _block_gain_terms(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        training: bool | None = None,
        include_storage: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        return [
            block.gain_terms(
                device=device, dtype=dtype, training=training, include_storage=include_storage
            )
            for block in self.blocks
        ]

    def _log_block_gain_product(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        terms = self._block_gain_terms(device=device, dtype=dtype, include_storage=False)
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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
            not bool(torch.isfinite(term["gamma"])) or not bool(torch.isfinite(term["ff_lip"]))
            for term in terms
        ):
            return torch.full((), float("inf"), device=device, dtype=dtype)
        factors = torch.stack(
            [(1.0 + term["ssm_branch_gain"]) * (1.0 + term["ff_branch_gain"]) for term in terms]
        )
        return torch.exp(torch.log(factors).sum())

    @torch.no_grad()
    def certified_gain_bound(self, gamma=None) -> torch.Tensor:
        """Return the composed upper bound used for the current train/eval mode."""
        if not self.use_cert_scaling:
            raise RuntimeError("certified_gain_bound requires a prescribed gamma.")

        device = self.encoder_w.device
        dtype = self.encoder_w.dtype
        encoder_norm, decoder_norm = self._encoder_decoder_norms()

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

    @property
    def is_lpv(self) -> bool:
        """True when the per-step matrices depend only on exogenous context.

        Every selective cell must be in ``select_input='context'`` mode. The map
        from input to output is then linear time-varying for a fixed context
        sequence, which is what turns the zero-state certificate into an
        incremental one.
        """
        if not self.blocks:
            return False
        return all(getattr(block.lru, "select_input", "both") == "context" for block in self.blocks)

    @torch.no_grad()
    def incremental_gain_bound(self, gamma=None) -> torch.Tensor:
        """Bound on ``||Q(e) - Q(e~)|| <= g ||e - e~||``, or ``inf`` if none holds.

        With an input-dependent selector the two inputs choose *different*
        per-step matrices, so the zero-state bound says nothing about the
        difference of two trajectories and this returns ``inf``.

        Under ``select_input='context'`` and a fixed context sequence every factor
        in the cascade has an incremental counterpart with the *same* constant:
        each cell is linear time-varying, so the difference of two trajectories
        obeys the same recursion and the same ``gamma``; each feedforward branch
        is ``L``-Lipschitz, which is already a statement about differences; and
        the encoder, decoder and residual gates are linear. Composing them gives
        the same ``k_i = (1 + b_i L_i)(1 + a_i gamma_i)`` per block, so the
        incremental gain equals :meth:`certified_gain_bound`.

        Note the stack is *not* linear even in this mode -- the feedforward
        branches are nonlinear, so superposition does not hold. What holds is the
        bound on differences, which is what the interconnection argument needs.

        This is what makes tracking about a nonzero equilibrium well posed and
        gives the interconnection argument something stronger than finite gain to
        work with. It holds per context trajectory: two runs compared under
        *different* contexts are not covered, which is the usual LPV situation
        where the scheduling signal is treated as an exogenous input.
        """
        device = self.encoder_w.device if self.use_cert_scaling else next(self.parameters()).device
        dtype = self.encoder_w.dtype if self.use_cert_scaling else next(self.parameters()).dtype
        if not self.use_cert_scaling or not self.is_lpv:
            return torch.full((), float("inf"), device=device, dtype=dtype)
        return self.certified_gain_bound(gamma=gamma)

    @torch.no_grad()
    def storage_weights(self, gamma=None) -> torch.Tensor:
        """Per-block weights of the stack's storage function.

        The certificate ``||f||^2 <= Gamma^2 ||e||^2`` holds from zero state. For a
        nonzero state it becomes a dissipation inequality that needs a storage
        function, and this returns its per-block weights ``w_i`` so that

            V_Q(s) = sum_i w_i * ||s_i||^2,
            V_Q(s_{t+1}) - V_Q(s_t) + ||f_t||^2 <= Gamma^2 ||e_t||^2.

        Writing ``c_i`` for the SSM residual storage weight, ``l_i`` for the
        feedforward factor and ``k_i`` for the block factor,

            w_i = (||D|| * sigma)^2 * prod_{j>i} k_j^2 * l_i^2 * c_i,

        which is block ``i``'s storage carried through the blocks above it, the
        decoder and its attenuation. Composed in log space for the same reason
        :meth:`certified_gain_bound` is: ``prod k_j^2`` overflows and ``sigma^2``
        underflows well inside float32 for deep stacks.

        Returns a ``(n_blocks,)`` tensor; empty when the stack has no blocks.
        """
        if not self.use_cert_scaling:
            raise RuntimeError("storage_weights requires a prescribed gamma.")

        device = self.encoder_w.device
        dtype = self.encoder_w.dtype
        if len(self.blocks) == 0:
            return torch.zeros(0, device=device, dtype=dtype)

        terms = self._block_gain_terms(device=device, dtype=dtype)
        tiny = torch.finfo(dtype).tiny
        log_kappa = torch.stack([torch.log(t["block_factor"].clamp_min(tiny)) for t in terms])
        log_ell = torch.stack([torch.log(t["ff_factor"].clamp_min(tiny)) for t in terms])
        log_c = torch.stack([torch.log(t["c_ssm"].clamp_min(tiny)) for t in terms])

        # suffix_i = sum_{j>i} log k_j
        total = log_kappa.sum()
        suffix = total - torch.cumsum(log_kappa, dim=0)

        _, decoder_norm = self._encoder_decoder_norms()
        log_scale = self._smooth_capped_log_scale_from_logs(
            gamma_t=self._effective_gamma_cap(gamma=gamma, device=device, dtype=dtype),
            log_gamma_prod=self._log_block_gain_product(device=device, dtype=dtype),
            temperature=self.config.cert_scale_temperature,
        )
        log_out = 2.0 * (torch.log(decoder_norm.clamp_min(tiny)) + log_scale)

        log_w = log_out + 2.0 * suffix + 2.0 * log_ell + log_c
        weights = torch.exp(log_w)
        # A non-finite log_scale means the decoder is fully attenuated: no state
        # energy reaches the output, so the storage vanishes rather than blows up.
        return torch.where(torch.isfinite(log_scale), weights, torch.zeros_like(weights))

    @torch.no_grad()
    def storage_value(
        self,
        state: Sequence[torch.Tensor | None] | None,
        gamma=None,
    ) -> torch.Tensor:
        """Evaluate ``V_Q(s) = sum_i w_i V_i(s_i)`` for a stack state.

        ``state`` is the per-block state list :meth:`forward` returns. Returns a
        ``(batch,)`` tensor, or a scalar zero for ``None`` / an empty stack.

        Cells with a state_energy() hook supply their own certified storage
        (e.g. x.T P x for full-metric defect). Other cells use squared norms.
        """
        weights = self.storage_weights(gamma=gamma)
        if state is None or weights.numel() == 0:
            return torch.zeros((), device=self.encoder_w.device, dtype=self.encoder_w.dtype)
        if len(state) != len(self.blocks):
            raise ValueError(
                f"state must provide exactly one entry per SSL block: "
                f"expected {len(self.blocks)}, got {len(state)}"
            )

        total = None
        for w_i, s_i, block in zip(weights, state, self.blocks):
            hook = getattr(block.lru, "state_energy", None)
            energy = (
                hook(s_i) if callable(hook)
                else _state_energy(s_i, device=weights.device, dtype=weights.dtype)
            )
            if energy is None:
                continue
            energy = energy.to(device=weights.device, dtype=weights.dtype)
            contribution = w_i * energy
            total = contribution if total is None else total + contribution
        if total is None:
            return torch.zeros((), device=weights.device, dtype=weights.dtype)
        return total

    @torch.no_grad()
    def initial_storage_bound(
        self,
        state: Sequence[torch.Tensor | None] | None,
        gamma=None,
    ) -> torch.Tensor:
        """``b_Q = sqrt(V_Q(s_0))``, the offset a nonzero initial state adds.

        This is the term the closed-loop finite-gain argument needs:
        ``||v||_T <= gamma_Q ||e||_T + b_Q(s_0)``. A chunk of a continuing
        trajectory starts from a nonzero state, so it is not a fresh zero-state
        experiment and this term does not vanish.
        """
        return torch.sqrt(self.storage_value(state, gamma=gamma).clamp_min(0.0))

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
            block_rows.append(
                {
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
                    "c_ssm": float(term["c_ssm"].detach().cpu()),
                }
            )

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
            encoder_norm_t, decoder_norm_t = self._encoder_decoder_norms()

            global_gamma = float(gamma_cap.detach().cpu())
            smooth_scale = float(smooth.detach().cpu())
            hard_scale = float(hard.detach().cpu())
            certified_bound = float(self.certified_gain_bound().detach().cpu())
            encoder_norm = float(encoder_norm_t.detach().cpu())
            decoder_norm = float(decoder_norm_t.detach().cpu())
            storage = self.storage_weights()
            for row, w_i in zip(block_rows, storage.detach().cpu().tolist()):
                row["storage_weight"] = w_i

        return {
            "mode": "train" if self.training else "eval",
            "use_cert_scaling": self.use_cert_scaling,
            "global_gamma": global_gamma,
            "gamma_prod": float(gamma_prod.detach().cpu()),
            "conservative_gamma_prod": float(conservative.detach().cpu()),
            "smooth_scale": smooth_scale,
            "hard_scale": hard_scale,
            # How much of the prescribed budget the blocks actually use. Well
            # below 1 means the decoder is being attenuated to compensate for an
            # oversized block product, which costs signal scale at no benefit --
            # it grows geometrically with depth at a fixed residual-gate init.
            # See ``auto_residual_init``.
            "budget_utilization": (
                None if global_gamma is None else float(gamma_prod) / global_gamma
            ),
            "certified_gain_bound": certified_bound,
            "encoder_norm": encoder_norm,
            "decoder_norm": decoder_norm,
            "n_blocks": len(block_rows),
            "blocks": block_rows,
        }

    @torch.no_grad()
    def certificate_fingerprint(self) -> str:
        """Hash every quantity the certificate depends on.

        Freezing deployment parameters is an assumption of the whole argument:
        the storage weights and the decoder attenuation are derived from the
        layer gains, so recomputing a cap after those gains move does not account
        for energy already stored under the old ones. This gives a value to
        assert on either side of a deployment, or across an online update, to
        show that nothing which determines the bound has changed.

        Covers the encoder and decoder weights, every block's gain terms, the
        prescribed gain and the resulting bound -- not the selector, gate or
        mixer weights, which the certificate is deliberately indifferent to.
        """
        import hashlib

        device = self.encoder_w.device
        dtype = self.encoder_w.dtype
        digest = hashlib.sha256()
        digest.update(f"cert-v1|{self.config.param}|{self.config.ff}|".encode())
        digest.update(f"{self.use_cert_scaling}|{self._prescribed_gamma}|".encode())
        for name in ("encoder_w", "decoder_w"):
            w = getattr(self, name)
            digest.update(name.encode())
            digest.update(w.detach().float().cpu().contiguous().numpy().tobytes())
        for index, term in enumerate(self._block_gain_terms(device=device, dtype=dtype)):
            digest.update(f"block{index}".encode())
            for key in ("gamma", "ff_lip", "alpha_ssm", "alpha_ff", "c_ssm", "block_factor"):
                digest.update(f"{key}={float(term[key].detach().cpu()):.12e}|".encode())
        if self.use_cert_scaling:
            digest.update(f"bound={float(self.certified_gain_bound()):.12e}".encode())
        return digest.hexdigest()

    @torch.no_grad()
    def freeze_certificate(self) -> str:
        """Freeze every parameter the certificate depends on; return its fingerprint.

        Leaves the selector, gate and mixer pathways trainable: the per-step
        renormalization means the bound holds for any value they take, so they
        remain free to adapt online. Encoder, decoder, cell gains, residual gates
        and feedforward Lipschitz budgets are fixed, because the bound is derived
        from them and a change invalidates the stored energy accounted for by
        :meth:`storage_weights`.
        """
        for name in ("encoder_w", "decoder_w"):
            weight = getattr(self, name, None)
            if isinstance(weight, nn.Parameter):
                weight.requires_grad_(False)
        for block in self.blocks:
            block.ssm_res_logit.requires_grad_(False)
            block.ff_res_logit.requires_grad_(False)
            raw_lip = getattr(block.ff, "raw_lip", None)
            if isinstance(raw_lip, nn.Parameter):
                raw_lip.requires_grad_(False)
            for attr in ("log_gamma", "gamma", "gamma_raw"):
                value = getattr(block.lru, attr, None)
                if isinstance(value, nn.Parameter):
                    value.requires_grad_(False)
        if isinstance(getattr(self, "gamma_t", None), nn.Parameter):
            self.gamma_t.requires_grad_(False)
        return self.certificate_fingerprint()

    _last_runtime_state = staticmethod(last_runtime_state)

    _smooth_capped_scale = staticmethod(smooth_capped_scale)

    _smooth_capped_log_scale_from_logs = staticmethod(smooth_capped_log_scale_from_logs)

    _smooth_capped_scale_from_logs = staticmethod(smooth_capped_scale_from_logs)

    def _capped_encoder_decoder(self):
        """Reuse spectral caps only in gradient-free eval, with fresh weights."""
        return self._eval_cached(
            "encoder_decoder",
            lambda: (
                self._spectrally_capped_weight(self.encoder_w),
                self._spectrally_capped_weight(self.decoder_w),
            ),
        )

    def _encoder_decoder_norms(self):
        """Reuse the same capped weights and their norms in gain diagnostics."""

        def compute():
            return tuple(
                torch.linalg.matrix_norm(w.float(), ord=2).to(dtype=w.dtype)
                for w in self._capped_encoder_decoder()
            )

        return self._eval_cached("encoder_decoder_norms", compute)


# Simple recurrent baselines


class PureLRUR(nn.Module):
    """Pure LRU block without scaffolding."""

    def __init__(
        self,
        n: int,
        gamma: float = None,
        param: str = "l2ru",
        init: str = "eye",
        learn_x0: bool = False,
    ):
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
        state: torch.Tensor | None = None,
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
    """nn.RNN baseline with the same input and reset conventions as DeepSSM.

    Inputs are (B,T,d_input), (T,d_input), or a single feature vector. Initial
    states accept a vector, (B,d_hidden), or the full (layers*directions,B,H).
    By default, return y and the final state of every layer/direction.

    return_state=True exposes the top-layer outputs with its initial state
    prepended, shaped (B,T+1,directions*H). return_last=True exposes only the
    top-layer final state, shaped (B,directions*H). Enabling both returns
    (y, trajectory, top_layer_last).
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
        dropout: float = 0.0,  # only applied if num_layers > 1 (PyTorch behavior)
        bidirectional: bool = False,
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
        self.state: torch.Tensor | None = None

        # Learnable initial hidden state: shape (num_layers * num_directions, 1, d_hidden)
        L = self.num_layers * self.num_directions
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(L, 1, self.d_hidden))
        else:
            self.register_buffer("x0_param", None)

    def _format_h0(self, h0: torch.Tensor | None, B: int, device, dtype) -> torch.Tensor:
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
            h0 = h0.unsqueeze(0)  # (1,B,H)
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
        state: torch.Tensor | None = None,  # h0
        *,
        return_state: bool = False,
        return_last: bool = False,
        reset_state: bool = True,
        detach_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        u3d = _normalize_to_3d(u)  # (B,T,D)
        B, T, D = u3d.shape
        if D != self.d_input:
            raise ValueError(f"Expected input dim {self.d_input}, got {D}")

        source_state = state
        if source_state is None:
            source_state = (
                self.x0_param if reset_state else self.state
            )  # None when learn_x0=False → zeros
        h0 = self._format_h0(source_state, B, u3d.device, u3d.dtype)

        # Run RNN
        out, hT = self.rnn(u3d, h0)  # out: (B,T,H*num_dir), hT: (L,B,H)
        self.state = hT.detach() if detach_state else hT

        # Project to output dim
        y = self.out_proj(out)  # (B,T,d_output)

        # nn.RNN exposes the top-layer sequence, not every layer's trajectory.
        h_seq = None
        if return_state:
            # top-layer h_t sequence is out; prepend the top-layer initial state
            top_layer_idx = self.num_directions * (self.num_layers - 1)
            h0_top = h0[top_layer_idx : top_layer_idx + self.num_directions]  # (num_dir,B,H)
            # Concatenate directions in the same channel order as nn.RNN's output.
            h_seq = torch.empty(B, T + 1, out.size(-1), device=u3d.device, dtype=u3d.dtype)
            h_seq[:, 0, :] = (
                torch.cat([h0_top[d] for d in range(self.num_directions)], dim=-1)
                if self.num_directions > 1
                else h0_top[0]
            )
            h_seq[:, 1:, :] = out

        # last hidden (top layer)
        h_last = None
        if return_last:
            top_layer = hT[-self.num_directions :]  # (num_dir,B,H)
            h_last = (
                torch.cat([top_layer[d] for d in range(self.num_directions)], dim=-1)
                if self.num_directions > 1
                else top_layer[0]
            )

        if return_state and return_last:
            return y, h_seq, h_last
        if return_state:
            return y, h_seq
        if return_last:
            return y, h_last
        return y, self.state

    def reset(self):
        self.state = reset_runtime_state(self.state, x0=self.x0_param)


__all__ = [
    "SSMConfig",
    "SSMConfigDict",
    "SSMParametrization",
    "SSL",
    "DeepSSM",
    "PureLRUR",
    "SimpleRNN",
    "LRU",
    "L2RU",
    "lruz",
    "L2BoundedLTICell",
    "Block2x2DenseL2SSM",
    "DefectL2SSM",
    "RobustMambaDiagSSM",
    "RobustMambaDiagLTI",
]

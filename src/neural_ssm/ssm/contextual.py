from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn as nn

from .layers import DeepSSM, SSMConfig
from .lti_cells import _normalize_to_3d


_VALID_CONTEXT_MODES = frozenset({"input", "gate", "mixer"})


def timewise_matrix_vector_product(matrices: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Apply ``A_t e_t`` at every batch/time index."""
    if matrices.shape[:-2] != vectors.shape[:-1] or matrices.shape[-1] != vectors.shape[-1]:
        raise ValueError(
            "Expected matrices (..., d_output, d_features) and vectors "
            f"(..., d_features), got {tuple(matrices.shape)} and {tuple(vectors.shape)}."
        )
    return torch.matmul(matrices, vectors.unsqueeze(-1)).squeeze(-1)


def _as_tuple(modes: Optional[Union[str, Sequence[str]]]) -> tuple[str, ...]:
    if modes is None:
        return ()
    if isinstance(modes, str):
        modes = (modes,)
    result = tuple(dict.fromkeys(str(mode).lower() for mode in modes))
    unknown = set(result) - set(_VALID_CONTEXT_MODES)
    if unknown:
        raise ValueError(
            f"Unknown context mode(s) {sorted(unknown)}. "
            f"Available modes are {sorted(_VALID_CONTEXT_MODES)}."
        )
    return result


def _make_mlp(
    d_input: int,
    d_output: int,
    *,
    hidden_dim: int,
    n_layers: int,
    activation: Callable[[], nn.Module],
    bias: bool = True,
) -> nn.Sequential:
    if n_layers < 0:
        raise ValueError("n_layers must be non-negative.")
    if n_layers == 0:
        return nn.Sequential(nn.Linear(d_input, d_output, bias=bias))

    layers: list[nn.Module] = [nn.Linear(d_input, hidden_dim, bias=bias), activation()]
    for _ in range(n_layers - 1):
        layers.extend((nn.Linear(hidden_dim, hidden_dim, bias=bias), activation()))
    layers.append(nn.Linear(hidden_dim, d_output, bias=bias))
    return nn.Sequential(*layers)


class _ContextFilter(nn.Module):
    """L2-admissible projection for the input-augmentation (additive) context path.

    Maps a bounded context channel into an l2 sequence so it can be fed as an
    ordinary SSM input. Two families:

    * multiplicative windows ``gamma_t * z_t`` -- ``finite_horizon`` (loss-free
      on the operating horizon, then zero), ``taper`` (smooth raised-cosine
      roll-off, kinder gradients), ``exponential``/``polynomial`` (anytime l2),
      ``none`` (no projection: sys-ID / finite-gain regime, NOT l2);
    * differencing ``z_t - z_{t-1}`` (``difference``) -- l2 for free on
      bounded-variation / switching context, loss-free, and self-vanishing once
      the context settles.
    """

    def __init__(
        self,
        mode: Optional[Union[str, nn.Module, Callable[..., torch.Tensor]]],
        *,
        horizon: Optional[int],
        decay: float,
        power: float,
        scale: float,
        trainable: bool,
        rho_max: float,
        ramp: Optional[int] = None,
    ):
        super().__init__()
        self.custom = mode if callable(mode) and not isinstance(mode, str) else None
        self.mode = (
            "none"
            if mode is None
            else str(mode).lower()
            if self.custom is None
            else "custom"
        )
        self.horizon = horizon
        self.ramp = None if ramp is None else int(ramp)
        self.scale = float(scale)
        self.trainable = bool(trainable)
        self.rho_max = float(rho_max)

        if self.mode == "auto":
            self.mode = "finite_horizon" if horizon is not None else "exponential"
        if self.mode not in {
            "none", "custom", "finite_horizon", "taper",
            "exponential", "polynomial", "difference",
        }:
            raise ValueError(
                "context_filter must be one of None, 'auto', 'finite_horizon', 'taper', "
                "'exponential', 'polynomial', 'difference', or a callable nn.Module."
            )
        if self.mode in {"finite_horizon", "taper"}:
            if horizon is None or int(horizon) <= 0:
                raise ValueError(f"{self.mode} context_filter requires a positive horizon.")
            if trainable:
                raise ValueError(f"{self.mode} context_filter cannot be trainable.")
            self.horizon = int(horizon)
        if self.mode == "taper":
            self.ramp = int(self.horizon) if self.ramp is None else int(self.ramp)
            if not 1 <= self.ramp <= int(self.horizon):
                raise ValueError("taper context_filter requires 1 <= ramp <= horizon.")
        if self.mode == "difference" and trainable:
            raise ValueError("difference context_filter cannot be trainable.")
        if self.mode == "exponential":
            if not 0.0 <= float(decay) < 1.0:
                raise ValueError("context_filter_decay must be in [0, 1).")
            if not 0.0 < self.rho_max <= 1.0:
                raise ValueError("trainable_filter_rho_max must be in (0, 1].")
            if trainable:
                initial = min(max(float(decay), 1e-6), self.rho_max - 1e-6)
                ratio = initial / self.rho_max
                self.raw_decay = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
            else:
                self.register_buffer("fixed_decay", torch.tensor(float(decay)))
        if self.mode == "polynomial":
            if float(power) <= 0.5:
                raise ValueError("polynomial context_filter requires power > 0.5.")
            if trainable:
                self.raw_power = nn.Parameter(torch.tensor(math.log(math.exp(power - 0.5) - 1.0)))
            else:
                self.register_buffer("fixed_power", torch.tensor(float(power)))

    @property
    def decay(self) -> Optional[torch.Tensor]:
        if self.mode != "exponential":
            return None
        if hasattr(self, "raw_decay"):
            return self.rho_max * torch.sigmoid(self.raw_decay)
        return self.fixed_decay

    @property
    def power(self) -> Optional[torch.Tensor]:
        if self.mode != "polynomial":
            return None
        if hasattr(self, "raw_power"):
            return 0.5 + torch.nn.functional.softplus(self.raw_power)
        return self.fixed_power

    def weights(
        self,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        time_offset: int,
    ) -> torch.Tensor:
        if self.mode == "none":
            return torch.ones(length, device=device, dtype=dtype) * self.scale
        steps = torch.arange(time_offset, time_offset + length, device=device, dtype=dtype)
        if self.mode == "finite_horizon":
            return self.scale * (steps < float(self.horizon)).to(dtype=dtype)
        if self.mode == "taper":
            flat_until = float(self.horizon - self.ramp)
            roll = 0.5 * (1.0 + torch.cos(math.pi * (steps - flat_until) / float(self.ramp)))
            window = torch.where(steps < flat_until, torch.ones_like(steps), roll)
            window = torch.where(steps < float(self.horizon), window, torch.zeros_like(steps))
            return self.scale * window
        if self.mode == "exponential":
            return self.scale * torch.pow(self.decay.to(device=device, dtype=dtype), steps)
        if self.mode == "polynomial":
            return self.scale * torch.pow(steps + 1.0, -self.power.to(device=device, dtype=dtype))
        raise RuntimeError(f"context_filter mode {self.mode!r} does not expose deterministic weights.")

    def forward(
        self,
        context: torch.Tensor,
        *,
        time_offset: int = 0,
        prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.custom is not None:
            try:
                return self.custom(context, time_offset=time_offset)
            except TypeError:
                if time_offset != 0:
                    raise TypeError("Custom context_filter must accept time_offset for streaming.")
                return self.custom(context)

        z = _normalize_to_3d(context)
        if self.mode == "difference":
            # z_t - z_{t-1}: l2 for bounded-variation context, vanishes once the
            # context settles. ``prev`` carries the previous chunk's last sample
            # for streaming (default: z_0, so the first increment is 0).
            prev_frame = z[:, :1] if prev is None else _normalize_to_3d(prev)[:, -1:]
            shifted = torch.cat((prev_frame, z[:, :-1]), dim=1)
            return self.scale * (z - shifted)

        window = self.weights(
            z.shape[1],
            device=z.device,
            dtype=z.dtype,
            time_offset=int(time_offset),
        )
        return z * window.reshape(1, -1, 1)

    def weight_l2_norm(self, length: Optional[int] = None, *, time_offset: int = 0) -> float:
        """``||window||_2`` over ``[time_offset, ...)`` (or a finite upper bound).

        Returns ``inf`` for modes that are not fixed l2 windows (``none``,
        ``custom``, ``difference``). Consumed by
        :meth:`ContextualDeepSSM.context_offset_bound`.
        """
        start = int(time_offset)
        scale = abs(self.scale)
        if self.mode in {"none", "custom", "difference"}:
            return float("inf")
        if self.mode in {"finite_horizon", "taper"}:
            span = max(int(self.horizon) - start, 0)
            if length is not None:
                span = min(span, max(int(length), 0))
            if span == 0:
                return 0.0
            w = self.weights(span, device=torch.device("cpu"), dtype=torch.float32, time_offset=start)
            return float(torch.linalg.vector_norm(w).item())
        if self.mode == "exponential":
            decay = float(self.decay)
            if decay == 0.0:
                return scale if (start == 0 and (length is None or length > 0)) else 0.0
            first = decay ** (2 * start)
            if length is None:
                return scale * math.sqrt(first / (1.0 - decay ** 2))
            length = max(int(length), 0)
            if length == 0:
                return 0.0
            total = first * (1.0 - decay ** (2 * length)) / (1.0 - decay ** 2)
            return scale * math.sqrt(max(total, 0.0))
        exponent = 2.0 * float(self.power)
        if length is not None:
            length = max(int(length), 0)
            total = sum((start + i + 1) ** (-exponent) for i in range(length))
            return scale * math.sqrt(total)
        base = float(start + 1)
        total_bound = base ** (-exponent) + base ** (1.0 - exponent) / (exponent - 1.0)
        return scale * math.sqrt(total_bound)


class _BoundedMixer(nn.Module):
    def __init__(
        self,
        d_disturbance: int,
        d_context: int,
        d_features: int,
        d_output: int,
        *,
        hidden_dim: int,
        n_layers: int,
        matrix_bound: float,
        include_disturbance: bool,
    ):
        super().__init__()
        self.d_disturbance = int(d_disturbance)
        self.d_context = int(d_context)
        self.d_features = int(d_features)
        self.d_output = int(d_output)
        self.matrix_bound = float(matrix_bound)
        self.include_disturbance = bool(include_disturbance)
        if self.matrix_bound <= 0.0:
            raise ValueError("mixer_bound must be positive.")
        d_in = self.d_context + (self.d_disturbance if self.include_disturbance else 0)
        if d_in <= 0:
            raise ValueError("Mixer needs context, disturbance, or both.")
        self.net = _make_mlp(
            d_in,
            self.d_output * self.d_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=nn.GELU,
        )

    @property
    def matrix_norm_bound(self) -> float:
        return self.matrix_bound

    def forward(self, disturbance: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        inputs = []
        if self.include_disturbance:
            inputs.append(disturbance)
        if self.d_context > 0:
            if context is None:
                raise ValueError("context is required for mixer mode.")
            inputs.append(context)
        x = torch.cat(inputs, dim=-1)
        matrices = self.net(x).reshape(*x.shape[:2], self.d_output, self.d_features)
        norm = torch.linalg.vector_norm(matrices, ord=2, dim=(-2, -1), keepdim=True)
        scale = torch.clamp(norm / self.matrix_bound, min=1.0)
        return matrices / scale


class _ContextGate(nn.Module):
    def __init__(
        self,
        d_disturbance: int,
        d_context: int,
        d_model: int,
        n_layers_core: int,
        *,
        hidden_dim: int,
        n_layers: int,
        per_channel: bool,
        include_disturbance: bool,
    ):
        super().__init__()
        self.d_context = int(d_context)
        self.d_model = int(d_model)
        self.n_layers_core = int(n_layers_core)
        self.per_channel = bool(per_channel)
        self.include_disturbance = bool(include_disturbance)
        self.gate_dim = self.d_model if self.per_channel else 1
        d_in = self.d_context + (d_disturbance if self.include_disturbance else 0)
        if self.n_layers_core == 0:
            self.net = None
            return
        if d_in <= 0:
            raise ValueError("Gate mode needs context, disturbance, or both.")
        self.net = _make_mlp(
            d_in,
            2 * self.n_layers_core * self.gate_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=nn.GELU,
        )

    def forward(
        self,
        disturbance: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> list[dict[str, torch.Tensor]]:
        if self.net is None:
            return []
        inputs = []
        if self.include_disturbance:
            inputs.append(disturbance)
        if self.d_context > 0:
            if context is None:
                raise ValueError("context is required for gate mode.")
            inputs.append(context)
        x = torch.cat(inputs, dim=-1)
        gates = torch.sigmoid(self.net(x))
        gates = gates.reshape(*x.shape[:2], self.n_layers_core, 2, self.gate_dim)
        return [
            {"ssm": gates[:, :, i, 0], "ff": gates[:, :, i, 1]}
            for i in range(self.n_layers_core)
        ]


class ContextualDeepSSM(nn.Module):
    """Context-enriched DeepSSM with input, gate, and mixer injection modes.

    ``context_modes`` can contain any combination of:

    - ``"input"``: concatenate ``L2``-filtered context to the SSM input.
    - ``"gate"``: use bounded context gates inside every residual SSM block.
    - ``"mixer"``: use a bounded timewise matrix ``A_t`` on the core features.

    The recurrent core is still a standard ``DeepSSM``. Therefore, when
    ``gamma`` is set, the usual certified recurrent parametrizations and
    Lipschitz feedforwards are enforced by ``DeepSSM`` itself.
    """

    def __init__(
        self,
        d_input: int,
        d_context: int,
        d_output: int,
        *,
        context_modes: Optional[Union[str, Sequence[str]]] = ("mixer",),
        d_features: Optional[int] = None,
        context_filter: Optional[Union[str, nn.Module, Callable[..., torch.Tensor]]] = "auto",
        context_filter_decay: float = 0.98,
        context_filter_power: float = 1.0,
        context_filter_scale: float = 1.0,
        horizon: Optional[int] = None,
        context_filter_ramp: Optional[int] = None,
        trainable_context_filter: bool = False,
        trainable_filter_rho_max: float = 0.999,
        mixer_bound: float = 1.0,
        mixer_hidden_dim: int = 64,
        mixer_layers: int = 2,
        mixer_include_disturbance: bool = True,
        gate_hidden_dim: int = 64,
        gate_layers: int = 2,
        gate_per_channel: bool = False,
        gate_include_disturbance: bool = False,
        context_encoder: Optional[nn.Module] = None,
        ssm_config: Optional[SSMConfig] = None,
        ssm_kwargs: Optional[dict[str, Any]] = None,
        **deep_ssm_kwargs: Any,
    ):
        super().__init__()
        if d_input <= 0 or d_context < 0 or d_output <= 0:
            raise ValueError(
                "d_input and d_output must be positive; d_context must be non-negative."
            )

        self.d_input = int(d_input)
        self.d_context = int(d_context)
        self.d_output = int(d_output)
        self.context_modes = _as_tuple(context_modes)
        self.context_encoder = context_encoder

        if self.context_modes and self.d_context == 0:
            if "input" in self.context_modes:
                raise ValueError("input context mode requires d_context > 0.")
            if "gate" in self.context_modes and not gate_include_disturbance:
                raise ValueError(
                    "gate mode with d_context=0 requires gate_include_disturbance=True."
                )
            if "mixer" in self.context_modes and not mixer_include_disturbance:
                raise ValueError(
                    "mixer mode with d_context=0 requires mixer_include_disturbance=True."
                )

        if ssm_kwargs is not None and deep_ssm_kwargs:
            raise ValueError("Pass either ssm_kwargs or DeepSSM keyword args, not both.")
        core_kwargs = dict(ssm_kwargs or deep_ssm_kwargs)
        if ssm_config is not None and core_kwargs:
            raise ValueError("Pass either ssm_config or DeepSSM keyword args, not both.")

        self.d_features = int(d_features or d_output)
        self.core_input_dim = (
            self.d_input + (self.d_context if "input" in self.context_modes else 0)
        )
        self.core_output_dim = self.d_features if "mixer" in self.context_modes else self.d_output
        self.core = DeepSSM(
            self.core_input_dim,
            self.core_output_dim,
            config=ssm_config,
            **core_kwargs,
        )

        self.context_filter = None
        if "input" in self.context_modes:
            self.context_filter = _ContextFilter(
                context_filter,
                horizon=horizon,
                decay=context_filter_decay,
                power=context_filter_power,
                scale=context_filter_scale,
                trainable=trainable_context_filter,
                rho_max=trainable_filter_rho_max,
                ramp=context_filter_ramp,
            )

        self.gate = None
        if "gate" in self.context_modes:
            self.gate = _ContextGate(
                self.d_input,
                self.d_context,
                self.core.config.d_model,
                self.core.config.n_layers,
                hidden_dim=gate_hidden_dim,
                n_layers=gate_layers,
                per_channel=gate_per_channel,
                include_disturbance=gate_include_disturbance,
            )

        self.mixer = None
        if "mixer" in self.context_modes:
            self.mixer = _BoundedMixer(
                self.d_input,
                self.d_context,
                self.core_output_dim,
                self.d_output,
                hidden_dim=mixer_hidden_dim,
                n_layers=mixer_layers,
                matrix_bound=mixer_bound,
                include_disturbance=mixer_include_disturbance,
            )

    def _encode_context(self, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if context is None:
            return None
        z = self.context_encoder(context) if self.context_encoder is not None else context
        z = _normalize_to_3d(z)
        if z.shape[-1] != self.d_context:
            raise ValueError(f"context last dimension must be {self.d_context}, got {z.shape[-1]}.")
        return z

    def _prepare_inputs(
        self,
        disturbance: torch.Tensor,
        context: Optional[torch.Tensor],
        *,
        time_offset: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        w = _normalize_to_3d(disturbance)
        if w.shape[-1] != self.d_input:
            raise ValueError(
                f"disturbance last dimension must be {self.d_input}, got {w.shape[-1]}."
            )

        z = self._encode_context(context)
        if self.context_modes and self.d_context > 0:
            if z is None:
                raise ValueError("context is required when context_modes is non-empty.")
            if z.shape[:2] != w.shape[:2]:
                raise ValueError(
                    "context and disturbance must have matching batch/time dimensions; "
                    f"got {tuple(z.shape[:2])} and {tuple(w.shape[:2])}."
                )

        filtered_context = None
        if "input" in self.context_modes:
            filtered_context = self.context_filter(z, time_offset=time_offset)
            core_input = torch.cat((w, filtered_context), dim=-1)
        else:
            core_input = w
        return w, z, core_input, filtered_context

    def forward(
        self,
        disturbance: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        state: Optional[Union[torch.Tensor, Sequence[Optional[torch.Tensor]]]] = None,
        gamma=None,
        mode: str = "scan",
        reset_state: bool = True,
        detach_state: bool = False,
        time_offset: int = 0,
        return_aux: bool = False,
    ):
        w, z, core_input, filtered_context = self._prepare_inputs(
            disturbance,
            context,
            time_offset=time_offset,
        )
        context_gates = self.gate(w, z) if self.gate is not None else None
        features, next_state = self.core(
            core_input,
            state=state,
            gamma=gamma,
            mode=mode,
            reset_state=reset_state,
            detach_state=detach_state,
            context_gates=context_gates,
        )

        mixer_matrices = None
        if self.mixer is not None:
            mixer_matrices = self.mixer(w, z)
            outputs = timewise_matrix_vector_product(mixer_matrices, features)
        else:
            outputs = features

        if not return_aux:
            return outputs, next_state
        return outputs, next_state, {
            "core_input": core_input,
            "filtered_context": filtered_context,
            "context_gates": context_gates,
            "features": features,
            "mixer": mixer_matrices,
        }

    @property
    def matrix_norm_bound(self) -> float:
        return 1.0 if self.mixer is None else self.mixer.matrix_norm_bound

    @torch.no_grad()
    def certified_gain_bound(self, gamma=None) -> torch.Tensor:
        """Return a conservative bound from the core input to the output."""
        bound = self.core.certified_gain_bound(gamma=gamma)
        return bound * float(self.matrix_norm_bound)

    @torch.no_grad()
    def additive_channel_gain(self, gamma=None) -> float:
        """l2 gain from the (filtered) additive context channel to the output.

        For *exogenous* additive context this is informational. For *endogenous*
        (in-loop) context fed via the ``input`` mode it is a state-feedback gain:
        closed-loop l2-stability then needs a small-gain margin
        ``additive_channel_gain * window_sup * plant_l2_gain < 1`` (``window_sup``
        is the projection's peak ``scale``). Prefer routing endogenous signals
        through the ``mixer`` mode, which is unconditionally safe.
        """
        return float(self.core.certified_gain_bound(gamma=gamma)) * float(self.matrix_norm_bound)

    @torch.no_grad()
    def context_offset_bound(
        self,
        context_amplitude_bound: float,
        length: Optional[int] = None,
        *,
        time_offset: int = 0,
        gamma=None,
    ) -> float:
        """Bound the additive context's contribution to ``||u||_2`` (the bias term).

        ``core_gain * ||A||_inf * ||window||_2 * sup_t||z_t||`` for the ``input``
        mode. Returns ``0`` when ``input`` is unused and ``inf`` for non-l2
        windows (``none``/``difference``) -- inspect the realised
        ``filtered_context`` norm via ``return_aux`` for those.
        """
        if "input" not in self.context_modes or self.context_filter is None:
            return 0.0
        if context_amplitude_bound < 0 or not math.isfinite(float(context_amplitude_bound)):
            raise ValueError("context_amplitude_bound must be finite and non-negative.")
        window_norm = self.context_filter.weight_l2_norm(length, time_offset=time_offset)
        if not math.isfinite(window_norm):
            return float("inf")
        gain = float(self.core.certified_gain_bound(gamma=gamma)) * float(self.matrix_norm_bound)
        return gain * window_norm * float(context_amplitude_bound)

    @torch.no_grad()
    def gain_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "context_modes": self.context_modes,
            "d_input": self.d_input,
            "d_context": self.d_context,
            "d_output": self.d_output,
            "d_features": self.d_features,
            "core_input_dim": self.core_input_dim,
            "core_output_dim": self.core_output_dim,
            "matrix_norm_bound": self.matrix_norm_bound,
        }
        if self.context_filter is not None:
            diagnostics["context_filter"] = self.context_filter.mode
            diagnostics["context_filter_l2_norm"] = self.context_filter.weight_l2_norm()
        if hasattr(self.core, "gain_diagnostics"):
            diagnostics["core"] = self.core.gain_diagnostics()
        if hasattr(self.core, "certified_gain_bound"):
            diagnostics["certified_gain_bound"] = float(self.certified_gain_bound().detach().cpu())
        return diagnostics

    def reset(self):
        self.core.reset()

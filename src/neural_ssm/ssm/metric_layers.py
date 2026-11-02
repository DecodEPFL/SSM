"""Metric-transfer SSMs: feature metrics, certified residual readout, and stack.

Each layer proves ||dx_next||²_P + ||dy||²_Q <= ||dx||²_P + ||du||²_Qprev.
Intermediate Q terms cancel across layers. Endpoints gamma² I and I give a
whole-network incremental l2 bound, without a product of scalar layer bounds.

The core protocol is normalized_block_matrices() -> (A_blocks, B, C, D),
certifying unit input/output gain. State storage is P=I unless the core exposes
state_energy(); full-metric defect and l2n use x.T P x. Dense A is allowed as one block.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.runtime import EvalCacheMixin, normalize_to_3d, resolve_runtime_state, sequence_output
from ..utils.scan import compute_block_lti_states
from .cells.lti import Block2x2DenseL2SSM, DefectL2SSM
from .cells.lti.defect import ThinContraction


@dataclass(frozen=True)
class MetricSSMConfig:
    """Settings for the metric stack; ff is 'residual' or 'none', not a GLU."""

    d_model: int = 16
    d_state: int = 64
    n_layers: int = 2
    param: str = "defect"
    gamma: float = 1.0
    activation: str = "tanh"
    ff: str = "residual"
    d_hidden: int = 32
    block_size: int = 2
    max_radius: float = 0.999
    factor_margin: float = 1e-3
    rho: float = 0.9
    max_phase: float = 0.04
    phase_center: float = 0.0
    random_phase: bool = True
    init_scale: float = 0.02
    init_feedthrough: float = 0.95
    metric_floor: float = 1e-4
    state_metric: str = "identity"  # defect/l2n: identity or full state storage


def _positive_raw(value: float, floor: float) -> float:
    return math.log(math.expm1(value - floor))


def _activate(x, name):
    if name == "tanh":
        return x.tanh()
    if name == "relu":
        return F.relu(x)
    if name == "arctan":
        return x.atan()
    return x  # identity; constructor rejects other names


def _thin_matrix(factors):
    left, values, right, _ = factors
    return (left * values) @ right.mT


def _initialize_diagonal(contraction, value):
    """Aligned thin bases give value times a rectangular identity at startup."""
    with torch.no_grad():
        contraction.left_raw.copy_(torch.eye(*contraction.left_raw.shape))
        contraction.right_raw.copy_(torch.eye(*contraction.right_raw.shape))
        contraction.values_raw.fill_(math.atanh(value / contraction.limit))


class FeatureMetric(nn.Module):
    """Learn Q and W with 0<Q<2V; terminal Q=I is fixed.

    Interior: S=L L.T, Q=2 V^.5 S(I+S)^-1 V^.5,
    W=.5 V^.5(I+S)V^.5. Cholesky solves avoid explicit inverses and SVDs.
    The positive diagonal floor is a stated finite parametrization margin.
    """

    def __init__(self, width, floor=1e-4, *, terminal=False):
        super().__init__()
        self.width, self.floor, self.terminal = width, floor, terminal
        self.register_buffer("identity", torch.eye(width))
        if not terminal:
            self.v_raw = nn.Parameter(torch.full((width,), _positive_raw(1.0, floor)))
            indices = torch.tril_indices(width, width, offset=-1)
            self.register_buffer("indices", indices, persistent=False)
            self.lower_raw = nn.Parameter(torch.zeros(indices.shape[1]))
            self.diag_raw = nn.Parameter(torch.full((width,), _positive_raw(1.0, floor)))

    def forward(self):
        if self.terminal:
            # For Q=I, the activation is already nonexpansive in Euclidean
            # energy. V=W=I is the tight choice; extra terminal metrics only
            # shrink the readout and introduce unnecessary parameters.
            return (self.identity,) * 4

        positive = F.softplus(self.v_raw) + self.floor
        v_half = positive.sqrt()
        lower = torch.diag(F.softplus(self.diag_raw) + self.floor)
        lower = lower.index_put((self.indices[0], self.indices[1]), self.lower_raw)
        s = lower @ lower.mT
        gram = self.identity + s
        chol = torch.linalg.cholesky(gram)
        # Solve against S, rather than subtracting (I+S)^-1 from I: this
        # preserves small metric eigenvalues without subtractive cancellation.
        q = 2 * torch.cholesky_solve(s, chol)
        q = 0.5 * (q + q.mT)
        q = v_half[:, None] * q * v_half[None, :]
        w = 0.5 * v_half[:, None] * gram * v_half[None, :]
        return q, torch.linalg.cholesky(q).mT, w, torch.linalg.cholesky(w).mT


class WeightedResidualFeedforward(nn.Module):
    """f(u)=u+O sigma(Hu), nonexpansive in the supplied feature metric Q.

    H=sqrt(2) Lambda^-.5 R F_Q, O=-sqrt(2) F_Q^-1 R.T Lambda^.5,
    where R is a contraction and Q=F_Q.T F_Q. Coupled weights certify the
    complete residual map directly; arbitrary GLU/MLP branches do not fit it.
    """

    def __init__(self, width, hidden, floor, activation, margin):
        super().__init__()
        self.floor, self.activation = floor, activation
        self.r = ThinContraction(hidden, width, 1 - margin, 0.1)
        self.lambda_raw = nn.Parameter(torch.full((hidden,), _positive_raw(1.0, floor)))

    def matrices(self, q_factor):
        r = _thin_matrix(self.r())
        root = (F.softplus(self.lambda_raw) + self.floor).sqrt()
        h = math.sqrt(2) * (r @ q_factor) / root[:, None]
        o = -math.sqrt(2) * torch.linalg.solve_triangular(q_factor, r.mT * root, upper=True)
        return h, o


@dataclass(frozen=True)
class LayerMatrices:
    """Prepared state blocks and feature projections; dense A is never assembled."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor
    q_prev: torch.Tensor
    q: torch.Tensor
    w: torch.Tensor
    ff_h: torch.Tensor | None
    ff_o: torch.Tensor | None


class MetricSSMLayer(nn.Module):
    """Contractive LTI core -> sector activation -> optional weighted residual."""

    def __init__(self, core, config, *, terminal=False):
        super().__init__()
        self.core = core
        self.activation = config.activation
        self.metric = FeatureMetric(config.d_model, config.metric_floor, terminal=terminal)
        self.ff = (
            WeightedResidualFeedforward(
                config.d_model,
                config.d_hidden,
                config.metric_floor,
                config.activation,
                config.factor_margin,
            )
            if config.ff == "residual"
            else None
        )

    def prepare(self, q_prev, input_factor):
        a, b, c, d = self.core.normalized_block_matrices()
        if a.ndim == 2:
            a = a.unsqueeze(0)  # a generic dense core is one scan block
        n, m = self.core.d_state, self.metric.width
        if (
            a.ndim != 3
            or a.shape[1] != a.shape[2]
            or not n <= a.shape[0] * a.shape[1] < n + a.shape[1]
            or b.shape != (n, m)
            or c.shape != (m, n)
            or d.shape != (m, m)
        ):
            raise ValueError("normalized_block_matrices returned incompatible dimensions.")
        q, q_factor, w, w_factor = self.metric()
        # Fold metric transforms into the projections once per forward, rather
        # than multiplying all batch/time features by extra m-by-m matrices.
        b = b @ input_factor
        c = torch.linalg.solve_triangular(w_factor, c, upper=True)
        d = torch.linalg.solve_triangular(w_factor, d @ input_factor, upper=True)
        ff_h, ff_o = self.ff.matrices(q_factor) if self.ff is not None else (None, None)
        return LayerMatrices(a, b, c, d, q_prev, q, w, ff_h, ff_o), q_factor

    def forward(self, inputs, initial, matrices, mode="scan"):
        a, b, c, d = matrices.a, matrices.b, matrices.c, matrices.d
        drives = F.linear(inputs, b)
        if mode == "scan":
            states = compute_block_lti_states(a, drives.transpose(0, 1), initial).transpose(0, 1)
        else:
            padding = a.shape[0] * a.shape[1] - initial.shape[-1]
            padded = F.pad(initial, (0, padding)) if padding else initial
            drives = F.pad(drives, (0, padding)) if padding else drives
            current = padded.reshape(initial.shape[0], a.shape[0], a.shape[1])
            trajectory = [initial]
            for drive in drives.unbind(1):
                current = (a @ current.unsqueeze(-1)).squeeze(-1) + drive.reshape_as(current)
                trajectory.append(current.flatten(1)[:, :initial.shape[-1]])
            states = torch.stack(trajectory, dim=1)
        outputs = _activate(F.linear(states[:, :-1], c) + F.linear(inputs, d), self.activation)
        if matrices.ff_h is not None:
            hidden = _activate(F.linear(outputs, matrices.ff_h), self.activation)
            outputs = outputs + F.linear(hidden, matrices.ff_o)
        return outputs, states


def _make_core(config):
    m, n = config.d_model, config.d_state
    if config.param == "defect":
        core = DefectL2SSM(
            n,
            m,
            m,
            block_size=config.block_size,
            max_radius=config.max_radius,
            factor_margin=config.factor_margin,
            init_rho=config.rho,
            init_scale=config.init_scale,
            max_phase=config.max_phase,
            phase_center=config.phase_center,
            random_phase=config.random_phase,
            state_metric=config.state_metric,
        )
        _initialize_diagonal(core.Z, config.init_feedthrough)
        return core
    core = Block2x2DenseL2SSM(n, m, m, exact_norm=True, energy_only=True,
                           state_metric=config.state_metric)
    core.init_on_circle(
        rho=config.rho,
        max_phase=config.max_phase,
        phase_center=config.phase_center,
        random_phase=config.random_phase,
        offdiag_scale=config.init_scale,
    )
    with torch.no_grad():
        core.K22_raw.copy_(config.init_feedthrough * torch.eye(m))
    return core


class MetricDeepSSM(EvalCacheMixin, nn.Module):
    """Deep SSM with a prescribed incremental l2 gain and learned feature metrics.

    Encoder and decoder are rectangular contractions. Between them each layer
    has its own state storage P and learns the outgoing feature Q shared with
    the next layer. Q_0=gamma² I,
    Q_L=I; terminal state energies enter the same global certificate.

    core_factory(n_state,n_input,n_output) may supply another certified LTI
    module implementing normalized_block_matrices(). This is a trusted algebraic
    contract, not a property implied by stable poles or checked by sampling.
    Factory cores use zero initial state and must expose d_state/input/output.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        *,
        d_model=16,
        d_state=64,
        n_layers=2,
        param="defect",
        gamma=1.0,
        activation="tanh",
        ff="residual",
        d_hidden=32,
        state_metric="identity",
        block_size=2,
        max_radius=0.999,
        factor_margin=1e-3,
        config: MetricSSMConfig | None = None,
        core_factory: Callable[[int, int, int], nn.Module] | None = None,
    ):
        super().__init__()
        cfg = config or MetricSSMConfig(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            param=param,
            gamma=gamma,
            activation=activation,
            ff=ff,
            d_hidden=d_hidden,
            state_metric=state_metric,
            block_size=block_size,
            max_radius=max_radius,
            factor_margin=factor_margin,
        )
        for value in (d_input, d_output, cfg.d_model, cfg.d_state, cfg.n_layers, cfg.d_hidden):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Dimensions and layer count must be positive integers.")
        if cfg.activation not in {"tanh", "relu", "arctan", "identity"}:
            raise ValueError("activation must be tanh, relu, arctan, or identity.")
        if cfg.ff not in {"residual", "none"}:
            raise ValueError("ff must be 'residual' or 'none'; GLUs need a different certificate.")
        if core_factory is None and cfg.param not in {"defect", "l2n"}:
            raise ValueError(
                "Built-in metric cores are defect and l2n; use core_factory for others."
            )
        if not math.isfinite(cfg.gamma) or cfg.gamma <= 0:
            raise ValueError("gamma must be finite and positive.")
        if cfg.state_metric not in {"identity", "full"}:
            raise ValueError("state_metric must be 'identity' or 'full'.")
        if not 0 < cfg.metric_floor < 0.5 or not 0 <= cfg.factor_margin < 1:
            raise ValueError("metric_floor must be in (0,.5) and factor_margin in [0,1).")
        if not 0 < cfg.init_feedthrough < 1 - cfg.factor_margin:
            raise ValueError("init_feedthrough must be in (0,1-factor_margin).")
        if (core_factory is None and cfg.param == "l2n"
                and cfg.state_metric == "identity" and cfg.d_state % 2):
            raise ValueError("l2n requires an even state dimension.")
        if core_factory is None and cfg.param == "l2n":
            if not math.isfinite(cfg.rho) or not 0 <= cfg.rho < 1:
                raise ValueError("l2n rho must be finite and in [0,1).")
            if (
                not math.isfinite(cfg.max_phase)
                or cfg.max_phase < 0
                or not math.isfinite(cfg.phase_center)
                or not math.isfinite(cfg.init_scale)
                or cfg.init_scale < 0
            ):
                raise ValueError(
                    "Phase settings and init_scale must be finite; widths/scales nonnegative."
                )
        self.config = cfg
        self.d_input, self.d_output = d_input, d_output
        self.register_buffer("gamma", torch.tensor(float(cfg.gamma)))
        self.register_buffer("identity", torch.eye(cfg.d_model))
        self.encoder = ThinContraction(cfg.d_model, d_input, 1.0, 0.0)
        self.decoder = ThinContraction(d_output, cfg.d_model, 1.0, 0.0)
        _initialize_diagonal(self.encoder, cfg.init_feedthrough)
        _initialize_diagonal(self.decoder, cfg.init_feedthrough)
        layers = []
        for index in range(cfg.n_layers):
            core = (
                core_factory(cfg.d_state, cfg.d_model, cfg.d_model)
                if core_factory is not None
                else _make_core(cfg)
            )
            if not isinstance(core, nn.Module) or not callable(
                getattr(core, "normalized_block_matrices", None)
            ):
                raise TypeError("Each core must implement normalized_block_matrices().")
            if (core.d_state, core.d_input, core.d_output) != (
                cfg.d_state,
                cfg.d_model,
                cfg.d_model,
            ):
                raise ValueError("Core dimensions must match the metric stack configuration.")
            layers.append(MetricSSMLayer(core, cfg, terminal=index == cfg.n_layers - 1))
        self.blocks = nn.ModuleList(layers)
        self.state = None

    def _prepare(self):
        if self.gamma.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "MetricDeepSSM parameters require float32 or float64; AMP may wrap forward."
            )
        # Cholesky/triangular solves and the contraction construction stay at
        # parameter precision even when the sequence projections use AMP.
        with torch.autocast(device_type=self.gamma.device.type, enabled=False):
            encoder, decoder = _thin_matrix(self.encoder()), _thin_matrix(self.decoder())
            input_factor, q_prev = self.gamma * self.identity, self.gamma.square() * self.identity
            prepared = []
            for block in self.blocks:
                matrices, input_factor = block.prepare(q_prev, input_factor)
                prepared.append(matrices)
                q_prev = matrices.q
            return encoder, tuple(prepared), decoder

    def prepared_matrices(self):
        """Build once per training forward; cache the entire bundle in no-grad eval."""
        return self._eval_cached("metric_stack", self._prepare)

    def forward(
        self,
        u,
        state: Sequence[torch.Tensor] | None = None,
        *,
        mode="scan",
        return_state=True,
        return_last=False,
        reset_state=True,
        detach_state=False,
        time_first=False,
    ):
        """Return y and x_0..x_T per layer; return_last exposes streaming states.

        Input is (B,T,d_input), or (T,d_input)/(d_input,). time_first swaps the
        first two axes of 3D inputs and outputs. State is a sequence of (B,n)
        tensors in the core's scan coordinates. Equal initial states give the difference
        certificate; nonzero initial states add their initial storage energy.
        """
        if mode not in {"scan", "loop"}:
            raise ValueError("MetricDeepSSM supports mode='scan' or 'loop'.")
        inputs = normalize_to_3d(u, time_first=time_first)
        if inputs.shape[-1] != self.d_input:
            raise ValueError("Input feature dimension does not match d_input.")
        if state is not None and len(state) != len(self.blocks):
            raise ValueError("Provide one initial state per layer.")
        encoder, prepared, decoder = self.prepared_matrices()
        outputs = F.linear(inputs, encoder)
        trajectories, last_states = [], []
        for index, (block, matrices) in enumerate(zip(self.blocks, prepared)):
            initial = resolve_runtime_state(
                state[index] if state is not None else None,
                self.state[index] if self.state is not None else None,
                reset_state=reset_state,
                batch_size=inputs.shape[0],
                n_state=self.config.d_state,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            outputs, states = block(outputs, initial, matrices, mode)
            last = states[:, -1]
            # A final-state view would keep its whole T-by-n trajectory alive.
            # Copy only B-by-n when no cross-call gradient graph is needed.
            if detach_state:
                last = last.detach().clone()
            elif not torch.is_grad_enabled():
                last = last.clone()
            last_states.append(last)
            if return_state:
                trajectories.append(states.transpose(0, 1) if time_first else states)
        self.state = tuple(last_states)
        outputs = F.linear(outputs, decoder)
        if time_first:
            outputs = outputs.transpose(0, 1)
        return sequence_output(
            outputs,
            tuple(trajectories),
            self.state,
            return_state=return_state,
            return_last=return_last,
        )

    def reset(self):
        self.state = None

    def certified_gain_bound(self):
        return self.gamma

    def incremental_gain_bound(self):
        return self.gamma

    def storage_value(self, state):
        """Sum of layer state energies; no depth-dependent storage weights."""
        if len(state) != len(self.blocks):
            raise ValueError("Provide one state per layer.")
        energies = []
        for block, x in zip(self.blocks, state):
            # A core may expose another basis when called on its own. Here we
            # use the coordinates of normalized_block_matrices(), e.g. L2N's z.
            hook = getattr(block.core, "scan_state_energy", None)
            if hook is None:
                hook = getattr(block.core, "state_energy", None)
            energies.append(hook(x) if callable(hook) else x.square().sum(dim=-1))
        return sum(energies)

    @torch.no_grad()
    def gain_diagnostics(self):
        return dict(
            certified_gain_bound=float(self.gamma),
            incremental_gain_bound=float(self.gamma),
            global_gamma=float(self.gamma),
            certificate="metric_transfer",
            core=self.config.param,
            activation=self.config.activation,
            ff=self.config.ff,
            n_layers=len(self.blocks),
            storage="sum of certified layer state energies",
        )

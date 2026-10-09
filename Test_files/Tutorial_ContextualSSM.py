"""ContextualDeepSSM, step by step.

Start after Tutorial_DeepSSM.py. Each numbered cell introduces one context idea.
The first example simply passes two short sequences to a model.
Install once from the project root: python -m pip install -e .
Run the whole tutorial: python Test_files/Tutorial_ContextualSSM.py
"""

# %% 1. Add a context sequence to the ordinary input
from dataclasses import replace

import torch
from torch import nn

from neural_ssm import ContextualDeepSSM, DeepSSM, SSMConfig

# u is the signal to process; z is extra information, such as an operating condition.
# Both have the same batch and time dimensions. Our context changes halfway through.
torch.manual_seed(7)
u = 0.2 * torch.randn(1, 8, 1)
z = torch.zeros(1, 8, 1)
z[:, 4:] = 1.0

# The "input" port concatenates context to the input features.
# For this short example, "none" leaves the eight context samples unchanged.
model = ContextualDeepSSM(
    d_input=1, d_context=1, d_output=1, context_modes="input", context_filter="none", param="lru"
)
y, _ = model(u, z)
print("Output shape:", y.shape)  # (1, 8, 1)


# %% 2. Add a gain bound and keep context only for a finite horizon
# SSMConfig controls the recurrent core. The wrapper's own arguments control context.
config = SSMConfig(
    d_model=4, d_state=8, n_layers=1, param="defect", ff="LGLU2", gamma=1.0,
    defect_state_metric="full", defect_max_radius=1.0, defect_factor_margin=0.0,
)
# Full P belongs to the LTI core and stays fixed during a sequence. The wrapper
# can still use changing context for its input, gate, and mixer ports below.
# To use weighted normalization instead, keep these context settings and set:
# config = replace(config, param="l2n", l2n_state_metric="full")
windowed = ContextualDeepSSM(
    1, 1, 1, ssm_config=config, context_modes="input", context_filter="finite_horizon", horizon=6
)
windowed.eval()
with torch.no_grad():
    y, _, details = windowed(u, z, return_aux=True)
print("Context after the window:", details["filtered_context"].flatten().tolist())
# The last two context samples are now zero. The gain applies to the COMBINED
# input [u, filtered z], so this port can produce output even when u is zero.


# %% 3. Let context gate the residual branches
# A gate changes how strongly each residual branch contributes, with weights in [0,1].
# This port conditions the model without adding context to the signal input.
gated = ContextualDeepSSM(1, 1, 1, ssm_config=config, context_modes="gate")
gated.eval()
with torch.no_grad():
    y, _ = gated(u, z)
    silent, _ = gated(torch.zeros_like(u), z)
print("Gated output shape:", y.shape)
torch.testing.assert_close(silent, torch.zeros_like(silent))
# Even with context present, zero input gives zero output from zero initial state.
# Add gate_per_channel=True if each feature should have its own gate.


# %% 4. Let context mix the output features
# The core produces four features. A context-dependent matrix maps them to one output.
# mixer_bound caps the matrix norm, so the final gain is at most 0.8 times the core gain.
mixed = ContextualDeepSSM(
    1, 1, 1, ssm_config=config, context_modes="mixer", d_features=4, mixer_bound=0.8
)
mixed.eval()
with torch.no_grad():
    y, _, details = mixed(u, z, return_aux=True)
print("Mixer shape:", details["mixer"].shape)  # (1, 8, 1, 4): one matrix per sample
print("Final gain bound:", mixed.certified_gain_bound().item())
# By default, the mixer sees both u and z. mixer_include_disturbance=False
# restricts its conditioning to context alone, as in the next combined example.


# %% 5. Let context select the recurrent dynamics
# TV/TVC can change their matrices at every sample. The wrapper supplies z to the selector.
# select_input="context" means the matrices are scheduled using z alone.
selected = ContextualDeepSSM(
    1, 1, 1, context_modes="select", ssm_config=replace(config, param="tv", select_input="context")
)
selected.eval()
with torch.no_grad():
    y, _ = selected(u, z)
print("Selected output shape:", y.shape)
print("Incremental gain:", selected.incremental_gain_bound().item())
# This also bounds output DIFFERENCES for two inputs sharing the same context
# sequence and initial state. Use the same exogenous context for that comparison.


# %% 6. Combine ports after understanding them individually
# This model selects dynamics, gates residuals, and mixes output features.
# Every scheduling pathway sees context alone, preserving the incremental certificate.
scheduled = ContextualDeepSSM(
    1,
    1,
    1,
    context_modes=("select", "gate", "mixer"),
    ssm_config=replace(config, param="tv", select_input="context"),
    gate_include_disturbance=False,
    mixer_include_disturbance=False,
    d_features=4,
    mixer_bound=0.8,
)
with torch.no_grad():
    y, _ = scheduled(u, z)
print("Combined incremental gain:", scheduled.incremental_gain_bound().item())
# Adding an "input" port is also supported; its context contribution is then part
# of the input energy, and the wrapper no longer reports this incremental certificate.

# To use all four ports, also supply the context's input window.
combined = ContextualDeepSSM(
    1,
    1,
    1,
    context_modes=("input", "select", "gate", "mixer"),
    ssm_config=replace(config, param="tv", select_input="context"),
    context_filter="finite_horizon",
    horizon=8,
    d_features=4,
    mixer_bound=0.8,
)
combined_y, _ = combined(u, z)
print("All four ports:", combined_y.shape)


# %% 7. Stream a windowed model in two chunks
# Reuse windowed from step 2. time_offset is the number of samples already processed.
# Advancing this clock keeps the six-sample window aligned across chunks.
with torch.no_grad():
    full, _ = windowed(u, z)
    first, state = windowed(u[:, :4], z[:, :4], detach_state=True)
    second, _ = windowed(u[:, 4:], z[:, 4:], state=state, time_offset=4, detach_state=True)
torch.testing.assert_close(torch.cat((first, second), dim=1), full)
print("Windowed streaming matches the full sequence.")


# %% 8. Use changes in context, then carry a change across a chunk boundary
# Difference filtering uses z[t] - z[t-1]. The first increment is zero by default.
# Here the context changes once, so only the fifth sample has a nonzero increment.
differenced = ContextualDeepSSM(
    1, 1, 1, ssm_config=config, context_modes="input", context_filter="difference"
)
differenced.eval()
with torch.no_grad():
    full, _, details = differenced(u, z, return_aux=True)
print("Context increments:", details["filtered_context"].flatten().tolist())
# For an ongoing context sequence, the increments need finite energy for the gain
# statement to give a finite budget; differencing does not ensure that automatically.

# The second chunk needs the last RAW context sample from the first chunk.
with torch.no_grad():
    first, state = differenced(u[:, :4], z[:, :4], detach_state=True)
    second, _ = differenced(
        u[:, 4:],
        z[:, 4:],
        state=state,
        time_offset=4,
        previous_context=z[:, 3:4],
        detach_state=True,
    )
torch.testing.assert_close(torch.cat((first, second), dim=1), full)
print("Differenced streaming keeps the change at the chunk boundary.")


# %% 9. Encode a context with more raw features
# Suppose each raw context sample has three features. A learned linear map reduces
# them to one context feature. d_context is the width AFTER this encoding.
raw_z = torch.cat((z, z.sin(), z.cos()), dim=-1)  # (1, 8, 3)
encoded = ContextualDeepSSM(
    1,
    1,
    1,
    ssm_config=config,
    context_modes="input",
    context_filter="difference",
    context_encoder=nn.Linear(3, 1),
)
encoded.eval()
with torch.no_grad():
    y, _ = encoded(u, raw_z)
print("Encoded-context output shape:", y.shape)
# When streaming this model, previous_context must also be a RAW sample of width 3.
# The wrapper applies the same encoder before computing the increment.


# %% 10. Try a different context window
# Finite horizon and difference were shown above. A taper fades out smoothly
# during its final context_filter_ramp samples, instead of cutting off abruptly.
tapered = ContextualDeepSSM(
    1,
    1,
    1,
    ssm_config=config,
    context_modes="input",
    context_filter="taper",
    horizon=8,
    context_filter_ramp=3,
)
tapered_y, _ = tapered(u, z)

# Exponential weighting uses decay**t; polynomial weighting uses (1+t)**(-power).
# These windows give bounded context finite energy; polynomial power must exceed 0.5.
exponential = ContextualDeepSSM(
    1,
    1,
    1,
    ssm_config=config,
    context_modes="input",
    context_filter="exponential",
    context_filter_decay=0.9,
)
exponential_y, _ = exponential(u, z)
polynomial = ContextualDeepSSM(
    1,
    1,
    1,
    ssm_config=config,
    context_modes="input",
    context_filter="polynomial",
    context_filter_power=1.0,
)
polynomial_y, _ = polynomial(u, z)
print("Window output shapes:", tapered_y.shape, exponential_y.shape, polynomial_y.shape)
# trainable_context_filter=True makes exponential decay or polynomial power trainable.
# "auto" chooses a finite horizon if horizon is supplied, otherwise exponential decay.


# %% 11. Advanced: supply your own context window
# A custom filter receives the context tensor. Accept time_offset so streaming
# uses the same absolute clock as a full-sequence call.
def custom_window(context, *, time_offset=0):
    steps = torch.arange(time_offset, time_offset + context.shape[1], device=context.device)
    return context / (1 + steps).reshape(1, -1, 1)


custom = ContextualDeepSSM(
    1, 1, 1, ssm_config=config, context_modes="input", context_filter=custom_window
)
custom_y, _ = custom(u, z)
print("Custom-window output shape:", custom_y.shape)
# The core gain still applies to [u, custom_window(z)]. For a custom window,
# the wrapper cannot infer a uniform context-energy bound on your behalf.


# %% 12. Advanced: pass context directly to DeepSSM's selector
# Use this when only the selective cells need context. Specify its width explicitly.
direct_config = replace(config, param="tv", select_context_dim=1, select_input="context")
direct = DeepSSM(1, 1, config=direct_config)
direct_y, _ = direct(u, select_context=z)
print("Direct selector output shape:", direct_y.shape)

# "both" lets the selector see the input AND context. Here we also try the TVC cell.
# The zero-state gain remains certified, but the incremental bound is not guaranteed:
# different inputs can select different matrices, even with the same context.
input_selected = ContextualDeepSSM(
    1,
    1,
    1,
    context_modes="select",
    ssm_config=replace(config, param="tvc", select_input="both", bcd_nonlinearity="tanh"),
)
input_selected_y, _ = input_selected(u, z)
print("Input-dependent selector's incremental bound:", input_selected.incremental_gain_bound())


# %% 13. Train the context encoder together with the recurrent core
# model.parameters() includes the encoder and every enabled context pathway.
# This is one ordinary training update; repeat these lines for a longer experiment.
encoded.train()
optimizer = torch.optim.Adam(encoded.parameters(), lr=3e-3)
optimizer.zero_grad(set_to_none=True)
prediction, _ = encoded(u, raw_z)
target = 0.2 * u + 0.1 * z
loss = (prediction - target).square().mean()
loss.backward()
optimizer.step()
print("Context-model training loss:", loss.item())

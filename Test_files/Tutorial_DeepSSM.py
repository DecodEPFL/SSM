"""DeepSSM, step by step.

Read from top to bottom, or run the numbered cells individually in your editor.
Start with steps 1–5; the remaining steps introduce optional model choices.
Install once from the project root: python -m pip install -e .
Run the whole tutorial: python Test_files/Tutorial_DeepSSM.py
"""

# %% 1. Create a model and pass a short sequence through it
from dataclasses import replace

import torch

from neural_ssm import DeepSSM, SSMConfig

# We have one sequence, eight time steps, and one input value at each step.
torch.manual_seed(7)
u = torch.randn(1, 8, 1)  # shape: (batch, time, input features)

# One input feature -> one output feature. LRU is a stable recurrent baseline.
model = DeepSSM(d_input=1, d_output=1, param="lru")
y, _ = model(u)  # The second return value is the recurrent state; we use it later.
print("Output shape:", y.shape)  # (1, 8, 1): one prediction at each time step


# %% 2. Choose the feature width, memory size, and number of layers
# d_model is the width between layers; d_state is the memory size of each layer.
# Neither has to match the number of input or output features.
model = DeepSSM(d_input=1, d_output=1, param="lru", d_model=4, d_state=8, n_layers=1)
y, state = model(u)
print("Number of layer states:", len(state))  # one state because n_layers=1
print("State shape:", state[0].shape)  # (1, 8): batch and memory size
# LRU states are complex tensors. Pass them back unchanged when continuing a stream.


# %% 3. Build a model with a prescribed gain bound
# Use a certified cell and a bounded feedforward branch together.
# gamma=1 means output energy <= input energy when starting from zero state.
model = DeepSSM(
    d_input=1, d_output=1, d_model=4, d_state=8, n_layers=1, param="defect", ff="LGLU2", gamma=1.0,
    defect_state_metric="full", defect_max_radius=1.0, defect_factor_margin=0.0,
)
# Full state storage lets different memory blocks share the gain certificate.
# The last two options allow the full open range of poles and contraction factors.
# This adds dense setup work; the recurrence still uses the fast small-block scan.
model.eval()  # Turn off dropout before checking the inference gain.
with torch.no_grad():
    y, _ = model(u)
print("Certified gain:", model.certified_gain_bound().item())
assert y.square().sum() <= u.square().sum() + 1e-5
# LRU can be used with gamma=None; it has no prescribed whole-stack gain bound.


# %% 4. Train on a simple input-to-output task
# The desired output is 0.3 times the current input. A target in one line lets us
# focus on the training steps; a task that needs memory comes in step 11.
train_u = torch.randn(4, 32, 1)  # four independent training sequences
train_y = 0.3 * train_u

# One CPU thread is sufficient for these tiny examples; this setting is optional.
torch.set_num_threads(1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
model.train()
for step in range(40):
    optimizer.zero_grad(set_to_none=True)  # Clear the previous gradients.
    prediction, _ = model(train_u)  # No state passed: start a new trajectory.
    loss = (prediction - train_y).square().mean()  # Mean squared prediction error.
    loss.backward()  # Differentiate through the whole sequence.
    optimizer.step()  # Update the trainable parameters.
    if step in (0, 39):
        print(f"Training step {step + 1}: MSE = {loss.item():.6f}")


# %% 5. Evaluate the trained model, then process the same sequence in two chunks
model.eval()  # Disables training-time dropout.
with torch.no_grad():  # No gradients are needed; inference matrices can be cached.
    full, _ = model(u)
    first, state = model(u[:, :4], detach_state=True)
    second, state = model(u[:, 4:], state=state, detach_state=True)

# The second chunk starts where the first ended. An explicit state overrides reset.
streamed = torch.cat((first, second), dim=1)
torch.testing.assert_close(streamed, full, rtol=2e-4, atol=2e-5)
print("Two chunks give the same predictions as the full sequence.")
# detach_state=True discards the gradient history at the chunk boundary.


# %% 6. Collect repeated settings in a configuration
# Direct constructor arguments are enough for a single model, as above.
# A configuration is useful when comparing models that share the same dimensions.
config = SSMConfig(
    d_model=4, d_state=8, n_layers=1, param="defect", ff="LGLU2", gamma=1.0,
    defect_state_metric="full", defect_max_radius=1.0, defect_factor_margin=0.0,
)
configured_model = DeepSSM(1, 1, config=config)
configured_y, _ = configured_model(u)
# replace() makes a NEW configuration: the original config remains available.

# To recover the cheaper original model, change only the storage choice.
# Identity storage with small blocks is more restrictive at a prescribed gain.
identity_model = DeepSSM(1, 1, config=replace(config, defect_state_metric="identity"))
identity_y, _ = identity_model(u)

# Full storage also supports odd memory sizes: the last mode is a real scalar.
odd_model = DeepSSM(1, 1, config=replace(config, d_state=9))
odd_y, _ = odd_model(u, mode="scan")

# Returned states stay in the scan coordinates. Ask the core for their energy
# instead of assuming it is the squared Euclidean norm.
core = configured_model.blocks[0].lru
P = core.storage_matrix()  # shape: (8, 8); useful for inspecting the certificate
_, final_states = configured_model(u)
stored_energy = core.state_energy(final_states[0])  # x.T @ P @ x, one value per batch


# %% 7. Try the other current parametrizations, changing one choice at a time
# First try the original L2N: rotation blocks, identity scan storage, even state size.
# Those rotations also support FFT convolution through mode="conv".
l2n = DeepSSM(1, 1, config=replace(config, param="l2n"))
l2n_y, _ = l2n(u, mode="conv")
print("L2N output shape:", l2n_y.shape)

# Now let the blocks share a dense state metric inside the normalization.
# The new mode learns general real blocks; use their fast real scan.
full_l2n = DeepSSM(1, 1, config=replace(config, param="l2n", l2n_state_metric="full"))
full_l2n_y, _ = full_l2n(u, mode="scan")
print("Full-metric L2N output shape:", full_l2n_y.shape)

# TV selects its dynamics from the input at each time step.
tv = DeepSSM(1, 1, config=replace(config, param="tv"))
tv_y, _ = tv(u, mode="scan")
print("TV output shape:", tv_y.shape)

# TVC also has a direct input-to-output term. tanh bounds the raw selector outputs.
tvc = DeepSSM(1, 1, config=replace(config, param="tvc", bcd_nonlinearity="tanh"))
tvc_y, _ = tvc(u, mode="scan")
print("TVC output shape:", tvc_y.shape)

# Defect blocks can be larger to allow more interactions within the state.
# The block size must divide d_state. Here eight states form two blocks of four.
larger_blocks = DeepSSM(1, 1, config=replace(config, defect_block_size=4))
block_y, _ = larger_blocks(u)
print("Larger defect blocks:", block_y.shape)


# %% 8. Change the feedforward branch while keeping the recurrent cell fixed
# LGLU2 was used above. These three alternatives also support a prescribed gamma.
# BLGLU2 shares a Lipschitz budget; BudgetedLGLU2 is another name for this choice.
budgeted = DeepSSM(1, 1, config=replace(config, ff="BLGLU2"))
budgeted_y, _ = budgeted(u)

# MBLIP mixes bounded branches; TLIP uses a bounded Sandwich network.
mixed = DeepSSM(1, 1, config=replace(config, ff="MBLIP"))
mixed_y, _ = mixed(u)
sandwich = DeepSSM(1, 1, config=replace(config, ff="TLIP"))
sandwich_y, _ = sandwich(u)
print("Alternative feedforward output shapes:", budgeted_y.shape, mixed_y.shape, sandwich_y.shape)

# Ordinary MLP/GLU branches use gamma=None. LGLU/LMLP are historical uncapped choices.
ordinary = DeepSSM(1, 1, config=replace(config, param="lru", ff="GLU", gamma=None))
ordinary_y, _ = ordinary(u)
print("Ordinary GLU output shape:", ordinary_y.shape)


# %% 9. Reproduce an experiment with a legacy parametrization
# These older constructions keep their original configuration keys and checkpoints.
# Prefer the current cells in step 7 when starting a new comparison.
l2ru = DeepSSM(1, 1, config=replace(config, param="l2ru", init="eye"))
l2ru_y, _ = l2ru(u, mode="scan")
zak = DeepSSM(1, 1, config=replace(config, param="zak", init="eye"))
zak_y, _ = zak(u, mode="scan")
l2nt = DeepSSM(1, 1, config=replace(config, param="l2nt"))
l2nt_y, _ = l2nt(u, mode="loop")  # This dense legacy cell uses the sequential loop.
print("Legacy output shapes:", l2ru_y.shape, zak_y.shape, l2nt_y.shape)


# %% 10. Adjust gain budgets and residual gates for a deeper model
# gamma remains the prescribed whole-stack target. train_gamma controls CELL gains.
# train_ff_lip controls the feedforward budgets; False fixes the internal budgets.
# Per-channel gates give each feature its own residual weight. Automatic initialization
# balances the initial gates against the gain budget in a deeper stack.
deep_config = replace(
    config,
    n_layers=4,
    train_gamma=False,
    train_ff_lip=False,
    per_channel_gates=True,
    auto_residual_init=True,
)
deep = DeepSSM(1, 1, config=deep_config)
deep_y, _ = deep(u)
print("Deeper model output shape:", deep_y.shape)


# %% 11. Advanced: learn a delay and keep gradients across a chunk boundary
# Now the target requires memory: y[t] = 0.3*u[t-1], with the first output zero.
delay_target = torch.zeros_like(train_u)
delay_target[:, 1:] = 0.3 * train_u[:, :-1]

# Reuse the trained model from steps 3–5. detach_state=False connects the chunks
# so the second chunk can backpropagate through the first. Use True for truncated BPTT.
model.train()
optimizer.zero_grad(set_to_none=True)
first, state = model(train_u[:, :16], detach_state=False)
second, _ = model(train_u[:, 16:], state=state, detach_state=False)
loss = (torch.cat((first, second), dim=1) - delay_target).square().mean()
loss.backward()
optimizer.step()
print("One training update across two connected chunks:", loss.item())

# To continue a stream through the model's INTERNAL state instead, call model.reset()
# once, then use reset_state=False on each chunk. Explicit states are easier to manage
# when one model serves several independent streams.
# Next: Tutorial_ContextualSSM.py introduces a second sequence for context.

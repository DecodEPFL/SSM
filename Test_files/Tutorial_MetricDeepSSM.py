"""MetricDeepSSM, step by step.

Start after Tutorial_DeepSSM.py. Learned feature metrics are handled by the model;
you do not have to assemble those matrices to build or train a network.
Install once from the project root: python -m pip install -e .
Run the whole tutorial: python Test_files/Tutorial_MetricDeepSSM.py
"""

# %% 1. Create a small metric model and obtain predictions
from dataclasses import replace

import torch

from neural_ssm import MetricDeepSSM, MetricSSMConfig

torch.manual_seed(7)
u = torch.randn(1, 8, 1)  # one sequence, eight samples, one input feature
model = MetricDeepSSM(1, 1, d_model=4, d_state=8, n_layers=1)
y = model(u, return_state=False)  # Request only predictions for this first example.
print("Output shape:", y.shape)  # (1, 8, 1)


# %% 2. Understand the two ways to request state
# By default, this stack returns a complete trajectory for each recurrent layer.
# The trajectory includes the initial state, hence time length 8+1.
y, trajectories = model(u)
print("Trajectory shape:", trajectories[0].shape)  # (1, 9, 8)

# For streaming, request only the last state: one (batch, d_state) tensor per layer.
y, state = model(u, return_state=False, return_last=True)
print("Final-state shape:", state[0].shape)  # (1, 8)
# These states use the recurrence coordinates. Pass them back unchanged.


# %% 3. Choose dimensions, a gain target, and the nonlinear branch
# This stack certifies an incremental gain: it bounds output differences for
# two input sequences starting from equal states. gamma is the prescribed target.
# d_hidden is the hidden width of the residual feedforward branch.
config = MetricSSMConfig(
    d_model=4, d_state=8, n_layers=2, d_hidden=6, param="defect", gamma=0.6,
    state_metric="full", max_radius=1.0, factor_margin=0.0,
)
# state_metric controls the memory energy P; feature metrics are separate.
# Full P couples the memory blocks while their recurrence remains blockwise.
model = MetricDeepSSM(1, 1, config=config)
print("Incremental gain bound:", model.certified_gain_bound().item())
# The defaults are activation="tanh" and ff="residual", the metric-aware branch.
# Weights and learned feature metrics are prepared internally during the forward pass.

# For cheaper setup, use identity state storage. This restricts the LTI core
# family at a prescribed gain when its A is split into small blocks.
identity_model = MetricDeepSSM(1, 1, config=replace(config, state_metric="identity"))
identity_y = identity_model(u, return_state=False)


# %% 4. Train the model on a simple input-to-output task
# Use y[t] = 0.2*u[t] first, so the data preparation is just one line.
# Replace train_y with measured targets or a delayed sequence for a memory task.
train_u = torch.randn(4, 32, 1)
train_y = 0.2 * train_u

torch.set_num_threads(1)  # Optional: one CPU thread suffices for these tiny matrices.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
model.train()
for step in range(40):
    optimizer.zero_grad(set_to_none=True)  # Clear gradients from the previous step.
    prediction = model(train_u, return_state=False)  # Start every sequence from zero.
    loss = (prediction - train_y).square().mean()  # Mean squared prediction error.
    loss.backward()  # Compute parameter gradients.
    optimizer.step()  # Learn the core, metrics, and feedforward.
    if step in (0, 39):
        print(f"Training step {step + 1}: MSE = {loss.item():.6f}")


# %% 5. Evaluate, then continue through a second chunk
model.eval()
with torch.no_grad():
    full, full_state = model(u, return_state=False, return_last=True)
    first, state = model(u[:, :4], return_state=False, return_last=True, detach_state=True)
    second, state = model(
        u[:, 4:], state=state, return_state=False, return_last=True, detach_state=True
    )
streamed = torch.cat((first, second), dim=1)
torch.testing.assert_close(streamed, full, rtol=2e-4, atol=2e-5)
print("Two chunks give the same predictions as the full sequence.")
# Pass detach_state=False instead when training through the chunk boundary.


# %% 6. Check both the output energy and the energy left in memory
# From zero initial state, the theorem also accounts for the final stored energy.
# full_state belongs to the full-sequence prediction made in step 5.
with torch.no_grad():
    output_energy = full.square().sum()
    memory_energy = model.storage_value(full_state).sum()
    allowed_energy = model.gamma.square() * u.square().sum()
assert output_energy + memory_energy <= allowed_energy + 1e-5
print("Output plus terminal memory energy fits the gain budget.")

# The incremental statement applies to two trajectories with the same initial state.
other_u = u + 0.1 * torch.randn_like(u)
with torch.no_grad():
    other_y = model(other_u, return_state=False)  # Default reset starts from zero again.
assert (other_y - full).norm() <= model.gamma * (other_u - u).norm() + 1e-5
print("Output differences also fit the incremental gain bound.")


# %% 7. Change one architecture choice at a time
# L2N is the other built-in core. The existing config's state_metric="full"
# now puts the dense certificate inside normalization and keeps the real scan.
l2n = MetricDeepSSM(1, 1, config=replace(config, param="l2n"))
l2n_y = l2n(u, return_state=False)
print("L2N output shape:", l2n_y.shape)

# For cheaper identity storage, change just that setting (use an even state size).
identity_l2n = MetricDeepSSM(1, 1, config=replace(config, param="l2n", state_metric="identity"))
identity_l2n_y = identity_l2n(u, return_state=False)
print("Identity-metric L2N output shape:", identity_l2n_y.shape)

# Replace tanh by ReLU. The other supported choices are "arctan" and "identity".
relu_model = MetricDeepSSM(1, 1, config=replace(config, activation="relu"))
relu_y = relu_model(u, return_state=False)
print("ReLU output shape:", relu_y.shape)

# Remove the residual feedforward branch while keeping the core and activation.
no_ff = MetricDeepSSM(1, 1, config=replace(config, ff="none"))
no_ff_y = no_ff(u, return_state=False)
print("No-feedforward output shape:", no_ff_y.shape)
# MetricDeepSSM accepts "residual" or "none" here; DeepSSM's GLU choices use
# a different certificate and are not interchangeable with this metric branch.


# %% 8. Use larger recurrent blocks or inspect a sequential execution
# MetricSSMConfig calls this setting block_size. It must divide d_state.
larger_blocks = MetricDeepSSM(1, 1, config=replace(config, block_size=4))
block_y = larger_blocks(u, return_state=False)
print("Larger-block output shape:", block_y.shape)

# "scan" is the default parallel execution. "loop" runs the recurrence sequentially.
model.eval()
with torch.no_grad():
    scan_y = model(u, mode="scan", return_state=False)
    loop_y = model(u, mode="loop", return_state=False)
torch.testing.assert_close(scan_y, loop_y, rtol=2e-4, atol=2e-5)
print("Scan and loop give the same predictions.")
# Keep metric-model parameters in float32 or float64. Autocast can wrap the forward
# pass while the internal metric preparation retains parameter precision.

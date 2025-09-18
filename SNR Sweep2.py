import torch
from typing import Tuple, Dict, Optional, Callable, Union
from SSM.ssm import DeepSSM, SSMConfig, PureLRUR
import math
from argparse import Namespace
import torch.nn as nn
import control


def estimate_l2_gain(model, input_shape, num_batches=10, batch_size=32, num_iterations=100, learning_rate=0.01):
    """
    Numerically estimates the L2 gain of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
        input_shape (tuple): The shape of a single input element (T, N).
        num_batches (int): Number of random batches to generate for estimation.
        batch_size (int): The batch size to use for each estimation.
        num_iterations (int): Number of optimization iterations to find max gain for each batch.
        learning_rate (float): Learning rate for the optimization.

    Returns:
        float: The estimated L2 gain of the model.
    """
    model.eval()  # Set the model to evaluation mode
    max_gain_per_batch = []

    for _ in range(num_batches):
        # Generate a random input batch
        B, T, N = batch_size, input_shape[0], input_shape[1]
        x = torch.randn(B, T, N, requires_grad=True)

        optimizer = torch.optim.Adam([x], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            # Ensure x is normalized to have unit L2 norm per input sequence for gain calculation
            # We want to find the input x_i that maximizes ||model(x_i)||_2 / ||x_i||_2
            # Here, we normalize x to unit L2 norm for each sequence, then maximize ||model(x)||_2
            x_normalized = x / (x.norm(p=2, dim=[1, 2], keepdim=True) + 1e-6)

            output, _ = model(x_normalized, mode="scan")

            # Calculate the L2 norm of the output for each batch element
            output_l2_norm = output.norm(p=2, dim=[1, 2])

            # We want to maximize this quantity, so we'll use a negative loss for gradient ascent
            loss = -output_l2_norm.mean()  # Maximize the mean L2 norm of the outputs

            loss.backward()
            optimizer.step()

        # After optimization, calculate the gain for the current batch
        with torch.no_grad():
            x_normalized_final = x / (x.norm(p=2, dim=[1, 2], keepdim=True) + 1e-6)
            output_final, _ = model(x_normalized_final, mode="scan")
            gain_values = output_final.norm(p=2, dim=[1, 2])
            max_gain_per_batch.append(gain_values.max().item())

    return max(max_gain_per_batch)


# Example Usage:
if __name__ == "__main__":
    # Define a simple PyTorch model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, N, Y):
            super(SimpleModel, self).__init__()
            self.linear1 = nn.Linear(N, 64)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(64, Y)

        def forward(self, x):
            # x shape: (B, T, N)
            B, T, N = x.shape
            x_reshaped = x.view(B * T, N)  # Reshape for linear layers
            out = self.linear1(x_reshaped)
            out = self.relu(out)
            out = self.linear2(out)
            out = out.view(B, T, -1)  # Reshape back to (B, T, Y)
            return out


    # Model parameters
    input_dim_N = 3
    output_dim_Y = 3
    sequence_length_T = 100


    # Another example: Identity-like model
    class IdentityModel(nn.Module):
        def __init__(self, N):
            super().__init__()
            self.linear = nn.Linear(N, N)
            self.linear.weight.data = torch.eye(N)
            self.linear.bias.data = torch.zeros(N)

        def forward(self, x):
            B, T, N = x.shape
            x_reshaped = x.view(B * T, N)
            out = self.linear(x_reshaped)
            out = out.view(B, T, -1)
            return out

        # Configs


    cfg_robust = {
        "n_u": 3, "n_y": 3, "d_model": 5, "d_state": 6, "n_layers": 1,
        "ff": "LMLP", "max_phase": math.pi / 50, "r_min": 0.7, "r_max": 0.98,
        "robust": True, "gamma": 20
    }
    cfg_robust = Namespace(**cfg_robust)

    LRUR = PureLRUR(3, 1.0)

    # Build models
    config_robust = SSMConfig(d_model=cfg_robust.d_model, d_state=cfg_robust.d_state, n_layers=cfg_robust.n_layers,
                              ff=cfg_robust.ff,
                              rmin=cfg_robust.r_min, rmax=cfg_robust.r_max, max_phase=cfg_robust.max_phase,
                              robust=cfg_robust.robust, gamma=cfg_robust.gamma)
    model_robust = DeepSSM(cfg_robust.n_u, cfg_robust.n_y, config_robust)

    identity_model = IdentityModel(input_dim_N)
    estimated_gain_identity = estimate_l2_gain(model_robust, (100, 3), num_batches=15, num_iterations=220)
    print(f"Estimated L2 Gain of the robust model: {estimated_gain_identity:.4f}")

    A, B, C, D = LRUR.lru.set_param()

    A = A.cpu().detach().numpy()
    B = B.cpu().detach().numpy()
    C = C.data.cpu().detach().numpy()
    D = D.cpu().detach().numpy()

    sys = control.ss(A, B, C, D, dt=1.0)
    # Compute the H∞ norm (L2 gain) and the peak frequency ω_peak
    gamma = control.norm(sys, p='inf')

cfg_robust

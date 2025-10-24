import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from os.path import join as pjoin
from torch import nn
import math
from argparse import Namespace
import torch
from tqdm import tqdm
from SSM.ssm import DeepSSM, SSMConfig
import control
import scipy.io as sio

# Load the MAT file
mat_data = sio.loadmat('dataBenchmark.mat')

# Convert to PyTorch tensors using torch.from_numpy()
u_train = torch.from_numpy(mat_data['uEst']).float()
y_train = torch.from_numpy(mat_data['yEst']).float()

u_val = torch.from_numpy(mat_data['uVal']).float()
y_val = torch.from_numpy(mat_data['yVal']).float()

seed = 66
torch.manual_seed(seed)

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

plt.close('all')

# set up a simple architecture
cfg = {
    "n_u": 1,
    "n_y": 1,
    "d_model": 5,  # 5
    "d_state": 11,  # 6
    "n_layers": 1,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi / 50,
    "r_min": 0.7,
    "r_max": 0.98,
    "d_amp": 8,
    "param": 'lru',
    "gamma": None,
    "init": 'rand'
}
cfg = Namespace(**cfg)

#torch.set_num_threads(10)

# Build model
config = SSMConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, dim_amp=cfg.d_amp, param=cfg.param, gamma=cfg.gamma,
                   init=cfg.init)
model = DeepSSM(cfg.n_u, cfg.n_y, config)
#model.cuda()

# lru = model.blocks[0].lru
# A, B, C, D = lru.set_param()
# real = lru.ss_real_matrices()
# A = real[0]
# B = real[1]
# C = real[2]
# D = real[3]
# sys = control.ss(A, B, C, D, dt=1.0)
# # Compute the H∞ norm (L2 gain) and the peak frequency ω_peak
# gamma = control.norm(sys, p='inf')


# Configure optimizer
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt.zero_grad()

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

MSE = nn.MSELoss()
#model = torch.compile(model)

# Variables for tracking the lowest loss
lowest_loss = float('inf')  # Initialize with a very large value
best_model_path = "best_model.pth"
LOSS = []

epochs = 3590  # Train loop
for itr in tqdm(range(epochs)):
    yRNN, _ = model(u_train, mode="scan")
    yRNN = torch.squeeze(yRNN)
    loss = MSE(yRNN, y_train)
    loss.backward()
    opt.step()
    # Check if this epoch has the lowest loss
    if loss.item() < lowest_loss:
        lowest_loss = loss.item()
        torch.save(model.state_dict(), best_model_path)  # Save model
    opt.zero_grad()
    if itr % 100 == 0:
        print(loss.item())
    LOSS.append(loss.item())
print(f"Training complete. Best model saved with loss: {lowest_loss:.4f}.")
checkpoint = {
    'model': model.state_dict(),
    'LOSS': np.array(LOSS),
    'cfg': cfg
}

torch.save(checkpoint, "ckpt.pt")

ym_train, _ = model(u_train)
ym_train = torch.squeeze(ym_train)

ym_val, _ = model(u_val)
ym_val = torch.squeeze(ym_val)

loss_val = MSE(ym_val, y_val.squeeze())

# Save loss sequence with unique filename
filename = f"loss3.npy"  # Customize filename (e.g., loss_hu_64.npy)
np.save(filename, np.array(LOSS))
print(f"Saved loss sequence to {filename}")

# Set publication-quality parameters
plt.rcParams.update({
    'font.family': 'serif',  # Professional font style
    'font.size': 8,  # Small font size for half-column figure
    'axes.linewidth': 0.8,  # Thin axes lines
    'lines.linewidth': 1.0,  # Thin plot lines
    'figure.dpi': 300,  # High resolution
    'savefig.dpi': 300,  # High resolution for saving
})

plt.figure('8')

fig, ax = plt.subplots(figsize=(5.25, 1.75))

# Plot losses with logarithmic y-axis
ax.plot(range(epochs), LOSS, label='Approach 1', color='blue', linestyle='-')

# Set logarithmic scale for y-axis
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')

# Clean up: remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save as PDF
plt.savefig('Loss.pdf', format='pdf', bbox_inches='tight')
plt.show()

print(f"Loss Validation single RNN: {loss_val}")

# Create figure with three horizontal subplots
fig, ax1 = plt.subplots(figsize=(6.25, 1.75))

# Subplot 1: Output validation
ax1.plot(y_val[:, 0].cpu().detach().numpy(), label='h', color='orange', linestyle='-')  # Solid data first
ax1.plot(ym_val[:].cpu().detach().numpy(), label='SSM', color='blue', linestyle=':')  # Dotted SSM on top
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$h_1$ [cm]')

# Save as PDF for LaTeX
# plt.savefig('comparison_figure.pdf', format='pdf', bbox_inches='tight')
plt.show()

fig2, ax1 = plt.subplots(figsize=(6.25, 1.75))

# Subplot 1: Output validation
ax1.plot(y_train[:, 0].cpu().detach().numpy(), label='h', color='orange', linestyle='-')  # Solid data first
ax1.plot(ym_train[:].cpu().detach().numpy(), label='SSM', color='blue', linestyle=':')  # Dotted SSM on top
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$h_1$ [cm]')

# Save as PDF for LaTeX
# plt.savefig('comparison_figure.pdf', format='pdf', bbox_inches='tight')
plt.show()

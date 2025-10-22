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

seed = 55
torch.manual_seed(seed)

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

plt.close('all')
# Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'dataset_sysID_3tanks.mat')
data = scipy.io.loadmat(filepath)

dExp, yExp, dExp_val, yExp_val, Ts = data['dExp'], data['yExp'], \
    data['dExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(dExp[0, 0], 1) * Ts - Ts, Ts)

t_end = t.size

u = torch.zeros(nExp, t_end, 1, device=device)
y = torch.zeros(nExp, t_end, 3, device=device)
inputnumberD = 1

for j in range(nExp):
    inputActive = (torch.from_numpy(dExp[0, j])).T
    u[j, :, :] = torch.unsqueeze(inputActive[:, inputnumberD], 1)
    y[j, :, :] = (torch.from_numpy(yExp[0, j])).T

# set up a simple architecture
cfg = {
    "n_u": 1,
    "n_y": 3,
    "d_model": 5,
    "d_state": 6,  # 6
    "n_layers": 1,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi / 50,
    "r_min": 0.7,
    "r_max": 0.98,
    "d_amp": 8,
    "param": 'l2ru',
    "gamma": None,
    "init": 'rand'
}
cfg = Namespace(**cfg)

#torch.set_num_threads(10)

# Build model
config = SSMConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, dim_amp=cfg.d_amp, param=cfg.param, gamma=cfg.gamma, init=cfg.init)
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

epochs = 4590  # Train loop
for itr in tqdm(range(epochs)):
    yRNN, _ = model(u, mode="scan")
    yRNN = torch.squeeze(yRNN)
    loss = MSE(yRNN, y)
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

t_end = yExp_val[0, 0].shape[1]

nExp = yExp_val.size

uval = torch.zeros(nExp, t_end, 1, device=device)
yval = torch.zeros(nExp, t_end, 3, device=device)

#model.load_state_dict(torch.load(best_model_path))
#model.eval()  # Set to evaluation mode


for j in range(nExp):
    inputActive = (torch.from_numpy(dExp_val[0, j])).T
    uval[j, :, :] = torch.unsqueeze(inputActive[:, inputnumberD], 1)
    yval[j, :, :] = (torch.from_numpy(yExp_val[0, j])).T

yRNN_val, _ = model(uval)
yRNN_val = torch.squeeze(yRNN_val)
yval = torch.squeeze(yval)

loss_val = MSE(yRNN_val, yval)

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
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.25, 1.75))

# Subplot 1: Output 0 validation
ax1.plot(yval[20:600, 0].cpu().detach().numpy(), label='h', color='orange', linestyle='-')  # Solid data first
ax1.plot(yRNN_val[20:600, 0].cpu().detach().numpy(), label='SSM', color='blue', linestyle=':')  # Dotted SSM on top
ax1.set_xlabel('Time [s]')
ax1.set_ylabel(r'$h_1$ [cm]')

# Subplot 2: Output 1 training
ax2.plot(yval[20:600, 1].cpu().detach().numpy(), label='h', color='orange', linestyle='-')  # Solid data first
ax2.plot(yRNN_val[20:600, 1].cpu().detach().numpy(), label='SSM', color='blue', linestyle=':')  # Dotted SSM on top
ax2.set_xlabel('Time [s]')
ax2.set_ylabel(r'$h_2$ [cm]')

# Subplot 3: Output 1 validation
ax3.plot(yval[20:600, 2].cpu().detach().numpy(), label='Ground truth', color='orange',
         linestyle='-')  # Solid data first
ax3.plot(yRNN_val[20:600, 2].cpu().detach().numpy(), label='L2RU', color='blue', linestyle=':')  # Dotted SSM on top
ax3.set_xlabel('Time [s]')
ax3.set_ylabel(r'$h_3$ [cm]')

handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=False)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend

# Save as PDF for LaTeX
plt.savefig('comparison_figure.pdf', format='pdf', bbox_inches='tight')
plt.show()

A, B, C, D = model.blocks[0].lru.set_param()
A = A.cpu().detach().numpy()
B = B.cpu().detach().numpy()
C = C.data.cpu().detach().numpy()
D = D.cpu().detach().numpy()
sys = control.ss(A, B, C, D, dt=1.0)
gamma = control.norm(sys, p='inf')

plt

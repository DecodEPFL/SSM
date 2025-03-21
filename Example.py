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
from lru.architectures import DeepSSM, DWNConfig

seed = 9
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
    "d_state": 5,
    "n_layers": 1,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi / 50,
    "r_min": 0.7,
    "r_max": 0.98,
    "gamma": True,
    "trainable": False,
    "gain": 2.4
}
cfg = Namespace(**cfg)

#torch.set_num_threads(10)

# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma, trainable=cfg.trainable, gain=cfg.gain)
model = DeepSSM(cfg.n_u, cfg.n_y, config)
#model.cuda()

# Configure optimizer
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
opt.zero_grad()

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

MSE = nn.MSELoss()
#model = torch.compile(model)

# Variables for tracking the lowest loss
lowest_loss = float('inf')  # Initialize with a very large value
best_model_path = "best_model.pth"
LOSS = []
# Train loop
for itr in tqdm(range(1500)):
    yRNN, _ = model(u, state=None, mode="scan")
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
ax.plot(range(1500), LOSS, label='Approach 1', color='blue', linestyle='-')

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

# plt.figure('9')
# plt.plot(yRNN[0, :, 0].cpu().detach().numpy(), label='SSM')
# plt.plot(y[0, :, 0].cpu().detach().numpy(), label='y train')
# plt.title("output 1 train single RNN")
# plt.legend()
# plt.show()
#
# plt.figure('10')
# plt.plot(yRNN_val[:, 0].cpu().detach().numpy(), label='SSM val')
# plt.plot(yval[:, 0].cpu().detach().numpy(), label='y val')
# plt.title("output 1 val single SSM")
# plt.legend()
# plt.show()
#
# plt.figure('11')
# plt.plot(yRNN[0, :, 1].cpu().detach().numpy(), label='SSM')
# plt.plot(y[0, :, 1].cpu().detach().numpy(), label='y train')
# plt.title("output 1 train single SSM")
# plt.legend()
# plt.show()
#
# plt.figure('12')
# plt.plot(yRNN_val[:, 1].cpu().detach().numpy(), label='SSM val')
# plt.plot(yval[:, 1].cpu().detach().numpy(), label='y val')
# plt.title("output 1 val single SSM")
# plt.legend()
# plt.show()
#
# plt.figure('13')
# plt.plot(yRNN[0, :, 2].cpu().detach().numpy(), label='SSM')
# plt.plot(y[0, :, 2].cpu().detach().numpy(), label='y train')
# plt.title("output 1 train single SSM")
# plt.legend()
# plt.show()
#
# plt.figure('14')
# plt.plot(yRNN_val[:, 2].cpu().detach().numpy(), label='SSM val')
# plt.plot(yval[:, 2].cpu().detach().numpy(), label='y val')
# plt.title("output 1 val single SSM")
# plt.legend()
# plt.show()

# plt.figure('15')
# plt.plot(d[inputnumberD, :].detach().numpy(), label='input train')
# plt.plot(dval[inputnumberD, :].detach().numpy(), label='input val')
# plt.title("input single REN")
# plt.legend()
# plt.show()

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

# Create a single figure -----------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.25, 2.75))

# Plot all data on the same axes
# h_1 (Output 0 validation)
ax.plot(yval[20:600, 0].cpu().detach().numpy(), color='orange', linestyle='-')  # Solid true data
ax.plot(yRNN_val[20:600, 0].cpu().detach().numpy(), color='blue', linestyle=':')  # Dotted SSM

# h_2 (Output 1 validation)
ax.plot(yval[20:600, 1].cpu().detach().numpy(), color='orange', linestyle='-')  # Solid true data
ax.plot(yRNN_val[20:600, 1].cpu().detach().numpy(), color='blue', linestyle=':')  # Dotted SSM

# h_3 (Output 2 validation)
ax.plot(yval[20:600, 2].cpu().detach().numpy(), label=r'$h$', color='orange', linestyle='-')  # Solid true data
ax.plot(yRNN_val[20:600, 2].cpu().detach().numpy(), label=r'L2RU', color='blue', linestyle=':')  # Dotted L2RU

# Labels
ax.set_xlabel('Time [s]')
ax.set_ylabel('Height [cm]')  # Generalized y-label since all are in cm

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=False)

# Add legend above the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

# Clean up: remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout to fit legend
plt.tight_layout(rect=[0, 0, 1, 0.9])

# Save as PDF in current directory
plt.savefig('combined_single_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Load loss data from files in current directory
loss_files = ['loss.npy', 'loss2.npy']
loaded_losses = {}

for file in loss_files:
    if os.path.exists(file):
        loaded_losses[file] = np.load(file)
    else:
        print(f"Warning: {file} not found. Skipping.")

# Check if both files were loaded
if len(loaded_losses) < 2:
    raise FileNotFoundError("Need both 'loss.npy' and 'loss2.npy' to compare.")

# Create figure (half-column width)
fig, ax = plt.subplots(figsize=(2.25, 1.75))

# Plot each loss sequence
epochs1 = np.arange(1, len(loaded_losses['loss.npy']) + 1)
epochs2 = np.arange(1, len(loaded_losses['loss2.npy']) + 1)

ax.plot(epochs1, loaded_losses['loss.npy'], label='Tuned init.', color='blue', linestyle='-')
ax.plot(epochs2, loaded_losses['loss2.npy'], label='Random init.', color='orange', linestyle='--')

# Set logarithmic scale for y-axis
ax.set_yscale('log')

# Labels
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')

# Add legend above the plot
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

# Clean up: remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout to fit legend
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save as PDF in current directory
plt.savefig('loss_comparison.pdf', format='pdf', bbox_inches='tight')
plt.show()

# # Create figure with three horizontal subplots
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.25, 1.75))
#
# # Subplot 1: Output 0 validation
# ax1.plot(yval[20:600, 0].cpu().detach().numpy(), label='h', color='orange', linestyle='-')  # Solid data first
# ax1.plot(yRNN_val[20:600, 0].cpu().detach().numpy(), label='SSM', color='blue', linestyle=':')  # Dotted SSM on top
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel(r'$h_1$ [cm]')
#
# # Subplot 2: Output 1 training
# ax2.plot(yval[20:600, 1].cpu().detach().numpy(), label='h', color='orange', linestyle='-')  # Solid data first
# ax2.plot(yRNN_val[20:600, 1].cpu().detach().numpy(), label='SSM', color='blue', linestyle=':')  # Dotted SSM on top
# ax2.set_xlabel('Time [s]')
# ax2.set_ylabel(r'$h_2$ [cm]')
#
# handles, labels = ax3.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=False)
#
# # Adjust layout to prevent overlap
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
#
# # Save as PDF for LaTeX
# plt.savefig('comparison_figure.pdf', format='pdf', bbox_inches='tight')
# plt.show()

# Create figure with two subplots side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 2.0))  # Total width ~7 inches

# --- Left Subplot: Loss Comparison ---
# Load loss data
loss_files = ['loss.npy', 'loss2.npy']
loaded_losses = {}
for file in loss_files:
    if os.path.exists(file):
        loaded_losses[file] = np.load(file)
    else:
        print(f"Warning: {file} not found.")

# Plot losses
if 'loss.npy' in loaded_losses:
    epochs1 = np.arange(1, len(loaded_losses['loss.npy']) + 1)
    ax2.plot(epochs1, loaded_losses['loss.npy'], label='Tuned init.', color='blue', linestyle='-')
if 'loss2.npy' in loaded_losses:
    epochs2 = np.arange(1, len(loaded_losses['loss2.npy']) + 1)
    ax2.plot(epochs2, loaded_losses['loss2.npy'], label='Random init.', color='orange', linestyle='--')

ax2.set_yscale('log')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Training loss')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Right Subplot: Combined Trajectories ---
# h_1 (Output 0 validation)
ax1.plot(yval[20:600, 0].cpu().detach().numpy(), color='orange', linestyle='-')
ax1.plot(yRNN_val[20:600, 0].cpu().detach().numpy(), color='blue', linestyle=':')

# h_2 (Output 1 validation)
ax1.plot(yval[20:600, 1].cpu().detach().numpy(), color='orange', linestyle='-')
ax1.plot(yRNN_val[20:600, 1].cpu().detach().numpy(), color='blue', linestyle=':')

# h_3 (Output 2 validation)
ax1.plot(yval[20:600, 2].cpu().detach().numpy(), label=r'Ground truth', color='orange', linestyle='-')
ax1.plot(yRNN_val[20:600, 2].cpu().detach().numpy(), label=r'L2RU', color='blue', linestyle=':')

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Height [cm]')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Subplot 3: Validation Loss vs. Parameters for Three Models ---
# Simulated data (replace with your actual data)
# Model 1: 11 points
param_counts1 = np.array([100, 200, 500, 1400, 2200, 3600, 4700, 5800, 8600, 10000, 14000])
val_losses1 = np.array([0.6, 0.34, 0.38, 0.48, 0.4, 0.44, 0.15, 0.22, 0.08, 0.1, 0.08])

# Model 2: 11 points
param_counts2 = np.array([1000, 1800, 2350, 2700, 4300, 6500, 8750, 9850, 11950, 14050, 17050])
val_losses2 = np.array([0.05, 0.03, 0.10, 0.09, 0.11, 0.027, 0.028, 0.031, 0.032, 0.035, 0.03])

# Model 3: 11 points
param_counts3 = np.array([1000, 1800, 2350, 2700, 4300, 6500, 8750, 9850, 11950, 14050, 17050])
val_losses3 = np.array([0.43, 0.033, 0.12, 0.11, 0.06, 0.033, 0.034, 0.035, 0.033, 0.035, 0.034])

# Define distinct colors and markers for each model
model_styles = [
    {'color': 'blue', 'marker': 'o', 'label': 'RNN'},  # Circle, blue
    {'color': 'orange', 'marker': 's', 'label': 'REN'},  # Square, orange
    {'color': 'green', 'marker': '^', 'label': 'L2RU'}  # Triangle, green
]

# Plot points for each model
for i, model in enumerate(model_styles):
    if i == 0:
        ax3.scatter(param_counts1, val_losses1, color=model['color'], marker=model['marker'],
                    label=model['label'], s=20)
    elif i == 1:
        ax3.scatter(param_counts2, val_losses2, color=model['color'], marker=model['marker'],
                    label=model['label'], s=20)
    elif i == 2:
        ax3.scatter(param_counts3, val_losses3, color=model['color'], marker=model['marker'],
                    label=model['label'], s=20)

#ax3.set_yscale('log')
ax3.set_xlabel('Num. of parameters')
ax3.set_ylabel('Val. Loss')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add legends above each subplot
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=1, frameon=False)

# Adjust layout to fit legends
#plt.tight_layout(rect=[0, 0, 1, 0.9])

# Add subplot labels (a) and (b)
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=6, va='top')
ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=6, va='top')
ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, fontsize=6, va='top')

# Save as PDF in current directory
plt.savefig('side_by_side_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Create a single figure sized for a column
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # ~3.5 inches wide, 2.5 inches tall

# Data for three models
# Model 1: 11 points
param_counts1 = np.array([100, 200, 500, 1400, 2200, 3600, 4700, 5800, 8600, 10000, 14000])
val_losses1 = np.array([0.6, 0.34, 0.38, 0.48, 0.4, 0.44, 0.15, 0.22, 0.08, 0.1, 0.08])

# Model 2: 11 points
param_counts2 = np.array([1000, 1800, 2350, 2700, 4300, 6500, 8750, 9850, 11950, 14050, 17050])
val_losses2 = np.array([0.05, 0.03, 0.10, 0.09, 0.11, 0.027, 0.028, 0.031, 0.032, 0.035, 0.03])

# Model 3: 11 points
param_counts3 = np.array([1000, 1800, 2350, 2700, 4300, 6500, 8750, 9850, 11950, 14050, 17050])
val_losses3 = np.array([0.43, 0.033, 0.12, 0.11, 0.06, 0.033, 0.034, 0.035, 0.033, 0.035, 0.034])

# Define distinct colors and markers for each model
model_styles = [
    {'color': 'blue', 'marker': 'o', 'label': 'RNN'},  # Circle, blue
    {'color': 'orange', 'marker': 's', 'label': 'REN'},  # Square, orange
    {'color': 'green', 'marker': '^', 'label': 'L2RU'}  # Triangle, green
]

# Plot points for each model
for i, model in enumerate(model_styles):
    if i == 0:
        ax.scatter(param_counts1, val_losses1, color=model['color'], marker=model['marker'],
                   label=model['label'], s=20)
    elif i == 1:
        ax.scatter(param_counts2, val_losses2, color=model['color'], marker=model['marker'],
                   label=model['label'], s=20)
    elif i == 2:
        ax.scatter(param_counts3, val_losses3, color=model['color'], marker=model['marker'],
                   label=model['label'], s=20)

# Axis labels and styling
ax.set_xlabel('Num. of Parameters')
ax.set_ylabel('Val. Loss')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend below the plot (suitable for column width)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)

# Save as PDF in current directory
plt.savefig('val_loss_vs_params.pdf', format='pdf', bbox_inches='tight')
plt.show()

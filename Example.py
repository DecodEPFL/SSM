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

seed = 2
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
    "n_layers": 2,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi,
    "r_min": 0.7,
    "r_max": 0.98,
    "gamma": True,
    "trainable": True,
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
for itr in tqdm(range(2500)):
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

plt.figure('8')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('9')
plt.plot(yRNN[0, :, 0].cpu().detach().numpy(), label='SSM')
plt.plot(y[0, :, 0].cpu().detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

plt.figure('10')
plt.plot(yRNN_val[:, 0].cpu().detach().numpy(), label='SSM val')
plt.plot(yval[:, 0].cpu().detach().numpy(), label='y val')
plt.title("output 1 val single SSM")
plt.legend()
plt.show()

plt.figure('11')
plt.plot(yRNN[0, :, 1].cpu().detach().numpy(), label='SSM')
plt.plot(y[0, :, 1].cpu().detach().numpy(), label='y train')
plt.title("output 1 train single SSM")
plt.legend()
plt.show()

plt.figure('12')
plt.plot(yRNN_val[:, 1].cpu().detach().numpy(), label='SSM val')
plt.plot(yval[:, 1].cpu().detach().numpy(), label='y val')
plt.title("output 1 val single SSM")
plt.legend()
plt.show()

plt.figure('13')
plt.plot(yRNN[0, :, 2].cpu().detach().numpy(), label='SSM')
plt.plot(y[0, :, 2].cpu().detach().numpy(), label='y train')
plt.title("output 1 train single SSM")
plt.legend()
plt.show()

plt.figure('14')
plt.plot(yRNN_val[:, 2].cpu().detach().numpy(), label='SSM val')
plt.plot(yval[:, 2].cpu().detach().numpy(), label='y val')
plt.title("output 1 val single SSM")
plt.legend()
plt.show()

# plt.figure('15')
# plt.plot(d[inputnumberD, :].detach().numpy(), label='input train')
# plt.plot(dval[inputnumberD, :].detach().numpy(), label='input val')
# plt.title("input single REN")
# plt.legend()
# plt.show()

print(f"Loss Validation single RNN: {loss_val}")

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from os.path import  join as pjoin
from torch import nn
import math
from argparse import Namespace
import torch
from lru.architectures import DWN, DWNConfig
from tqdm import tqdm

dtype = torch.float
device = torch.device("cuda")

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

u = torch.zeros(nExp, t_end, 1, device = device)
y = torch.zeros(nExp, t_end, 3, device = device)
inputnumberD = 1

for j in range(nExp):
    inputActive = (torch.from_numpy(dExp[0, j])).T
    u[j, :, :] = torch.unsqueeze(inputActive[:, inputnumberD], 1)
    y[j, :, :] = (torch.from_numpy(yExp[0, j])).T

seed = 2
torch.manual_seed(seed)

# very small architecture
cfg = {
    "n_u": 1,
    "n_y": 3,
    "d_model": 5,
    "d_state": 5,
    "n_layers": 4,
    "ff": "LMLP",  # GLU | MLP | LMLP
    "max_phase": math.pi,
    "r_min": 0.7,
    "r_max": 0.9,
    "gamma": True,
}
cfg = Namespace(**cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
#torch.set_num_threads(10)

# Build model
config = DWNConfig(d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers, ff=cfg.ff, rmin=cfg.r_min,
                   rmax=cfg.r_max, max_phase=cfg.max_phase, gamma=cfg.gamma)
model = DWN(cfg.n_u, cfg.n_y, config)
model.cuda()

# Configure optimizer
opt = torch.optim.AdamW(model.parameters(), lr=2e-2)
opt.zero_grad()


total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")


MSE = nn.MSELoss()

LOSS = []
# Train loop
for itr in tqdm(range(800)):
    yRNN = model(u, state=None, mode="scan")
    yRNN = torch.squeeze(yRNN)
    loss = MSE(yRNN, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    if itr % 100 == 0:
        print(loss.item())
    LOSS.append(loss.item())

checkpoint = {
    'model': model.state_dict(),
    'LOSS': np.array(LOSS),
    'cfg': cfg
}

torch.save(checkpoint, "ckpt.pt")


t_end = yExp_val[0, 0].shape[1]

nExp = yExp_val.size

uval = torch.zeros(nExp, t_end, 1,  device = device)
yval = torch.zeros(nExp, t_end, 3,  device = device)

for j in range(nExp):
    inputActive = (torch.from_numpy(dExp_val[0, j])).T
    uval[j, :, :] = torch.unsqueeze(inputActive[:, inputnumberD], 1)
    yval[j, :, :] = (torch.from_numpy(yExp_val[0, j])).T

yRNN_val = model(uval)
yRNN_val = torch.squeeze(yRNN_val)
yval = torch.squeeze(yval)

loss_val = MSE(yRNN_val, yval)

plt.figure('8')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('9')
plt.plot(yRNN[0, :, 0].cpu().detach().numpy(), label='REN')
plt.plot(y[0, :, 0].cpu().detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

plt.figure('10')
plt.plot(yRNN_val[:, 0].cpu().detach().numpy(), label='REN val')
plt.plot(yval[:, 0].cpu().detach().numpy(), label='y val')
plt.title("output 1 val single RNN")
plt.legend()
plt.show()

plt.figure('11')
plt.plot(yRNN[0, :, 1].cpu().detach().numpy(), label='REN')
plt.plot(y[0, :, 1].cpu().detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

plt.figure('12')
plt.plot(yRNN_val[:, 1].cpu().detach().numpy(), label='REN val')
plt.plot(yval[:, 1].cpu().detach().numpy(), label='y val')
plt.title("output 1 val single REN")
plt.legend()
plt.show()

plt.figure('13')
plt.plot(yRNN[0, :, 2].cpu().detach().numpy(), label='REN')
plt.plot(y[0, :, 2].cpu().detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

plt.figure('14')
plt.plot(yRNN_val[:, 2].cpu().detach().numpy(), label='REN val')
plt.plot(yval[:, 2].cpu().detach().numpy(), label='y val')
plt.title("output 1 val single RNN")
plt.legend()
plt.show()

# plt.figure('15')
# plt.plot(d[inputnumberD, :].detach().numpy(), label='input train')
# plt.plot(dval[inputnumberD, :].detach().numpy(), label='input val')
# plt.title("input single REN")
# plt.legend()
# plt.show()

print(f"Loss Validation single RNN: {loss_val}")
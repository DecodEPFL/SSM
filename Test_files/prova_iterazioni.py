import numpy as np
import torch
from pathlib import Path
import neural_ssm
from torch.utils.data import DataLoader, random_split, Subset, SubsetRandomSampler, Dataset
from src.neural_ssm.ssm.lru import DeepSSM
import matplotlib.pyplot as plt
torch.manual_seed(2)

def ensure_3d(x):
    """ensures that tensors have dimension (batch, time, input_dim)."""
    if x.ndim == 1:
        # Case: sequence 1D -> (1, T, 1)
        x = x.unsqueeze(0).unsqueeze(-1)
    elif x.ndim == 2:
        # Case: batvh or sequence 2D -> (batch, T, 1)
        x = x.unsqueeze(-1)
    return x



patient = 1

# Carica e ricrea il loader
train_dataset = torch.load('train_dataset.pth', map_location="cpu")

checkpoint = torch.load('batch_data.pth', map_location="cpu")
u0_batch = checkpoint['u0_batch']
u1_batch = checkpoint['u1_batch']
y_batch = checkpoint['y_batch']
time = checkpoint['time']

# --------------------------------------------------------

SSM_0 = DeepSSM(d_input= 1, d_output=1, d_model = 8,  n_layers = 7, ff='GLU',param='lru', d_state=8,  d_hidden=12, dim_amp=3)
SSM_1 = DeepSSM(d_input= 1, d_output=1, d_model = 8,  n_layers = 7, ff='GLU',param='l2n', d_state=8,  d_hidden=12, dim_amp=3)

checkpoint = torch.load( "trained_models_lru.pth", map_location="cpu")

#SSM_0.load_state_dict(checkpoint["SSM_0_state_dict"])
SSM_1.load_state_dict(checkpoint["SSM_1_state_dict"])

SSM_0.eval(); SSM_1.eval()
SSM_0.reset; SSM_1.reset


# --------------Plot identification results for G-----------------


u0_batch, u1_batch, y_batch, time = ensure_3d(u0_batch), ensure_3d(u1_batch), ensure_3d(y_batch), ensure_3d(time)

T=1200
#u1_batch = torch.randn(12,T,1)

y0_hat, _ = SSM_0(u0_batch)
y1_hat, _ = SSM_1(u = u1_batch, mode = 'loop')
y_hat = y1_hat

y_hat_np_1 = y_hat.detach().cpu().numpy()
    

SSM_0.reset; SSM_1.reset


# --------------------------------------

y_hat_list = []
y0_hat_list = []
y1_hat_list = []

SSM_0_x = None; SSM_1_x = None


for i in range(T):

    #y0, SSM_0_x = SSM_0(u = u0_batch[:,i:i+1,:], state = SSM_0_x)
    y1, SSM_1_x = SSM_1(u = u1_batch[:,i:i+1,:], state = SSM_1_x, mode = 'loop')
    
    y = y1
    
    y_hat_list.append(y)  # Store output
    #y0_hat_list.append(y0)
    y1_hat_list.append(y1)
    
    
y_hat = torch.cat(y_hat_list, dim=1)  # Shape: (batch_size, horizon, output_dim)
#y0_hat = torch.cat(y0_hat_list, dim=1)  # Shape: (batch_size, horizon, output_dim)
y1_hat = torch.cat(y1_hat_list, dim=1)  # Shape: (batch_size, horizon, output_dim)


# --------------------------------------------------

y_hat_np_2 = y_hat.detach().cpu().numpy()


plt.figure()
plt.plot(np.abs(y_hat_np_1[0,:,0]-y_hat_np_2[0,:,0]), label = "run")
#plt.plot(y_hat_np_2[0,:,0], label = "iteration")
plt.grid()
plt.legend()
plt.show()



SSM_1.reset
y1, SSM_1_x = SSM_1(u1_batch[0,:8,:], state = None)  # 0 e 1
print(f'y1_t0 = {y1[0,0,0]}')
print(f'y1_t1 = {y1[0,1,0]}')
print(f'SSM_1_x_t1 = {SSM_1_x[0][0,:] }')



SSM_1.reset
y1_t0, SSM_1_x_t0 = SSM_1(u = u1_batch[0,:1,:], state = None)
y1_t1, SSM_1_x_t1 = SSM_1(u = u1_batch[0,1:2,:], state = SSM_1_x_t0)
y1_t2, SSM_1_x_t2 = SSM_1(u = u1_batch[0,2:3,:], state = SSM_1_x_t1)
y1_t3, SSM_1_x_t3 = SSM_1(u = u1_batch[0,3:4,:], state = SSM_1_x_t2)
print(f'y1_t0 = {y1_t0[0,0,0]}')
print(f'y1_t1 = {y1_t1[0,0,0]}')
print(f'SSM_0_x[0] = {SSM_1_x_t1[0][0,:] }')

SSM_1
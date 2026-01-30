import numpy as np
import torch
import time
from l2cell import L2BoundedLTICell
from src.neural_ssm.ssm.lru import DeepSSM, SSMConfig, PureLRUR
from l2cellblocks import Block2x2DenseL2SSM
import matplotlib.pyplot as plt

d_state=4
d_input=23
torch.manual_seed(32)
cell = Block2x2DenseL2SSM(d_state=d_state, d_input=d_input, d_output=d_input, gamma=1)


lru = PureLRUR(n=2, param='l2ru', gamma=.2)

u1 = 5*torch.rand(5,180,2)+torch.tensor(5.)
u2 = 5*torch.ones(5,180,2)+torch.tensor(3.)

#u = torch.cat((u1,  u2), 1)

#a,b = lru(u)

u=torch.randn(1,100,1)



ssm = DeepSSM(d_input=1, d_output=1, d_model=8, d_state=8, n_layers=7, ff='GLU', param='tv', gamma=None, nl_layers=5)

a,b = ssm(u = u, state = None)

a=a.detach().cpu().numpy()

x = None
yy=[]
for t in range(100):
    y, x = ssm(u = u[0,t:t+1,:], state = x)
    yy.append(y)

yy=torch.cat(yy, dim=1)
yy = yy.detach().cpu().numpy()

plt.figure()
plt.plot(a[0,:,0], label = "run")
plt.plot(yy[0,:,0], label = "iteration")
plt.grid()
plt.legend()
plt.show()



a0, b0 = ssm(u=u[:,:1,:], state = None, mode = 'loop')
a1, b1 = ssm(u=u[:, 1:2, :], state = b0, mode = 'loop')
a2, b2 = ssm(u=u[:,2:3,:], state = b1, mode = 'loop')
a3, b3 = ssm(u=u[:,3:4,:], state = b2, mode = 'loop')



ud= torch.zeros_like(u)
ud[:,0,:] = u[:,0,:]

for k in range(u.size(1)-1):
    ud[:,k+1,:] = torch.abs(u[:,k+1,:]-u[:,k,:])

yd= torch.zeros_like(a)
yd[:,0,:] = a[:,0,:]

for k in range(a.size(1)-1):
    yd[:,k+1,:] = torch.abs(a[:,k+1,:]-a[:,k,:])

plt.plot(yd[2,:,0].detach().numpy())
plt.show()

plt.plot(a[2,:,0].detach().numpy())
plt.show()


ssm = DeepSSM(d_input=3, d_output=3, d_model=6, d_state=8, n_layers=2, ff='GLU', param='tv', gamma=1)

a,b=ssm(u = torch.randn((3,1,3)), state = None)

SSM_1_101 = DeepSSM(d_input=1, d_output=1, d_model = 8, d_state=8, n_layers = 7, ff='GLU', param='l2n',
                    max_phase_b=0.04,  d_hidden=12, dim_amp=3)

# Initialize A with eigenvalues ≈ 0.99
# target spectrum: 3 slow modes, 3 faster, some negative
eigvals_target = .95*torch.ones(d_state, dtype=torch.float64)
#cell.init_near_identity()
#cell.init_orthogonal_spectrum(eigvals_target, offdiag_scale=0.8)

# cell.init_on_circle(
#     rho=0.99,
#     max_phase=0.5,            # small spread
#     phase_center=0.3,          # center angle ≈ 17°
#     random_phase=True,
# )


#cell.init_orthogonal_spectrum(eigvals_target, offdiag_scale=1e-3)

#A, B, C, D, P = cell.compute_ssm_matrices()
A, B, C, D, P = cell.compute_dense_matrices()

# 1) Check BRL matrix
M = cell.bounded_real_matrix_x()
max_eig = torch.linalg.eigvalsh(M).max().item()
print("max eig of M:", max_eig)  # should be <= 0 (up to ~1e-10)

# 2) Use your control code
A_np = A.detach().cpu().numpy()
B_np = B.detach().cpu().numpy()
C_np = C.detach().cpu().numpy()
D_np = D.detach().cpu().numpy()

import control

sys = control.ss(A_np, B_np, C_np, D_np, dt=1.0)
gamma_hinf = control.norm(sys, p='inf')
print("control H∞:", gamma_hinf)


u = torch.randn(44, 3000, d_input)
y, x_last = cell(u, mode = 'scan')  # loop mode, L2-gain constrained



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cell = cell.to(device).eval()
u = u.to(device)

# optional: no gradients for speed measurement
@torch.no_grad()
def run_mode(mode: str, iters: int = 10):
    # warmup
    for _ in range(3):
        y, x_last = cell(u, mode=mode)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y, x_last = cell(u, mode=mode)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / iters  # average seconds per call


loop_time = run_mode("loop", iters=20)
scan_time = run_mode("scan", iters=20)

print(f"loop: {loop_time*1000:.3f} ms / call")
print(f"scan: {scan_time*1000:.3f} ms / call")




gamma_hinf
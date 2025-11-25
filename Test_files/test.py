import numpy as np
import torch
import time
from l2cell import L2BoundedLTICell

from l2cellblocks import Block2x2DenseL2SSM

d_state=4
d_input=23
torch.manual_seed(32)
cell = Block2x2DenseL2SSM(d_state=d_state, d_input=d_input, d_output=d_input, gamma=1)


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
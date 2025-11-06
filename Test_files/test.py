import numpy as np
import torch

from l2cell import L2BoundedLTICell

torch.manual_seed(39)
cell = L2BoundedLTICell(d_state=3, d_input=11, d_output=11, gamma=44)


# Initialize A with eigenvalues ≈ 0.99
# target spectrum: 3 slow modes, 3 faster, some negative
eigvals_target = torch.tensor([0.98, 0.95, -0.9], dtype=torch.float64)
cell.init_orthogonal_spectrum(eigvals_target, offdiag_scale=0.8)

#cell.init_orthogonal_spectrum(eigvals_target, offdiag_scale=1e-3)

A, B, C, D, P = cell.compute_ssm_matrices()

# 1) Check BRL matrix
M = cell.bounded_real_matrix()
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


u = torch.randn(44, 300, 11)
y, x_last = cell(u)  # loop mode, L2-gain constrained

gamma_hinf
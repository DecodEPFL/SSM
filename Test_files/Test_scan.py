import torch
import math
import time
import matplotlib.pyplot as plt

# -------------------------
# Parallel prefix (safe)
# -------------------------
def parallel_scan_affine(M: torch.Tensor, v: torch.Tensor):
    """
    Inclusive parallel prefix for affine pairs (M, v) with column-vector convention.

    Args:
        M: (seq_len, batch, D, D)
        v: (seq_len, batch, D)

    Returns:
        M_p: (seq_len, batch, D, D)  where M_p[t] = M[t] @ M[t-1] @ ... @ M[0]
        v_p: (seq_len, batch, D)    where v_p[t] = M[t] @ M[t-1] @ ... @ M[1] @ v[0] + ... + v[t]
    """
    n = M.shape[0]
    if n == 0:
        return M, v

    # Work on copies so we don't mutate inputs
    M_p = M.clone()
    v_p = v.clone()

    offset = 1
    # Doubling rounds
    while offset < n:
        # slice left = M_p[offset:]  (length n-offset)
        # slice right = M_p[:n-offset] (length n-offset)
        left = M_p[offset:].clone()   # shape (n-offset, batch, D, D)
        right = M_p[: n - offset].clone()
        # new_M[i] for i >= offset equals left[i-offset] @ right[i-offset]
        new_M_tail = torch.matmul(left, right)  # (n-offset, batch, D, D)

        # For v: new_v[i] = left[i-offset] @ v_p[i-offset] + v_p[i]
        right_v = v_p[: n - offset].unsqueeze(-1).clone()   # (n-offset, batch, D, 1)
        transformed = torch.matmul(left, right_v).squeeze(-1)   # (n-offset, batch, D)
        new_v_tail = transformed + v_p[offset:].clone()        # (n-offset, batch, D)

        # Reconstruct full arrays without in-place overlapping writes
        if offset == 0:
            M_p = new_M_tail
            v_p = new_v_tail
        else:
            M_p = torch.cat([M_p[:offset], new_M_tail], dim=0)
            v_p = torch.cat([v_p[:offset], new_v_tail], dim=0)

        offset <<= 1

    return M_p, v_p


# -------------------------
# High-level wrapper
# -------------------------
def compute_linear_recurrence_parallel(A, B, u, x0):
    """
    Compute x_t for t=1..T where x_{t} = A_t @ x_{t-1} + B_t @ u_{t-1}

    Conventions (column-vector):
      - A: (seq_len, D, D) or (D, D)
      - B: (seq_len, D, D) or (D, D)
      - u: (seq_len, batch, D)   (u_0 .. u_{T-1})
      - x0: (batch, D)

    Returns:
      x: (seq_len, batch, D)  -> x[0] = x_1, ... x[T-1] = x_T
    """
    seq_len = u.shape[0]
    batch_size = u.shape[1]
    D = u.shape[2]

    # ensure A,B have time dimension
    if A.dim() == 2:
        A = A.unsqueeze(0).expand(seq_len, -1, -1).contiguous()
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(seq_len, -1, -1).contiguous()

    # shape (seq_len, batch, D, D)
    M = A.unsqueeze(1).expand(-1, batch_size, -1, -1).contiguous()
    B_exp = B.unsqueeze(1).expand(-1, batch_size, -1, -1).contiguous()

    # v_t = B_t @ u_t  (u is (seq_len, batch, D) -> unsqueeze -> (seq_len, batch, D, 1))
    v = torch.matmul(B_exp, u.unsqueeze(-1)).squeeze(-1)  # (seq_len, batch, D)

    # compute prefix
    M_p, v_p = parallel_scan_affine(M, v)  # M_p: (seq_len, batch, D, D), v_p: (seq_len, batch, D)

    # compute x_t = M_p[t] @ x0 + v_p[t]
    # x0.unsqueeze(-1) is (batch, D, 1) -- will broadcast across seq_len
    x = torch.matmul(M_p, x0.unsqueeze(-1)).squeeze(-1) + v_p  # (seq_len, batch, D)
    return x


# -------------------------
# Sequential reference
# -------------------------
def compute_linear_recurrence_sequential(A, B, u, x0):
    seq_len = u.shape[0]
    batch_size, D = x0.shape
    device = x0.device
    x = torch.zeros(seq_len, batch_size, D, device=device, dtype=x0.dtype)
    current_x = x0.clone()
    A_time = (A.dim() == 3)
    B_time = (B.dim() == 3)

    for t in range(seq_len):
        A_t = A[t] if A_time else A
        B_t = B[t] if B_time else B
        input_term = torch.matmul(B_t, u[t].unsqueeze(-1)).squeeze(-1)
        current_x = torch.matmul(A_t, current_x.unsqueeze(-1)).squeeze(-1) + input_term
        x[t] = current_x
    return x


# -------------------------
# Small test driver
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    seq_len = 120
    D = 3
    batch_size = 7
    device = torch.device("cpu")

    import torch


    # --------- Helpers ---------
    def spectral_radius(A: torch.Tensor) -> float:
        """Return spectral radius (max abs eigenvalue). Works for real or batched real matrices."""
        vals = torch.linalg.eigvals(A)
        return float(vals.abs().max().item())


    def operator_norm2(A: torch.Tensor) -> float:
        """Return operator 2-norm (largest singular value)."""
        return float(torch.linalg.norm(A, ord=2).item())


    # --------- Method 1: Scale random matrix by spectral radius ----------
    def random_stable_by_scaling(n: int, rho: float = 0.99, device=None, dtype=torch.float32, seed=None):
        """
        Generate random matrix and scale so its spectral radius <= rho.
        Result may be non-normal.
        """
        if seed is not None:
            torch.manual_seed(seed)
        A = torch.randn(n, n, device=device, dtype=dtype)
        vals = torch.linalg.eigvals(A)
        maxabs = float(vals.abs().max().item())
        if maxabs == 0:
            return A  # degenerate; rare
        if maxabs > rho:
            A = A * (rho / maxabs)
        return A


    # --------- Method 2: Orthogonal similarity to diagonal (normal, low transient) ----------
    def random_stable_normal(n: int, rho: float = 0.99, device=None, dtype=torch.float32, seed=None):
        """
        Build A = Q D Q^T with Q orthogonal (from QR) and D diagonal with entries in (-rho, rho).
        This yields a normal matrix (no transient amplification beyond spectral radius).
        """
        if seed is not None:
            torch.manual_seed(seed)
        M = torch.randn(n, n, device=device, dtype=dtype)
        Q, R = torch.linalg.qr(M)
        # generate real eigenvalues in [-rho, rho]
        d = (2 * rho) * (torch.rand(n, device=device, dtype=dtype) - 0.5)
        D = torch.diag(d)
        A = Q @ D @ Q.T
        return A


    # --------- Method 3: Controlled non-normal via similarity A = S D S^{-1} ----------
    def random_stable_similarity(n: int, rho: float = 0.99, cond_S: float = 10.0, device=None, dtype=torch.float32,
                                 seed=None):
        """
        Create A = S D S^{-1} where D diagonal of eigenvalues with abs<=rho and S has desired condition number ~cond_S.
        cond_S controls non-normality (larger -> more transient amplification possible).
        """
        if seed is not None:
            torch.manual_seed(seed)
        # random orthogonal for U and V
        U = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))[0]
        V = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))[0]

        # construct singular values for S between 1 and cond_S
        sv_min = 1.0
        sv_max = float(cond_S)
        s_vals = torch.linspace(sv_max, sv_min, steps=n, device=device, dtype=dtype)
        S = U @ torch.diag(s_vals) @ V.T

        # diagonal eigenvalues inside unit disk (can be complex but keep real for simplicity)
        eigs = (2 * rho) * (torch.rand(n, device=device, dtype=dtype) - 0.5)  # in [-rho, rho]
        D = torch.diag(eigs)

        Sinv = torch.linalg.inv(S)
        A = S @ D @ Sinv
        return A


    # --------- Batch / sequence generator ----------
    def generate_stable_sequence(seq_len: int, n: int, rho: float = 0.99, method: str = 'scaling',
                                 cond_S: float = 10.0, device=None, dtype=torch.float32, seed=None):
        """
        Generate a sequence of stable matrices A[t], shape (seq_len, n, n).
        method in {'scaling', 'normal', 'similarity'}.
        """
        As = []
        for t in range(seq_len):
            s = None if seed is None else seed + t
            if method == 'scaling':
                A = random_stable_by_scaling(n, rho=rho, device=device, dtype=dtype, seed=s)
            elif method == 'normal':
                A = random_stable_normal(n, rho=rho, device=device, dtype=dtype, seed=s)
            elif method == 'similarity':
                A = random_stable_similarity(n, rho=rho, cond_S=cond_S, device=device, dtype=dtype, seed=s)
            else:
                raise ValueError("unknown method")
            As.append(A.unsqueeze(0))
        return torch.cat(As, dim=0)  # (seq_len, n, n)


    # Random stable-ish A (scale down)
    A = generate_stable_sequence(seq_len, D, rho=0.98, method='scaling', seed=5)
    B = 0.5 * torch.randn(D, D, device=device)   # constant B
    u = 0.2 * torch.randn(seq_len, batch_size, D, device=device)
    x0 = torch.randn(batch_size, D, device=device)

    t0 = time.time()
    x_seq = compute_linear_recurrence_sequential(A, B, u, x0)
    t1 = time.time()
    x_par = compute_linear_recurrence_parallel(A, B, u, x0)
    t2 = time.time()

    print(f"Sequential time: {t1-t0:.4f}s, Parallel time: {t2-t1:.4f}s")
    # Compare (use a reasonable tolerance due to FP rounding)
    # Try double precision first if you want near-exact equality
    print("max abs diff (float32):", (x_seq - x_par).abs().max().item())
    print("allclose (atol=1e-5):", torch.allclose(x_seq, x_par, atol=1e-5))

    # If diff is large, try double-precision run to see whether it's rounding
    A_d = A.double()
    B_d = B.double()
    u_d = u.double()
    x0_d = x0.double()
    xs_seq_d = compute_linear_recurrence_sequential(A_d, B_d, u_d, x0_d)
    xp_par_d = compute_linear_recurrence_parallel(A_d, B_d, u_d, x0_d)
    print("max abs diff (float64):", (xs_seq_d - xp_par_d).abs().max().item())

    x_seq_np = x_seq[:, 0, :].numpy()
    x_par_np = x_par[:, 0, :].numpy()
    t = range(1, seq_len + 1)

    maxdiff = float(torch.max(torch.abs(x_seq - x_par)).item())

    # Plot
    plt.figure(figsize=(10, 6))
    for dim in range(D):
        plt.plot(t, x_seq_np[:, dim], label=f"seq dim {dim}")
        plt.plot(t, x_par_np[:, dim], label=f"par dim {dim}", linestyle='--')
    plt.xlabel("time step (t)")
    plt.ylabel("state value")
    plt.title(f"State trajectories (seq vs parallel) — max abs diff = {maxdiff:.3e}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    A_d

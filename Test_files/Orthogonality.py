import numpy as np
import matplotlib.pyplot as plt

def make_scaled_orthogonal(n, rho=0.99, rng=None):
    """A = rho * Q, Q orthogonal (Haar-ish via QR)."""
    rng = np.random.default_rng() if rng is None else rng
    G = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(G)
    # Fix sign ambiguity to make Q closer to Haar
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return rho * Q

def make_random_unit_column_matrix(n, rng=None, stabilize=True, rho=0.99, stable_by="spectral_radius"):
    """
    Build A by drawing n random unit vectors in R^n as columns.
    Columns are unit norm but not orthonormal. Optionally rescale to be stable.

    stabilize=True rescales A so that:
      - stable_by="spectral_radius": spectral radius(A) == rho (uses eigvals, O(n^3))
      - stable_by="opnorm": ||A||_2 approx == rho (power iteration, cheaper)
    """
    rng = np.random.default_rng() if rng is None else rng

    # Draw columns ~ N(0,I), then normalize each column to unit norm
    A = rng.standard_normal((n, n))
    col_norms = np.linalg.norm(A, axis=0) + 1e-12
    A = A / col_norms  # each column has ||a_j||2 = 1

    if not stabilize:
        return A

    if stable_by == "spectral_radius":
        eig = np.linalg.eigvals(A)
        rad = np.max(np.abs(eig)) + 1e-12
        A = (rho / rad) * A
        return A

    if stable_by == "opnorm":
        # Power iteration to estimate ||A||_2
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v) + 1e-12
        for _ in range(50):
            v = A @ v
            nv = np.linalg.norm(v) + 1e-12
            v /= nv
        # Rayleigh-ish estimate of ||A||_2
        Av = A @ v
        op = np.linalg.norm(Av) + 1e-12
        A = (rho / op) * A
        return A

    raise ValueError("stable_by must be 'spectral_radius' or 'opnorm'")


def make_complex_diagonal_real(n, rho_range=(0.90, 0.999), rng=None):
    """
    Real representation of a complex diagonal normal matrix:
    blockdiag_i [ rho_i * R(theta_i) ] (2x2 rotation-contraction blocks).
    Requires even n; if odd, last state is a real decay.
    """
    rng = np.random.default_rng() if rng is None else rng
    A = np.zeros((n, n))
    m = n // 2
    rhos = rng.uniform(rho_range[0], rho_range[1], size=m)
    thetas = rng.uniform(0.0, 2*np.pi, size=m)
    for i, (rho, th) in enumerate(zip(rhos, thetas)):
        c, s = np.cos(th), np.sin(th)
        blk = rho * np.array([[c, -s],
                              [s,  c]])
        A[2*i:2*i+2, 2*i:2*i+2] = blk
    if n % 2 == 1:
        A[-1, -1] = rng.uniform(rho_range[0], rho_range[1])
    return A

def make_random_stable_iid(n, rho=0.99, rng=None):
    """
    Dense random A, then rescale to spectral radius rho.
    (O(n^3) eigvals; keep n moderate or replace with sparse eigs.)
    """
    rng = np.random.default_rng() if rng is None else rng
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    eig = np.linalg.eigvals(A)
    rad = np.max(np.abs(eig))
    A = (rho / (rad + 1e-12)) * A
    return A

def make_strongly_nonnormal(n, rho=0.99, alpha=5.0):
    """
    Schur-stable but very non-normal: A = rho*(I + alpha*N),
    where N is strictly upper shift (nilpotent).
    Eigenvalues all rho, but can have large transient / alignment.
    """
    N = np.zeros((n, n))
    N[np.arange(n-1), np.arange(1, n)] = 1.0  # superdiagonal ones
    A = rho * (np.eye(n) + alpha * N)
    return A

def lag_matrix(A, b, K):
    """
    Build V = [b, Ab, A^2 b, ..., A^{K-1} b] without matrix powers.
    """
    n = A.shape[0]
    V = np.zeros((n, K), dtype=A.dtype)
    v = b.astype(A.dtype, copy=True)
    for k in range(K):
        V[:, k] = v
        v = A @ v
    return V

def orthogonality_metrics(V):
    """
    V: n x K (columns are lag vectors).
    Returns normalized Gram matrix and summary metrics.
    """
    # Normalize columns
    norms = np.linalg.norm(V, axis=0) + 1e-12
    U = V / norms
    G = U.T @ U  # K x K
    K = G.shape[0]

    # Off-diagonal statistics
    off = G - np.eye(K, dtype=G.dtype)
    off_abs = np.abs(off)
    max_off = np.max(off_abs[np.triu_indices(K, k=1)])
    mean_off = np.mean(off_abs[np.triu_indices(K, k=1)])

    # Spectral deviation from orthonormality
    # (How far is G from I in operator norm?)
    spec_dev = np.linalg.norm(off, ord=2)

    return G, {"max_offdiag": float(max_off),
               "mean_offdiag": float(mean_off),
               "spec_dev": float(spec_dev)}

def experiment(n=256, K=128, p_out=None, rho=0.99, seed=0):
    """
    Compare orthogonality of lag terms for several A choices.
    If p_out is not None, we measure orthogonality of h_k = C A^k b in R^{p_out}.
    Otherwise we measure g_k = A^k b in R^n.
    """
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(n)
    b /= np.linalg.norm(b) + 1e-12

    # Optional output projection C
    if p_out is not None:
        C = rng.standard_normal((p_out, n)) / np.sqrt(n)
    else:
        C = None

    As = {
        "scaled_orthogonal": make_scaled_orthogonal(n, rho=rho, rng=rng),
        "complex_diag_real": make_complex_diagonal_real(n, rho_range=(0.90, rho), rng=rng),
        "random_stable_iid": make_random_stable_iid(n, rho=rho, rng=rng),
        "strongly_nonnormal": make_strongly_nonnormal(n, rho=rho, alpha=5.0),

        # NEW: columns are random unit vectors (then rescaled to be stable)
        "unit_column_matrix": make_random_unit_column_matrix(
            n, rng=rng, stabilize=True, rho=rho, stable_by="spectral_radius"
        ),
    }


    results = {}
    grams = {}
    for name, A in As.items():
        V = lag_matrix(A, b, K)      # g_k in state
        if C is not None:
            V = C @ V                # h_k in output space
        G, stats = orthogonality_metrics(V)
        results[name] = stats
        grams[name] = G
    return results, grams

def plot_gram_heatmap(G, title="", vmin=-1.0, vmax=1.0, savepath=None):
    """
    Heatmap of normalized Gram matrix G = U^T U (cosine similarities).
    Off-diagonals near 0 => quasi-orthogonal lag directions.
    """
    plt.figure()
    plt.imshow(G, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("lag index")
    plt.ylabel("lag index")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=150)
    plt.show()

def random_unit_vectors_matrix(n, k=None, rng=None):
    """n x k matrix with i.i.d. random unit vectors as columns."""
    rng = np.random.default_rng() if rng is None else rng
    k = n if k is None else k
    X = rng.standard_normal((n, k))
    X /= (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
    return X

def haar_orthogonal(n, rng=None):
    """Haar-ish random orthogonal matrix via QR of Gaussian."""
    rng = np.random.default_rng() if rng is None else rng
    G = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(G)
    Q = Q @ np.diag(np.sign(np.diag(R)))  # fix sign ambiguity
    return Q

def matrix_orthogonality_stats(X):
    """
    X: n x k with unit-norm columns (approximately or exactly).
    Returns pairwise and global orthogonality / conditioning metrics.
    """
    k = X.shape[1]
    G = X.T @ X

    # Pairwise off-diagonal cosine similarities
    off = G - np.eye(k)
    iu = np.triu_indices(k, k=1)
    max_offdiag = float(np.max(np.abs(off[iu])))
    mean_offdiag = float(np.mean(np.abs(off[iu])))

    # Global deviation from orthonormality
    spec_dev = float(np.linalg.norm(off, ord=2))  # ||G - I||_2

    # Singular values (basis quality)
    s = np.linalg.svd(X, compute_uv=False)
    s_min, s_max = float(np.min(s)), float(np.max(s))
    cond = float(s_max / (s_min + 1e-12))

    return {
        "max_offdiag": max_offdiag,
        "mean_offdiag": mean_offdiag,
        "spec_dev": spec_dev,
        "s_min": s_min,
        "s_max": s_max,
        "cond": cond,
    }

def demo_random_vectors_vs_qr(ns=(64, 128, 256, 256), seed=0):
    rng = np.random.default_rng(seed)
    print("Comparing X (random unit vectors) vs Q (QR orthonormal basis)\n")
    for n in ns:
        X = random_unit_vectors_matrix(n, k=n, rng=rng)  # n random unit vectors
        Q = haar_orthogonal(n, rng=rng)  # true orthonormal basis

        sx = matrix_orthogonality_stats(X)
        sq = matrix_orthogonality_stats(Q)

        print(f"n={n}")
        print("  X=random unit vectors:")
        print(f"    max_offdiag={sx['max_offdiag']:.4f}  mean_offdiag={sx['mean_offdiag']:.4f}")
        print(
            f"    ||X^T X - I||_2={sx['spec_dev']:.4f}  s_min={sx['s_min']:.4f}  s_max={sx['s_max']:.4f}  cond={sx['cond']:.1f}")
        print("  Q=QR orthonormal basis:")
        print(f"    max_offdiag={sq['max_offdiag']:.4f}  mean_offdiag={sq['mean_offdiag']:.4f}")
        print(
            f"    ||Q^T Q - I||_2={sq['spec_dev']:.4e}  s_min={sq['s_min']:.4f}  s_max={sq['s_max']:.4f}  cond={sq['cond']:.1f}")
        print()

if __name__ == "__main__":
    n=8
    kk=64
    # Try state-space lag orthogonality
    res_state, _ = experiment(n=n, K=kk, p_out=None, rho=0.99, seed=0)
    print("State-space lag orthogonality (g_k = A^k b):")
    for k, v in res_state.items():
        print(f"  {k:>18s}  max_offdiag={v['max_offdiag']:.4f}  "
              f"mean_offdiag={v['mean_offdiag']:.4f}  spec_dev={v['spec_dev']:.4f}")

    print("\nOutput-space lag orthogonality (h_k = C A^k b), with p_out=128:")
    res_out, _ = experiment(n=n, K=kk, p_out=128, rho=0.99, seed=0)
    for k, v in res_out.items():
        print(f"  {k:>18s}  max_offdiag={v['max_offdiag']:.4f}  "
              f"mean_offdiag={v['mean_offdiag']:.4f}  spec_dev={v['spec_dev']:.4f}")

# --- State-space lag orthogonality heatmaps: g_k = A^k b ---
    res_state, grams_state = experiment(n=n, K=kk, p_out=None, rho=0.99, seed=0)
    print("State-space lag orthogonality (g_k = A^k b):")
    for k, v in res_state.items():
        print(f"  {k:>18s}  max_offdiag={v['max_offdiag']:.4f}  "
              f"mean_offdiag={v['mean_offdiag']:.4f}  spec_dev={v['spec_dev']:.4f}")
    for name, G in grams_state.items():
        plot_gram_heatmap(G, title=f"State lag Gram: {name}")


    #demo_random_vectors_vs_qr()
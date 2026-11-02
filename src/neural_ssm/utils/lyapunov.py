"""Full state metrics for small-block LTI systems, outside the recurrence.

The Lyapunov equation splits into independent block-pair equations. Weighted
defect completion needs only Cholesky and triangular solves: neither a dense
energy-coordinate A nor the assembled contraction matrix is required.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def block_left_multiply(blocks: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply blockdiag(blocks) to a matrix, allowing a final scalar block."""
    rows, cols = matrix.shape
    padding = blocks.shape[0] * blocks.shape[1] - rows
    padded = F.pad(matrix, (0, 0, 0, padding)) if padding else matrix
    result = blocks @ padded.reshape(blocks.shape[0], blocks.shape[1], cols)
    return result.reshape(-1, cols)[:rows]


def _upper_block_pairs(matrix, count, size):
    """Pack the upper pairs, padding a final scalar with zero dummy entries."""
    padding = count * size - matrix.shape[0]
    if padding:
        matrix = F.pad(matrix, (0, padding, 0, padding))
    rows, cols = torch.triu_indices(count, count, device=matrix.device)
    pairs = matrix.reshape(count, size, count, size).permute(0, 2, 1, 3)
    return pairs[rows, cols].reshape(-1, size * size, 1)


def _symmetric_from_pairs(solution, count, size, n):
    """Reflect upper block solutions without doubling the diagonal blocks."""
    rows, cols = torch.triu_indices(count, count, device=solution.device)
    solution = solution.reshape(-1, size, size)
    upper = solution.new_zeros(count, count, size, size).index_put((rows, cols), solution)
    pairs = upper + upper.transpose(0, 1).mT
    diagonal = torch.arange(count, device=solution.device)
    positions = diagonal * (2 * count - diagonal + 1) // 2
    diagonal_values = solution[positions]
    pairs = pairs.index_put(
        (diagonal, diagonal), 0.5 * (diagonal_values + diagonal_values.mT)
    )
    return pairs.permute(0, 2, 1, 3).reshape(count * size, count * size)[:n, :n]


def _lyapunov_equations(blocks):
    count, size, _ = blocks.shape
    rows, cols = torch.triu_indices(count, count, device=blocks.device)
    # Row-major vec(A_i.T P_ij A_j) has coefficients A_i[c,a]*A_j[d,b].
    product = torch.einsum("kca,kdb->kabcd", blocks[rows], blocks[cols])
    return torch.eye(size * size, device=blocks.device, dtype=blocks.dtype) - product.reshape(
        -1, size * size, size * size
    )


class _BlockLyapunov(torch.autograd.Function):
    """Reuse the small LU factors for an exact implicit backward.

    The adjoint solves W-A W A.T=G, then dQ=W and dA=2 P A W.
    Computing only dA's diagonal blocks avoids differentiating every entry of
    the block-pair Kronecker coefficients and reduces saved training tensors.
    """

    @staticmethod
    def forward(ctx, blocks, q):
        count, size, _ = blocks.shape
        # Stable blocks make each system invertible. The unchecked factorization
        # avoids GPU synchronization; numerical failures remain visible in P.
        lu, pivots, _ = torch.linalg.lu_factor_ex(_lyapunov_equations(blocks), check_errors=False)
        solution = torch.linalg.lu_solve(lu, pivots, _upper_block_pairs(q, count, size))
        p = _symmetric_from_pairs(solution, count, size, q.shape[0])
        ctx.save_for_backward(blocks, p, lu, pivots)
        return p

    @staticmethod
    def backward(ctx, gradient):
        blocks, p, lu, pivots = ctx.saved_tensors
        count, size, _ = blocks.shape
        n = p.shape[0]
        g = 0.5 * (gradient + gradient.mT)
        if torch.is_grad_enabled():
            # Higher derivatives need a live graph through the adjoint solve.
            w = solve_block_lyapunov(blocks.mT, g)
        else:
            solution = torch.linalg.lu_solve(
                lu, pivots, _upper_block_pairs(g, count, size), adjoint=True
            )
            w = _symmetric_from_pairs(solution, count, size, n)
        aw = block_left_multiply(blocks, w)
        padding = count * size - n
        if padding:
            p, aw = F.pad(p, (0, padding, 0, padding)), F.pad(aw, (0, padding, 0, padding))
        p_pairs = p.reshape(count, size, count, size).permute(0, 2, 1, 3)
        aw_pairs = aw.reshape(count, size, count, size).permute(0, 2, 1, 3)
        grad_blocks = 2 * torch.einsum("ijac,jicd->iad", p_pairs, aw_pairs)
        return grad_blocks, w


def solve_block_lyapunov(blocks: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Solve P - A.T P A = Q for stable small blocks and symmetric Q.

    Only upper block pairs are solved, then reflected. Each solve has b*b
    unknowns (four for b=2), rather than n*n unknowns. A padded final scalar has
    zero dummy dynamics and is cropped out. The implicit backward reuses the
    LU factors; higher derivatives rebuild a differentiable adjoint solve.
    This helper is for small blocks, not a dense Kronecker Lyapunov solver.
    """
    return _BlockLyapunov.apply(blocks, 0.5 * (q + q.mT))


def weighted_defect_completion(blocks, r, p, x, y, residual):
    """Return unit-gain B,C,D with state storage P, where R.T R=Q.

    Set T=R^-T A.T P and H.T H=P+T.T T. Then B=H^-1 X,
    C=Y R, D=-Y T B+residual. These are defect-completion coordinates with
    orthogonal rotations absorbed into the free thin contractions X and Y.
    The positive sum defining H avoids subtracting nearly equal dense matrices.
    """
    at_p = block_left_multiply(blocks.mT, p)
    t = torch.linalg.solve_triangular(r.mT, at_p, upper=False)
    gram = p + t.mT @ t
    h = torch.linalg.cholesky(0.5 * (gram + gram.mT)).mT
    b = torch.linalg.solve_triangular(h, x, upper=True)
    return b, y @ r, -y @ (t @ b) + residual

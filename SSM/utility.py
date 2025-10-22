import torch
import torch.nn as nn
import torch.nn.functional as F


# Lower Triangular Matrix with Positive Diagonal (for Cholesky-like parameterization)

class LowerTriangularPositiveDiag(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        n_params = n * (n + 1) // 2
        self.params = nn.Parameter(torch.randn(n_params))

        self.register_buffer('tril_indices',
                             torch.tril_indices(n, n, offset=0))

        # Precompute diagonal indices in the flattened vector
        diag_idx = torch.arange(n, dtype=torch.long) + 1
        self.register_buffer('diag_indices',
                             (diag_idx * (diag_idx + 1)) // 2 - 1)

    def forward(self):
        params = self.params.clone()
        # Apply softplus to diagonal elements for positivity
        params[self.diag_indices] = F.softplus(params[self.diag_indices])

        L = torch.zeros(self.n, self.n, device=self.params.device)
        L[self.tril_indices[0], self.tril_indices[1]] = params
        return L

# Skew-Symmetric Matrix Parameterization

class SkewSymmetricEfficient(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        n_params = n * (n - 1) // 2
        self.params = nn.Parameter(torch.randn(n_params))

        self.register_buffer('triu_indices',
                             torch.triu_indices(n, n, offset=1))

    def forward(self):
        A = torch.zeros(self.n, self.n, device=self.params.device)
        # Fill upper triangular
        A[self.triu_indices[0], self.triu_indices[1]] = self.params
        # Fill lower triangular with negatives
        A[self.triu_indices[1], self.triu_indices[0]] = -self.params
        return A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as param
from deel import torchlip
from ..static_layers.generic_layers import LayerConfig



class L2BoundedLinearExact(nn.Module):
    """
    Linear map y = x @ W^T with ||W||_2 <= bound.

    Supports:
      - exact spectral norm via SVD
      - optional power iteration fallback
      - inputs of shape (..., d_in)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        bound: float = 1.0,
        exact_norm: bool = True,
        power_iters: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.bound = float(bound)
        self.exact_norm = bool(exact_norm)
        self.power_iters = max(1, int(power_iters))
        self.eps = float(eps)

        self.W_raw = nn.Parameter(torch.empty(self.d_out, self.d_in))
        nn.init.orthogonal_(self.W_raw)

        if not self.exact_norm:
            self.register_buffer("_u", F.normalize(torch.randn(self.d_out), dim=0))

    def _sigma_exact(self, W: torch.Tensor) -> torch.Tensor:
        # SVD is more version-compatible than matrix_norm(ord=2)
        try:
            return torch.linalg.svdvals(W)[0].clamp_min(self.eps)
        except Exception:
            # Older PyTorch fallback
            return torch.svd(W).S[0].clamp_min(self.eps)

    def _sigma_power_iter(self, W: torch.Tensor) -> torch.Tensor:
        u = self._u
        for _ in range(self.power_iters):
            v = F.normalize(W.T @ u, dim=0, eps=self.eps)
            u = F.normalize(W @ v, dim=0, eps=self.eps)
        sigma = torch.abs(u @ (W @ v)).clamp_min(self.eps)
        with torch.no_grad():
            self._u.copy_(u)
        return sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W_raw

        # Safer for some GPU/dtype combinations
        W_for_norm = W.float() if W.dtype in (torch.float16, torch.bfloat16) else W

        if self.exact_norm:
            sigma = self._sigma_exact(W_for_norm).to(dtype=W.dtype, device=W.device)
        else:
            sigma = self._sigma_power_iter(W_for_norm).to(dtype=W.dtype, device=W.device)

        scale = torch.clamp(sigma / self.bound, min=1.0)
        W_eff = W / scale

        return F.linear(x, W_eff, bias=None)




class TLIP(nn.Module):
    """ Standard MLP with Lipschitz-bounded static_layers """

    def __init__(self, config: LayerConfig):
        super().__init__()
        # Pre-compute hidden dimension for efficiency
        self.hidden_dim = config.d_hidden
        layers = nn.ModuleList()
        layers.append(torchlip.SpectralLinear(config.d_input, self.hidden_dim))
        layers.append(torchlip.GroupSort2(2))

        for i in range(config.n_layers):
            layers.append(torchlip.SpectralLinear(self.hidden_dim, self.hidden_dim))
            layers.append(torchlip.GroupSort2(2))
        layers.append(torchlip.SpectralLinear(self.hidden_dim, config.d_output))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


# Manchester lipschitz bounded MLPs


def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)


class FirstChannel(nn.Module):
    def __init__(self, cout, scale=1.0):
        super().__init__()
        self.cout = cout
        self.scale = scale

    def forward(self, x):
        xdim = len(x.shape)
        if xdim == 4:
            return self.scale * x[:, :self.cout, :, :]
        elif xdim == 2:
            return self.scale * x[:, :self.cout]
        elif xdim == 3:
            return self.scale * x[:, :, :]


class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features + out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T)  # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x


class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features + out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:])  # B*h
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5)  # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi)  # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T)  # sqrt(2) A^top \Psi z
        return x


class LMLP(nn.Module):
    """ Implements a Lipschitz.-bounded MLP with sandwich static_layers. The square root
    # of the Lipschitz bound is given by the scale parameter, by default, set to 1. """

    def __init__(self, config: LayerConfig):
        super().__init__()
        # Pre-compute hidden dimension for efficiency
        hidden_dim = config.d_hidden

        # Layer construction using list comprehension
        layers = nn.ModuleList()
        layers.append(FirstChannel(config.d_input, scale=config.lip))
        layers.append(SandwichFc(config.d_input, hidden_dim, bias=False, scale=config.lip))
        for i in range(config.n_layers):
            layers.append(SandwichFc(hidden_dim, hidden_dim, bias=False, scale=config.lip))
        layers.append(SandwichLin(hidden_dim, config.d_output, bias=False, scale=config.lip))

        # Only add dropout if needed
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as param


class L2BoundedGLU(nn.Module):
    """
    GLU block with an L2-bounded linear map:
      - value branch:  a  (unconstrained; spectral norm on W ensures ||a||_2 <= ||x||_2)
      - gate branch:   sigmoid(b)
      - per-channel trainable Lipschitz scaling via diag(lip_vec)

    The global Euclidean Lipschitz constant is <= max(lip_vec),
    which is exposed as the scalar property `lip` for certification.
    """
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.eps = 1e-6

        d_input = int(config.d_input)
        lip_init = float(config.lip)
        lip0 = max(lip_init, self.eps)

        # Per-channel positive trainable Lipschitz levels
        # We parametrize lip_vec = softplus(raw_lip) + eps
        raw_init = torch.log(torch.expm1(torch.full((d_input,), lip0)))
        self.raw_lip = nn.Parameter(raw_init.clone().detach())

        # Linear map to 2d, intended to satisfy ||W||_2 <= 1
        self.lin = param.spectral_norm(nn.Linear(d_input, 2 * d_input, bias=False))

    @property
    def lip_vec(self) -> torch.Tensor:
        """
        Per-channel positive scaling vector used in forward.
        Shape: (d_input,)
        """
        return F.softplus(self.raw_lip) + self.eps

    @property
    def lip(self) -> torch.Tensor:
        """
        Scalar global l2-Lipschitz bound of the whole block.
        Since forward applies diag(lip_vec), the exact Euclidean operator norm
        of that diagonal scaling is max(lip_vec).
        """
        return self.lip_vec.max()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        a, b = h.chunk(2, dim=-1)

        # Standard GLU gating: value branch left unconstrained (spectral norm on W
        # already ensures ||a||_2 <= ||x||_2), sigmoid gate in (0,1).
        y = a * torch.sigmoid(b)

        # Per-channel scaling; broadcast over batch/time dimensions
        return y * self.lip_vec


class L2BoundedGLUv2(nn.Module):
    """
    GLU-like FF with a rigorous global l2 bound.

    y = tanh(Ax) * sigmoid(Bx)
    with
        ||A||_2 <= a_bound
        ||B||_2 <= b_bound
        a_bound + 0.25 * b_bound <= 1

    Then the pre-scaled block is 1-Lipschitz, and after multiplying by
    diag(lip_vec), the global Lipschitz constant is <= max(lip_vec).
    """
    def __init__(self, config: LayerConfig, a_bound: float = 0.75, b_bound: float = 1.0):
        super().__init__()
        self.eps = 1e-6

        d_input = int(config.d_input)
        lip_init = float(config.lip)
        lip0 = max(lip_init, self.eps)

        if a_bound + 0.25 * b_bound > 1.0 + 1e-8:
            raise ValueError("Need a_bound + 0.25 * b_bound <= 1 for a 1-Lipschitz pre-block.")

        raw_init = torch.log(torch.expm1(torch.full((d_input,), lip0)))
        self.raw_lip = nn.Parameter(raw_init.clone().detach())

        self.A = L2BoundedLinearExact(d_input, d_input, bound=a_bound, exact_norm=True)
        self.B = L2BoundedLinearExact(d_input, d_input, bound=b_bound, exact_norm=True)

    @property
    def lip_vec(self) -> torch.Tensor:
        return F.softplus(self.raw_lip) + self.eps

    @property
    def lip(self) -> torch.Tensor:
        return self.lip_vec.max()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(self.A(x))
        b = torch.sigmoid(self.B(x))
        y = a * b
        return y * self.lip_vec
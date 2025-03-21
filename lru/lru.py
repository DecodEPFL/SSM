import math
import torch
import torch.nn as nn
from .scan_utils import associative_scan, binary_operator_diag
import torch.jit as jit



class LRU(nn.Module):
    """ Linear Recurrent Unit. The LRU is simulated using Parallel Scan (fast!) when
     "scan" is set to True (default) in the forward pass, otherwise recursively (slow)."""

    def __init__(
            self, in_features: int, out_features: int, state_features: int, rmin=0.9, rmax=1.0, max_phase=6.283
    ):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / math.sqrt(in_features)
        )
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(
                torch.sqrt(torch.ones_like(lambda_abs) - torch.square(lambda_abs))
            )
        )
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))  # N, U
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))  # H, N

        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)

        lambda_re = lambda_abs * torch.cos(lambda_phase)
        lambda_im = lambda_abs * torch.sin(lambda_phase)
        lambdas = torch.complex(lambda_re, lambda_im)
        #lambdas = lambda_abs*torch.exp(1j*lambda_phase)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B = gammas * self.B
        return lambdas, B, self.C, self.D

    def ss_real_matrices(self, to_numpy=True):

        lambdas, B, self.C, self.D = self.ss_params()

        lambdas_full = torch.zeros(2 * self.state_features, device=lambdas.device, dtype=lambdas.dtype)
        lambdas_full[::2] = lambdas
        lambdas_full[1::2] = lambdas.conj()

        # First convert to complex conjugate system....
        A_full = torch.diag(lambdas_full)
        B_full = torch.zeros((2 * self.state_features, self.in_features), device=lambdas.device, dtype=lambdas.dtype)
        B_full[::2] = B
        B_full[1::2] = B.conj()
        C_full = torch.zeros((self.out_features, 2 * self.state_features), device=lambdas.device, dtype=lambdas.dtype)
        C_full[:, ::2] = 0.5 * self.C  # we take the real part of the complex conjugate system as output...
        C_full[:, 1::2] = 0.5 * self.C.conj()
        D_full = self.D

        # Then apply transformation to real domain
        T_block = torch.tensor([[1, 1], [1j, -1j]], device=lambdas.device, dtype=lambdas.dtype)
        T_block_inv = torch.linalg.inv(T_block)
        T_full = torch.block_diag(*([T_block] * self.state_features))
        T_full_inv = torch.block_diag(*([T_block_inv] * self.state_features))

        A_real = (T_full @ A_full @ T_full_inv).real
        B_real = (T_full @ B_full).real
        C_real = (C_full @ T_full_inv).real
        D_real = D_full

        ss_real_params = [A_real, B_real, C_real, D_real]
        if to_numpy:
            ss_real_params = [ss_real_param.detach().numpy() for ss_real_param in ss_real_params]

        return (*ss_real_params,)

    def forward_loop(self, input, state=None):

        # Input size: (B, L, H)
        lambdas, B, C, D = self.ss_params()
        output = torch.empty(
            [i for i in input.shape[:-1]] + [self.out_features], device=self.B.device
        )

        states = []
        for u_step in input.split(1, dim=1):  # 1 is the time dimension

            u_step = u_step.squeeze(1)
            state = lambdas * state + u_step.to(B.dtype) @ B.T
            states.append(state)

        states = torch.stack(states, 1)
        output = (states @ C.mT).real + input @ D.T

        return output, states

    @torch.compiler.disable
    def forward_scan(self, input, state=None):

        # Only handles input of size (B, L, H) Batched parallel scan, borrows heavily from
        # https://colab.research.google.com/drive/1RgIv_3WAOW53CS0BnT7_782VKTYis9WG?usp=sharing which in turn borrows
        # from https://github.com/i404788/s5-pytorch
        lambdas, B, C, D = self.ss_params()

        # lambdas is shape (N,) but needs to be repeated to shape (L, N),
        # since input_sequence has shape (B, L, H).
        lambda_elements = lambdas.tile(input.shape[1], 1)
        # Calculate B@u for each step u of each input sequence in the batch.
        # Bu_elements will have shape (B, L, N)
        Bu_elements = input.to(B.dtype) @ B.T
        if state is not None:
            Bu_elements[:, 0, :] = Bu_elements[:, 0, :] + lambdas * state
            # Vmap the associative scan since Bu_elements is a batch of B sequences.
        # Recall that Lambda_elements has been repeated L times to (L, N),
        # while Bu_seq has shape (B, L, N)
        inner_state_fn = lambda Bu_seq: associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]
        # inner_states will be of shape (B, L, N)
        inner_states = torch.vmap(inner_state_fn)(Bu_elements)

        # y = (inner_states @ self.C.T).real + input_sequences * self.D
        y = (inner_states @ C.T).real + input @ D.T
        return y, inner_states

    def forward(self, input, gamma=None, state=None, mode="scan"):

        if state is None:
            state = torch.view_as_complex(
                torch.zeros((self.state_features, 2), device=input.device)
            )  # default initial state, size N

        match mode:
            case "scan":
                y, st = self.forward_scan(input, state)
            case "loop":
                y, st = self.forward_loop(input, state)
        return y, st


# WORK IN PROGRESS

class LRU_Robust(jit.ScriptModule):
    """ Implements a Linear Recurrent Unit (LRU) with trainable or prescribed l2 gain gamma.
    No parallel scan implementation available at the moment. """

    def __init__(self, state_features: int, trainable: bool):
        super().__init__()
        #self.trainable = trainable
        self.state_features = state_features
        self.register_buffer('state', torch.zeros(state_features))
        self.register_buffer('ID', torch.eye(state_features))

        self.alpha = nn.Parameter(torch.tensor(4.1))  # controls the initialization of the matrix A:
        # the larger the alpha at initialization, the closer the eigenvalues of A will be
        # to the boundary of the unitary circle at initialization. This helps the SSM to obtain long memory properties.

        self.gamma = nn.Parameter(30 * torch.tensor(1.1))  # l2 gain
        self.epsilon = nn.Parameter(torch.tensor(-99.9))  # Regularization

        self.Skew = nn.Parameter(0.01 * torch.randn(state_features, state_features))

        # Define each block of X as a parameter
        self.X11 = nn.Parameter(torch.eye(state_features))
        self.X22 = nn.Parameter(0.01 * torch.eye(state_features))
        self.X21 = nn.Parameter(torch.eye(state_features))

        self.C = nn.Parameter(torch.eye(state_features))
        self.D = nn.Parameter(torch.eye(state_features))

        # self.X11 = nn.Parameter(torch.randn(state_features, state_features))
        # self.X22 = nn.Parameter(torch.randn(state_features, state_features))
        # self.X21 = nn.Parameter(torch.randn(state_features, state_features))
        #
        # self.C = nn.Parameter(torch.randn(state_features, state_features))
        # self.D = nn.Parameter(torch.randn(state_features, state_features))

    @jit.script_method
    def set_param(self):  # Parameter update for l2 gain (free param)

        gamma = self.gamma
        # Auxiliary Parameters
        X11 = self.X11
        X22 = self.X22

        Sk = self.Skew - self.Skew.T
        Q = (self.ID - Sk) @ torch.linalg.inv(self.ID + Sk)
        Z = self.X21 @ self.X21.T + X22 @ X22.T + self.D.T @ self.D + torch.exp(self.epsilon) * self.ID
        beta = gamma ** 2 * torch.sigmoid(self.alpha) / torch.norm(Z, 2)
        H11 = X11 @ X11.T + self.C.T @ self.C + beta * torch.exp(self.epsilon) * self.ID
        H12 = torch.sqrt(beta) * (X11 @ self.X21.T + self.C.T @ self.D)
        V = Z * beta - gamma ** 2 * self.ID
        R = H12 @ torch.linalg.inv(V.T) @ H12.T
        CR = torch.linalg.cholesky(-R)
        CRH = torch.linalg.cholesky(-R + H11)

        # Parameters

        A = torch.linalg.inv(CRH).T @ Q @ CR.T
        B = A @ torch.linalg.inv(H12.T) @ V.T
        C = self.C
        D = torch.sqrt(beta) * self.D

        # # Assemble H*H.T in block form
        # HHt = torch.cat([
        #     torch.cat([HHt_11, HHt_12], dim=1),
        #     torch.cat([HHt_21, HHt_22n], dim=1)
        # ], dim=0)
        #P = -Atilde @ R @ Atilde.T
        #la= torch.abs(torch.linalg.eigvals(A))
        # lp = torch.linalg.eigvals(self.P)
        # row1 = torch.cat([-A.T@P@ A+P, -A.T@P@B], dim=1)
        # row2 = torch.cat([-(A.T@P@B).T, -B.T@P@B+(gamma**2*self.ID)], dim=1)
        # M = torch.cat([row1, row2], dim=0)
        # eigs = torch.linalg.eigvals(M)

        #eigs
        return A, B, C, D

    #state: Optional[torch.Tensor]
    @jit.script_method
    def forward(self, input, state=None, gamma=None, mode: str = "scan"):
        # Input size: (B, L, H)
        state = torch.zeros(self.state_features, device=self.C.device)
        A, B, C, D = self.set_param()
        if input.dim() == 1:
            input = input.unsqueeze(0).unsqueeze(0)
        # output = torch.empty(
        #     [i for i in input.shape[:-1]] + [self.state_features], device=self.C.device
        # )

        states = []
        for u_step in input.split(1, dim=1):  # 1 is the time dimension

            u_step = u_step.squeeze(1)
            state = state @ A.T + u_step @ B.T
            states.append(state)

        states = torch.stack(states, 1)
        output = states @ C.mT + input @ D.T
        return output, states

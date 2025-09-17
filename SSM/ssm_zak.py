import torch
import torch.nn as nn
import torch.jit as jit
from SSM.scan_utils import associative_scan, binary_operator_diag, compute_linear_recurrence_parallel


class LRU_Robust(jit.ScriptModule):
    """ Implements a Linear Recurrent Unit (LRU) with trainable or prescribed l2 gain gamma. """
    def __init__(self, state_features: int, input_features: int, output_features: int):
        super().__init__()
        self.state_features = state_features
        self.input_features = input_features
        self.output_features = output_features
        self.gamma = nn.Parameter(torch.tensor(22.2))
        # initialize the internal state (will be resized per-batch at first forward)
        self.state = torch.zeros(self.state_features)
        self.register_buffer('ID', torch.eye(state_features))
        self.alpha = nn.Parameter(torch.tensor(4.1))  # controls the initialization of the matrix A:
        self.epsilon = nn.Parameter(torch.tensor(-99.9))  # Regularization
        self.Skew = nn.Parameter(0.01 * torch.randn(state_features, state_features))
        # Learnable parameters
        self.X = nn.Parameter(torch.eye(state_features))
        self.Yb = nn.Parameter(torch.randn(2*state_features, input_features+output_features))
        self.Dt = nn.Parameter(torch.randn(output_features, input_features))
        self.C = nn.Parameter(torch.eye(state_features))
        self.D = nn.Parameter(torch.eye(state_features))

    @jit.script_method
    def set_param(self):
        nx=self.state_features
        nu=self.input_features
        ny=self.output_features
        # Parameter update for l2 gain (free param)
        gamma = self.gamma
        # Auxiliary Parameters
        M=torch.block_diag(torch.ones((nx,nu)), torch.ones((nx,ny)))
        X11 = self.X11
        X22 = self.X22
        Sk = self.Skew - self.Skew.T
        Q = (self.ID - Sk) @ torch.linalg.inv(self.ID + Sk)
        Z = self.X21 @ self.X21.T + X22 @ X22.T + self.D.T @ self.D + torch.exp(self.epsilon) * self.ID
        beta = gamma ** 2 * torch.sigmoid(self.alpha) / torch.linalg.matrix_norm(Z, 2)
        H11 = X11 @ X11.T + self.C.T @ self.C + beta * torch.exp(self.epsilon) * self.ID
        H12 = torch.sqrt(beta) * (X11 @ self.X21.T + self.C.T @ self.D)
        V = Z * beta - gamma ** 2 * self.ID
        R = H12 @ torch.linalg.inv(V.T) @ H12.T
        CR = torch.linalg.cholesky(-R)
        CRH = torch.linalg.cholesky(-R + H11)
        # LTI system matrices
        A = torch.linalg.inv(CRH).T @ Q @ CR.T
        B = A @ torch.linalg.inv(H12.T) @ V.T
        C = self.C
        D = torch.sqrt(beta) * self.D
        return A, B, C, D

    def forward(self, input: torch.Tensor, state=None, mode: str = "loop") -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward for LRU_Robust.
        Args:
            input: (B, L, H) inputs where H == state_features
            state: optional initial state of shape (B, N). If not provided, self.state is used.
            mode: "loop" (sequential, TorchScript-safe) or "scan" (use compute_linear_recurrence_parallel)
        Returns:
            output: (B, L, H)
            states: (B, L, H) (states after each input, i.e., x_{1..L})
        """
        # Normalize input shape
        if input.dim() == 1:
            input = input.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        elif input.dim() == 2:
            input = input.unsqueeze(0)  # (1, L, H)
        batch_size = input.shape[0]
        L = input.shape[1]
        N = self.state_features
        # Ensure stored state has a batch dimension and correct device/dtype
        # Initialize state if needed
        if self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.state_features, device=input.device)
        x0 = self.state
        A, B, C, D = self.set_param()
        if mode == "scan":
            # Use the parallel scan implementation
            # Adjust input to (L, B, N)
            u = input.permute(1, 0, 2)
            states = compute_linear_recurrence_parallel(A, B, u, x0)  # (L, B, N)
            states = states.permute(1, 0, 2)  # (B, L, N)
            # update stored state to last state (detach to avoid retaining history)
            last_state = states[:, -1, :].detach()
            self.state = last_state
            # outputs: y_t = C x_t + D u_t
            outputs = torch.matmul(states, C.transpose(-1, -2)) + torch.matmul(input, D.transpose(-1, -2))
            return outputs, states
        else:
            # fallback sequential loop (TorchScript-safe)
            # ensure x is (B, N)
            x = x0
            states_list = []
            for u_step in input.split(1, dim=1):  # iterate over time steps
                u = u_step.squeeze(1)  # (B, N)
                # x_{k+1} = A x_k + B u_k
                x = torch.matmul(x, A.transpose(-1, -2)) + torch.matmul(u, B.transpose(-1, -2))
                states_list.append(x)
            states = torch.stack(states_list, dim=1)  # (B, L, N)
            # update stored state (detach)
            self.state = states[:, -1, :].detach()
            outputs = torch.matmul(states, C.transpose(-1, -2)) + torch.matmul(input, D.transpose(-1, -2))
            return outputs, states

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """ Reset the internal state. By default resets to a single zero state vector.
        If you need per-batch reset, pass batch_size and device.
        """
        if device is None:
            device = torch.device('cpu')
        self.state = torch.zeros(batch_size, self.state_features, device=device)
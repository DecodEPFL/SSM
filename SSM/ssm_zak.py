import torch
import torch.nn as nn
import torch.jit as jit
from SSM.scan_utils import associative_scan, binary_operator_diag, compute_linear_recurrence_parallel


class lruz(jit.ScriptModule):
    """ Implements a Linear Recurrent Unit (LRU) with trainable or prescribed l2 gain gamma. """

    def __init__(self, input_features: int, output_features: int, state_features: int, rmin=0.9,
                 rmax=1.0, max_phase=6.283, gamma: float = None):
        super().__init__()
        self.A = torch.empty(2)
        self.C = torch.empty(2)
        self.B = torch.empty(2)
        self.state_features = state_features
        self.input_features = input_features
        self.output_features = output_features

        self._rmin_rmax_diff = rmax - rmin
        self._rmin_rmax_sum = rmax + rmin
        self._rmin_squared = rmin ** 2
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * self._rmin_rmax_sum * self._rmin_rmax_diff + self._rmin_squared))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        if gamma is not None:  # in this case the l2 gain of the system is fixed
            self.gamma = torch.tensor(gamma)
        else:  # in this case the l2 gain is learnable (default)
            self.gamma = nn.Parameter(torch.tensor(2.2))
        # initialize the internal state (will be resized per-batch at first forward)
        self.state = torch.tensor(0.0)
        self.register_buffer('ID', torch.eye(state_features))
        self.register_buffer('IDu', torch.eye(input_features))
        self.register_buffer('IDy', torch.eye(output_features))
        self.register_buffer('Inu', torch.ones((state_features, input_features)))
        self.register_buffer('Iny', torch.ones((state_features, output_features)))
        self.register_buffer('Znu', torch.zeros((state_features, input_features)))
        self.register_buffer('Zny', torch.zeros((state_features, output_features)))
        # Learnable parameters
        self.X2b = nn.Parameter(torch.randn(2 * state_features, input_features + output_features))
        self.D = nn.Parameter(torch.randn(output_features, input_features))

    def ss_real_matrices(self, to_numpy=True):
        A, B, C, D = self.set_param()
        lambdas = torch.diagonal(A)
        device, dtype = lambdas.device, lambdas.dtype
        state_features_2 = 2 * self.state_features

        lambdas_conjugate = torch.stack([lambdas, lambdas.conj()], dim=1).flatten()
        A_full = torch.diag(lambdas_conjugate)
        B_conjugate = torch.stack([B, B.conj()], dim=1).view(state_features_2, self.input_features)
        C_half = 0.5 * C
        C_conjugate = torch.stack([C_half, C_half.conj()], dim=2).view(self.output_features, state_features_2)

        # build a small 2x2 transform on the current device / dtype (do NOT cache to self)
        T_block = torch.tensor([[1, 1], [1j, -1j]], device=device, dtype=dtype)
        T_block_inv = torch.linalg.inv(T_block)

        T_full = torch.block_diag(*([T_block] * self.state_features))
        T_full_inv = torch.block_diag(*([T_block_inv] * self.state_features))

        A_real = (T_full @ A_full @ T_full_inv).real
        B_real = (T_full @ B_conjugate).real
        C_real = (C_conjugate @ T_full_inv).real
        D_real = D

        ss_real_params = [A_real, B_real, C_real, D_real]
        if to_numpy:
            ss_real_params = [param.detach().cpu().numpy() for param in ss_real_params]
        return tuple(ss_real_params)

    def set_param(self):
        nx = self.state_features
        nu = self.input_features
        ny = self.output_features
        epsilon = 0.01
        alpha = 1 - epsilon

        # Create A
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        A = torch.diag(lambda_abs * torch.exp(1j * lambda_phase))

        Q = torch.conj(A).T @ A + epsilon * self.ID

        X11 = torch.cat((Q, Q @ A), dim=1)
        X12 = torch.cat((torch.conj(A).T @ Q, Q), dim=1)
        X1 = torch.cat((X11, X12), dim=0)

        X4_offdiagonal = self.gamma * alpha * self.D.T / torch.linalg.matrix_norm(self.D, 2)

        X4_row1 = torch.cat(
            (self.gamma * self.IDu, X4_offdiagonal), dim=1)
        X4_row2 = torch.cat(
            (X4_offdiagonal.T,
             self.gamma * self.IDy),
            dim=1)
        X4 = torch.cat((X4_row1, X4_row2), dim=0)

        M1 = torch.cat((self.Inu, self.Zny), dim=1)
        M2 = torch.cat((self.Znu, self.Iny), dim=1)
        M = torch.cat((M1, M2), dim=0)

        X2t = self.X2b * M

        eta_1 = torch.linalg.matrix_norm(torch.linalg.inv(X1) @ X2t.to(torch.complex64), ord=2)
        eta_2 = torch.linalg.matrix_norm(X2t @ torch.linalg.inv(X4), ord=2)

        eta = torch.maximum(torch.maximum(eta_1, eta_2), torch.tensor(1.0))

        X2 = X2t/eta

        B = torch.linalg.inv(Q) @ X2[:nx, :nu].to(torch.complex64)
        C = torch.conj(X2[-nx:, -ny:]).T.to(torch.complex64)
        D = X4_offdiagonal.T

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        return A, B, C, D

    def forward_loop(self, input, state=None, set_param: bool = True):

        batch_size, seq_len, _ = input.shape

        if set_param:
            self.set_param()
        lambdas = torch.diagonal(self.A)

        # State computation using pre-converted input
        input_B_dtype = input.to(self.B.dtype)
        B_T = self.B.mT  # Cache transpose

        # Optimized loop with pre-allocated tensor for states
        inner_states = torch.empty(batch_size, seq_len, self.state_features,
                                   device=input.device, dtype=torch.complex64)

        # Vectorized state updates
        current_state = self.state
        for t, u_step in enumerate(input_B_dtype.unbind(dim=1)):
            inner_states[:, t] = current_state
            current_state = lambdas * current_state + u_step @ B_T

        self.state = current_state.detach()  # Update the internal state

        # Output computation using all inner states
        output = (inner_states @ self.C.mT).real + input @ self.D.T

        return output, inner_states

    @torch.compiler.disable
    def forward_scan(self, input, state=None, set_param: bool = True):
        """
        Computes the LRU output using a parallel scan.

        Args:
            input (torch.Tensor): (B, L, H) input sequence.
            state (torch.Tensor, optional): (B, N) initial state. If None, a zero state is used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - (B, L, H_out) output sequence.
                - (B, L, N) sequence of internal states.
        """

        batch_size, seq_len, _ = input.shape
        A, B, C, D = self.set_param()
        lambdas = torch.diagonal(A)

        # Pre-compute input transformation
        Bu_elements = input.to(B.dtype) @ B.mT

        # Incorporate the initial state into the first element of the sequence
        Bu_elements[:, 0, :] += lambdas * self.state

        # Define the scan function for vmap
        lambda_elements = lambdas.expand(seq_len, -1)

        def scan_fn(Bu_seq):
            return associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]

        # Apply the scan over the batch dimension
        scanned_states = torch.vmap(scan_fn)(Bu_elements)

        # Prepend the initial state to get the full state sequence
        inner_states = torch.cat([self.state.unsqueeze(1), scanned_states[:, :-1, :]], dim=1)

        # Update the internal state of the LRU module to the last state
        self.state = scanned_states[:, -1, :].detach()

        # Compute the final output
        output = (inner_states @ C.mT).real + input @ D.T
        return output, inner_states

    def forward(self, input, gamma=None, state=None, set_param: bool = True, mode="scan"):

        if input.dim() == 1:
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 2:
            input = input.unsqueeze(0)
        elif input.dim() > 3:
            raise ValueError(f"Invalid input dimensions {input.dim()}, expected 1, 2, or 3.")

        if state is not None:
            self.state = state
        else:
            self.state = torch.zeros(input.shape[0], self.state_features, device=input.device, dtype=torch.complex64)
        # forward pass
        if mode == "scan":
            return self.forward_scan(input, self.state, set_param)
        elif mode in ["loop", "loop_efficient"]:
            return self.forward_loop(input, self.state, set_param)
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = None

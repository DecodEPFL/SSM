# Pytorch port of associative scan

# @title PyTorch associative/parallel scan
# Taken from https://github.com/i404788/s5-pytorch/blob/74e2fdae00b915a62c914bf3615c0b8a4279eb84/s5/jax_compat.py#L50-L134
import torch
from jax.tree_util import tree_flatten, tree_unflatten
from typing import overload, Callable, Iterable, List, TypeVar, Any, Literal, Union, Sequence, Tuple, Optional
from functools import partial
import math

"""
Jax-Pytorch ported functions, mostly interfaces are kept the same but unsupported features are removed:
* Jax-Keyed RNGs are sampled from global RNG
* Canonical/Named shapes/dtypes/etc are now regular shapes,dtypes
"""

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@overload
def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> List[T]: ...


@overload
def safe_map(f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]) -> List[T]: ...


@overload
def safe_map(f: Callable[[T1, T2, T3], T], __arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3]) -> List[
    T]: ...


@overload
def safe_map(f: Callable[..., T], __arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any],
             __arg4: Iterable[Any], *args) -> List[T]: ...


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))


def slice_along_axis(start, end, stride=None, axis=0):
    return (slice(None),) * axis + (slice(start, end, stride),)


# Pytorch impl. of jax.lax.associative_scan
def associative_scan(operator, elems, axis=0, reverse=False):
    if not callable(operator):
        raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = tree_flatten(elems)

    nd = elems_flat[0].ndim
    if axis < 0:
        axis += nd
    if not (0 <= axis < nd):
        raise ValueError(f"axis={axis} out of bounds for ndim={nd}")

    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat):
        # Lower `fn` to operate on flattened sequences of elems.
        a = tree_unflatten(tree, a_flat)
        b = tree_unflatten(tree, b_flat)
        c = operator(a, b)
        c_flat, _ = tree_flatten(c)
        return c_flat

    assert axis >= 0 or axis < elems_flat[0].ndim, "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[axis]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs of elements.
        reduced_elems = combine(
            [elem[slice_along_axis(0, -1, stride=2, axis=axis)] for elem in elems],
            [elem[slice_along_axis(1, None, stride=2, axis=axis)] for elem in elems])

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = combine(
                [e[slice_along_axis(0, -1, axis=axis)] for e in odd_elems],
                [e[slice_along_axis(2, None, stride=2, axis=axis)] for e in elems])
        else:
            even_elems = combine(
                odd_elems,
                [e[slice_along_axis(2, None, stride=2, axis=axis)] for e in elems])

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([elem[slice_along_axis(0, 1, axis=axis)], result], dim=axis)
            if result.shape.numel() > 0 and elem.shape[axis] > 0 else
            result if result.shape.numel() > 0 else
            elem[slice_along_axis(0, 1, axis=axis)]  # Jax allows/ignores concat with 0-dim, Pytorch does not
            for (elem, result) in zip(elems, even_elems)]

        return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

    scans = _scan(elems_flat)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(tree, scans)


def _interleave(a, b, axis):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[axis] == b.shape[axis] + 1):
        pad = [0, 0] * b.ndim
        pad[(b.ndim - axis - 1) * 2 + 1] = 1  # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis + 1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis + 1)
    if b_trunc:
        # TODO: find torch alternative for slice_along axis for torch.jit.script to work
        interleaved = interleaved[slice_along_axis(0, b.shape[axis] + a.shape[axis] - 1, axis=axis)]
    return interleaved


# Taken from https://github.com/i404788/s5-pytorch/blob/74e2fdae00b915a62c914bf3615c0b8a4279eb84/s5/s5_model.py
@torch.jit.script
def binary_operator_diag(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    # return A_j * A_i, A_j * b_i + b_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)


# Parallel scan for non diagonal matrices

# -------------------------
# Parallel prefix (safe)
# -------------------------
def parallel_scan_affine(M: torch.Tensor, v: torch.Tensor):
    """
    Inclusive parallel prefix for affine pairs (M, v) in the recurrence

        x_{t+1} = M_t @ x_t + v_t,   t = 0..T-1

    Args:
        M: (T, batch, D, D)
        v: (T, batch, D)

    Returns:
        M_p: (T, batch, D, D)
            M_p[t] = M_t @ M_{t-1} @ ... @ M_0
        v_p: (T, batch, D)
            v_p[t] = sum_{k=0}^t ( M_t @ ... @ M_{k+1} @ v_k )
                    (with empty products taken as identity)
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
        # left  = M_p[offset:]     (length n-offset)
        # right = M_p[:n-offset]   (length n-offset)
        left = M_p[offset:].clone()  # (n-offset, batch, D, D)
        right = M_p[: n - offset].clone()  # (n-offset, batch, D, D)

        # new_M[i] for i >= offset equals left[i-offset] @ right[i-offset]
        new_M_tail = torch.matmul(left, right)  # (n-offset, batch, D, D)

        # For v: new_v[i] = left[i-offset] @ v_p[i-offset] + v_p[i]
        right_v = v_p[: n - offset].unsqueeze(-1).clone()  # (n-offset, batch, D, 1)
        transformed = torch.matmul(left, right_v).squeeze(-1)  # (n-offset, batch, D)
        new_v_tail = transformed + v_p[offset:].clone()  # (n-offset, batch, D)

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
    Parallel solution of the linear recurrence

        x_{t+1} = A_t @ x_t + B_t @ u_t,    t = 0..T-1
        x_0 given.

    Conventions (column-vector, time-major):
        A: (T, D_state, D_state) or (D_state, D_state)
           state transition
        B: (T, D_state, D_in) or (D_state, D_in)
           input matrix
        u: (T, batch, D_in)              inputs u_0 .. u_{T-1}
        x0: (batch, D_state)             initial state x_0

    Returns:
        states: (T+1, batch, D_state)
            states[0]     = x_0
            states[t + 1] = x_{t+1} for t = 0..T-1
    """
    # u: (T, batch, D_in)
    seq_len, batch_size, D_in = u.shape

    # infer state dim from A or x0
    if A.dim() == 2:
        D_state = A.shape[0]
        assert A.shape[1] == D_state, "A must be square (D_state, D_state)"
    else:
        # A: (T, D_state, D_state)
        assert A.shape[0] == seq_len, "time dimension of A must match u"
        D_state = A.shape[1]
        assert A.shape[2] == D_state, "A must be square along last two dims"

    # check / infer B shape
    if B.dim() == 2:
        # (D_state, D_in_B)
        assert B.shape[0] == D_state, "B first dim must match state dim"
        D_in_B = B.shape[1]
    else:
        # (T, D_state, D_in_B)
        assert B.shape[0] == seq_len, "time dimension of B must match u"
        assert B.shape[1] == D_state, "B second dim must match state dim"
        D_in_B = B.shape[2]

    assert D_in_B == D_in, f"Input dim mismatch: u has {D_in}, B has {D_in_B}"

    # ensure A,B have time dimension
    if A.dim() == 2:
        # (D_state, D_state) -> (T, D_state, D_state)
        A = A.unsqueeze(0).expand(seq_len, -1, -1).contiguous()
    if B.dim() == 2:
        # (D_state, D_in) -> (T, D_state, D_in)
        B = B.unsqueeze(0).expand(seq_len, -1, -1).contiguous()

    # shape for affine scan:
    # M_t = A_t  (T, D_state, D_state) -> (T, batch, D_state, D_state)
    M = A.unsqueeze(1).expand(-1, batch_size, -1, -1).contiguous()

    # B_t: (T, D_state, D_in) -> (T, batch, D_state, D_in)
    B_exp = B.unsqueeze(1).expand(-1, batch_size, -1, -1).contiguous()

    # v_t = B_t @ u_t
    # u: (T, batch, D_in) -> (T, batch, D_in, 1)
    v = torch.matmul(B_exp, u.unsqueeze(-1)).squeeze(-1)  # (T, batch, D_state)

    # compute prefix for x_{t+1} = M_t ... M_0 x_0 + v_p[t]
    M_p, v_p = parallel_scan_affine(M, v)  # M_p: (T, batch, D_state, D_state), v_p: (T, batch, D_state)

    # x_{t+1} = M_p[t] @ x0 + v_p[t]
    x_next = torch.matmul(M_p, x0.unsqueeze(-1)).squeeze(-1) + v_p  # (T, batch, D_state)

    # assemble full trajectory: [x_0, x_1, ..., x_T]
    states = torch.empty(
        seq_len + 1, batch_size, D_state,
        device=u.device, dtype=u.dtype,
    )
    states[0] = x0
    states[1:] = x_next

    return states



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


# Test


def prefix_scan(x, prefix_func, dim, pad_value=0):
    """
    Apply prefix_func in parallel over sequence, left to right, executing
    log2(seq length) iterations. Implemented by Franz A. Heinsen, 2024.

    Args:
        x: tensor of shape [*preceding_dims, seq_len, *operand_dims].
        prefix_func: broadcastable binary associative function.
        dim: dimension over which to compute the parallel scan.
        pad_value: for padding sequences to a power of two. Default: 0.

    Output:
        y: tensor of shape [*preceding_dims, seq_len, *operand_dims].

    Sample use:
    >>> n, d = (100, 1024)
    >>> x = torch.randn(n, d, d) / (d**0.5)       # n square matrices
    >>> y = prefix_scan(x, torch.matmul, dim=-3)  # cumulative matmul
    """
    x = x.movedim(dim, -1)  # for easier indexing
    other_dims, seq_len = (x.shape[:-1], x.size(-1))
    n_powers_of_2 = int(math.ceil(math.log2(seq_len)))
    n_pads = 2 ** n_powers_of_2 - seq_len
    x = torch.nn.functional.pad(x, (0, n_pads), value=pad_value)
    for n in (2 ** torch.arange(n_powers_of_2)).tolist():
        x = x.view(*other_dims, -1, n * 2)
        last_on_L = x[..., (n - 1):n]
        last_on_L = last_on_L.movedim((-2, -1), (dim - 1, dim))
        all_on_R = x[..., n:]
        all_on_R = all_on_R.movedim((-2, -1), (dim - 1, dim))
        updated_on_R = prefix_func(last_on_L, all_on_R)
        updated_on_R = updated_on_R.movedim((dim - 1, dim), (-2, -1))
        x = torch.cat([x[..., :n], updated_on_R], dim=-1)
    x = x.view(*other_dims, -1)
    x = x[..., :seq_len]
    y = x.movedim(-1, dim)  # put dims back in orig order
    return y


def reduce_scan(x, reduce_func, dim):
    """
    Apply reduce_func in parallel over sequence, left to right, executing
    log2(seq length) iterations. Implemented by Franz A. Heinsen, 2024.

    Args:
        x: tensor of shape [*preceding_dims, seq_len, *operand_dims].
        reduce_func: broadcastable binary associative function.
        dim: dimension over which to compute the parallel scan.

    Output:
        y: tensor of shape [*preceding_dims, *operand_dims].

    Sample use:
    >>> n, d = (100, 1024)
    >>> x = torch.randn(n, d, d) / (d**0.5)       # n square matrices
    >>> y = reduce_scan(x, torch.matmul, dim=-3)  # matmul of all matrices
    """
    x = x.movedim(dim, -1)  # for easier indexing
    other_dims, seq_len = (x.shape[:-1], x.size(-1))
    n_powers_of_2 = int(math.ceil(math.log2(seq_len)))
    for _ in range(n_powers_of_2):
        if x.size(-1) % 2 == 0:
            leftover = None
        else:
            leftover = x[..., -1:]
            x = x[..., :-1]
        x = x.view(*other_dims, -1, 2)
        operands_on_L = x[..., 0].movedim(-1, dim)
        operands_on_R = x[..., 1].movedim(-1, dim)
        x = reduce_func(operands_on_L, operands_on_R)
        x = x.movedim(dim, -1)
        if leftover is not None:
            x = torch.cat([x, leftover], dim=-1)
    y = x.squeeze(-1)
    return y


import torch.nn.functional as F


# pip install git+https://github.com/glassroom/torch_parallel_scan.git


def compute_linear_recurrence_parallel_scan(
        A: torch.Tensor,
        B: torch.Tensor,
        u: torch.Tensor,
        x0: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel solution of the LTI recurrence using torch_parallel_scan:

        x_{t+1} = A x_t + B u_t,   t = 0..T-1

    Shapes (matches your existing compute_linear_recurrence_parallel):
        A:   (D, D)          constant (not time-varying)
        B:   (D, D)          constant (not time-varying)
        u:   (T, B, D)       inputs u_t
        x0:  (B, D)          initial state x_0

    Returns:
        states: (T+1, B, D)
            states[0]     = x_0
            states[t + 1] = x_{t+1}
    """
    T, batch_size, D = u.shape
    device = u.device
    dtype = u.dtype

    A = A.to(device=device, dtype=dtype)
    B = B.to(device=device, dtype=dtype)
    x0 = x0.to(device=device, dtype=dtype)

    # Trivial case
    if T == 0:
        return x0.unsqueeze(0)  # (1, B, D)

    # We’ll work in row-vector form:
    #   x_{t+1} = x_t @ A_row + u_t @ B_row
    A_row = A.mT  # (D, D)
    B_row = B.mT  # (D, D)

    # Reorder u to (B, T, D) to be batch-major for prefix_scan setup
    u_bt = u.permute(1, 0, 2)  # (B, T, D)

    # 1) Compute affine term: b_t = u_t @ B_row  (row-vector convention)
    #    u_bt: (B, T, D), B_row: (D, D)  -> b: (B, T, D)
    b = u_bt @ B_row

    # 2) Constant A_row per (batch, time): shape (B, T, D, D)
    A_block = A_row.view(1, 1, D, D).expand(batch_size, T, D, D).contiguous()

    # 3) Build augmented matrices mod_W[b, t] ∈ R^{(D+1)x(D+1)} s.t.
    #       [x_t, 1] @ mod_W[b, t] = [x_{t+1}, 1]
    #
    #   Block form:
    #       mod_W = [[ A_row, 0 ],
    #                [ b_t,   1 ]]
    #
    W_pad = F.pad(A_block, (0, 1), value=0.0)  # (B, T, D,   D+1)
    b_pad = F.pad(b, (0, 1), value=1.0).unsqueeze(-2)  # (B, T, 1,   D+1)
    mod_W = torch.cat([W_pad, b_pad], dim=-2)  # (B, T, D+1, D+1)

    # 4) Parallel prefix over time dimension (dim=1 is T)
    #    cum_mod_W[b, t] = mod_W[b, 0] @ ... @ mod_W[b, t]
    cum_mod_W = prefix_scan(mod_W, torch.matmul, dim=1)  # (B, T, D+1, D+1)

    # 5) Augmented initial state: [x0, 1]  -> (B, D+1)
    mod_x0 = F.pad(x0, (0, 1), value=1.0)

    # 6) Apply cumulative affine maps in parallel:
    #       mod_x[b, t] = [x0[b], 1] @ cum_mod_W[b, t] = [x_{t+1}, 1]
    #    Use einsum for batched row-vector @ matrix:
    mod_x = torch.einsum('bd, btdk -> btk', mod_x0, cum_mod_W)  # (B, T, D+1)

    # 7) Drop homogeneous coordinate and build full trajectory [x_0, ..., x_T]
    x = mod_x[..., :-1]  # (B, T, D)

    states_btD = torch.empty(batch_size, T + 1, D,
                             device=device, dtype=dtype)
    states_btD[:, 0, :] = x0
    states_btD[:, 1:, :] = x

    # Return as (T+1, B, D) to match your existing call-site
    states_TBD = states_btD.permute(1, 0, 2)
    return states_TBD


# scan for blocks

@torch.jit.script
def binary_operator_block2x2(
    q_i: Tuple[torch.Tensor, torch.Tensor],
    q_j: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binary operator for parallel scan of linear recurrence with 2x2 block-diagonal A.

    q_i: (A_i, b_i)
        A_i: (..., 2, 2)
        b_i: (..., 2)

    q_j: (A_j, b_j)
        A_j: (..., 2, 2)
        b_j: (..., 2)

    Represents composition of affine maps:
        x -> A_i x + b_i
        x -> A_j x + b_j

    Composition:
        x -> A_j (A_i x + b_i) + b_j
           = (A_j A_i) x + (A_j b_i + b_j)
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    # A_out = A_j @ A_i  (batched 2x2 matmul)
    A_out = torch.matmul(A_j, A_i)  # (..., 2, 2)

    # b_out = A_j @ b_i + b_j  (batched matvec)
    b_i_vec = b_i.unsqueeze(-1)             # (..., 2, 1)
    Ab_i = torch.matmul(A_j, b_i_vec).squeeze(-1)  # (..., 2)
    b_out = Ab_i + b_j

    return A_out, b_out


def compute_linear_recurrence_parallel_block2x2(
    A: torch.Tensor,
    B: torch.Tensor,
    u: torch.Tensor,
    x0: torch.Tensor,
):
    """
    Parallel solution of the linear recurrence with 2x2 block-diagonal A:

        x_{t+1} = A @ x_t + B @ u_t,    t = 0..T-1
        x_0 given.

    Shapes (time-major, column-vector convention):
        A:  (D_state, D_state)    block-diagonal in 2x2 blocks
        B:  (D_state, D_in)
        u:  (T, batch, D_in)      inputs u_0 .. u_{T-1}
        x0: (batch, D_state)      initial state x_0

    Returns:
        states: (T+1, batch, D_state)
            states[0]     = x_0
            states[t + 1] = x_{t+1} for t = 0..T-1
    """
    # u: (T, B, D_in)
    T, Ba, D_in = u.shape

    # infer state dim and number of 2x2 blocks
    D_state = A.shape[0]
    assert A.shape[1] == D_state, "A must be square (D_state, D_state)"
    assert D_state % 2 == 0, "D_state must be even for 2x2 blocks"
    n_blocks = D_state // 2

    # sanity check B
    assert B.shape[0] == D_state, "B first dim must match state dim"
    D_in_B = B.shape[1]
    assert D_in_B == D_in, f"Input dim mismatch: u has {D_in}, B has {D_in_B}"

    device = u.device
    dtype = u.dtype

    # ------------------------------------------------------------------
    # 1) Extract 2x2 blocks of A: (n_blocks, 2, 2)
    # ------------------------------------------------------------------
    # Here we assume A is already block-diagonal in 2x2 blocks, so we can view it.
    # If A was constructed that way (as in our Block2x2DenseL2SSM), this holds.
    # 1) Extract 2x2 blocks of A: (n_blocks, 2, 2)
    D_state = A.shape[0]
    assert A.shape[1] == D_state
    assert D_state % 2 == 0, "D_state must be even"
    n_blocks = D_state // 2

    A_blocks = torch.stack(
        [A[2 * i:2 * i + 2, 2 * i:2 * i + 2] for i in range(n_blocks)],
        dim=0,
    )  # (n_blocks, 2, 2)  # (n_blocks, 2, 2)

    # ------------------------------------------------------------------
    # 2) Compute per-time input contribution v_t = B @ u_t
    # ------------------------------------------------------------------
    # u: (T, B, D_in), B: (D_state, D_in)
    # v: (T, B, D_state)
    v = torch.einsum("tbd,sd->tbs", u, B)  # or u @ B.T with reshapes

    # reshape v into 2-dim blocks: (T, B, n_blocks, 2)
    b_seq = v.view(T, Ba, n_blocks, 2)

    # ------------------------------------------------------------------
    # 3) Build A_seq for each time (broadcast A_blocks over T,B)
    # ------------------------------------------------------------------
    # A_blocks: (n_blocks, 2, 2) -> (T, B, n_blocks, 2, 2)
    A_seq = A_blocks.view(1, 1, n_blocks, 2, 2).expand(T, Ba, n_blocks, 2, 2).contiguous()

    # ------------------------------------------------------------------
    # 4) Parallel prefix-scan over time using associative_scan
    # ------------------------------------------------------------------
    # elems is a pytree: (A_seq, b_seq), axis=0 is time
    (A_prefix, b_prefix) = associative_scan(
        binary_operator_block2x2,
        (A_seq, b_seq),
        axis=0,
        reverse=False,
    )
    # A_prefix[t], b_prefix[t] represent the composed affine map from x_0-blocks to x_{t+1}-blocks:
    #   z_{t+1} = A_prefix[t] @ z_0 + b_prefix[t]
    # Shapes:
    #   A_prefix: (T, B, n_blocks, 2, 2)
    #   b_prefix: (T, B, n_blocks, 2)

    # ------------------------------------------------------------------
    # 5) Apply composed maps to initial state x0 (in block form)
    # ------------------------------------------------------------------
    # x0: (B, D_state) -> (B, n_blocks, 2)
    x0_blocks = x0.view(Ba, n_blocks, 2)

    # Broadcast x0 over time: (T, B, n_blocks, 2)
    x0_blocks_exp = x0_blocks.unsqueeze(0).expand(T, -1, -1, -1)

    # x_{t+1_blocks} = A_prefix[t] @ x0_blocks + b_prefix[t]
    # A_prefix: (T,B,n_blocks,2,2), x0_blocks_exp: (T,B,n_blocks,2,1)
    x_next_blocks = torch.matmul(
        A_prefix,
        x0_blocks_exp.unsqueeze(-1)
    ).squeeze(-1) + b_prefix  # (T, B, n_blocks, 2)

    # reshape back to flat state: (T, B, D_state)
    x_next = x_next_blocks.reshape(T, Ba, D_state)

    # ------------------------------------------------------------------
    # 6) Assemble full trajectory [x_0, x_1, ..., x_T]
    # ------------------------------------------------------------------
    states = torch.empty(
        T + 1, Ba, D_state,
        device=device,
        dtype=dtype,
    )
    states[0] = x0
    states[1:] = x_next

    return states

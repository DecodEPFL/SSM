# Pytorch port of associative scan

# @title PyTorch associative/parallel scan
# Taken from https://github.com/i404788/s5-pytorch/blob/74e2fdae00b915a62c914bf3615c0b8a4279eb84/s5/jax_compat.py#L50-L134
import torch
from jax.tree_util import tree_flatten, tree_unflatten
from typing import overload, Callable, Iterable, List, TypeVar, Any, Literal, Union, Sequence, Tuple, Optional
from functools import partial

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
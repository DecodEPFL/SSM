# python
# file: src/neural_ssm/ssm/cache_utils.py
"""Eval-only caching of expensive parameter-derived tensors (spectral-norm
normalizations, certified state-space matrices, ...).

Several certified cells recompute an exact spectral norm (an SVD via
``torch.linalg.matrix_norm(ord=2)`` / ``svdvals``) plus Cholesky/inverse/solve
*every forward* to enforce their l2-gain certificate. During training this must
be recomputed each step (the weights change and gradients must flow through the
normalization, so the exact guarantee is preserved). In eval the weights are
fixed, so the result is constant and the SVD is pure waste — it can be computed
once and reused.

:class:`EvalCacheMixin` factors out exactly the caching pattern already used by
``L2BoundedLinearExact`` and ``DeepSSM._capped_encoder_decoder`` so it can be
shared by the remaining cells. The cached value is the *same* tensor a fresh
computation would produce in eval (detached), so model outputs are unchanged.
"""
from __future__ import annotations

from typing import Any, Callable

import torch


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (tuple, list)):
        return type(value)(
            v.detach() if isinstance(v, torch.Tensor) else v for v in value
        )
    return value


class EvalCacheMixin:
    """Mixin providing eval-only caching of parameter-derived computations.

    Use ``self._eval_cached(key, compute)`` inside a method. ``compute`` is a
    zero-arg callable returning a tensor (or tuple of tensors).

      * Training mode, or any grad-enabled call: always recompute (and drop the
        cached entry) so autograd sees the live computation and the exact
        certificate is preserved bit-for-bit.
      * Eval mode with grad disabled: compute once, cache a detached copy, reuse.

    The whole cache is cleared on any ``train()`` / ``eval()`` toggle. The cache
    lives in ``self.__dict__`` (not as a registered buffer), so it never enters
    ``state_dict`` and does not interfere with ``nn.Module.__setattr__``.
    """

    def _eval_cached(self, key: str, compute: Callable[[], Any]) -> Any:
        # Training or any grad-enabled call: recompute so autograd sees the live
        # computation (never serve a detached / possibly-stale-graph tensor).
        if self.training or torch.is_grad_enabled():
            cache = self.__dict__.get("_eval_cache")
            if cache is not None:
                cache.pop(key, None)
            return compute()
        # Eval + no-grad: reuse a detached copy, guarded by the in-place _version
        # of the module's parameters so it auto-invalidates if a weight changes
        # (e.g. load_state_dict before inference), not only on a train()/eval() flip.
        version = self._param_version()
        cache = self.__dict__.setdefault("_eval_cache", {})
        hit = cache.get(key)
        if hit is not None and hit[0] == version:
            return hit[1]
        value = _detach_tree(compute())
        cache[key] = (version, value)
        return value

    def _param_version(self) -> tuple:
        return tuple(p._version for p in self.parameters(recurse=True))

    def train(self, mode: bool = True):
        self.__dict__.pop("_eval_cache", None)
        return super().train(mode)

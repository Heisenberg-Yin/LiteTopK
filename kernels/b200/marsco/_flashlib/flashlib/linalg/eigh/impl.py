"""eigh dispatcher.

By default ``eigh(A)`` is **exact** (cuSOLVER / MKL on the input dtype) --
no precision is lost beyond what the input already carries. ``tol`` opts
into approximation paths (Halko subspace iteration, QDWH spectral D&C).

Routing rule (formerly in ``route.py``):

  * ``backend`` overrides everything.
  * ``tol`` is None / ``<= 0``: exact.
      * Truncated (``K`` given) : exact full eigh + slice top-K.
      * Full                   : ``triton_eigh`` (CPU MKL trick for
                                  small N) or cuSOLVER otherwise.
  * ``tol`` is given: pick the fastest variant whose published residual
    fits.
      * ``K`` given AND favourable shape (``K*4 < N`` AND
        ``N >= 256``) AND ``tol >= 1e-4`` -> Halko.
      * ``N >= 5120`` AND ``tol >= 8e-4`` -> QDWH-NS.
      * ``N >= 5120`` AND ``tol >= 3e-3`` -> QDWH.
      * Otherwise: cuSOLVER (exact is always fast enough at this point).

The Halko shape gate is internal -- callers just hand in ``(K, tol)``;
they no longer reach into ``halko.py``.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib import _hw
from flashlib.linalg.eigh import cusolver, jacobi, halko
from flashlib.linalg.eigh.triton import triton_eigh


# Per-variant published residual (relative). Used by the tol router AND
# by cost.py.
_RESIDUAL_PREFERENCE = [
    ("cusolver", 1e-7),
    ("jacobi",   1e-6),
    ("halko",    1e-4),
    ("qdwh_ns",  8e-4),
    ("qdwh",     3e-3),
]

_QDWH_MIN_N = 5120


def _halko_is_favourable(N: int, K: Optional[int]) -> bool:
    """Halko helps when ``K << N`` AND N is large enough to amortize."""
    return K is not None and halko.should_use_halko(N, K)


def _route(
    *,
    N: int,
    K: Optional[int] = None,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
    hw: Optional[_hw.HwProps] = None,
) -> str:
    """Pick the eigh variant. Returns the bare name (no ``eigh_`` prefix).

    See module docstring for the rule. The same function is called by
    runtime dispatch (impl) and the cost shim (cost.py).

    ``jacobi`` is reachable only via explicit ``backend="jacobi"`` --
    it is a single-CTA Triton cyclic-Jacobi kernel that beats cuSOLVER
    at very small N (N <= 16) but is slower beyond that. No C++
    toolchain / ninja is required on first call (the previous
    ``load_inline`` CUDA implementation was retired in favour of a
    pure-Triton kernel under :mod:`flashlib.linalg.eigh.triton.jacobi`).
    """
    del hw
    if backend is not None:
        return backend
    if tol is None or tol <= 0:
        return "cusolver"
    if _halko_is_favourable(N, K) and tol >= 1e-4:
        return "halko"
    if N >= _QDWH_MIN_N:
        if tol >= 3e-3:
            return "qdwh"
        if tol >= 8e-4:
            return "qdwh_ns"
    return "cusolver"


def route_op_name(*, N: int, K: Optional[int] = None,
                  tol: Optional[float] = None,
                  hw: Optional[_hw.HwProps] = None) -> str:
    """Canonical ``eigh_<variant>`` label used by :mod:`flashlib.info`."""
    return "eigh_" + _route(N=N, K=K, tol=tol, hw=hw)


def eigh(
    A: torch.Tensor,
    K: Optional[int] = None,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
    n_iter: int = 5,
    p: int = 30,
    **kwargs,
):
    """Symmetric eigendecomposition.

    Args:
        A: ``(N, N)`` symmetric tensor.
        K: optional truncation -- return only top-K eigenpairs (ascending).
            With ``tol`` loose enough this triggers Halko.
        tol: residual tolerance.

            * ``None`` (default) **-> EXACT** in the input dtype (always
              cuSOLVER -- the Triton ``jacobi`` backend is opt-in via
              ``backend="jacobi"`` since cuSOLVER beats it past N ~ 16).
            * Otherwise pick the fastest variant whose declared residual
              fits, optionally Halko if ``K`` is set + shape favours it.
        backend: explicit ``"cusolver" | "jacobi" | "qdwh" | "qdwh_ns" |
            "halko"`` override.
        n_iter, p: Halko power-iter and oversample (only used when Halko
            is selected).
        **kwargs: forwarded to the chosen variant.

    Returns:
        ``(eigvals, eigvecs)`` ascending; both sliced to top-K if ``K``
        was supplied.
    """
    if not isinstance(A, torch.Tensor) or A.ndim != 2:
        raise ValueError("eigh expects a 2D tensor")
    N = A.size(0)
    chosen = _route(N=N, K=K, tol=tol, backend=backend)

    if chosen == "halko":
        if K is None:
            raise ValueError(
                "eigh(backend='halko') requires K -- Halko is a truncated path."
            )
        return halko.halko_eigh(A, K=K, n_iter=n_iter, p=p)

    if chosen == "jacobi":
        eigvals, eigvecs = jacobi.eigh(A, **kwargs)
    elif chosen == "qdwh":
        from flashlib.linalg.eigh.qdwh import qdwh_eig
        eigvals, eigvecs = qdwh_eig(A, **kwargs)
    elif chosen == "qdwh_ns":
        from flashlib.linalg.eigh.qdwh_ns import qdwh_eig_ns
        eigvals, eigvecs = qdwh_eig_ns(A, **kwargs)
    else:  # cusolver / default exact
        # Use the small-D CPU-MKL trick automatically (it IS the exact
        # path -- input dtype preserved, just dispatched to the faster
        # solver for small N).
        eigvals, eigvecs = triton_eigh(A)

    if K is not None:
        eigvals = eigvals[-K:]
        eigvecs = eigvecs[:, -K:]
    return eigvals, eigvecs


__all__ = ["eigh", "route_op_name"]

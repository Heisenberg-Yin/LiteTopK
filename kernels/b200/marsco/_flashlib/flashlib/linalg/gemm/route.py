"""GEMM precision/throughput routing rules.

GEMM routing is qualitatively different from the shape-routed primitives:
the dominant decision dimension is the user-supplied ``tol`` (residual
tolerance) rather than (M, N, K). This file is the single source of
truth for the tol-driven pick that ``flashlib.linalg.gemm.gemm()``
performs, and the same pick is consumed by the info API.

Variants and their published RMS-rel residuals are kept in
``flashlib/linalg/gemm/__init__.py``'s ``_RESIDUAL_PREFERENCE``; this
module imports those values at call time to avoid duplication.
"""
from __future__ import annotations

from typing import Optional

from flashlib import _hw


def route(
    *,
    M: int,
    N: int,
    K: int,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
    hw: Optional[_hw.HwProps] = None,
) -> str:
    """Pick the GEMM variant. Returns the bare variant name.

    Cache-aware fields used: none today. Variant ordering is dominated
    by precision tier, then by measured H200 throughput. To re-tune the
    throughput tie-breaker on a different GPU::

        python -m benchmarks.tune.gemm
        python -m benchmarks.tune.derive.gemm

    then update the ``_THROUGHPUT_TF`` dict in
    [flashlib/linalg/gemm/__init__.py](__init__.py).
    """
    del M, N, K, hw  # currently unused; reserved for shape-aware tweaks
    if backend is not None:
        return backend
    # Defer to the per-tol picker living next to the variant table.
    from flashlib.linalg.gemm import _pick_by_tol
    return _pick_by_tol(tol)

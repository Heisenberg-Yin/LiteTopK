"""CuteDSL backend stub for flash linear regression.

The bespoke Hopper WGMMA kernel that previously lived in
``hopper_bf16_gemm.py`` was removed during the 2026-05 ``flashlib``
layout refactor: GEMM dispatch is now centralised in
:mod:`flashlib.linalg.gemm`, which means a primitive should not carry
its own GEMM implementation. The CuteDSL entry points here remain so
existing callers (tests, benches, the ``flash_linear_regression_cutedsl``
mapping in :mod:`flashlib`) keep working; they now route through
:func:`flashlib.linalg.gemm.gemm` for the dominant ``X.T @ X`` (and
through the Triton fused kernels for the cast / refinement steps), so
the practical behaviour matches the Triton backend until a CuteDSL
``gemm_bf16`` variant is added to :mod:`flashlib.linalg.gemm`.
"""
from __future__ import annotations

import torch

from flashlib.linalg.gemm import gemm as _flash_gemm
from flashlib.primitives.linear_regression.triton.fused_kernels import (
    cast_and_xty,
    fused_refine_from_bf16,
)


def cutedsl_xtx(X_bf: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute X.T @ X via :func:`flashlib.linalg.gemm.gemm` (bf16 path).

    The legacy CuTeDSL Hopper WGMMA kernel was removed; this is now a
    thin wrapper that routes the GEMM through the central dispatcher.
    Kept under the ``cutedsl`` namespace so existing ``cutedsl_xtx``
    callers continue to work.
    """
    assert X_bf.is_cuda and X_bf.is_contiguous() and X_bf.dtype == torch.bfloat16
    Xt = X_bf.transpose(0, 1).contiguous()
    XtX = _flash_gemm(Xt, X_bf, tol=1e-3)
    if out is not None:
        out.copy_(XtX)
        return out
    return XtX


def cutedsl_linear_regression(X: torch.Tensor, y: torch.Tensor,
                               n_refine: int = 1, *, tol: float = 1e-3):
    """End-to-end linear regression -- ``flashlib.gemm`` for the dominant
    GEMM plus the Triton fused cast / Xty / refine kernels.

    Drop-in replacement for :func:`triton_linear_regression`; same
    correctness profile and (currently) the same performance, since both
    route through :mod:`flashlib.linalg.gemm`.
    """
    assert X.is_cuda and X.ndim == 2 and y.ndim == 1
    X = X.contiguous()
    y = y.contiguous()
    N, D = X.shape

    X_bf, Xty = cast_and_xty(X, y)
    XtX = cutedsl_xtx(X_bf)

    eps = 1e-3 * XtX.diagonal().mean()
    XtX_reg = XtX + eps * torch.eye(D, device=X.device, dtype=torch.float32)
    try:
        L = torch.linalg.cholesky(XtX_reg)
    except Exception:
        return torch.linalg.solve(XtX_reg, Xty)

    w = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)

    for _ in range(n_refine):
        Xtr = fused_refine_from_bf16(X_bf, y, w)
        delta = torch.cholesky_solve(Xtr.unsqueeze(1), L).squeeze(1)
        w = w + delta

    return w


_CUTEDSL_AVAILABLE = True


def cutedsl_available() -> bool:
    """Backend is always available now (cuBLAS-backed via flashlib.gemm)."""
    return True

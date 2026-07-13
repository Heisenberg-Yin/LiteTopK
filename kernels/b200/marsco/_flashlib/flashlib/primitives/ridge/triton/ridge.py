"""Ridge Regression via H200-optimized normal equations.

Solves L2-regularised least squares
  ``w* = argmin_w ||X w - y||^2 + alpha ||w||^2``
via the closed-form normal equations ``(X.T X + alpha I) w = X.T y``.

Pipeline
--------
1. ``X.T X`` and ``X.T y`` via :func:`flashlib.linalg.gemm.gemm` —
   the precision is selected by ``tol`` (default ``tol=1e-3`` -> bf16
   tensor cores, ~720 TFLOPS effective on H200; ``tol=None`` -> fp32).
2. Tikhonov: ``XtX.diagonal().add_(alpha)``.
3. Cholesky factorisation (alpha makes the matrix SPD even with bf16
   round-off; LU fallback if it ever fails).
4. ``n_refine`` alpha-aware iterative-refinement steps in fp32::

      r   = y - X @ w
      Xtr = X.T @ r - alpha * w     (alpha-aware augmentation)
      w   = w + chol_solve(Xtr)

This module no longer carries an explicit ``exact`` flag or a
private bf16 cast helper — both used to live here as ``if exact: ...``
branches and a manual ``X.to(torch.bfloat16)`` cache. The single
``tol`` knob plus :mod:`flashlib.linalg.gemm`'s tol-routed dispatch
covers the same surface area:

  * ``tol=None`` -> ``gemm_fp32`` (the old ``exact=True`` path)
  * ``tol=1e-3`` -> ``gemm_bf16`` (the old default)
  * ``tol=1e-5`` -> ``gemm_3xbf16`` (CuTeDSL fused, +6 bits, 228 TF)
"""
import torch

from flashlib.linalg.gemm import gemm as _flash_gemm


def triton_ridge_regression(X: torch.Tensor, y: torch.Tensor,
                            alpha: float = 1.0, *,
                            tol: "float | None" = None,
                            n_refine: int = 1):
    """Solve ``(X.T X + alpha I) w = X.T y``.

    Args:
        X: (N, D) float32 CUDA tensor.
        y: (N,) float32 CUDA tensor.
        alpha: regularisation strength.
        tol: residual tolerance for the dominant ``X.T X`` and ``X.T y``
            GEMMs forwarded to :func:`flashlib.linalg.gemm.gemm`.
            ``None`` (default) keeps the input dtype intact (exact);
            pass ``tol=1e-3`` to opt into bf16 GEMMs with iterative
            refinement.
        n_refine: number of alpha-aware iterative-refinement steps.
            Refinement always runs in fp32 (the precision-critical part);
            one step is enough on every measured shape.

    Returns:
        w: (D,) fp32 weight vector.
    """
    assert X.is_cuda and X.ndim == 2 and y.ndim == 1
    N, D = X.shape

    # tol=None -> EXACT: refinement runs in IEEE fp32. tol>0 -> opt into
    # TF32 for the direct ``X @ w`` / ``X.T @ r`` ops (refine is the hot
    # loop). Always restored on exit.
    use_tf32 = tol is not None and tol > 0
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        # Dominant GEMMs - precision picked by tol via flashlib.gemm.
        Xt = X.transpose(0, 1).contiguous()
        XtX = _flash_gemm(Xt, X, tol=tol)
        XtX.diagonal().add_(alpha)
        try:
            L = torch.linalg.cholesky(XtX)
        except Exception:
            # Cholesky failure (rare; alpha and bf16 noise both leave
            # the matrix non-PD). Fall back to LU on the same XtX --
            # no refinement, but still alpha-correct.
            Xty_fallback = X.T @ y
            return torch.linalg.solve(XtX, Xty_fallback)

        Xty = _flash_gemm(Xt, y.unsqueeze(1), tol=tol).squeeze(1)
        w = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)

        for _ in range(n_refine):
            r = y - X @ w
            Xtr = X.T @ r
            Xtr -= alpha * w
            delta = torch.cholesky_solve(Xtr.unsqueeze(1), L).squeeze(1)
            w = w + delta

        return w
    finally:
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


flash_ridge_regression = triton_ridge_regression

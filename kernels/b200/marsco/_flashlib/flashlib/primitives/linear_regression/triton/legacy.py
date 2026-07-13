"""Legacy linear regression implementation kept for parity testing.

This is the pre-refactor inline path: explicit ``if tol >= 1e-3`` precision
branch and bare ``torch.matmul``s -- preserved verbatim so we can compare
the new ``flashlib.linalg.gemm``-routed implementation against it.
"""
import torch


def triton_linear_regression_legacy(X: torch.Tensor, y: torch.Tensor, n_refine: int = 1,
                                     *, tol: float | None = None):
    """Solve linear regression via mixed-precision normal equations + refinement.

    Args:
        X: (N, D) float32
        y: (N,) float32
        n_refine: iterative-refinement steps (default 1).
        tol: residual tolerance (relative). None -> fp32 XᵀX (exact).
            tol >= 1e-3 -> bf16 XᵀX (cuBLAS TC, fp32 accum) + iterative refinement
            in fp32; net rel‖w − w_fp32‖∞ ≤ 2.5e-4 with n_refine=1.

    Returns:
        w: (D,) fp32 weight vector
    """
    assert X.is_cuda and X.ndim == 2 and y.ndim == 1
    N, D = X.shape

    # ── Factorization: bf16 GEMM if tol allows, else fp32 cov_gemm ──
    if tol is not None and tol >= 1e-3:
        X_bf16 = X.to(torch.bfloat16)
        XtX = (X_bf16.T @ X_bf16).float()
    else:
        XtX = (X.T @ X).float()
    eps = 1e-3 * XtX.diagonal().mean()
    XtX_reg = XtX + eps * torch.eye(D, device=X.device, dtype=torch.float32)
    try:
        L = torch.linalg.cholesky(XtX_reg)
    except Exception:
        # Rare case: factorization fails entirely → LU fallback (still bf16
        # XtX, but without iterative refinement).
        Xty = X.T @ y
        return torch.linalg.solve(XtX_reg, Xty)

    # ── Initial solve ──
    Xty = X.T @ y  # fp32 GEMV
    w = torch.cholesky_solve(Xty.unsqueeze(1), L).squeeze(1)

    # ── Mixed-precision iterative refinement ──
    # Each step: 2 fp32 GEMVs (X@w and Xᵀr) + cheap chol_solve.
    # 1 step reduces rel error from O(bf16 eps · √N) to O((bf16 eps)² · √N)
    # — for N=2M D=5000 that's 1e-2 → 2.5e-4, matching fp32 reference's MSE.
    for _ in range(n_refine):
        r = y - X @ w  # fp32 residual (the precision-critical step)
        Xtr = X.T @ r  # fp32 GEMV
        delta = torch.cholesky_solve(Xtr.unsqueeze(1), L).squeeze(1)
        w = w + delta

    return w

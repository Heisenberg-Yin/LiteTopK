"""Pure Newton-Schulz polar / matrix-sign for symmetric A, with optional
QDWH-Cholesky warm start.

Polynomial: Polar Express minimax-optimal odd quintic
    p_t(x) = a_t x + b_t x³ + c_t x⁵
applied via X_new = X @ (a_t I + b_t M + c_t M²),  M = X^T X.

Reference: Amsel, Pouya, Freund, Nakatsukasa, Tropp.
"The Polar Express: Optimal Matrix Sign Methods and Their Application
to the Muon Algorithm" (arXiv:2505.16932, 2025).

Coefficient tables come from `polar_express_coeffs.optimal_composition`.

Public:
  polar_polar_express(A, ...)        — pure NS, no Cholesky.
  polar_polar_express_warm(A, ...)   — n_qdwh QDWH-Cholesky warm start
                                       + Polar Express tail.
"""
import torch

from flashlib.linalg.gemm.triton.triton_mm import mm_3xbf16
from flashlib.linalg.polar.qdwh_hybrid import _spectral_norm_estimate, _qdwh_chol_step
from flashlib.linalg.polar.express_coeffs import (
    POLAR_COEFFS_1E6, POLAR_TAIL_1QDWH, POLAR_TAIL_2QDWH,
)


def polar_polar_express(A, alpha=None, tol=1e-4, alpha_scale=1.01,
                        coeffs=POLAR_COEFFS_1E6, matmul=mm_3xbf16):
    """Polar factor sign(A) of symmetric A via Polar Express quintic iteration.

    Returns X_k ≈ sign(A) with ‖X_k^T X_k - I‖_F / sqrt(n) < tol or
    after len(coeffs) iters.

    Each iteration:
        M  = X^T X          (1 matmul)
        M2 = M @ M          (1 matmul)
        poly = a*I + b*M + c*M²   (scalar FMAs, no matmul)
        X  = X @ poly       (1 matmul)
    """
    n = A.size(0)
    device, dtype = A.device, A.dtype
    if alpha is None:
        alpha = _spectral_norm_estimate(A)
    if alpha == 0.0:
        return A.clone()
    X = A / (alpha * alpha_scale)
    I_n = torch.eye(n, device=device, dtype=dtype)
    inv_sqrt_n = 1.0 / (n ** 0.5)

    for t, (a, b, c) in enumerate(coeffs):
        M = matmul(X.T.contiguous(), X)
        M = 0.5 * (M + M.T)

        err = torch.linalg.norm(M - I_n).item() * inv_sqrt_n
        if err < tol:
            break

        M2 = matmul(M, M)
        poly = a * I_n + b * M + c * M2
        X = matmul(X, poly)

    return X


def polar_polar_express_warm(A, n_qdwh=1, L0=1e-5, alpha=None,
                             matmul=mm_3xbf16, syrk_fp64_thresh=16384):
    """Polar factor via `n_qdwh` QDWH-Cholesky iters + Polar Express tail.

    Brings Cholesky back into the critical path. The iter-1 Cholesky
    has cond(Z) ≈ 2.3e6 — same precision cliff that motivated splitting
    qdwh-NS off in the first place.

    Measured cost breakdown at N=8192:
      pure PE (12×3=36 matmul)              ~256 ms
      1 QDWH + PE tail (4×3=12 matmul)      ~160 ms  ← n_qdwh=1
      2 QDWH + PE tail (1×3=3 matmul)       ~125 ms  ← n_qdwh=2
      polar_qdwh_hybrid (2 QDWH + 2 KL)     ~148 ms

    n_qdwh=2 beats the polar_qdwh_hybrid path by ~15% in raw speed
    but carries a higher-residual / occasional-FAIL profile.
    """
    n = A.size(0)
    device, dtype = A.device, A.dtype
    if alpha is None:
        alpha = _spectral_norm_estimate(A)
    if alpha == 0.0:
        return A.clone()

    X = A / alpha
    I_n = torch.eye(n, device=device, dtype=dtype)

    L = L0
    syrk_fp64_iter1 = n >= syrk_fp64_thresh
    for i in range(n_qdwh):
        if i == 0:
            X, L, _ = _qdwh_chol_step(X, L, I_n, syrk_mm=None,
                                      invert_solve=False,
                                      syrk_fp64=syrk_fp64_iter1)
        else:
            from flashlib.linalg.gemm.triton.triton_mm import mm_tf32_lt as _mm_tf32_iter
            X, L, _ = _qdwh_chol_step(X, L, I_n, syrk_mm=_mm_tf32_iter,
                                      invert_solve=True)

    # Rescale post-QDWH so σ_max ≤ 1 before the tail starts.
    post_alpha = _spectral_norm_estimate(X)
    X = X / post_alpha

    if n_qdwh == 1:
        tail = POLAR_TAIL_1QDWH
    elif n_qdwh == 2:
        tail = POLAR_TAIL_2QDWH
    else:
        tail = POLAR_COEFFS_1E6

    for a, b, c in tail:
        M = matmul(X.T.contiguous(), X)
        M = 0.5 * (M + M.T)
        M2 = matmul(M, M)
        poly = a * I_n + b * M + c * M2
        X = matmul(X, poly)

    return X


# Back-compat aliases used by qdwh_ns.py.
_polar_ns = polar_polar_express
_polar_warmstart_pe = polar_polar_express_warm

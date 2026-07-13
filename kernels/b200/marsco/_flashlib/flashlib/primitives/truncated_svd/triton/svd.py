"""Truncated SVD via cuBLAS GEMMs + ``flashlib.linalg.eigh``.

Auto-dispatches between two paths:

1. N >= D (cov path):
     gram = X.T @ X (cuBLAS TF32 GEMM)
     eigh(gram, K=K, tol=tol)
     S = sqrt(lambda); Vh = eigvecs.T (descending)

2. D >  N (dual path):
     G = X @ X.T  (cuBLAS TF32 GEMM)
     eigh(G, K=K, tol=tol)
     V = X.T @ U; column-normalise; S = sqrt(lambda)

Mathematically identical to PCA (without centering):
  SVD singular values = sqrt(eigenvalues of X.T @ X)
  SVD right singular vectors = eigenvectors of X.T @ X

Precision is owned by ``flashlib.linalg.eigh.eigh`` -- pass ``tol=None``
(default) for an exact path (cuSOLVER / MKL), pass ``tol >= 1e-4`` to opt
into Halko subspace iteration when shape favours it.
"""
import torch

from flashlib.linalg.eigh import eigh


def _triton_svd_cov(X: torch.Tensor, K: int, *, tol=None):
    """Truncated SVD via cuBLAS TF32 cov GEMM + eigh on D x D."""
    N, D = X.shape
    gram = X.T @ X                                # cuBLAS TF32 GEMM
    top_eigvals, top_eigvecs = eigh(gram, K=K, tol=tol)
    S = torch.sqrt(top_eigvals.clamp(min=0)).flip(0)
    Vh = top_eigvecs.T.flip(0)                    # (K, D), descending
    return S, Vh


def _triton_svd_dual(X: torch.Tensor, K: int, *, tol=None):
    """Truncated SVD via cuBLAS gram GEMM + eigh on N x N + projection."""
    N, D = X.shape
    G = X @ X.T                                   # cuBLAS TF32 GEMM
    top_eigvals, U = eigh(G, K=K, tol=tol)
    V = X.T @ U                                   # cuBLAS TF32 GEMM
    V = V / V.norm(dim=0, keepdim=True).clamp(min=1e-10)
    S = torch.sqrt(top_eigvals.clamp(min=0)).flip(0)
    Vh = V.T.flip(0)                              # (K, D), descending
    return S, Vh


def triton_truncated_svd(X: torch.Tensor, K: int, *, tol=None):
    """Truncated SVD: picks the path whose eigh dimension is smaller.

    All ``torch.matmul`` calls run with TF32 enabled on Hopper/Ampere
    (the same internal precision the prior in-house Triton kernels used);
    the global flag is restored on exit.

    Args:
        X: ``(N, D)`` input on CUDA.
        K: number of singular components.
        tol: residual tolerance. ``None`` (default) -> exact eigh on the
            cov / Gram matrix. Otherwise: Halko if ``tol >= 1e-4`` AND
            ``K*4 < M`` AND ``M >= 256`` (``M = D`` for cov, ``N`` for
            dual); QDWH variants for very large ``N``.

    Returns:
        S: ``(K,)`` top-K singular values, descending.
        Vh: ``(K, D)`` top-K right singular vectors, rows.
    """
    # tol=None -> EXACT: keep PyTorch's default IEEE matmul (no TF32).
    # tol>0    -> caller opted into a lossy budget: TF32 enabled for
    # auxiliary ``@`` ops; restored on exit.
    use_tf32 = tol is not None and tol > 0
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        N, D = X.shape
        if D <= N:
            return _triton_svd_cov(X, K, tol=tol)
        return _triton_svd_dual(X, K, tol=tol)
    finally:
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


flash_truncated_svd = triton_truncated_svd

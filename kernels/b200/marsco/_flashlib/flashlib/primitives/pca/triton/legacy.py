"""Legacy PCA path -- in-house Triton GEMMs + (cuSOLVER/MKL) eigh.

Kept for parity testing against the cuBLAS-TF32 + Halko path that the
``triton/pca.py`` entry point now uses.
"""

import torch
from flashlib.linalg.cov_gemm import cov_gemm as triton_cov_gemm
from flashlib.linalg.gram_gemm import gram_gemm as triton_gram_gemm
from flashlib.linalg.ab_gemm import ab_gemm as triton_ab_gemm
from flashlib.linalg.eigh import eigh as triton_eigh


def _triton_pca_cov_legacy(X: torch.Tensor, K: int, *, tol: float | None = None):
    """PCA via cov_gemm + eigh; tol cascades to eigh's variant pick."""
    N, D = X.shape
    cov = triton_cov_gemm(X)
    cov /= N
    eigenvalues, eigenvectors = triton_eigh(cov, tol=tol)
    return eigenvalues[-K:], eigenvectors[:, -K:]


def _triton_pca_dual_legacy(X: torch.Tensor, K: int, *, tol: float | None = None):
    """PCA via gram_gemm + eigh + ab_gemm; tol cascades to eigh."""
    N, D = X.shape
    G = triton_gram_gemm(X)
    G /= N
    eigvals, eigvecs = triton_eigh(G, tol=tol)
    K_actual = min(K, eigvals.shape[0])
    U = eigvecs[:, -K_actual:]
    top_eigvals = eigvals[-K_actual:]
    V = triton_ab_gemm(X, U)
    col_norms = V.norm(dim=0, keepdim=True).clamp(min=1e-10)
    V = V / col_norms
    return top_eigvals, V


def triton_pca_legacy(X: torch.Tensor, K: int, *, tol: float | None = None):
    """PCA via Triton kernels — auto-selects N>>D vs D>>N path; eigh routed by tol.

    Args:
        X: (N, D) float32 data (centered).
        K: number of components.
        tol: residual tolerance (relative). Cascaded to flashlib.linalg.eigh.
            None (default) -> cuSOLVER syevd / Jacobi (~1e-7 grade).
            1e-3 -> QDWH spectral D&C for D >= 5120 (~1.3× faster, ~3e-3 residual).
            See flashlib.linalg.eigh for the full Pareto frontier.

    Returns:
        eigenvalues: (K,) ascending.
        eigenvectors: (D, K) columns, ascending.
    """
    N, D = X.shape
    if N >= 4 * D:
        return _triton_pca_cov_legacy(X, K, tol=tol)
    else:
        return _triton_pca_dual_legacy(X, K, tol=tol)


flash_pca_legacy = triton_pca_legacy

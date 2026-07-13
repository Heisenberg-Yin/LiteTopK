"""Legacy truncated SVD path -- in-house Triton GEMMs + (cuSOLVER/MKL) eigh.

Kept for parity testing against the cuBLAS-TF32 + Halko path that the
``triton/svd.py`` entry point now uses.
"""


import torch
from flashlib.linalg.cov_gemm import cov_gemm as triton_cov_gemm
from flashlib.linalg.gram_gemm import gram_gemm as triton_gram_gemm
from flashlib.linalg.ab_gemm import ab_gemm as triton_ab_gemm
from flashlib.linalg.eigh import eigh as triton_eigh


def _triton_svd_cov_legacy(X: torch.Tensor, K: int):
    """Truncated SVD via Triton covariance GEMM + exact eigendecomposition.

    For N >> D: single Triton kernel computes D×D Gram (X.T @ X),
    then exact eigendecomposition extracts top-K eigenpairs.
    Singular values = sqrt(eigenvalues).
    """
    N, D = X.shape

    # Step 1: Gram matrix X.T @ X via Triton tall-skinny GEMM — single kernel
    gram = triton_cov_gemm(X)  # (D, D)

    # Step 2: Eigendecomposition — CPU MKL for D ≤ 512, cuSOLVER for D > 512
    eigenvalues, eigenvectors = triton_eigh(gram)

    # Top-K eigenvalues → singular values (descending)
    top_eigvals = eigenvalues[-K:]
    S = torch.sqrt(top_eigvals.clamp(min=0)).flip(0)
    Vh = eigenvectors[:, -K:].T.flip(0)  # (K, D), descending

    return S, Vh


def _triton_svd_dual_legacy(X: torch.Tensor, K: int):
    """Truncated SVD via Triton Gram GEMM + exact eigendecomposition + projection.

    For D >> N: NEVER materializes D×D.
      1. G = triton_gram_gemm(X)         — single Triton kernel, N×N
      2. eigh(G)                         — exact, N×N is small
      3. V = triton_ab_gemm(X, U_K)     — single Triton kernel, project to D-space

    Eigenvalues of X.T @ X == eigenvalues of X @ X.T (dual identity).
    Right singular vectors recovered via projection + normalization.
    """
    N, D = X.shape

    # Step 1: Gram matrix via Triton GEMM — single kernel, reads X once
    G = triton_gram_gemm(X)  # (N, N) = X @ X.T

    # Step 2: Eigendecomposition — CPU MKL for N ≤ 512, cuSOLVER for N > 512
    eigvals, eigvecs = triton_eigh(G)

    # Step 3: Select top K and project to D-space via Triton kernel
    K_actual = min(K, eigvals.shape[0])
    U = eigvecs[:, -K_actual:]       # (N, K_actual)
    top_eigvals = eigvals[-K_actual:]

    V = triton_ab_gemm(X, U)         # (D, K_actual) — single Triton kernel

    # Normalize to unit right singular vectors
    col_norms = V.norm(dim=0, keepdim=True).clamp(min=1e-10)
    V = V / col_norms

    # Singular values (descending)
    S = torch.sqrt(top_eigvals.clamp(min=0)).flip(0)
    Vh = V.T.flip(0)  # (K, D), descending

    return S, Vh


def triton_truncated_svd_legacy(X: torch.Tensor, K: int):
    """Truncated SVD via Triton kernels — auto-selects algorithm by data shape.

    Picks the path whose eigh dimension is smaller:
    - D <= N: covariance + eigh on D×D
    - D >  N: dual-space Gram + eigh on N×N + projection (avoids D²)

    Both paths use exact eigendecomposition.

    Args:
        X: (N, D) float32 data
        K: number of singular components

    Returns:
        S: (K,) top-K singular values (descending)
        Vh: (K, D) top-K right singular vectors (rows)
    """
    N, D = X.shape
    if D <= N:
        return _triton_svd_cov_legacy(X, K)
    else:
        return _triton_svd_dual_legacy(X, K)


flash_truncated_svd_legacy = triton_truncated_svd_legacy

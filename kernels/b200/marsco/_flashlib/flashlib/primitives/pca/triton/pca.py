"""PCA via cuBLAS GEMMs + (cuSOLVER / MKL / Halko) eigh — auto-dispatches:

1. N >> D (cov path):
     cov = X.T @ X / N (cuBLAS TF32 GEMM)
     → (D ≤ 512) MKL eigh on CPU (avoids cuSOLVER's ~5 ms launch overhead)
     → (D > 512) cuSOLVER eigh on GPU
2. D >> N (dual path):
     gram = X @ X.T / N (cuBLAS TF32 GEMM)
     → (K*4 < N)  Halko subspace iteration eigh — orders of magnitude faster
                  than the full O(N³) eigh when only top-K are needed
     → (K*4 ≥ N) MKL/cuSOLVER eigh as in the cov path
     → V = X.T @ U (cuBLAS GEMM)
     → per-column normalize

The GEMM stage uses cuBLAS TF32 matmul (`torch.matmul` with TF32 enabled);
the dual path's eigh uses Halko subspace iteration when the shape
favours it (``K*4 < N`` AND ``N >= 256``), otherwise the full cuSOLVER
syevd. Input is fp32 throughout; TF32 internal compute matches what
``tl.dot`` already used on H100/H200.
"""

import sys
import os

import torch

from flashlib.linalg.eigh import eigh


# ─── Path 1: Covariance + eigh (N >> D) ─────────────────────────────────────

def _triton_pca_cov(X: torch.Tensor, K: int, *, tol=None):
    """PCA via cuBLAS TF32 cov GEMM + eigendecomposition.

    Output cov is the full (D, D) covariance matrix in fp32 (cuBLAS does
    not expose a SYRK-style symmetric-output GEMM via torch); the lower
    triangle work is cuBLAS-internal and amortized in the tile schedule.

    Precision is controlled entirely by ``tol``: ``tol=None`` (default)
    runs the exact eigh path; passing a loose ``tol`` lets the underlying
    :func:`flashlib.linalg.eigh.eigh` switch to Halko subspace iteration
    when shape favours it (``K*4 < D`` AND ``D >= 256``).
    """
    N, D = X.shape
    cov = (X.T @ X) / N                        # cuBLAS TF32 GEMM
    return eigh(cov, K=K, tol=tol)


# ─── Path 2: Dual-Space Gram + eigh + Projection (D >> N) ───────────────────

def _triton_pca_dual(X: torch.Tensor, K: int, *, tol=None):
    """PCA via cuBLAS Gram GEMM + eigh + cuBLAS projection.

    For D >> N: NEVER materializes D x D.
      1. G = X @ X.T / N    (cuBLAS TF32 GEMM, output N x N)
      2. eigh(G, K=K, tol=tol) -- exact by default; Halko on loose tol +
                                  favourable shape.
      3. V = X.T @ U_K      (cuBLAS TF32 GEMM, output D x K)
      4. column-normalize   -- recovers unit eigenvectors of X^T X / N.

    Eigenvalues of X.T @ X / N and X @ X.T / N coincide (dual identity).
    """
    N, D = X.shape
    G = (X @ X.T) / N                          # cuBLAS TF32 GEMM
    top_eigvals, U = eigh(G, K=K, tol=tol)
    V = X.T @ U                                # cuBLAS TF32 GEMM
    V = V / V.norm(dim=0, keepdim=True).clamp(min=1e-10)
    return top_eigvals, V


# ─── Auto-dispatch ───────────────────────────────────────────────────────────

def triton_pca(X: torch.Tensor, K: int, *, tol=None):
    """PCA via cuBLAS + ``flashlib.linalg.eigh`` -- auto-selects by data shape.

    - N >= 4*D: covariance + eigh (D×D matrix, L2-cached small for D ≤ 1024)
    - N <  4*D: dual-space Gram + eigh + projection (N×N Gram, no D²
                materialization)

    Precision contract:
      * ``tol=None`` -> all matmuls run in PyTorch's default IEEE fp32
        (cuBLAS pure-fp32 path, ~67 TFLOPS on H100). Halko is OFF.
      * ``tol>0``    -> opt into a lossy budget. We set
        ``allow_tf32=True`` for the duration so the cov/Gram GEMMs run on
        TF32 tensor cores (~225 TFLOPS, ~3-4x faster) and Halko / bf16
        backends become eligible. The flag is restored on exit.

    Args:
        X: (N, D) float32 data (centered)
        K: number of components
        tol: residual tolerance.

            * ``None`` (default) -> EXACT eigh on the cov / Gram matrix
              (cuSOLVER / MKL). Halko subspace iteration is NEVER used
              automatically -- you must opt in via ``tol``.
            * ``tol >= 1e-4`` AND favourable shape (``K*4 < M`` AND
              ``M >= 256`` where ``M = D`` for cov, ``N`` for dual)
              -> Halko subspace iteration; ~26-80x speedup over cuSOLVER.

    Returns:
        eigenvalues: (K,) top-K eigenvalues (ascending)
        eigenvectors: (D, K) top-K eigenvectors (columns, ascending)
    """
    # tol=None -> EXACT: keep PyTorch's default IEEE matmul (no TF32).
    # tol>0    -> user opted into a lossy budget: enable TF32 globally
    # for the duration so the direct ``X @ y`` ops in helper paths can
    # use tensor cores. The global flag is restored on exit.
    use_tf32 = tol is not None and tol > 0
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        N, D = X.shape
        if N >= 4 * D:
            return _triton_pca_cov(X, K, tol=tol)
        return _triton_pca_dual(X, K, tol=tol)
    finally:
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


# Public alias used by the examples/ folder.
flash_pca = triton_pca

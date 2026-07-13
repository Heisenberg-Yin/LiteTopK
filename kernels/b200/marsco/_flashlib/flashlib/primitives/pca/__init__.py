"""PCA primitive -- covariance / Gram + eigh, auto-dispatch by N/D ratio.

Public API:
    flash_pca(X, K, *, tol=None, backend=None)
        -> (eigenvalues, eigenvectors), ascending.

Default is **exact in input dtype**: ``flashlib.linalg.eigh.eigh`` runs
cuSOLVER / MKL on the cov / Gram matrix. Pass ``tol >= 1e-4`` to opt
into Halko subspace iteration when the shape favours it (~26-80x
speedup over cuSOLVER at large N + small K).
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.pca import cost
from flashlib.primitives.pca.impl import flash_pca
from flashlib.primitives.pca.triton import (
    triton_pca,
    _triton_pca_cov,
    _triton_pca_dual,
)


cutedsl_pca = lazy_attr("flashlib.primitives.pca.cutedsl", "cutedsl_pca")
cutedsl_cov_gemm = lazy_attr("flashlib.primitives.pca.cutedsl", "cutedsl_cov_gemm")
cutedsl_gram_gemm = lazy_attr("flashlib.primitives.pca.cutedsl", "cutedsl_gram_gemm")
flash_pca_cutedsl = lazy_attr("flashlib.primitives.pca.cutedsl", "flash_pca_cutedsl")


__all__ = [
    "flash_pca",
    "triton_pca",
    "cutedsl_pca",
    "cutedsl_cov_gemm",
    "cutedsl_gram_gemm",
    "flash_pca_cutedsl",
    "cost",
]

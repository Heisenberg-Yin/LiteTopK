"""PCA dispatcher.

Defaults to the Triton (cuBLAS-TF32 + ``flashlib.linalg.eigh``) entry
point, exact in the input dtype. ``backend="cutedsl"`` selects the
CuteDSL-wrapped GEMM path (currently a thin cuBLAS wrapper -- see
``cutedsl/gemm.py``).
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.pca.cutedsl import flash_pca_cutedsl
from flashlib.primitives.pca.triton import triton_pca


def flash_pca(
    X: torch.Tensor,
    K: int,
    *,
    tol: Optional[float] = None,
    backend: Optional[str] = None,
):
    """PCA top-K eigenpairs (auto cov vs dual path).

    Args:
        X: (N, D) input on CUDA.
        K: number of components.
        tol: residual tolerance.

            * ``None`` (default) **-> EXACT** in input dtype: uses
              :func:`flashlib.linalg.eigh.eigh` with ``tol=None``
              (cuSOLVER / MKL on the cov / Gram matrix).
            * ``tol >= 1e-4`` AND favourable shape -> Halko subspace
              iteration internally; ~26-80x speedup over cuSOLVER.
        backend: ``"cutedsl"`` to route through the CuteDSL backend.
    """
    if backend == "cutedsl":
        return flash_pca_cutedsl(X, K)
    return triton_pca(X, K, tol=tol)

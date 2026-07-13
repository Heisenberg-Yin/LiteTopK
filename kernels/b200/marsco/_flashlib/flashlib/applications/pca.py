"""PCA — sklearn-style wrapper around primitives.pca."""
from __future__ import annotations

import torch

from flashlib.primitives.pca import flash_pca


class PCA:
    """Principal Component Analysis.

    Args:
        n_components: number of components to keep.

    Attributes set after fit():
        components_       : (n_components, D) fp32 — principal axes (rows = comps), descending order.
        explained_variance_ : (n_components,) fp32 — top-K eigenvalues, descending.
        mean_             : (D,) fp32 — column means subtracted from X.
    """

    def __init__(self, n_components: int, *,
                 backend: str | None = None, tol: float | None = None):
        self.n_components = int(n_components)
        self.backend = backend
        self.tol = tol
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None

    def fit(self, X: torch.Tensor, **kwargs) -> "PCA":
        if not X.is_cuda or X.ndim != 2:
            raise ValueError("PCA requires a 2D CUDA tensor")
        self.mean_ = X.mean(dim=0)
        Xc = X - self.mean_
        for k, v in (("backend", self.backend), ("tol", self.tol)):
            if v is not None:
                kwargs.setdefault(k, v)
        eigvals_asc, eigvecs_asc = flash_pca(Xc, K=self.n_components, **kwargs)
        # cuML/sklearn convention: descending order.
        eigvals = torch.flip(eigvals_asc, dims=[0]).contiguous()
        eigvecs = torch.flip(eigvecs_asc, dims=[1]).contiguous()  # (D, K)
        self.explained_variance_ = eigvals
        self.components_ = eigvecs.T.contiguous()  # (K, D)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.components_ is None:
            raise RuntimeError("PCA not fitted; call fit() first.")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)


__all__ = ["PCA"]

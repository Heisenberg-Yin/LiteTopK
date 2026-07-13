"""t-SNE primitive — flash gradient kernel (no N^2 Q materialisation).

Public API:
    flash_tsne(X, n_iter=1000, lr=200.0, perplexity=30.0, *, backend=None)
        -> (N, 2) embedding. Routes to the Triton backend by default.
    triton_tsne                  — alias of flash_tsne with no dispatch
    triton_tsne_gradient_only    — gradient loop only (P precomputed; bench)
    cutedsl_tsne_perplex_bisect  — CuteDSL P-matrix bisection variant
    cutedsl_compute_p_matrix     — CuteDSL end-to-end P-matrix entry point
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.tsne import cost
from flashlib.primitives.tsne.impl import (
    flash_tsne,
    triton_tsne,
    triton_tsne_gradient_only,
)


cutedsl_tsne_perplex_bisect = lazy_attr(
    "flashlib.primitives.tsne.cutedsl", "cutedsl_tsne_perplex_bisect",
)
cutedsl_compute_p_matrix = lazy_attr(
    "flashlib.primitives.tsne.cutedsl", "cutedsl_compute_p_matrix",
)
cutedsl_available = lazy_attr(
    "flashlib.primitives.tsne.cutedsl", "cutedsl_available",
)


__all__ = [
    "flash_tsne",
    "triton_tsne",
    "triton_tsne_gradient_only",
    "cutedsl_tsne_perplex_bisect",
    "cutedsl_compute_p_matrix",
    "cutedsl_available",
    "cost",
]

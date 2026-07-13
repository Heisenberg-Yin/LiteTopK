"""tsne triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file.
"""
from flashlib.primitives.tsne.triton.grad import (
    triton_tsne_qsum,
    triton_tsne_grad,
)
from flashlib.primitives.tsne.triton.grad_blocked import (
    block_p_matrix,
    triton_tsne_grad_blocked,
)
from flashlib.primitives.tsne.triton.train import (
    _compute_p_matrix,
    triton_tsne,
    triton_tsne_gradient_only,
)

__all__ = [
    "triton_tsne_qsum",
    "triton_tsne_grad",
    "block_p_matrix",
    "triton_tsne_grad_blocked",
    "triton_tsne",
    "triton_tsne_gradient_only",
]

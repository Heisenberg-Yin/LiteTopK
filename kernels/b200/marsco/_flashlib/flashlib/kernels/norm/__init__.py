"""Forward LayerNorm / RMSNorm kernels (small normalized-dim optimized).

    from flashlib.kernels.norm import flash_rmsnorm, flash_layernorm
    from flashlib import flash_rmsnorm                 # also at top level

Multi-row-per-CTA forward kernels that saturate HBM bandwidth when the
normalized dimension is small (e.g. per-head QK-norm over ``head_dim``),
where PyTorch eager assigns one CTA per row and under-utilizes the GPU.
First introduced in Sparse VideoGen (ICML 2025, arXiv:2502.01776); see
``triton/norm.py`` for the provenance note.
"""
from flashlib.kernels.norm.triton import flash_rmsnorm, flash_layernorm

__all__ = ["flash_rmsnorm", "flash_layernorm"]

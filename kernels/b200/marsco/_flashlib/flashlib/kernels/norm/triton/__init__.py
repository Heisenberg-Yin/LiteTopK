"""norm triton backend (forward LayerNorm / RMSNorm).

``@triton.jit`` kernels stay private to ``norm.py``; call them via the
``flash_rmsnorm`` / ``flash_layernorm`` Python wrappers re-exported here.
"""
from flashlib.kernels.norm.triton.norm import flash_rmsnorm, flash_layernorm

__all__ = ["flash_rmsnorm", "flash_layernorm"]

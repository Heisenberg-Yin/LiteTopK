"""connected_components triton backend (connected_components).

Re-exports top-level functions/classes/constants from each component
file. ``@triton.jit`` / ``@cute.jit`` kernels stay private to their
file (call them via the Python wrapper that lives next to them).
"""
from flashlib.kernels.connected_components.triton.connected_components import (
    connected_components,
)

__all__ = [
    "connected_components",
]

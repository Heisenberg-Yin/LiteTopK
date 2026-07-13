"""connected_components(rows, cols, N) -> (N,) int32 component labels."""
from flashlib.kernels.connected_components.triton import connected_components
from flashlib.kernels.connected_components import cost

# Backward-compatible alias for existing dbscan import path.
flash_cc_from_edges = connected_components

__all__ = ["connected_components", "flash_cc_from_edges", "cost"]

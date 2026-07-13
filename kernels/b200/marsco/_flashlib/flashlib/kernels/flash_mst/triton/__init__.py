"""flash_mst Triton backend (Boruvka-style MST + connected components).

Component file:
    flash_mst.py — full Boruvka MST/CC kernels + ``flash_mst`` and
                   ``flash_cc_from_edges`` wrappers (kept together
                   because they share most sub-kernels).
"""
from flashlib.kernels.flash_mst.triton.flash_mst import (
    flash_cc_from_edges,
    flash_mst,
)

__all__ = [
    "flash_cc_from_edges",
    "flash_mst",
]

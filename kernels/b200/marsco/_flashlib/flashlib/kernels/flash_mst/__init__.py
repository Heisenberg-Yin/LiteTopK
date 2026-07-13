"""flash_mst — GPU-resident Boruvka MST kernels.

Public API:
    flash_mst(MRD)                          -> (N-1, 3) MST edge list
    flash_cc_from_edges(rows, cols, N, ...) -> (N,) int32 CC labels

Algorithm (Boruvka, all on GPU):

    1. Per-component argmin via packed-int64 atomic_min
       (weight_bits << 32 | dst_idx) — `_per_component_argmin_v2_kernel`
    2. Concurrent union-find via atomic_cas — `_concurrent_uf_kernel`
    3. Pointer-jumping (`_pointer_jump_kernel`) flattens the parent forest
       in parallel — replaces the per-iter ``parent[parent.to(int64)]``
       PyTorch loop.

Persistent state (allocated ONCE at entry) is reused across rounds so
even shapes with many Boruvka iterations don't pay alloc cost.

Both ``flash_dbscan`` and ``flash_hdbscan`` route through this module
(directly for hdbscan; via ``flashlib.kernels.connected_components`` for
dbscan).

This module **replaces** the slower union-find that previously lived in
``flashlib.kernels.connected_components.triton`` — that module still
exists but is now a thin delegating wrapper.
"""
from __future__ import annotations

from flashlib.kernels.flash_mst.triton import (
    flash_cc_from_edges,
    flash_mst,
)
from flashlib.kernels.flash_mst import cost


__all__ = ["flash_mst", "flash_cc_from_edges", "cost"]

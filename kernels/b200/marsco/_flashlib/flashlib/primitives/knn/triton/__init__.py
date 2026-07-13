"""knn triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file.

Kernel layout:

* :mod:`_common`   -- shared helpers: ``_next_pow2``, micro-bench,
  IEEE-sortable u32 transform, ``_INF_PACKED`` sentinel.
* :mod:`sortmerge` -- packed-uint64 sort-merge top-K (x²-free). Routed
  to only for the small-Q + medium-K Pattern-A corner.
* :mod:`insert`    -- iterative argmin-insert top-K (x²-free). The
  general path -- BN in {8, 16, 32, 64, 128} covers everything from
  small-Q search to large-Q build.
* :mod:`_row_norm` -- ``_fast_row_sq`` / ``_get_or_compute_csq`` for
  the CuteDSL FA3 wrapper (it needs ``c_sq`` as a kernel argument).
* :mod:`dispatch`  -- host-side router. Single public entry
  :func:`flash_knn_triton` runs through :func:`_heuristic_config`
  unconditionally -- the per-CTA-count check inside the heuristic
  picks single-pass vs M-split. Per-shape config (BN/BM/mode/mps/
  ns_pipe) also falls out of :func:`_heuristic_config`.
"""
from flashlib.primitives.knn.triton._common import (
    _next_pow2,
    _bench_quick,
)
from flashlib.primitives.knn.triton._row_norm import (
    _fast_row_sq,
    _get_or_compute_csq,
)
from flashlib.primitives.knn.triton.dispatch import (
    flash_knn_triton,
    flash_knn_triton_small_n,
    flash_knn_triton_large_n,
)

__all__ = [
    "flash_knn_triton",
    "flash_knn_triton_small_n",
    "flash_knn_triton_large_n",
]

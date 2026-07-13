"""random_forest triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file (call them via the
Python wrapper that lives next to them).
"""
from flashlib.primitives.random_forest.triton.histogram import (
    triton_rf_histogram_split,
    triton_rf_histogram,
)
from flashlib.primitives.random_forest.triton.rf_kernels import (
    _build_node_histograms,
    _build_node_histograms_subfeat,
    _build_node_histograms_subfeat_hybrid,
    _build_node_histograms_hybrid,
    _build_node_histograms_ranged,
    _find_best_splits_triton,
    _find_best_splits_subfeat,
    _split_counts_fused,
    _hist_subtract_fused,
    _partition_samples_fused,
)

__all__ = [
    "triton_rf_histogram_split",
    "triton_rf_histogram",
]

"""Pairwise distance kernels — Euclidean (squared) and L2.

    from flashlib.kernels.distance import pairwise_l2, pairwise_l2sq
    from flashlib import pairwise_l2                  # also at top level
"""
from flashlib.kernels.distance.triton import pairwise_l2, pairwise_l2sq
from flashlib.kernels.distance import cost

__all__ = ["pairwise_l2", "pairwise_l2sq", "cost"]

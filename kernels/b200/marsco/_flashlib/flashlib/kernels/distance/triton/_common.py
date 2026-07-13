"""Shared helpers used by all flash-distance Triton kernels."""
import math


def _round_to_bucket(n):
    """Round n up to nearest power-of-2 bucket (autotuner key stability)."""
    if n <= 0:
        return 1
    return 1 << math.ceil(math.log2(max(n, 1)))

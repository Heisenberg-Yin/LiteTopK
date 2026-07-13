"""multinomial_nb cutedsl backend (nb).

Re-exports top-level functions/classes/constants from each component
file. ``@triton.jit`` / ``@cute.jit`` kernels stay private to their
file (call them via the Python wrapper that lives next to them).
"""
from flashlib.primitives.multinomial_nb.cutedsl.nb import (
    _CUTEDSL_AVAILABLE,
    _CUTE_IMPORT_ERROR,
    BLOCK_N,
    BLOCK_C,
    BLOCK_K,
    THR_M,
    THR_N,
    ITEMS_M,
    _try_init_cutedsl,
    precompile_cutedsl_for_params,
    cutedsl_multinomial_nb_predict_argmax,
    cutedsl_multinomial_nb_predict_jll,
    cutedsl_multinomial_nb,
    cutedsl_available,
)

__all__ = [
    "BLOCK_N",
    "BLOCK_C",
    "BLOCK_K",
    "THR_M",
    "THR_N",
    "ITEMS_M",
    "precompile_cutedsl_for_params",
    "cutedsl_multinomial_nb_predict_argmax",
    "cutedsl_multinomial_nb_predict_jll",
    "cutedsl_multinomial_nb",
    "cutedsl_available",
]

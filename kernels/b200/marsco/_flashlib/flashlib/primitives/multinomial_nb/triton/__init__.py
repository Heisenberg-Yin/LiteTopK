"""multinomial_nb triton backend.

Re-exports the public Python wrappers from each component file.
``@triton.jit`` kernels stay private to their file.
"""
from flashlib.primitives.multinomial_nb.triton.nb import (
    flash_multinomial_nb_fit,
    flash_multinomial_nb_predict_log_proba_unnormalized,
    flash_multinomial_nb_predict,
    flash_multinomial_nb,
)
from flashlib.primitives.multinomial_nb.triton.nb_core import (
    nb_count_features,
)

__all__ = [
    "flash_multinomial_nb_fit",
    "flash_multinomial_nb_predict_log_proba_unnormalized",
    "flash_multinomial_nb_predict",
    "flash_multinomial_nb",
    "nb_count_features",
]

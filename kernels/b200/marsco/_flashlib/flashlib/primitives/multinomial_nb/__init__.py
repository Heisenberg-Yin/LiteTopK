"""Multinomial Naive Bayes primitive — text/tabular feature counts to log-probabilities.

Public API:
    flash_multinomial_nb(X_train, y_train, X_test, n_classes, *, tol=None, backend=None)
        -> labels (or (labels, log_proba) when ``return_log_proba=True``).
        ``tol`` selects the predict GEMM precision via
        :func:`flashlib.linalg.gemm.gemm` (default ``None`` -> fp32 IEEE).
    flash_multinomial_nb_fit                      — fit-only entry point
    flash_multinomial_nb_predict                  — predict-only entry point
    flash_multinomial_nb_predict_log_proba_unnormalized
                                                  — joint log-likelihood
    cutedsl_multinomial_nb                        — CuteDSL fused-predict
    cutedsl_multinomial_nb_predict_argmax         — CuteDSL argmax kernel
    cutedsl_multinomial_nb_predict_jll            — CuteDSL JLL kernel
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.multinomial_nb import cost
from flashlib.primitives.multinomial_nb.impl import flash_multinomial_nb
from flashlib.primitives.multinomial_nb.triton.nb import (
    flash_multinomial_nb_fit,
    flash_multinomial_nb_predict,
    flash_multinomial_nb_predict_log_proba_unnormalized,
)


cutedsl_multinomial_nb = lazy_attr(
    "flashlib.primitives.multinomial_nb.cutedsl", "cutedsl_multinomial_nb",
)
cutedsl_multinomial_nb_predict_argmax = lazy_attr(
    "flashlib.primitives.multinomial_nb.cutedsl", "cutedsl_multinomial_nb_predict_argmax",
)
cutedsl_multinomial_nb_predict_jll = lazy_attr(
    "flashlib.primitives.multinomial_nb.cutedsl", "cutedsl_multinomial_nb_predict_jll",
)
cutedsl_available = lazy_attr(
    "flashlib.primitives.multinomial_nb.cutedsl", "cutedsl_available",
)


__all__ = [
    "flash_multinomial_nb",
    "flash_multinomial_nb_fit",
    "flash_multinomial_nb_predict",
    "flash_multinomial_nb_predict_log_proba_unnormalized",
    "cutedsl_multinomial_nb",
    "cutedsl_multinomial_nb_predict_argmax",
    "cutedsl_multinomial_nb_predict_jll",
    "cutedsl_available",
    "cost",
]

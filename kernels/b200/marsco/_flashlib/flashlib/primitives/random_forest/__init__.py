"""Random Forest primitive -- quantile binning + level-wise BFS tree growth.

The full classifier lives in :mod:`flashlib.primitives.random_forest.impl`
(``FlashRandomForestClassifier``); the heavy Triton kernels powering the
histogram / best-split / partition stages live under
:mod:`flashlib.primitives.random_forest.triton.rf_kernels`.
"""
from flashlib._lazy import lazy_attr
from flashlib.primitives.random_forest import cost
from flashlib.primitives.random_forest.impl import FlashRandomForestClassifier
from flashlib.primitives.random_forest.triton import (
    triton_rf_histogram_split,
    triton_rf_histogram,
)


flash_random_forest = FlashRandomForestClassifier

cutedsl_predict_classifier = lazy_attr(
    "flashlib.primitives.random_forest.cutedsl",
    "cutedsl_predict_classifier",
)
CuteDSLRandomForestClassifier = lazy_attr(
    "flashlib.primitives.random_forest.cutedsl",
    "CuteDSLRandomForestClassifier",
)


__all__ = [
    "flash_random_forest",
    "FlashRandomForestClassifier",
    "triton_rf_histogram_split",
    "triton_rf_histogram",
    "cutedsl_predict_classifier",
    "CuteDSLRandomForestClassifier",
    "cost",
]

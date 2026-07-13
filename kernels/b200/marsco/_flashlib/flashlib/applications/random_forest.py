"""RandomForestClassifier — sklearn-style class.

The actual classifier lives in primitives.random_forest.FlashRandomForestClassifier;
this module just re-exports it under the sklearn-style name.
"""
from flashlib.primitives.random_forest import FlashRandomForestClassifier as RandomForestClassifier

__all__ = ["RandomForestClassifier"]

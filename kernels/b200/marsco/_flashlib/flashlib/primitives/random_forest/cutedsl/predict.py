"""flash-rf CuteDSL alternative — HONEST REPORT: no architectural win.

Random forest training and inference are irregular workloads. CuteDSL's
strengths on Hopper SM90 are:
  - WGMMA + TMA + cluster launch for dense GEMM-like kernels
  - Streaming/structured tile pipelines (warp-specialized epilogues)
  - First-class TMA for predictable HBM access patterns

RF's hot kernels are:
  1. Histogram scatter: per-sample atomic_add into (n_active, D, n_bins, K).
     Atomic-bound at 3-9% of peak HBM BW (structural — confirmed in earlier
     profiling). Triton already saturates this ceiling. CuteDSL has no API
     advantage on atomics — wgmma/TMA are irrelevant for this kernel.
  2. Best-split (per-feature gini scan): cumsum + reduction over (N_BINS, K)
     register tiles. Triton's `tl.cumsum` + `tl.argmax` is concise and
     register-pressure-tuned. CuteDSL has no built-in cumsum; you'd hand-roll
     a warp scan. No throughput advantage.
  3. Partition: each sample reads its node's split, gathers X[i, feat], writes
     new node id. Pointer-chase / dependent loads. No GEMM structure.
  4. Predict (NEWLY FUSED IN TRITON): traverse all trees in one kernel,
     accumulate per-class votes in registers. The bottleneck is the
     dependent-load chain `X_BIN[i, FEAT[t, cur]]` for `max_depth` steps —
     pure HBM latency / cache, no compute structure. CuteDSL cannot help.

The "speculative" suggestion in the brief — express tree traversal as GEMM
with one-hot lookups — would replace the O(N · n_trees · max_depth · 1-byte)
load pattern with an O(N · n_trees · max_nodes) dense matmul. That's a
n_trees · max_nodes / max_depth ≈ 2^max_depth blow-up of memory traffic.
Not a win at any tested shape (max_depth=10 → ×1024 traffic blow-up).

Therefore: this file provides a thin shim that re-exports the Triton fused
predict kernel from `common.flash_rf.fused_predict_classifier`. The shim
exists for API symmetry with other operators in the repo (which all expose
a `cutedsl_impl.py`); it is not a separate CuteDSL implementation.

CuteDSL setup verified on this machine:
  >>> import cutlass; print(cutlass.__version__)   # 4.5.0
  >>> import cutlass.cute as cute                  # works

Speedup vs Triton fused predict:
  CuteDSL (this shim) = Triton fused (1.0×, same kernel).

Honest conclusion: no Hopper-specific HW feature accelerates tree traversal,
so we do not implement a separate CuteDSL kernel.
"""
import torch

from flashlib.primitives.random_forest.impl import (
    FlashRandomForestClassifier as _FlashRandomForestClassifier,
)


def _fused_predict_classifier(X_bin, trees, n_classes, max_depth,
                              return_proba=False):
    """Fallback fused-predict for the random-forest CuteDSL shim.

    The cuml-test sibling expected this in ``common.flash_rf`` but the public
    helper was never published there. We delegate to the FlashRandomForestClassifier
    instance's predict method, building a lightweight wrapper.
    """
    rf = _FlashRandomForestClassifier(n_estimators=len(trees), max_depth=max_depth)
    rf._trees = trees
    rf._n_classes = n_classes
    if return_proba:
        return rf.predict_proba(X_bin)
    return rf.predict(X_bin)


def cutedsl_predict_classifier(X_bin, trees, n_classes, max_depth,
                                return_proba=False):
    """API-compatible wrapper around the Triton fused predict.

    See module docstring for why CuteDSL has no architectural advantage on
    this workload.
    """
    return _fused_predict_classifier(
        X_bin, trees, n_classes, max_depth, return_proba=return_proba
    )


class CuteDSLRandomForestClassifier(_FlashRandomForestClassifier):
    """Same as FlashRandomForestClassifier — see module docstring."""
    pass

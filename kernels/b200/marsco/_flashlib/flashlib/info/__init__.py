"""flashlib.info -- informative cost API. Agent-friendly: no GPU, fast.

Single-variant op
-----------------

    import flashlib.info as info
    est = info.estimate("kmeans", shape=(1_000_000, 128),
                        params={"K": 100})
    print(est.summary_line())
    #   kmeans  4.42 ms  bound=memory  410 GB/s  ( 11% peak)  [calibrated]

Multi-variant ops (precision / performance Pareto frontier)
-----------------------------------------------------------

    info.variants("eigh", shape=(8192, 8192))      # every variant
    info.pareto  ("eigh", shape=(8192, 8192))      # only Pareto-optimal
    info.list_variant_families()                    # ['eigh', 'gemm', ...]

Compare against external references (cuml / sklearn / cublas)
-------------------------------------------------------------

    info.compare("kmeans", shape=(500_000, 64), params={"K": 64})
    # -> {"flashlib": Estimate(...), "references": {"cuml": {...}, ...}}

Derived performance numbers on every Estimate
---------------------------------------------

    est.achieved_tflops     # flops / (runtime_ms * 1e9)
    est.achieved_gbs        # bytes_moved / (runtime_ms * 1e6)
    est.arithmetic_intensity
    est.utilization_pct     # achieved / hardware peak (requires device + dtype)

Confidence tier (most reliable first)
-------------------------------------

* ``"calibrated"`` -- consulted measured sustained TFLOPS / GB/s for the
  op_class on this device (see roofline._SUSTAINED_TFLOPS).
* ``"measured"``   -- hard-coded a regression fit from benchmark data.
* ``"roofline"``   -- vendor peak * default-efficiency.
* ``"heuristic"``  -- coarse guess (e.g. first-call CuteDSL compile).

Pure stdlib: importing this module does NOT load torch/triton. Each
op's cost.py is lazy-loaded on first call to ``estimate()``. The
single deferred ``import torch`` is gated by
:func:`flashlib.info.roofline.detect_device` and only runs when the
caller leaves ``device`` unspecified.
"""
from flashlib.info.estimate import Estimate, Variant
from flashlib.info.roofline import (
    canonicalize_dtype,
    detect_device,
    is_calibrated,
    list_devices,
    list_op_classes,
)
from flashlib.info.registry import list_ops, list_variant_families, list_variants
from flashlib.info.dispatch import (
    compare,
    estimate,
    pareto,
    recommend,
    summary,
    variants,
)

__all__ = [
    # dispatchers
    "estimate", "recommend", "variants", "pareto", "compare", "summary",
    # registry helpers
    "list_ops", "list_variant_families", "list_variants",
    # roofline helpers
    "list_devices", "list_op_classes", "detect_device",
    "canonicalize_dtype", "is_calibrated",
    # dataclasses
    "Estimate", "Variant",
]

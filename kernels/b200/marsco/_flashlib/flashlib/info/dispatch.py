"""info.estimate / info.recommend / info.variants / info.pareto / info.compare
   / info.summary dispatchers.

All take ``tol`` as a first-class kwarg. The cost function for each op
accepts ``tol`` directly and cascades it to sub-op cost functions, so
the returned ``Estimate.subops`` carries the full call-stack tree.

Registry resolution supports two forms:
  - ``"pkg.module"``           -> calls ``module.estimate(...)`` /
                                  ``module.recommend(...)``
  - ``"pkg.module:funcname"``  -> calls ``module.<funcname>(...)`` for
                                  estimate; ``module.recommend_<funcname>(...)``
                                  for recommend (falls back to
                                  ``module.recommend`` if absent).

The dispatcher also stamps ``dtype`` and ``device`` onto the returned
:class:`Estimate` so derived properties (``achieved_tflops`` /
``utilization_pct``) work end-to-end without the caller plumbing them
manually.
"""
from __future__ import annotations

import importlib

from flashlib.info.estimate import Estimate, Variant, is_pareto_optimal
from flashlib.info.registry import resolve, list_variants
from flashlib.info.roofline import canonicalize_dtype, detect_device


# ---------------------------------------------------------------------------
# Reference cost table -- approximate cuML / sklearn / torch wall-clock
# numbers for the headline shapes. Used by :func:`compare` so agents can
# see "flashlib X.Yms vs cuml Z.Wms" without running anything.
#
# Each entry is (reference_lib, runtime_ms_at_typical_shape). The
# "typical shape" assumption is documented next to each entry; the
# helper just scales linearly with the shape's dominant axis when the
# user's shape differs (rough, but agent-useful).
#
# Sourced from benchmarks/results/vs_cuml_full.md.
# ---------------------------------------------------------------------------
_REFERENCE_RUNTIME_MS: dict[str, dict[str, dict]] = {
    # op_name -> reference_lib -> {anchor_shape, anchor_runtime_ms, scaling}
    "kmeans": {
        "cuml":     {"anchor": (500_000, 64, 64),   "rt": 6.5,
                     "scaling": "linear-N*K"},
        "sklearn":  {"anchor": (500_000, 64, 64),   "rt": 1800.0,
                     "scaling": "linear-N*K"},
    },
    "knn": {
        "cuml":     {"anchor": (1, 8192, 65536, 256), "rt": 14.1,
                     "scaling": "linear-N*M*D"},
        "torch":    {"anchor": (1, 8192, 65536, 256), "rt": 14.1,
                     "scaling": "linear-N*M*D"},
    },
    "dbscan": {
        "cuml":     {"anchor": (200_000, 32), "rt": 87.0,
                     "scaling": "quadratic-N"},
    },
    "hdbscan": {
        "cuml":     {"anchor": (10_000, 16), "rt": 65.0,
                     "scaling": "quadratic-N"},
    },
    "pca": {
        "cuml":     {"anchor": (1_000_000, 512, 32), "rt": 26.3,
                     "scaling": "linear-N*D + cubic-D"},
    },
    "truncated_svd": {
        "cuml":     {"anchor": (1_000_000, 512, 32), "rt": 25.7,
                     "scaling": "linear-N*D + cubic-D"},
    },
    "linear_regression": {
        "cuml":     {"anchor": (1_000_000, 512), "rt": 11.6,
                     "scaling": "linear-N*D + cubic-D"},
        "sklearn":  {"anchor": (1_000_000, 512), "rt": 111.3,
                     "scaling": "linear-N*D + cubic-D"},
    },
    "ridge": {
        "cuml":     {"anchor": (1_000_000, 512), "rt": 15.1,
                     "scaling": "linear-N*D + cubic-D"},
    },
    "logistic_regression": {
        "cuml":     {"anchor": (100_000, 200), "rt": 1.13,
                     "scaling": "linear-N*D"},
    },
    "multinomial_nb": {
        "cuml":     {"anchor": (200_000, 500, 20), "rt": 0.30,
                     "scaling": "linear-N*D"},
    },
    "standard_scaler": {
        "sklearn":  {"anchor": (5_000_000, 64), "rt": 2.94,
                     "scaling": "linear-N*D"},
    },
    "spectral_clustering": {
        "sklearn":  {"anchor": (8_000, 16), "rt": 180.0,
                     "scaling": "quadratic-N + cubic-N(eigh)"},
    },
    "tsne": {
        "cuml":     {"anchor": (10_000, 64), "rt": 1100.0,  # exact path
                     "scaling": "quadratic-N"},
    },
    "umap": {
        "cuml":     {"anchor": (50_000, 64), "rt": 720.0,
                     "scaling": "linear-N*D"},
    },
    "random_forest": {
        "cuml":     {"anchor": (100_000, 200), "rt": 35.0,
                     "scaling": "linear-N*log(N)*D"},
    },
    "gemm": {
        "cublas":   {"anchor": (4096, 4096, 4096), "rt": 0.36,
                     "scaling": "cubic-N (assumes square)"},
    },
    "eigh": {
        "cusolver": {"anchor": (4096, 4096), "rt": 60.0,
                     "scaling": "cubic-N"},
    },
}


def _shape_scale(scaling: str, anchor: tuple, shape) -> float:
    """Rough multiplicative scaling factor between ``anchor`` and ``shape``.

    Only the scaling kinds that actually appear in
    :data:`_REFERENCE_RUNTIME_MS` are implemented; unknown kinds return
    ``1.0`` so the anchor runtime is used verbatim.
    """
    import math
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    if len(shape) != len(anchor):
        return 1.0
    try:
        if scaling == "linear-N*K":
            # (N, D, K)
            n_ratio = shape[0] / anchor[0]
            k_ratio = shape[2] / anchor[2]
            return n_ratio * k_ratio
        if scaling == "linear-N*M*D":
            B, N, M, D = shape
            Ba, Na, Ma, Da = anchor
            return (N * M * D) / (Na * Ma * Da)
        if scaling == "quadratic-N":
            return (shape[0] / anchor[0]) ** 2
        if scaling == "linear-N*D + cubic-D":
            # PCA / LR style: cov + eigh
            N, D = shape[0], shape[1]
            Na, Da = anchor[0], anchor[1]
            cov_ratio = (N * D * D) / (Na * Da * Da)
            eigh_ratio = (D ** 3) / (Da ** 3)
            # Cov dominates at large N; pick the larger.
            return max(cov_ratio, eigh_ratio)
        if scaling == "linear-N*D":
            return (shape[0] * shape[1]) / (anchor[0] * anchor[1])
        if scaling == "linear-N*log(N)*D":
            n_ratio = (shape[0] * math.log2(max(shape[0], 2))) / (
                anchor[0] * math.log2(max(anchor[0], 2))
            )
            d_ratio = shape[1] / anchor[1]
            return n_ratio * d_ratio
        if scaling == "quadratic-N + cubic-N(eigh)":
            return (shape[0] / anchor[0]) ** 3
        if scaling == "cubic-N (assumes square)":
            return (shape[0] / anchor[0]) ** 3
    except (IndexError, ZeroDivisionError, TypeError):
        return 1.0
    return 1.0


def _split(target: str) -> tuple[str, str]:
    if ":" in target:
        return tuple(target.split(":", 1))
    return target, ""


def _load_estimate(op: str):
    target = resolve(op)
    modpath, fnname = _split(target)
    mod = importlib.import_module(modpath)
    if fnname:
        return getattr(mod, fnname)
    return mod.estimate


def _load_recommend(op: str):
    target = resolve(op)
    modpath, fnname = _split(target)
    mod = importlib.import_module(modpath)
    if fnname:
        specific = getattr(mod, f"recommend_{fnname}", None)
        if specific is not None:
            return specific
    return mod.recommend


def _stamp(est: Estimate, dtype: str, device: str) -> Estimate:
    """Tag estimate (and every sub-op recursively) with dtype/device."""
    if est.dtype is None:
        est.dtype = canonicalize_dtype(dtype)
    if est.device is None:
        est.device = device
    for s in est.subops:
        _stamp(s, dtype, device)
    return est


def estimate(op: str, shape, params=None, tol: float | None = None,
             dtype: str = "float32", device: str | None = None,
             **kwargs) -> Estimate:
    """Predict runtime / flops / bytes / peak-memory for an op.

    ``tol`` is the universal tolerance: ``tol=None`` means "exact"
    (most accurate). The cost function picks the routed variant based
    on ``tol`` AND cascades it to sub-op cost calls. Returned
    Estimate's ``subops`` carries the full call-stack tree of
    sub-primitive estimates, each itself under the same tol.

    ``device``: when ``None`` (default), probe the actual CUDA device
    via :func:`flashlib.info.roofline.detect_device` (deferred torch
    import). Pass an explicit key (``"H100"`` / ``"H200"`` / ``"A100"``)
    to override.

    Use ``est.print_tree()`` to display the call-stack and
    ``est.summary_line()`` for a one-line at-a-glance view.
    """
    if device is None:
        device = detect_device("H100")
    dtype = canonicalize_dtype(dtype)
    fn = _load_estimate(op)
    est = fn(shape=shape, params=params, tol=tol, dtype=dtype, device=device,
             **kwargs)
    return _stamp(est, dtype, device)


def recommend(op: str, shape, params=None, tol: float | None = None,
              dtype: str = "float32", device: str | None = None,
              **kwargs) -> dict:
    """Return suggested kernel hyperparameters / variant choice for the op."""
    if device is None:
        device = detect_device("H100")
    dtype = canonicalize_dtype(dtype)
    fn = _load_recommend(op)
    return fn(shape=shape, params=params, tol=tol, dtype=dtype, device=device,
              **kwargs)


def variants(op_family: str, shape, params=None, tol: float | None = None,
             dtype: str = "float32", device: str | None = None,
             **kwargs) -> list[Variant]:
    """Estimate every registered variant of ``op_family``.

    Unavailable variants (registry entry resolves to a function whose
    cost module raises) come back with ``runtime_ms=inf`` and a note
    explaining the failure -- ``pareto`` then drops them automatically.
    """
    if device is None:
        device = detect_device("H100")
    out = []
    for v in list_variants(op_family):
        try:
            est = estimate(v, shape=shape, params=params, tol=tol, dtype=dtype,
                           device=device, **kwargs)
        except Exception as e:
            est = Estimate(
                op_name=v, runtime_ms=float("inf"),
                bound="latency", confidence="heuristic",
                dtype=canonicalize_dtype(dtype), device=device,
                notes=[f"variant {v!r} unavailable: {type(e).__name__}: {e}"],
            )
        out.append(Variant(name=v, estimate=est))
    return out


def pareto(op_family: str, shape, params=None, tol: float | None = None,
           dtype: str = "float32", device: str | None = None,
           **kwargs) -> list[Variant]:
    """Filter ``variants()`` to those on the (runtime, residual) Pareto front.

    Returned list is sorted by ascending runtime (fastest first).
    """
    all_variants = variants(op_family, shape=shape, params=params, tol=tol,
                            dtype=dtype, device=device, **kwargs)
    estimates = [v.estimate for v in all_variants]
    front = [v for v in all_variants if is_pareto_optimal(v.estimate, estimates)]
    front.sort(key=lambda v: v.estimate.runtime_ms)
    return front


def compare(op: str, shape, params=None, tol: float | None = None,
            dtype: str = "float32", device: str | None = None,
            references: list[str] | None = None,
            **kwargs) -> dict:
    """Compare flashlib's predicted runtime against external references.

    Returns a dict::

        {
            "flashlib":  Estimate,
            "references": {
                "cuml":     {"runtime_ms": 14.1, "speedup": 5.4, "anchor": ...},
                "sklearn":  {...},
                ...
            },
            "shape": (...),
            "dtype": "fp32",
            "device": "H200",
        }

    The reference numbers come from the calibrated
    :data:`_REFERENCE_RUNTIME_MS` table -- they are scaled from the
    anchor shape to the requested shape with a per-op scaling rule
    (linear in N*M*D for KNN, cubic in N for eigh / spectral-clustering,
    etc.). This is rough but enough for "is flashlib worth using here?"
    triage without touching a GPU. Pass ``references=["cuml"]`` to
    restrict the comparison set.
    """
    if device is None:
        device = detect_device("H100")
    fl_est = estimate(op, shape=shape, params=params, tol=tol,
                      dtype=dtype, device=device, **kwargs)
    refs = _REFERENCE_RUNTIME_MS.get(op, {})
    if references is not None:
        refs = {k: v for k, v in refs.items() if k in references}
    ref_out: dict[str, dict] = {}
    for ref_name, info in refs.items():
        scale = _shape_scale(info["scaling"], info["anchor"], shape)
        ref_rt = info["rt"] * scale
        speedup = (
            ref_rt / fl_est.runtime_ms if fl_est.runtime_ms > 0 else float("inf")
        )
        ref_out[ref_name] = {
            "runtime_ms": ref_rt,
            "speedup": speedup,
            "anchor": info["anchor"],
            "scaling": info["scaling"],
            "note": (
                "approximation scaled from anchor shape; "
                "see benchmarks/results/vs_cuml_full.md for actuals"
            ),
        }
    return {
        "flashlib": fl_est,
        "references": ref_out,
        "shape": shape,
        "dtype": canonicalize_dtype(dtype),
        "device": device,
    }


def summary(op: str, shape, params=None, tol: float | None = None,
            dtype: str = "float32", device: str | None = None,
            **kwargs) -> str:
    """One-line agent-friendly summary string.

    Equivalent to
    ``estimate(op, shape, ...).summary_line()`` -- exposed at the
    dispatcher level so it can be called without touching the
    :class:`Estimate` object directly.
    """
    return estimate(op, shape=shape, params=params, tol=tol,
                    dtype=dtype, device=device, **kwargs).summary_line()

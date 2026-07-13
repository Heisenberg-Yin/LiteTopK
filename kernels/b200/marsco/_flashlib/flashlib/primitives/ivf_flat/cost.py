"""Cost model for IVF-Flat -- composes kmeans (build) + knn (coarse) + fine-scan.

The estimate is a two-phase tree mirroring an end-to-end
``flash_ivf_flat`` call:

* **build** -- one-time:
    - ``kmeans``  : coarse quantizer trained on a ``min(M, nlist*256)`` sample
                    (reuses :mod:`flashlib.primitives.kmeans.cost`).
    - ``assign``  : one x²-free pass over all ``M`` rows vs ``nlist`` centroids.
    - ``layout``  : bincount + cumsum + argsort + cell-contiguous reorder
                    (bandwidth-bound over ``M`` rows).
* **search** -- per query batch (the steady-state cost):
    - ``coarse``  : ``flash_knn`` top-``nprobe`` over the ``nlist`` centroids
                    (reuses :mod:`flashlib.primitives.knn.cost`).
    - ``fine``    : the fused ragged-list scan -- ``2*nq*nprobe*(M/nlist)*D``
                    FLOPs, candidate-vector reads + partial-topk writes;
                    bandwidth-bound (``op_class="ivf_flat_search"``).

The shape contract is ``shape = (M, D)`` (the database) with the search
workload supplied via ``params``:

    params = {"nlist": .., "nprobe": .., "k": .., "nq": .., "niter": ..}
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype in ("fp32", "float32", "tf32", "float") else 2


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def common(shape, params):
    M, D = shape
    params = params or {}
    nlist = int(params.get("nlist", 1024))
    nlist = max(1, min(nlist, M))
    nprobe = int(params.get("nprobe", 8))
    nprobe = max(1, min(nprobe, nlist))
    k = int(params.get("k", 10))
    nq = int(params.get("nq", 10_000))
    niter = int(params.get("niter", params.get("max_iters", 20)))
    train_size = int(params.get("train_size") or min(M, nlist * 256))
    train_size = max(min(train_size, M), nlist)
    return M, D, nlist, nprobe, k, nq, niter, train_size


# ── build phase ───────────────────────────────────────────────────────────
def _build_subops(M, D, nlist, niter, train_size, dtype, device):
    db = _dtype_bytes(dtype)

    km = _est("kmeans", shape=(train_size, D),
              params={"K": nlist, "max_iters": niter},
              dtype=dtype, device=device)
    km.op_name = "ivf_flat.build.kmeans"

    # Full-database assignment: one x²-free (M, nlist, D) pass.
    assign_flops = 2 * M * nlist * D
    assign_bytes = (M * D + nlist * D) * db + M * 4
    a_rt, a_bound = roofline(assign_flops, assign_bytes, dtype, device,
                             op_type="kmeans", n_launches=1)
    assign = Estimate(
        op_name="ivf_flat.build.assign", runtime_ms=a_rt,
        flops=assign_flops, bytes_moved=assign_bytes,
        memory_peak_gb=M * D * db / 1e9, bound=a_bound,
        confidence="calibrated", n_kernel_launches=1,
        notes=[f"assign M={M} rows to nlist={nlist} centroids (x²-free)"],
        dtype=dtype, device=device,
    )

    # CSR layout: bincount + cumsum + argsort + gather reorder of (M, D).
    layout_bytes = M * D * db * 2 + M * 4 * 3
    l_rt, l_bound = roofline(0.0, layout_bytes, dtype, device,
                             op_type="elementwise", n_launches=4)
    layout = Estimate(
        op_name="ivf_flat.build.layout", runtime_ms=l_rt,
        flops=0.0, bytes_moved=layout_bytes,
        memory_peak_gb=M * D * db / 1e9, bound=l_bound,
        confidence="roofline", n_kernel_launches=4,
        notes=["bincount + cumsum + argsort + cell-contiguous reorder"],
        dtype=dtype, device=device,
    )
    return [km, assign, layout]


# ── search phase ──────────────────────────────────────────────────────────
def _search_subops(M, D, nlist, nprobe, k, nq, dtype, device):
    db = _dtype_bytes(dtype)

    coarse = _est("knn", shape=(1, nq, nlist, D), params={"k": nprobe},
                  dtype=dtype, device=device)
    coarse.op_name = "ivf_flat.search.coarse"

    # Fine scan: scan nprobe lists of avg length M/nlist per query.
    cand = nq * nprobe * (M / max(nlist, 1))
    topk_pad = _next_pow2(k)
    fine_flops = 2 * cand * D
    fine_bytes = cand * D * db + nq * nprobe * topk_pad * 8
    f_rt, f_bound = roofline(fine_flops, fine_bytes, dtype, device,
                             op_type="ivf_flat_search", n_launches=1)
    fine = Estimate(
        op_name="ivf_flat.search.fine", runtime_ms=f_rt,
        flops=fine_flops, bytes_moved=fine_bytes,
        memory_peak_gb=nq * nprobe * topk_pad * 8 / 1e9, bound=f_bound,
        confidence="calibrated", n_kernel_launches=1,
        suggested_config={"nprobe": nprobe, "k": k},
        notes=[
            f"fused ragged-list scan: ~{cand:.0f} candidate distances "
            f"(nq={nq}, nprobe={nprobe}, M/nlist={M/max(nlist,1):.0f}); "
            "no (nq x candidates) HBM matrix",
        ],
        dtype=dtype, device=device,
    )
    return [coarse, fine]


def _compose(name, subops, *, tol, dtype, device, notes, suggested):
    total_rt = sum(s.runtime_ms for s in subops)
    flops = sum(s.flops for s in subops)
    bytes_moved = sum(s.bytes_moved for s in subops)
    # Bound = the dominant sub-op's bound.
    dominant = max(subops, key=lambda s: s.runtime_ms)
    return Estimate(
        op_name=name, runtime_ms=total_rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=max((s.memory_peak_gb for s in subops), default=0.0),
        bound=dominant.bound, confidence="calibrated",
        n_kernel_launches=sum(s.n_kernel_launches for s in subops),
        suggested_config=suggested, subops=subops, notes=notes,
        tol=tol, dtype=dtype, device=device,
    )


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """End-to-end IVF-Flat cost (build + search) as a sub-op tree."""
    M, D, nlist, nprobe, k, nq, niter, train_size = common(shape, params)
    build = _compose(
        "ivf_flat.build",
        _build_subops(M, D, nlist, niter, train_size, dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"one-time build: M={M}, D={D}, nlist={nlist}, niter={niter}"],
        suggested={"nlist": nlist, "niter": niter, "train_size": train_size},
    )
    search = _compose(
        "ivf_flat.search",
        _search_subops(M, D, nlist, nprobe, k, nq, dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"per-batch search: nq={nq}, k={k}, nprobe={nprobe}"],
        suggested={"nprobe": nprobe, "k": k},
    )
    return _compose(
        "ivf_flat", [build, search], tol=tol, dtype=dtype, device=device,
        notes=[
            f"M={M}, D={D}, nlist={nlist}, nprobe={nprobe}, k={k}, nq={nq}",
            "recall fixed by (nlist, nprobe); iso-recall vs reference IVF-Flat",
        ],
        suggested={"nlist": nlist, "nprobe": nprobe, "k": k},
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Suggest ``(nlist, nprobe)`` -- FAISS-style ``nlist ~ sqrt(M)``."""
    M, _D, nlist, nprobe, k, _nq, _ni, _ts = common(shape, params)
    sqrt_nlist = int(max(1, round(M ** 0.5)))
    return {
        "nlist": params.get("nlist", sqrt_nlist) if params else sqrt_nlist,
        "nprobe": nprobe,
        "k": k,
    }


# ── GPU op-name shim ───────────────────────────────────────────────────────
# IVF-Flat has a single GPU backend (Triton); this is the canonical op_name
# the runtime reports on CUDA (see impl.route_op_name) and what
# info.estimate("ivf_flat_triton") resolves to. The torch fallback is a CPU
# reference, not a Pareto variant, so it is intentionally not registered.
def estimate_ivf_flat_triton(shape, params=None, tol=None, dtype="float32",
                             device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "ivf_flat_triton"
    est.tol = tol
    return est

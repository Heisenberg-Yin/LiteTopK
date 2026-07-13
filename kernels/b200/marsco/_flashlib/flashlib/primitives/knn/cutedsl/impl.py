"""Flash-KNN CuteDSL implementation: FA3 fully-fused path (x²-free).

This module exposes a SINGLE public entry point:

    cutedsl_flash_knn(x, c, k, *, autotune=False, ...)

It is FA3-style end-to-end fused: TMA bulk-tensor loads for X / C,
Hopper SM90 WGMMA for the cross-term GEMM, and a per-thread top-K
register heap -- no ``(B, N, M)`` distance matrix is ever materialised
in HBM. The kernel scores ``s = c_sq[m] − 2·<x, c>`` directly; the
``x_sq`` term is constant per row so dropping it preserves argmin-K
while saving one HBM tensor, one register-file pass, and a ``d >= 0``
clamp.

Indices only
------------

The fused kernel writes only ``(B, N, k) int32`` indices -- the true
``||x − c[idx]||²`` distances are recovered (cheaply) by
:func:`flashlib.kernels.distance.triton_knn_gather_sqdist` outside the
fused pass. This matches the contract of the new
:func:`flashlib.primitives.knn.flash_knn` entry point.

Public API
----------
``cutedsl_flash_knn(x, c, k)``
    FA3-style fully-fused KNN (Hopper-only). Falls back to the Triton
    :func:`flashlib.primitives.knn.triton.flash_knn_triton` path for
    any shape outside the FA3 sweet spot (B != 1, fp32 input,
    ``D % 16 != 0``, ``D > 512``, ``k > k_max``, or first-call compile
    failure). Returns ``(B, N, k) int32`` indices.

``cutedsl_available()``
    Cheap probe for whether the cutlass-dsl Python bindings imported.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.linalg.gemm import storage_dtype_for
from flashlib.primitives.knn.triton._row_norm import _get_or_compute_csq
from flashlib.primitives.knn.triton.dispatch import flash_knn_triton


# =============================================================================
# CuteDSL availability
# =============================================================================

_CUTEDSL_AVAILABLE = False
_CUTE_IMPORT_ERROR: Optional[Exception] = None


def _try_init_cutedsl() -> bool:
    """Lazy import probe; returns True iff cutlass-dsl is importable."""
    global _CUTEDSL_AVAILABLE, _CUTE_IMPORT_ERROR
    if _CUTEDSL_AVAILABLE:
        return True
    if _CUTE_IMPORT_ERROR is not None:
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        import cutlass.cute.runtime as cute_rt  # noqa: F401

        globals()["cutlass"] = cutlass
        globals()["cute"] = cute
        globals()["cute_rt"] = cute_rt
        _CUTEDSL_AVAILABLE = True
        return True
    except Exception as e:  # noqa: BLE001
        _CUTE_IMPORT_ERROR = e
        return False


def cutedsl_available() -> bool:
    return _try_init_cutedsl()


# =============================================================================
# Compiled-kernel + DLPack handle caches (shared across call sites).
# =============================================================================

_kernel_cache: dict = {}
_dlpack_cache: dict = {}


def _cached_from_dlpack(t: torch.Tensor):
    """Memoise ``cute_rt.from_dlpack`` per ``(ptr, shape, stride, dtype)``."""
    import cutlass.cute.runtime as cute_rt
    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    val = _dlpack_cache.get(key)
    if val is None:
        val = cute_rt.from_dlpack(t)
        _dlpack_cache[key] = val
    return val


def _trim_dlpack_cache(max_entries: int = 64):
    if len(_dlpack_cache) > max_entries:
        for k in list(_dlpack_cache.keys())[:-max_entries]:
            del _dlpack_cache[k]


# =============================================================================
# Shape-only heuristic (default; one CuteDSL compile per shape, no sweep)
# =============================================================================
#
# Picks the config along the **build vs search** axis. The per-shape
# autotune sweeps on H200 split cleanly:
#
#   * Build shapes (N >= 10K AND 0.5 <= M/N <= 2): non-WS with larger
#     tiles wins. WS3's pipelined topK doesn't pay off when every CTA
#     already has plenty of M-rows -- the pipeline depth costs
#     register / SMEM that hurts WGMMA throughput.
#       - N >= 50K (very-large build): BM=256
#       - else (10K-50K build):        BM=128
#       - K=4    -> ``perthread`` strategy
#       - K >= 16 -> ``sortmerge`` strategy
#       - Narrow D (<=128) wants perthread regardless of K
#         (autotune confirmed it on D=64, K=16).
#       - BN: 128 default; 64 when BM=256 (narrow D) or
#             when sortmerge + BM=256.
#   * Search shapes (small N OR M >> N): WS3 + smem_perthread wins
#     by a wide margin.
#       - K <= 16: BM=64  BN=128
#       - K  > 16: BM=128 BN=64
#
# All configs compile in ~5-8 s (one config) vs ~5-10 min for the
# full sweep.


def _heuristic_fa3_config(N: int, M: int, D: int, K_PAD: int) -> dict:
    is_build = (N >= 10_000) and (0.5 <= M / N <= 2.0)
    if is_build:
        BM = 256 if N >= 50_000 else 128
        # Strategy depends on (D, K) regime. The autotune-derived rule
        # from upstream flash-knn is:
        #   * Narrow D (<= 128) + K <= 16: ``perthread`` wins
        #     (autotune confirmed K=4, K=16 at D=64).
        #   * Narrow D (<= 128) + K >= 32: ``sortmerge`` -- the
        #     per-thread sequential bitonic in CuteDSL's perthread
        #     grows linearly with K_PAD and at K=32 + BM=256 the
        #     bitonic dominates the kernel (12x slower than sortmerge
        #     here on H200). flashlib's kernel `auto` strategy already
        #     flips at K>16, so this matches its empirical breakpoint.
        #   * Wide D (>= 192) + K <= 4: ``perthread`` (autotune).
        #   * Wide D (>= 192) + K >= 16: ``sortmerge`` (autotune
        #     confirmed BM=128 BN=128 wins at D=256 K ∈ {16, 32}).
        if D <= 128 and K_PAD <= 16:
            strat = "perthread"
            BN = 128 if BM == 128 else 64
        elif D > 128 and K_PAD <= 4:
            strat = "perthread"
            BN = 128
        else:
            strat = "sortmerge"
            BN = 128 if BM == 128 else 64
        return dict(
            BM=BM, BN=BN,
            use_ws=False, topk_strategy=strat,
            use_ws3=False, use_ws4=False, dist_stage=1,
        )

    if K_PAD <= 16:
        return dict(
            BM=64, BN=128,
            use_ws=True, topk_strategy="smem_perthread",
            use_ws3=True, use_ws4=False, dist_stage=3,
        )
    return dict(
        BM=128, BN=64,
        use_ws=True, topk_strategy="smem_perthread",
        use_ws3=True, use_ws4=False, dist_stage=3,
    )


# =============================================================================
# Per-shape autotune (opt-in -- sweeps the full FA3 grid).
# =============================================================================
#
# All gated strategies use native fp32 compares that work for the signed
# score (``c_sq - 2 * cross`` -- ``x_sq`` is never materialised).

_autotune_cache: dict = {}
_heuristic_cache: dict = {}


def _autotune_fa3(x2d, c2d, c_sq_1d, out_i, k_pad: int, *, verbose: bool = False):
    """Compile + bench every fitting (BM, BN, use_ws, strategy) FA3 config."""
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import cutlass.cute as cute_mod
    from cutlass.cute.runtime import from_dlpack as _from_dlpack

    from flashlib.primitives.knn.cutedsl.fused_kernel import HopperFlashKnnFused

    N, D = x2d.shape
    M = c2d.shape[0]
    smem_capacity = torch.cuda.get_device_properties(
        x2d.device
    ).shared_memory_per_block_optin
    bytes_per = 2

    def _fits(bm, bn):
        return (bm * D * bytes_per + bn * D * bytes_per + 1024) <= smem_capacity

    if k_pad <= 3:
        strategies = ("insert", "perthread", "sortmerge")
    else:
        strategies = ("perthread", "sortmerge")

    bms = (64, 128, 256)
    bns = (64, 128, 256)

    def _is_known_hang(bm, bn, use_ws, strat):
        # Pre-existing FA3 hang in the underlying kernel -- BM=256 WS2
        # candidates wedge the GPU during warmup on H200. They never win
        # the autotune anyway (BM=128 WS2 consistently dominates), so skip.
        if bm == 256 and use_ws:
            return True
        return False

    candidates = [
        (bm, bn, ws, strat, False, False, 1)
        for bm in bms
        for bn in bns
        for ws in (False, True)
        for strat in strategies
        if _fits(bm, bn) and not _is_known_hang(bm, bn, ws, strat)
    ]
    ws3_tiles = [(64, 64), (64, 128), (128, 64), (128, 128)]
    for bm, bn in ws3_tiles:
        if _fits(bm, bn):
            for stage in (3, 2):
                candidates.append(
                    (bm, bn, True, "smem_perthread", True, False, stage)
                )
    ws4_tiles = [(64, 64), (64, 128), (128, 64)]
    for bm, bn in ws4_tiles:
        if _fits(bm, bn) and bm * bn <= 128 * 64:
            candidates.append(
                (bm, bn, True, "smem_perthread", True, True, 2)
            )

    stream = cuda_drv.CUstream(0)
    best = None
    best_t = float("inf")

    for BM, BN, use_ws, strat, use_ws3, use_ws4, dist_stage in candidates:
        try:
            kernel = HopperFlashKnnFused(
                acc_dtype=cutlass.Float32,
                m_block_size=BM, n_block_size=BN,
                k_pad=k_pad, use_ws=use_ws,
                topk_strategy=strat,
                use_ws3=use_ws3, use_ws4=use_ws4, dist_stage=dist_stage,
            )
            compiled = cute_mod.compile(
                kernel,
                _from_dlpack(x2d), _from_dlpack(c2d),
                _from_dlpack(c_sq_1d), _from_dlpack(out_i),
                stream,
            )
        except Exception as exc:
            if verbose:
                ws_tag = (
                    "ws4" if use_ws4 else
                    ("ws3" if use_ws3 else ("ws2" if use_ws else "no "))
                )
                print(
                    f"  fa3 skip BM={BM} BN={BN} {ws_tag} "
                    f"strat={strat}: {str(exc).splitlines()[0][:80]}"
                )
            continue

        x_dl = _cached_from_dlpack(x2d)
        c_dl = _cached_from_dlpack(c2d)
        cs_dl = _cached_from_dlpack(c_sq_1d)
        oi_dl = _cached_from_dlpack(out_i)
        try:
            for _ in range(3):
                compiled(x_dl, c_dl, cs_dl, oi_dl, stream)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(8):
                compiled(x_dl, c_dl, cs_dl, oi_dl, stream)
            e.record()
            torch.cuda.synchronize()
            t_us = s.elapsed_time(e) / 8 * 1000
        except Exception:
            continue
        if verbose:
            tag = (
                "WS4" if use_ws4 else
                ("WS3" if use_ws3 else ("WS " if use_ws else "non"))
            )
            print(
                f"  fa3 BM={BM:>3} BN={BN:>3} {tag} stg={dist_stage} "
                f"strat={strat:<14}: {t_us:>7.0f} us"
            )
        if t_us < best_t:
            best_t = t_us
            best = (BM, BN, use_ws, strat, use_ws3, use_ws4, dist_stage, compiled)

    if best is None:
        raise RuntimeError(
            f"fa3 autotune found no working config for "
            f"N={N} D={D} M={M} k_pad={k_pad}"
        )
    return (
        best[0], best[1], best[2], best[3], best[4], best[5], best[6],
        best_t, best[7], stream,
    )


# =============================================================================
# Public entry point
# =============================================================================

def cutedsl_flash_knn(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    tol: Optional[float] = None,
    autotune: bool = False,
    autotune_verbose: bool = False,
    k_max: int = 32,
) -> torch.Tensor:
    """Hopper FA3 fully-fused kNN. Returns ``(B, N, k) int32`` indices.

    Args:
        x: (B, N, D) -- bf16 / fp16 native. fp32 input + ``tol`` selecting
            bf16/fp16 storage is cast once; otherwise fp32 falls back to
            Triton (FA3 path requires 16-bit storage).
        c: (B, M, D) -- same dtype as x, D contiguous.
        k: number of neighbours. ``k > k_max`` falls back to Triton.
        tol: residual tolerance.
            * ``None`` (default) -- preserve input dtype (EXACT path).
            * Otherwise the standard
              :func:`flashlib.linalg.gemm.storage_dtype_for` cast is
              applied to ``x``/``c`` once internally.
        autotune: if False (default), pick a single FA3 config from
            :func:`_heuristic_fa3_config` -- first call pays one
            CuteDSL compile (~5-8 s) instead of ~5-10 min for the full
            sweep. ``True`` runs the brute-force search and caches the
            winner (also racing against Triton to handle FA3-loses-to-
            tl.sort shapes).
        autotune_verbose: print per-candidate timings during autotune.
        k_max: gate threshold; ``k > k_max`` routes to Triton.

    Returns:
        ``(B, N, k) int32`` indices, sorted ascending by squared L2
        (ties broken by index). True distances per neighbour are
        recovered by
        :func:`flashlib.kernels.distance.triton_knn_gather_sqdist`.
    """
    assert x.is_cuda and c.is_cuda
    target_dtype = storage_dtype_for(tol)
    if target_dtype is not None and x.dtype != target_dtype:
        x = x.to(target_dtype)
        c = c.to(target_dtype)
    B, N, D = x.shape
    M = c.shape[1]
    assert c.shape == (B, M, D)
    assert 1 <= k <= M

    if (
        not _try_init_cutedsl()
        or B != 1
        or x.dtype not in (torch.float16, torch.bfloat16)
        or c.dtype != x.dtype
        or D % 16 != 0
        or D > 512
        or k > k_max
    ):
        return flash_knn_triton(x, c, k)

    # Shape-based fast-path opt-out (heuristic mode only): when Triton
    # clearly wins, don't pay the FA3 compile + dispatch tax. Autotune
    # mode races against Triton dynamically and caches the winner.
    #
    # Empirical regime (H200 / bf16) where Triton beats FA3:
    #   * N < 8192 -- FA3 warp-per-row epilogue under-fed.
    #   * D < 192 AND N < 50_000 -- narrow-D mid-N shapes where Triton
    #     keeps the cross on-chip cheaper than FA3's TMA + WGMMA stack.
    #     At very-large N (build shapes like (1, 100K, 100K, 64, *))
    #     FA3 non-WS BM=256 still beats Triton, so the carve-out is
    #     bounded.
    #   * M < 50K AND k < 32 AND not_build -- small DB + small K = FA3
    #     overhead dominates; Triton wins by 1.3-1.5x. The "not_build"
    #     carve-out lets large-N square shapes like (1, 10K, 10K, 256, *)
    #     through to FA3 where autotune finds wins.
    is_build = (N >= 10_000) and (0.5 <= M / N <= 2.0)
    if not autotune and (
        N < 8192
        or (D < 192 and N < 50_000)
        or (M < 50_000 and k < 32 and not is_build)
    ):
        return flash_knn_triton(x, c, k)

    c_sq = _get_or_compute_csq(c).view(M).contiguous()

    x2d = x.view(N, D)
    c2d = c.view(M, D)
    if not x2d.is_contiguous(): x2d = x2d.contiguous()
    if not c2d.is_contiguous(): c2d = c2d.contiguous()

    K_PAD = int(k)
    out_i = torch.empty((N, K_PAD), device=x.device, dtype=torch.int32)

    ac_key = (N, M, D, K_PAD, x.dtype)

    if not autotune:
        cached_h = _heuristic_cache.get(ac_key)
        if cached_h is None:
            import cuda.bindings.driver as cuda_drv
            import cutlass
            import cutlass.cute as cute_mod
            from cutlass.cute.runtime import from_dlpack as _from_dlpack
            from flashlib.primitives.knn.cutedsl.fused_kernel import (
                HopperFlashKnnFused,
            )

            cfg = _heuristic_fa3_config(N, M, D, K_PAD)
            stream = cuda_drv.CUstream(0)
            try:
                kernel = HopperFlashKnnFused(
                    acc_dtype=cutlass.Float32,
                    m_block_size=cfg["BM"], n_block_size=cfg["BN"],
                    k_pad=K_PAD, use_ws=cfg["use_ws"],
                    topk_strategy=cfg["topk_strategy"],
                    use_ws3=cfg["use_ws3"], use_ws4=cfg["use_ws4"],
                    dist_stage=cfg["dist_stage"],
                )
                compiled = cute_mod.compile(
                    kernel,
                    _from_dlpack(x2d), _from_dlpack(c2d),
                    _from_dlpack(c_sq), _from_dlpack(out_i),
                    stream,
                )
                cached_h = (compiled, stream)
                _heuristic_cache[ac_key] = cached_h
            except Exception:
                _heuristic_cache[ac_key] = "triton"
                return flash_knn_triton(x, c, k)
        elif cached_h == "triton":
            return flash_knn_triton(x, c, k)

        compiled, stream = cached_h
        try:
            compiled(
                _cached_from_dlpack(x2d), _cached_from_dlpack(c2d),
                _cached_from_dlpack(c_sq), _cached_from_dlpack(out_i),
                stream,
            )
        except Exception:
            return flash_knn_triton(x, c, k)
        return out_i.view(B, N, K_PAD)

    cached = _autotune_cache.get(ac_key)
    if cached is None:
        try:
            (
                BM, BN, use_ws, strat, use_ws3, use_ws4, dist_stage,
                _t_us, compiled, stream,
            ) = _autotune_fa3(
                x2d, c2d, c_sq, out_i, K_PAD,
                verbose=autotune_verbose,
            )
        except Exception:
            _autotune_cache[ac_key] = "triton"
            return flash_knn_triton(x, c, k)

        # Race against the Triton flash_knn_triton path so the wrapper
        # benefits from whichever Triton kernel (M-split or single-pass)
        # is best for this shape.
        try:
            for _ in range(3):
                flash_knn_triton(x, c, k)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(8):
                flash_knn_triton(x, c, k)
            e.record()
            torch.cuda.synchronize()
            tri_us = s.elapsed_time(e) / 8 * 1000
        except Exception:
            tri_us = float("inf")
        if tri_us < _t_us:
            _autotune_cache[ac_key] = "triton"
            if autotune_verbose:
                print(
                    f"  fa3 vs triton: triton wins "
                    f"({tri_us:.0f} us < {_t_us:.0f} us) -- caching fallback"
                )
            return flash_knn_triton(x, c, k)
        _autotune_cache[ac_key] = (compiled, stream)
        if autotune_verbose:
            tag = (
                "WS4" if use_ws4 else
                ("WS3" if use_ws3 else ("WS" if use_ws else "non-WS"))
            )
            stg = f" stg={dist_stage}" if (use_ws3 or use_ws4) else ""
            print(
                f"  fa3 winner: BM={BM} BN={BN} {tag}{stg} "
                f"strat={strat} ({_t_us:.0f} us, triton {tri_us:.0f} us)"
            )
    elif cached == "triton":
        return flash_knn_triton(x, c, k)
    else:
        compiled, stream = cached

    try:
        compiled(
            _cached_from_dlpack(x2d), _cached_from_dlpack(c2d),
            _cached_from_dlpack(c_sq), _cached_from_dlpack(out_i),
            stream,
        )
    except Exception:
        return flash_knn_triton(x, c, k)

    return out_i.view(B, N, K_PAD)

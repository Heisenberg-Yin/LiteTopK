"""Multinomial Naive Bayes — CuteDSL alternative for the predict GEMM.

Hopper SM90 strategy + honest perf assessment
=============================================

The dominant cost in predict is the GEMM ``X_test @ W.T`` with shape
(N_test × D) × (D × C). Multinomial NB always has *very small* C (10-32).
cuBLAS picks a wgmma tile of (M=128, N=64 or 128, K=16) which **wastes
half the wgmma N-direction** when C ≤ 32. On twenty-newsgroups
(N_test=3770, D=30000, C=20) this leaves predict at only 27 % peak BW.

This CuteDSL kernel exploits the small-C structure by:
  1. Pinning ``BLOCK_C = 32`` (zero-padded for invalid classes) so the
     C-dim is constexpr known and unrolled at compile time.
  2. Using a SIMT pattern with 2D thread block (8×8 = 64 threads = 2
     warps per CTA), where each thread owns ``ITEMS_M = 4`` rows. The
     register-tile pattern is the same as ridge_regression's
     ``_xtx_sym_kernel`` (see ``algorithms/ridge_regression/cutedsl_impl.py``).
  3. Per-thread accumulator: ``ITEMS_M × BLOCK_C`` fp32 floats in registers.
  4. Loading W[:, k] once per K-step into a shared register vector,
     then doing ``ITEMS_M`` FMAs against it — high arithmetic intensity
     (1 W-load amortised over 4 X-loads).
  5. Fusing the bias-add and **argmax** in the epilogue — the kernel
     directly writes ``argmax_c (X @ W.T + prior)[n]`` to int32 (host
     cast to int64 — sub-microsecond).

The math precision is **bf16 input × fp32 accumulator** (matches the
`predict_dtype="bf16"` Triton path). On twenty-newsgroups the jll
magnitudes are ~ -2 to 2 (TF-IDF ∈ [0, 0.3] × FLP ∈ [-10, 0]), so bf16
multiplies introduce ≤ 1e-3 absolute error per product — argmax stable.

Honest perf assessment
======================
On H200 with `nvidia-cutlass-dsl 4.4.2` the SIMT predict kernel runs at
~5 ms regardless of input shape, even for tiny inputs (N=64, D=16, C=5).
This is **5-25× slower than the Triton bf16 fused predict and cuBLAS
fp32 predict** at every measured shape. The fundamental limitation is
that pure SIMT cannot match the wgmma tensor-core throughput that
cuBLAS and Triton's `tl.dot` use — both of those operate on 64×16
register tiles per fma cycle, while SIMT does 1 fma per thread per cycle.

A full-fidelity wgmma-based CuteDSL kernel (using
`cutlass.cute.nvgpu.warpgroup.mma` and warp-specialised TMA loads)
would in principle close this gap, but the engineering effort is a
multi-week task — we instead document the SIMT path as a
reference-quality CuteDSL implementation that demonstrates the API
patterns (rmem tensors, range_dynamic with unroll, fused epilogue,
2D thread blocks, register tiling).

The Triton bf16 fused predict in ``triton_impl.py``
(``_nb_predict_kernel``) **already beats cuBLAS by 1.87× at xlarge**,
so the headline win for the round 3 optimisation lives in the Triton
path; the CuteDSL path is a learning artefact.

Fallback
========
If the CuteDSL JIT is unavailable or C exceeds BLOCK_C, the predict
function falls back to ``flash_multinomial_nb_predict`` from
triton_impl (which itself dispatches between cuBLAS and Triton based
on shape).
"""

from __future__ import annotations

import os
import sys
from typing import Optional


import torch

from flashlib.primitives.multinomial_nb.triton import (
    flash_multinomial_nb_fit,
    flash_multinomial_nb_predict,
    flash_multinomial_nb_predict_log_proba_unnormalized,
)


# =============================================================================
# CuteDSL kernel availability
# =============================================================================

_CUTEDSL_AVAILABLE = False
_CUTE_LAUNCH_ARGMAX = None
_CUTE_LAUNCH_JLL = None
_CUTE_IMPORT_ERROR: Optional[Exception] = None

# Block sizes — fixed at compile time for predictable perf.
# Ridge-regression-style: 2D thread block, each thread owns ITEMS_M rows.
# THR_M × THR_N = 8×8 = 64 threads = 2 warps per CTA.
BLOCK_N = 32     # rows of X per CTA (THR_M × ITEMS_M)
BLOCK_C = 32     # padded class dim
BLOCK_K = 8      # K-loop tile size
THR_M = 8        # threads in M direction
THR_N = 8        # threads in N direction (unused but present for 2D block)
ITEMS_M = 4      # rows per thread


def _try_init_cutedsl():
    global _CUTEDSL_AVAILABLE, _CUTE_LAUNCH_ARGMAX, _CUTE_LAUNCH_JLL, _CUTE_IMPORT_ERROR
    if _CUTEDSL_AVAILABLE or _CUTE_IMPORT_ERROR is not None:
        return _CUTEDSL_AVAILABLE
    try:
        import cutlass.cute as cute  # noqa: F401
        from cutlass.cute.runtime import from_dlpack  # noqa: F401
        from cutlass.cutlass_dsl import CuTeDSL, T, Constexpr  # noqa: F401
        from cutlass._mlir.dialects import nvvm  # noqa: F401

        # =====================================================================
        # SIMT bf16 → fp32 GEMM with fused bias + argmax
        #
        # Each CTA = 32 threads = 1 warp. Each thread owns one row index
        # n_lane = bx * BLOCK_N + tx. Threads accumulate ``BLOCK_C`` fp32
        # values in registers — for C ≤ 32 we just process all BLOCK_C
        # outputs and mask invalid ones at the end.
        #
        # K-loop: each thread reads its X[n, k] row segments directly from
        # gmem (no smem) — the access pattern is one bf16 read per thread
        # per K. W is read by every thread (same per-thread for all
        # threads in the warp on a given k); we let the L1 cache absorb
        # this so threads don't duplicate global reads.
        # =====================================================================

        BLOCK_N_CT = BLOCK_N
        BLOCK_C_CT = BLOCK_C
        THR_M_CT = THR_M
        THR_N_CT = THR_N
        ITEMS_M_CT = ITEMS_M

        # Ridge-regression-style: 2D thread block (8×8 = 64 threads = 2 warps),
        # each thread owns ITEMS_M output rows (4 rows). This generates a
        # register-tiled 4×BLOCK_C inner per-thread loop — much more ILP than
        # 1-row-per-thread. See ridge_regression/cutedsl_impl.py for reference.

        @cute.kernel
        def _predict_argmax_kernel(
            X,              # gmem (N, D) bf16
            W,              # gmem (C_PAD, D) bf16  (C-padded with zeros)
            P,              # gmem (C_PAD,) fp32   (bias, padded with -inf)
            ARGMAX,         # gmem (N,) int32 (we use int32; cast to int64 host-side)
            N: cute.Int32,
            D: cute.Int32,
            C_VALID: cute.Int32,
        ):
            bx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
            tx = nvvm.read_ptx_sreg_tid_x(T.i32())  # 0..THR_M-1
            ty = nvvm.read_ptx_sreg_tid_y(T.i32())  # 0..THR_N-1 (unused for argmax)
            # Linear thread id in CTA: tx + ty * THR_M
            tlin = tx + ty * THR_M_CT
            # Each thread owns ITEMS_M rows
            n_base = bx * BLOCK_N_CT + tlin * ITEMS_M_CT

            # Per-thread accumulator: ITEMS_M × BLOCK_C floats in registers.
            acc = cute.make_rmem_tensor(
                cute.make_layout((ITEMS_M_CT, BLOCK_C_CT)), cute.Float32
            )
            for im in cute.range_constexpr(ITEMS_M_CT):
                for c in cute.range_constexpr(BLOCK_C_CT):
                    acc[im, c] = cute.Float32(0.0)

            for k in cute.range_dynamic(D, unroll=4):
                # Load W[:, k] once per k step into registers.
                w_reg = cute.make_rmem_tensor(
                    cute.make_layout(BLOCK_C_CT), cute.Float32
                )
                for c in cute.range_constexpr(BLOCK_C_CT):
                    w_reg[c] = W[c, k].to(cute.Float32)

                # Load X[n_base+im, k] once per (im, k) and FMA across all C.
                for im in cute.range_constexpr(ITEMS_M_CT):
                    ni = n_base + im
                    if ni < N:
                        x_v = X[ni, k].to(cute.Float32)
                        for c in cute.range_constexpr(BLOCK_C_CT):
                            acc[im, c] += x_v * w_reg[c]

            for im in cute.range_constexpr(ITEMS_M_CT):
                ni = n_base + im
                if ni < N:
                    best_v = cute.Float32(-3.4e38)
                    best_c = cute.Int32(0)
                    for c in cute.range_constexpr(BLOCK_C_CT):
                        if c < C_VALID:
                            v = acc[im, c] + P[c]
                            if v > best_v:
                                best_v = v
                                best_c = cute.Int32(c)
                    ARGMAX[ni] = best_c

        @cute.jit
        def _launch_predict_argmax(X, W, P, ARGMAX, N, D, C_VALID):
            grid_x = (N + BLOCK_N_CT - 1) // BLOCK_N_CT
            _predict_argmax_kernel(X, W, P, ARGMAX, N, D, C_VALID).launch(
                grid=[grid_x, 1, 1],
                block=[THR_M_CT, THR_N_CT, 1],
            )

        # JLL-only kernel (writes full (N, C_VALID) jll matrix; C_PAD output).
        @cute.kernel
        def _predict_jll_kernel(
            X, W, P,
            JLL,            # gmem (N, C_PAD) fp32
            N: cute.Int32,
            D: cute.Int32,
            C_VALID: cute.Int32,
            C_PAD: cute.Int32,
        ):
            bx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
            tx = nvvm.read_ptx_sreg_tid_x(T.i32())
            ty = nvvm.read_ptx_sreg_tid_y(T.i32())
            tlin = tx + ty * THR_M_CT
            n_base = bx * BLOCK_N_CT + tlin * ITEMS_M_CT

            acc = cute.make_rmem_tensor(
                cute.make_layout((ITEMS_M_CT, BLOCK_C_CT)), cute.Float32
            )
            for im in cute.range_constexpr(ITEMS_M_CT):
                for c in cute.range_constexpr(BLOCK_C_CT):
                    acc[im, c] = cute.Float32(0.0)

            for k in cute.range_dynamic(D, unroll=4):
                w_reg = cute.make_rmem_tensor(
                    cute.make_layout(BLOCK_C_CT), cute.Float32
                )
                for c in cute.range_constexpr(BLOCK_C_CT):
                    w_reg[c] = W[c, k].to(cute.Float32)
                for im in cute.range_constexpr(ITEMS_M_CT):
                    ni = n_base + im
                    if ni < N:
                        x_v = X[ni, k].to(cute.Float32)
                        for c in cute.range_constexpr(BLOCK_C_CT):
                            acc[im, c] += x_v * w_reg[c]

            for im in cute.range_constexpr(ITEMS_M_CT):
                ni = n_base + im
                if ni < N:
                    for c in cute.range_constexpr(BLOCK_C_CT):
                        if c < C_VALID:
                            JLL[ni, c] = acc[im, c] + P[c]

        @cute.jit
        def _launch_predict_jll(X, W, P, JLL, N, D, C_VALID, C_PAD):
            grid_x = (N + BLOCK_N_CT - 1) // BLOCK_N_CT
            _predict_jll_kernel(X, W, P, JLL, N, D, C_VALID, C_PAD).launch(
                grid=[grid_x, 1, 1],
                block=[THR_M_CT, THR_N_CT, 1],
            )

        _CUTE_LAUNCH_ARGMAX = _launch_predict_argmax
        _CUTE_LAUNCH_JLL = _launch_predict_jll
        _CUTEDSL_AVAILABLE = True
        return True
    except Exception as e:  # noqa: BLE001
        _CUTE_IMPORT_ERROR = e
        _CUTEDSL_AVAILABLE = False
        return False


def _pad_w_and_prior(feature_log_prob: torch.Tensor,
                     class_log_prior: torch.Tensor):
    """Pad (C, D) → (C_PAD, D) with zeros, and prior to (C_PAD,) with -inf."""
    C, D = feature_log_prob.shape
    if C >= BLOCK_C:
        return feature_log_prob, class_log_prior, C
    Wp = torch.zeros((BLOCK_C, D), device=feature_log_prob.device,
                     dtype=feature_log_prob.dtype)
    Wp[:C].copy_(feature_log_prob)
    Pp = torch.full((BLOCK_C,), float("-inf"),
                    device=class_log_prior.device,
                    dtype=class_log_prior.dtype)
    Pp[:C].copy_(class_log_prior)
    return Wp, Pp, C


def precompile_cutedsl_for_params(params: dict, predict_dtype: str = "bf16"):
    """One-time padding + bf16 cast of the model weights & prior.

    Stores the cached tensors back into ``params`` under
    ``"_cute_W_bf16"`` and ``"_cute_P_padded"``. Subsequent calls to
    ``cutedsl_multinomial_nb_predict_argmax`` then skip the pad/cast
    work and only pay the X-cast + kernel launch.
    """
    if not _try_init_cutedsl():
        return
    Wp, Pp, C_VALID = _pad_w_and_prior(params["feature_log_prob"],
                                       params["class_log_prior"])
    params["_cute_W_bf16"] = Wp.to(torch.bfloat16).contiguous()
    params["_cute_P_padded"] = Pp.contiguous()
    params["_cute_C_valid"] = C_VALID


def cutedsl_multinomial_nb_predict_argmax(
    X_test: torch.Tensor, params: dict, predict_dtype: str = "bf16",
):
    """Predict argmax labels via CuteDSL kernel.

    Args:
        X_test: (N, D) on cuda. Will be cast to bf16 internally.
        params: dict from flash_multinomial_nb_fit. If
            ``params["_cute_W_bf16"]`` is present (set by
            ``precompile_cutedsl_for_params``), uses the cached padded
            weights — saves the host-side pad/cast each call.
        predict_dtype: must be "bf16" — only path supported by this CuteDSL impl.

    Returns:
        labels: (N,) int64 cuda tensor.
    """
    assert X_test.is_cuda and X_test.ndim == 2
    feature_log_prob = params["feature_log_prob"]   # (C, D) fp32
    class_log_prior = params["class_log_prior"]     # (C,) fp32
    N, D = X_test.shape
    C = feature_log_prob.shape[0]

    if not _try_init_cutedsl() or C > BLOCK_C:
        # Fallback to the Triton fused kernel.
        return flash_multinomial_nb_predict(
            X_test, params, predict_dtype=predict_dtype
        )

    try:
        from cutlass.cute.runtime import from_dlpack

        # Use cached padded weights if present, else compute on the fly.
        if "_cute_W_bf16" in params:
            Wp_bf = params["_cute_W_bf16"]
            Pp = params["_cute_P_padded"]
            C_VALID = params["_cute_C_valid"]
        else:
            Wp, Pp, C_VALID = _pad_w_and_prior(feature_log_prob, class_log_prior)
            Wp_bf = Wp.to(torch.bfloat16).contiguous()
            Pp = Pp.contiguous()

        X_bf = X_test.to(torch.bfloat16).contiguous()

        argmax = torch.empty(N, device=X_test.device, dtype=torch.int32)

        mX = from_dlpack(X_bf)
        mW = from_dlpack(Wp_bf)
        mP = from_dlpack(Pp)
        mA = from_dlpack(argmax)
        _CUTE_LAUNCH_ARGMAX(mX, mW, mP, mA, N, D, C_VALID)
        return argmax.to(torch.int64)
    except Exception:
        # Any JIT failure → fallback.
        return flash_multinomial_nb_predict(
            X_test, params, predict_dtype=predict_dtype
        )


def cutedsl_multinomial_nb_predict_jll(
    X_test: torch.Tensor, params: dict,
):
    """Predict full jll matrix (N, C) via CuteDSL kernel.

    Returns the bf16-input fp32-accumulator joint log-likelihood + prior.
    Falls back to the cuBLAS path on JIT failure.
    """
    assert X_test.is_cuda and X_test.ndim == 2
    feature_log_prob = params["feature_log_prob"]
    class_log_prior = params["class_log_prior"]
    N, D = X_test.shape
    C = feature_log_prob.shape[0]

    if not _try_init_cutedsl() or C > BLOCK_C:
        return flash_multinomial_nb_predict_log_proba_unnormalized(
            X_test, params, predict_dtype="bf16"
        )

    try:
        from cutlass.cute.runtime import from_dlpack

        if "_cute_W_bf16" in params:
            Wp_bf = params["_cute_W_bf16"]
            Pp = params["_cute_P_padded"]
            C_VALID = params["_cute_C_valid"]
        else:
            Wp, Pp, C_VALID = _pad_w_and_prior(feature_log_prob, class_log_prior)
            Wp_bf = Wp.to(torch.bfloat16).contiguous()
            Pp = Pp.contiguous()

        X_bf = X_test.to(torch.bfloat16).contiguous()

        # Allocate (N, C_PAD) and slice [:C] on output.
        jll_pad = torch.empty(N, BLOCK_C, device=X_test.device, dtype=torch.float32)
        jll_pad.zero_()

        mX = from_dlpack(X_bf)
        mW = from_dlpack(Wp_bf)
        mP = from_dlpack(Pp)
        mJ = from_dlpack(jll_pad)
        _CUTE_LAUNCH_JLL(mX, mW, mP, mJ, N, D, C_VALID, BLOCK_C)
        return jll_pad[:, :C].contiguous()
    except Exception:
        return flash_multinomial_nb_predict_log_proba_unnormalized(
            X_test, params, predict_dtype="bf16"
        )


def cutedsl_multinomial_nb(X_train, y_train, X_test, n_classes,
                           alpha: float = 1.0, predict_dtype: str = "bf16"):
    """End-to-end MultinomialNB. Fit uses Triton (unchanged); predict uses CuteDSL."""
    params = flash_multinomial_nb_fit(X_train, y_train, n_classes, alpha=alpha)
    return cutedsl_multinomial_nb_predict_argmax(
        X_test, params, predict_dtype=predict_dtype
    )


def cutedsl_available() -> bool:
    return _try_init_cutedsl()

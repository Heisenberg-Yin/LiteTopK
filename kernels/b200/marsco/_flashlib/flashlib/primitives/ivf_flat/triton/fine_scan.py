"""IVF-Flat fused fine-scan kernel (elementwise / online path).

This is the IVF analogue of the flash_knn iterative-insert kernel: a
generalization from "scan a contiguous corpus chunk" to "scan a *ragged*
inverted list" addressed through the CSR ``list_offsets``.

For each ``(query, probed-list[, split])`` the kernel:

  1. reads the list's half-open row range
     ``[offsets[c] + lo, offsets[c] + hi)`` of the cell-contiguous
     ``data`` (fully coalesced, no gather of an index array),
  2. computes the **true** squared-L2 distance ``||q - data[m]||^2``
     directly in fp32 (no x²-free + post-hoc gather needed: this path is
     bandwidth-bound on the candidate reads, where tensor cores buy
     nothing, and the direct difference avoids ``x² + y² - 2xy``
     cancellation), and
  3. maintains an on-chip top-K register heap with the flash_knn
     argmin/argmax insert loop.

Hard contract (identical to flash_knn): **no ``(nq x candidates)``
distance matrix is ever materialised in HBM** -- only the standard
flash-decode partial buffer ``(nq, nprobe * n_splits, TOPK_PAD)`` is
written, then merged host-side with a single ``torch.topk``.

One query per program (``program_id(0)``); the ``probe`` and ``split``
grid axes (``program_id(1/2)``) give parallelism. ``n_splits`` chops long
lists into wave-filling sub-ranges for the small-batch / online regime
where ``nq * nprobe`` alone would not saturate the SMs -- the same
wave-count idea as flash_knn's M-split decode. The batch-throughput
regime (many queries sharing a list) is handled by the companion
group-by-list GEMM kernel in :mod:`...ivf_flat.triton.fine_scan_gemm`.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import _next_pow2


@triton.jit
def _ivf_fine_scan_kernel(
    q_ptr, data_ptr, probed_ptr, offsets_ptr,
    pv_ptr, pi_ptr,
    stride_q_n, stride_q_d,
    stride_d_m, stride_d_d,
    stride_pr_n, stride_pr_p,
    stride_pv_n, stride_pv_p, stride_pv_k,
    stride_pi_n, stride_pi_p, stride_pi_k,
    M,
    D: tl.constexpr, K: tl.constexpr,
    N_SPLITS: tl.constexpr,
    BM: tl.constexpr, BD: tl.constexpr,
    TOPK_PAD: tl.constexpr, MAX_STEPS: tl.constexpr, MAX_CHUNKS: tl.constexpr,
):
    """Grid: ``(nq, nprobe, N_SPLITS)``. One query, one probed list, one split.

    Writes ``(TOPK_PAD,)`` partial vals/idxs to slot
    ``pid_p * N_SPLITS + pid_s`` for query ``pid_n``. ``idx`` is the
    *stored-row* position into ``data`` (mapped to original ids host-side).
    """
    pid_n = tl.program_id(0).to(tl.int64)
    pid_p = tl.program_id(1)
    pid_s = tl.program_id(2)

    # Which inverted list this (query, probe-slot) scans.
    c = tl.load(probed_ptr + pid_n * stride_pr_n + pid_p * stride_pr_p).to(tl.int64)
    start = tl.load(offsets_ptr + c)
    end = tl.load(offsets_ptr + c + 1)
    list_len = end - start

    # Per-split sub-range within the list.
    split_len = (list_len + N_SPLITS - 1) // N_SPLITS
    lo = pid_s.to(tl.int64) * split_len
    hi = tl.minimum(list_len, lo + split_len)

    bm_range = tl.arange(0, BM)
    k_range = tl.arange(0, TOPK_PAD)
    topk_vals = tl.full([TOPK_PAD], float("inf"), dtype=tl.float32)
    topk_idx = tl.full([TOPK_PAD], -1, dtype=tl.int32)
    topk_max = tl.max(topk_vals)

    for ci in range(MAX_CHUNKS):
        within = lo + ci * BM + bm_range.to(tl.int64)        # (BM,) within-list idx
        valid = within < hi                                  # (BM,)
        pos = start + within                                 # (BM,) stored-row pos
        pos_safe = tl.minimum(tl.maximum(pos, 0), M - 1)

        acc = tl.zeros([BM], dtype=tl.float32)
        for d_start in range(0, D, BD):
            d_offs = d_start + tl.arange(0, BD)
            d_mask = d_offs < D
            q_vec = tl.load(
                q_ptr + pid_n * stride_q_n + d_offs.to(tl.int64) * stride_q_d,
                mask=d_mask, other=0.0,
            ).to(tl.float32)                                 # (BD,)
            c_blk = tl.load(
                data_ptr + pos_safe[:, None] * stride_d_m
                + d_offs[None, :].to(tl.int64) * stride_d_d,
                mask=valid[:, None] & d_mask[None, :], other=0.0,
            ).to(tl.float32)                                 # (BM, BD)
            diff = q_vec[None, :] - c_blk
            acc += tl.sum(diff * diff, axis=1)               # (BM,)

        score = tl.where(valid, acc, float("inf"))           # (BM,)
        base = start + lo + ci * BM                          # scalar stored-row base

        # Iterative argmin-insert into the register top-K (flash_knn body,
        # collapsed to a single query row).
        chunk_best = tl.min(score)
        if chunk_best < topk_max:
            _active = tl.full([1], 1, dtype=tl.int32)
            for _step in range(MAX_STEPS):
                if tl.max(_active) > 0:
                    row_min = tl.min(score)
                    row_arg = tl.argmin(score, axis=0)
                    if row_min < topk_max:
                        worst = tl.argmax(topk_vals, axis=0)
                        repl = k_range == worst
                        topk_vals = tl.where(repl, row_min, topk_vals)
                        topk_idx = tl.where(
                            repl, (base + row_arg.to(tl.int64)).to(tl.int32), topk_idx
                        )
                        topk_max = tl.max(topk_vals)
                        score = tl.where(bm_range == row_arg, float("inf"), score)
                    else:
                        _active = tl.full([1], 0, dtype=tl.int32)

    pslot = pid_p * N_SPLITS + pid_s
    tl.store(
        pv_ptr + pid_n * stride_pv_n + pslot * stride_pv_p + k_range * stride_pv_k,
        topk_vals,
    )
    tl.store(
        pi_ptr + pid_n * stride_pi_n + pslot * stride_pi_p + k_range * stride_pi_k,
        topk_idx,
    )


def _pick_n_splits(nq: int, nprobe: int, max_list_len: int, sm_count: int) -> int:
    """Chop lists into enough splits to roughly fill ~2 waves of SMs.

    For large query batches ``nq * nprobe`` already saturates the GPU and
    ``n_splits == 1`` (each program owns a whole list). For tiny batches
    (online / single-query search) we raise ``n_splits`` so the SMs are
    not left idle -- the flash-decode wave-targeting idea, applied to the
    list axis.
    """
    base = max(1, nq * nprobe)
    target = 2 * max(sm_count, 1)
    n_splits = (target + base - 1) // base
    n_splits = max(1, min(n_splits, 64, max(1, max_list_len)))
    return int(n_splits)


def ivf_fine_scan(
    Qp: torch.Tensor,
    data: torch.Tensor,
    probed: torch.Tensor,
    list_offsets: torch.Tensor,
    k: int,
    *,
    max_list_len: int,
    BM: int = 128,
    n_splits: Optional[int] = None,
):
    """Launch the fused fine-scan + host-side merge.

    Args:
        Qp: ``(nq, Dp)`` queries (working dtype; padded to ``Dp``).
        data: ``(M, Dp)`` cell-contiguous database.
        probed: ``(nq, nprobe)`` int32 inverted-list ids (from coarse search).
        list_offsets: ``(nlist + 1,)`` int64 CSR offsets.
        k: neighbours per query.
        max_list_len: max inverted-list length (bounds the static chunk loop).

    Returns:
        ``(vals, pos)`` -- ``vals`` ``(nq, k)`` true squared-L2 (fp32),
        ``pos`` ``(nq, k)`` int64 stored-row positions into ``data``
        (``-1`` where fewer than ``k`` candidates were available).
    """
    assert Qp.is_cuda and data.is_cuda and probed.is_cuda and list_offsets.is_cuda
    nq, Dp = Qp.shape
    M = data.shape[0]
    nprobe = probed.shape[1]

    Qp = Qp.contiguous()
    data = data.contiguous()
    probed = probed.contiguous().to(torch.int32)
    list_offsets = list_offsets.contiguous().to(torch.int64)

    if n_splits is None:
        sm_count = torch.cuda.get_device_properties(Qp.device).multi_processor_count
        n_splits = _pick_n_splits(nq, nprobe, max_list_len, sm_count)
    n_splits = max(1, min(int(n_splits), max(1, max_list_len)))

    TOPK_PAD = _next_pow2(k)
    BD = min(_next_pow2(Dp), 128)
    MAX_STEPS = min(k, BM)
    per_split_max = (max_list_len + n_splits - 1) // n_splits
    MAX_CHUNKS = max(1, (per_split_max + BM - 1) // BM)

    P = nprobe * n_splits
    partial_vals = torch.full((nq, P, TOPK_PAD), float("inf"),
                              device=Qp.device, dtype=torch.float32)
    partial_idx = torch.full((nq, P, TOPK_PAD), -1,
                             device=Qp.device, dtype=torch.int32)

    grid = (nq, nprobe, n_splits)
    _ivf_fine_scan_kernel[grid](
        Qp, data, probed, list_offsets,
        partial_vals, partial_idx,
        Qp.stride(0), Qp.stride(1),
        data.stride(0), data.stride(1),
        probed.stride(0), probed.stride(1),
        partial_vals.stride(0), partial_vals.stride(1), partial_vals.stride(2),
        partial_idx.stride(0), partial_idx.stride(1), partial_idx.stride(2),
        M,
        D=Dp, K=k,
        N_SPLITS=n_splits,
        BM=BM, BD=BD,
        TOPK_PAD=TOPK_PAD, MAX_STEPS=MAX_STEPS, MAX_CHUNKS=MAX_CHUNKS,
        num_warps=4,
    )

    # Stage-2: merge the P partial top-Ks per query (no HBM cross matrix).
    pv = partial_vals.view(nq, -1)
    pi = partial_idx.view(nq, -1)
    vals, sel = pv.topk(k, dim=-1, largest=False, sorted=True)
    pos = pi.gather(-1, sel).to(torch.int64)
    pos = torch.where(vals.isinf(), torch.full_like(pos, -1), pos)
    return vals, pos


__all__ = ["ivf_fine_scan", "_pick_n_splits"]

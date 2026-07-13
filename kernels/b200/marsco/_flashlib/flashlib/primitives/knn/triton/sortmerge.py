"""flash_knn — Triton sort-merge top-K kernel (x²-free, signed score).

The kernel computes ``argmin-K`` over the *signed* shifted distance

    s(n, m) = ||c[m]||² − 2·⟨x[n], c[m]⟩      (== ||x[n] − c[m]||² − ||x[n]||²)

The ``||x||²`` term is constant per row and so does not affect the top-K
selection over ``m``. Dropping it eliminates the ``x_sq`` HBM tensor, its
precompute pass, its per-tile load, one fp32 ADD per accumulator element,
and the ``dist >= 0`` underflow clamp. The trade-off is that ``s`` is no
longer non-negative; the packed-uint64 sort-merge top-K therefore uses
an IEEE-sortable u32 transform (positives flip just the sign bit, negatives
flip every bit) so ascending uint32 order matches ascending fp32 order on
signed inputs.

Public name: ``_flash_knn_sortmerge_kernel`` — called from
``flashlib/primitives/knn/triton/dispatch.py``. Stage 1 writes signed
fp32 scores + int32 idxs to a partial buffer; Stage 2 either returns
``partial[:, 0]`` directly (single split) or runs ``torch.topk(...,
largest=False)`` to reduce across splits.

Same kernel covers both ``small-N M-split flash-decode`` (multiple
splits per query block) and ``large-N single-pass`` (M_PER_SPLIT == M)
— the host-side dispatcher just chooses the grid + M_PER_SPLIT.
"""
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import (
    _INF_PACKED,
    _fp32_to_sortable_u32,
    _sortable_u32_to_fp32,
)


@triton.jit
def _flash_knn_sortmerge_kernel(
    x_ptr, c_ptr,
    partial_val_ptr, partial_idx_ptr,
    stride_x_b, stride_x_n, stride_x_d,
    stride_c_b, stride_c_m, stride_c_d,
    stride_pv_b, stride_pv_s, stride_pv_n, stride_pv_k,
    stride_pi_b, stride_pi_s, stride_pi_n, stride_pi_k,
    N: tl.constexpr, M: tl.constexpr,
    D: tl.constexpr, K: tl.constexpr,
    M_PER_SPLIT: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr,
    D_INNER: tl.constexpr,
    TOPK_PAD: tl.constexpr,
    NUM_STAGES_PIPE: tl.constexpr = 2,
):
    """Packed-uint64 sort-merge top-K with IEEE-sortable u32 score bits.

    Grid: ``(num_m_splits, ceil(N/BN), B)``. ``BM == TOPK_PAD``.

    Two compile-time paths gated on ``D_INNER >= D``:
      * persistent x tile + pipelined M-loop (single tile covers all of D)
      * D-split with sequential M-loop (D-chunked GEMM for wide D)

    ``NUM_STAGES_PIPE`` controls the M-loop ``tl.range`` pipelining
    depth (host-side dispatch tunes it per shape; defaults to 2 if not
    provided -- matches the pre-port hard-coded value).
    """
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BN
    n_offs = (n_start + tl.arange(0, BN)).to(tl.int64)
    n_mask = n_offs < N

    topk_packed = tl.full([BN, TOPK_PAD], _INF_PACKED, dtype=tl.uint64)
    bm_range = tl.arange(0, BM)
    m_base = pid_s.to(tl.int64) * M_PER_SPLIT

    if D_INNER >= D:
        d_offs = tl.arange(0, D_INNER).to(tl.int64)
        d_mask = d_offs < D
        x_tile = tl.load(
            x_ptr + pid_b * stride_x_b
            + n_offs[:, None] * stride_x_n
            + d_offs[None, :] * stride_x_d,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        for m_local in tl.range(0, M_PER_SPLIT, BM, num_stages=NUM_STAGES_PIPE):
            m_start = m_base + m_local
            m_offs = m_start + bm_range.to(tl.int64)
            m_mask = m_offs < M

            c_tile = tl.load(
                c_ptr + pid_b * stride_c_b
                + m_offs[:, None] * stride_c_m
                + d_offs[None, :] * stride_c_d,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0)

            c_f = c_tile.to(tl.float32)
            c_sq_tile = tl.sum(c_f * c_f, axis=1)
            cross = tl.dot(x_tile, tl.trans(c_tile)).to(tl.float32)
            score = c_sq_tile[None, :] - 2.0 * cross
            score = tl.where(m_mask[None, :], score, float('inf'))

            chunk_best = tl.min(score)
            topk_worst_packed = tl.max(topk_packed)
            topk_worst_val = _sortable_u32_to_fp32(
                (topk_worst_packed >> 32).to(tl.uint32))
            if chunk_best < topk_worst_val:
                score_sortable = _fp32_to_sortable_u32(score)
                idx_vals = (m_start + bm_range.to(tl.int64)).to(tl.int32)
                idx_bits = idx_vals.to(tl.uint32).to(tl.uint64)
                chunk_packed = (score_sortable.to(tl.uint64) << 32) | idx_bits[None, :]

                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
                chunk_packed = tl.sort(chunk_packed, dim=1)
                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)

                merged = tl.minimum(topk_packed, chunk_packed)
                topk_packed = tl.sort(merged, dim=1)

    else:
        for m_local in tl.range(0, M_PER_SPLIT, BM, num_stages=1):
            m_start = m_base + m_local
            m_offs = m_start + bm_range.to(tl.int64)
            m_mask = m_offs < M

            cross = tl.zeros([BN, BM], dtype=tl.float32)
            c_sq_tile = tl.zeros([BM], dtype=tl.float32)
            for d_start in range(0, D, D_INNER):
                d_offs = (d_start + tl.arange(0, D_INNER)).to(tl.int64)
                d_mask = d_offs < D

                x_sub = tl.load(
                    x_ptr + pid_b * stride_x_b
                    + n_offs[:, None] * stride_x_n
                    + d_offs[None, :] * stride_x_d,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0)

                c_sub = tl.load(
                    c_ptr + pid_b * stride_c_b
                    + m_offs[:, None] * stride_c_m
                    + d_offs[None, :] * stride_c_d,
                    mask=m_mask[:, None] & d_mask[None, :], other=0.0)

                cross += tl.dot(x_sub, tl.trans(c_sub)).to(tl.float32)
                c_f = c_sub.to(tl.float32)
                c_sq_tile += tl.sum(c_f * c_f, axis=1)

            score = c_sq_tile[None, :] - 2.0 * cross
            score = tl.where(m_mask[None, :], score, float('inf'))

            chunk_best = tl.min(score)
            topk_worst_packed = tl.max(topk_packed)
            topk_worst_val = _sortable_u32_to_fp32(
                (topk_worst_packed >> 32).to(tl.uint32))
            if chunk_best < topk_worst_val:
                score_sortable = _fp32_to_sortable_u32(score)
                idx_vals = (m_start + bm_range.to(tl.int64)).to(tl.int32)
                idx_bits = idx_vals.to(tl.uint32).to(tl.uint64)
                chunk_packed = (score_sortable.to(tl.uint64) << 32) | idx_bits[None, :]

                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)
                chunk_packed = tl.sort(chunk_packed, dim=1)
                chunk_packed = chunk_packed ^ tl.full([1], 0xFFFFFFFFFFFFFFFF, dtype=tl.uint64)

                merged = tl.minimum(topk_packed, chunk_packed)
                topk_packed = tl.sort(merged, dim=1)

    topk_score_sortable = (topk_packed >> 32).to(tl.uint32)
    topk_score = _sortable_u32_to_fp32(topk_score_sortable)
    topk_idx = (topk_packed & 0xFFFFFFFF).to(tl.int32)

    k_offs = tl.arange(0, TOPK_PAD).to(tl.int64)
    k_mask = k_offs < K
    write_mask = n_mask[:, None] & k_mask[None, :]
    pid_s_i64 = pid_s.to(tl.int64)
    tl.store(partial_val_ptr + pid_b * stride_pv_b + pid_s_i64 * stride_pv_s
             + n_offs[:, None] * stride_pv_n + k_offs[None, :] * stride_pv_k,
             topk_score, mask=write_mask)
    tl.store(partial_idx_ptr + pid_b * stride_pi_b + pid_s_i64 * stride_pi_s
             + n_offs[:, None] * stride_pi_n + k_offs[None, :] * stride_pi_k,
             topk_idx, mask=write_mask)

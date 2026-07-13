"""flash_knn — Triton iterative-insert top-K kernel (x²-free, signed score).

Same x²-free signed score as ``sortmerge.py`` — see that module's docstring
for the rationale. The insert kernel differs in two ways:

  * ``BM`` is decoupled from ``TOPK_PAD`` (sortmerge requires ``BM == TOPK_PAD``).
    The autotune sweeps ``BM ∈ {64, 128, 256}`` independently.
  * Top-K is maintained as an unsorted ``(BN, TOPK_PAD)`` register array;
    each chunk does at most ``MAX_STEPS = min(K, BM)`` argmin->insert
    passes. fp32 ``<`` works natively on signed scores, so only the
    trailing output sort uses the sortable-u32 transform.

Empirically the insert kernel wins on virtually every shape past the
trivial ``K == 1`` case; sortmerge is kept for completeness (autotune
still considers it, and it's the natural fit when ``K`` matches a
hardware-sortable tile size).

Public name: ``_flash_knn_insert_kernel`` — called from
``flashlib/primitives/knn/triton/dispatch.py``.
"""
import triton
import triton.language as tl

from flashlib.primitives.knn.triton._common import (
    _fp32_to_sortable_u32,
    _sortable_u32_to_fp32,
)


@triton.jit
def _flash_knn_insert_kernel(
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
    MAX_STEPS: tl.constexpr,
    NUM_STAGES_PIPE: tl.constexpr = 2,
    METRIC_IP: tl.constexpr = 0,
):
    """Iterative-insert top-K on signed shifted distance.

    Grid: ``(num_m_splits, ceil(N/BN), B)``.
    Two compile-time paths gated on ``D_INNER >= D`` (persistent x +
    pipelined M-loop vs D-split with sequential M-loop). The M-loop
    pipelining depth is exposed as ``NUM_STAGES_PIPE`` so the
    dispatcher can tune it per shape (see :mod:`dispatch` for the
    heuristic rules; defaults to 2 if not provided).
    """
    pid_s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BN
    n_offs = (n_start + tl.arange(0, BN)).to(tl.int64)
    n_mask = n_offs < N

    topk_vals = tl.full([BN, TOPK_PAD], float('inf'), dtype=tl.float32)
    topk_idxs = tl.full([BN, TOPK_PAD], -1, dtype=tl.int32)
    topk_max = tl.full([BN], float('inf'), dtype=tl.float32)
    k_range = tl.arange(0, TOPK_PAD)
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

            # NOTE (B200 experiment, kept for the record): transposing this
            # dot to corpus-major (M dim = BM >= 64) makes Triton 3.5 lower it
            # to tcgen05/MMAv5 instead of Ampere mma.sync -- but measured
            # SLOWER (heuristic cfg 5.1 -> 59 ms; autotuned 3.73 -> 3.98 ms at
            # 1M/k128): without TMA + MMAv5-aware pipelining the kernel is
            # bound by its cp.async load pipeline + insert epilogue, not MMA
            # throughput. Query-major mma.sync stays.
            cross = tl.dot(x_tile, tl.trans(c_tile)).to(tl.float32)
            if METRIC_IP:
                # inner-product top-k: rank by -x.c (argmin == argmax IP);
                # the corpus-norm term is dropped entirely.
                score = -cross
            else:
                c_f = c_tile.to(tl.float32)
                c_sq_tile = tl.sum(c_f * c_f, axis=1)
                score = c_sq_tile[None, :] - 2.0 * cross
            score = tl.where(m_mask[None, :], score, float('inf'))

            chunk_best = tl.min(score)
            threshold_worst = tl.max(topk_max)
            if chunk_best < threshold_worst:
                _active = tl.full([1], 1, dtype=tl.int32)
                for _step in range(MAX_STEPS):
                    if tl.max(_active) > 0:
                        row_min = tl.min(score, axis=1)
                        row_argmin = tl.argmin(score, axis=1)
                        do_insert = row_min < topk_max
                        n_inserts = tl.sum(do_insert.to(tl.int32))
                        if n_inserts > 0:
                            topk_argmax = tl.argmax(topk_vals, axis=1)
                            replace_mask = (k_range[None, :] == topk_argmax[:, None])
                            insert_mask = do_insert[:, None] & replace_mask
                            topk_vals = tl.where(insert_mask, row_min[:, None], topk_vals)
                            topk_idxs = tl.where(
                                insert_mask,
                                (m_start + row_argmin.to(tl.int64))[:, None].to(tl.int32),
                                topk_idxs,
                            )
                            topk_max = tl.max(topk_vals, axis=1)
                            used_mask = (bm_range[None, :] == row_argmin[:, None])
                            score = tl.where(used_mask & do_insert[:, None], float('inf'), score)
                        _active = tl.where(
                            n_inserts > 0,
                            tl.full([1], 1, dtype=tl.int32),
                            tl.full([1], 0, dtype=tl.int32),
                        )

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
                if METRIC_IP == 0:
                    c_f = c_sub.to(tl.float32)
                    c_sq_tile += tl.sum(c_f * c_f, axis=1)

            if METRIC_IP:
                score = -cross
            else:
                score = c_sq_tile[None, :] - 2.0 * cross
            score = tl.where(m_mask[None, :], score, float('inf'))

            chunk_best = tl.min(score)
            threshold_worst = tl.max(topk_max)
            if chunk_best < threshold_worst:
                _active = tl.full([1], 1, dtype=tl.int32)
                for _step in range(MAX_STEPS):
                    if tl.max(_active) > 0:
                        row_min = tl.min(score, axis=1)
                        row_argmin = tl.argmin(score, axis=1)
                        do_insert = row_min < topk_max
                        n_inserts = tl.sum(do_insert.to(tl.int32))
                        if n_inserts > 0:
                            topk_argmax = tl.argmax(topk_vals, axis=1)
                            replace_mask = (k_range[None, :] == topk_argmax[:, None])
                            insert_mask = do_insert[:, None] & replace_mask
                            topk_vals = tl.where(insert_mask, row_min[:, None], topk_vals)
                            topk_idxs = tl.where(
                                insert_mask,
                                (m_start + row_argmin.to(tl.int64))[:, None].to(tl.int32),
                                topk_idxs,
                            )
                            topk_max = tl.max(topk_vals, axis=1)
                            used_mask = (bm_range[None, :] == row_argmin[:, None])
                            score = tl.where(used_mask & do_insert[:, None], float('inf'), score)
                        _active = tl.where(
                            n_inserts > 0,
                            tl.full([1], 1, dtype=tl.int32),
                            tl.full([1], 0, dtype=tl.int32),
                        )

    val_sortable = _fp32_to_sortable_u32(topk_vals)
    val_bits = val_sortable.to(tl.uint64)
    idx_bits = topk_idxs.to(tl.uint32).to(tl.uint64)
    packed = (val_bits << 32) | idx_bits
    packed = tl.sort(packed, dim=1)
    topk_score = _sortable_u32_to_fp32((packed >> 32).to(tl.uint32))
    topk_idx = (packed & 0xFFFFFFFF).to(tl.int32)

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

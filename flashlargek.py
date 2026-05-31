"""flash_knn — large-K fused kernel: streaming threshold + bucket-partitioned buffer.

Single-matmul large-K top-K. The two-pass design recomputes the expensive
``x·cᵀ`` matmul twice; here the histogram and the candidate writeback happen in
one pass, so the matmul runs once.

The hard part of a fused single pass is bounding memory while staying exact
(the exact threshold T = ``argmin{b : cumsum(hist,b) ≥ k}`` is only known once
the histogram is complete). This file solves it with three ideas:

  1. **Streaming threshold.** Each row's gate starts at the last bucket and,
     after every program's ``atomic_add`` into the histogram, is re-derived from
     the (partial) histogram and folded in with ``atomic_min`` (it only
     tightens). Because counts only grow, every partial threshold is ≥ the
     final T, so the gate is ≥ T at all times → every true candidate
     (``bucket ≤ T``) is always written. The gate also stops writing buckets
     far above T, cutting junk traffic.

  2. **Bucket-partitioned buffer.** Candidates are stored in a per-row,
     per-bucket region ``cand[row, bucket, slot]`` (not a shared per-row append
     buffer). High-bucket junk overflows *its own* region and can never evict a
     low-bucket true candidate — which is what broke the shared-buffer version
     on large M.

  3. **k slots per bucket → exactness.** If a bucket has ≥ k items then
     ``cumsum`` there is ≥ k, so T is at or below it. Contrapositive: every
     bucket **strictly below T holds < k items**, so a k-slot region never
     overflows for buckets < T → the whole "definitely-in" set (buckets < T) is
     kept exactly. The boundary bucket == T keeps its first k (we need only
     k − count_below ≤ k from it; within one bucket the scores are ~ties).

Memory is ``[CHUNK, 256, k]`` which is independent of the query count N: the
host driver processes queries in **chunks of ~2·BN**, reusing the buffer, so the
``×N`` blow-up never happens.

Score / bucketing (monotonic in squared L2):
    s       = ‖c‖² − 2⟨x,c⟩            (same argmin-K as ‖x−c‖²)
    bucket  = clamp(int((s − origin)·inv_delta), 0, 255)

Returns ``(N, K)`` int32 indices (and squared-L2 distances). True squared L2 is
recovered on host via ``‖x−c‖² = s + ‖x‖²``.
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from _common import _next_pow2
from _row_norm import _get_or_compute_csq

_NUM_BUCKETS = 256
_BUCKET_EPS = 1.0e-20


# ────────────────────────── fused pass ─────────────────────────────────


@triton.jit
def _flashlargek_fused_kernel(
    x_ptr,
    c_ptr,
    csq_ptr,
    origin_ptr,
    inv_delta_ptr,
    bcount_ptr,
    threshold_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    stride_x_n,
    stride_x_d,
    stride_c_m,
    stride_c_d,
    stride_csq_m,
    stride_o_n,
    stride_bc_n,
    stride_bc_k,
    stride_t_n,
    stride_cv_n,
    stride_cv_k,
    stride_cv_s,
    stride_ci_n,
    stride_ci_k,
    stride_ci_s,
    N: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    K_CAP: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    D_INNER: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    NUM_STAGES_PIPE: tl.constexpr = 2,
):
    """One pass over [BN queries × BM corpus]: score → bucket → hist + writeback.

    ``bcount[row, bucket]`` is both the histogram and the per-bucket slot
    counter: ``slot = atomic_add(bcount, 1)`` returns the within-bucket index.
    The streaming gate (re-derived from bcount's cumsum, atomic_min) decides
    which buckets to write; a lane writes ``cand[row, bucket, slot]`` when
    ``bucket ≤ gate`` and ``slot < K_CAP``.

    Grid: ``(ceil(M/BM), ceil(N/BN))`` where N is the *chunk* size.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offs = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
    n_mask = n_offs < N
    m_offs = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    m_mask = m_offs < M

    # ── score = ‖c‖² − 2⟨x,c⟩ (single matmul) ──
    if D_INNER >= D:
        d_offs = tl.arange(0, D_INNER).to(tl.int64)
        d_mask = d_offs < D
        x_tile = tl.load(
            x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        c_tile = tl.load(
            c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        cross = tl.dot(x_tile, tl.trans(c_tile)).to(tl.float32)
    else:
        cross = tl.zeros([BN, BM], dtype=tl.float32)
        for d_start in range(0, D, D_INNER):
            d_offs = (d_start + tl.arange(0, D_INNER)).to(tl.int64)
            d_mask = d_offs < D
            x_sub = tl.load(
                x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
                mask=n_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            c_sub = tl.load(
                c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            cross += tl.dot(x_sub, tl.trans(c_sub)).to(tl.float32)

    csq = tl.load(csq_ptr + m_offs * stride_csq_m, mask=m_mask, other=0.0)
    score = csq[None, :] - 2.0 * cross

    origin = tl.load(origin_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    inv_delta = tl.load(inv_delta_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)

    bucket_f = (score - origin[:, None]) * inv_delta[:, None]
    bucket_raw = bucket_f.to(tl.int32)
    in_range = bucket_raw < NUM_BUCKETS
    bucket_i = tl.minimum(tl.maximum(bucket_raw, 0), NUM_BUCKETS - 1)

    valid = n_mask[:, None] & m_mask[None, :] & in_range

    # histogram + slot counter (one atomic): slot = pre-increment value
    bc_off = n_offs[:, None] * stride_bc_n + bucket_i.to(tl.int64) * stride_bc_k
    slot = tl.atomic_add(bcount_ptr + bc_off, 1, sem="relaxed", mask=valid)
    slot = slot.to(tl.int64)

    # streaming threshold: re-derive the gate from the (partial) histogram and
    # fold it in with atomic_min (it only tightens; stays ≥ the final T).
    bkt = tl.arange(0, NUM_BUCKETS)
    hrow = tl.load(
        bcount_ptr
        + n_offs[:, None] * stride_bc_n
        + bkt[None, :].to(tl.int64) * stride_bc_k,
        mask=n_mask[:, None],
        other=0,
    ).to(
        tl.int32
    )  # (BN, NUM_BUCKETS)
    cum = tl.cumsum(hrow, axis=1)
    reached = cum >= K
    cand_b = tl.where(reached, bkt[None, :], NUM_BUCKETS)
    my_thr = tl.min(cand_b, axis=1)
    my_thr = tl.minimum(my_thr, NUM_BUCKETS - 1).to(tl.int32)
    th_off = n_offs * stride_t_n
    old_thr = tl.atomic_min(threshold_ptr + th_off, my_thr, sem="relaxed", mask=n_mask)
    gate = tl.minimum(old_thr, my_thr)

    # writeback into the per-(row, bucket) region (k slots/bucket)
    store_mask = valid & (bucket_i <= gate[:, None]) & (slot < K_CAP)
    cv_off = (
        n_offs[:, None] * stride_cv_n
        + bucket_i.to(tl.int64) * stride_cv_k
        + slot * stride_cv_s
    )
    ci_off = (
        n_offs[:, None] * stride_ci_n
        + bucket_i.to(tl.int64) * stride_ci_k
        + slot * stride_ci_s
    )
    tl.store(cand_val_ptr + cv_off, score, mask=store_mask)
    tl.store(cand_idx_ptr + ci_off, m_offs[None, :].to(tl.int32), mask=store_mask)


# ────────────────────────── host driver ────────────────────────────────


def _heuristic_flashlargek_config(*, N: int, M: int, D: int) -> dict:
    """Shape-only heuristic. BN is capped at 64 (the streaming pass holds a
    ``[BN, 256]`` cumsum tile per program); more num_warps spreads it."""
    if N <= 16:
        BN = 16
    elif N <= 32:
        BN = 32
    else:
        BN = 64
    BM = 64 if D >= 256 else 128
    D_INNER = _next_pow2(D) if D <= 64 else 64
    return dict(BN=BN, BM=BM, D_INNER=D_INNER, num_warps=8, num_stages_pipe=2)


@torch.no_grad()
def _sample_range_per_row(
    x: torch.Tensor,
    c: torch.Tensor,
    csq: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-query bucket range ``[min, k-th smallest]`` from the first
    ``min(M, max(k, 16384))`` corpus rows, using ``s = ‖c‖² − 2⟨x,c⟩``. The true
    full-corpus k-th smallest is ≤ the sample's, so it sits inside the range.
    Returns ``(bucket_origin, bucket_inv_delta)`` of shape ``(N,)``."""
    M = c.shape[0]
    # S must be ≥ k: the range max is the sample's k-th smallest, and only with
    # S ≥ k does the subset argument (true k-th ≤ sample k-th) hold, so the range
    # is guaranteed to contain the true top-K. A 16384 floor keeps small-k well
    # sampled. (The old min(16384, ...) capped S and broke exactness for k>16384.)
    S = min(M, max(k, 16384))
    k_eff = min(k, S)  # == k since S ≥ k

    c_sample = c[:S, :].contiguous()
    csq_sample = csq[:S]
    cross = torch.matmul(x.float(), c_sample.float().t())  # (N, S)
    s_sample = csq_sample.unsqueeze(0) - 2.0 * cross

    if k_eff < S:
        vals = torch.topk(s_sample, k=k_eff, dim=-1, largest=False, sorted=True).values
    else:
        vals, _ = s_sample.sort(dim=-1)
    min_score = vals[..., 0]
    max_score = vals[..., k_eff - 1]
    span = torch.clamp(max_score - min_score, min=_BUCKET_EPS)
    # Use (NUM_BUCKETS-1)/span so the range max (the k-th smallest) lands on the
    # last valid bucket 255, not bucket 256 which in_range would drop. Matters
    # when S == M (small corpus): then max_score == the true k-th, and that
    # boundary item must be kept.
    return min_score.contiguous(), ((_NUM_BUCKETS - 1) / span).contiguous()


def _gather_topk_chunk(
    bcount: torch.Tensor,
    cand_val: torch.Tensor,
    cand_idx: torch.Tensor,
    k: int,
    K_CAP: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bucket-select over a chunk's [R, 256, K_CAP] partitioned buffer.

    Exact T per row from the histogram, then **gap-free compaction**: the real
    candidates live at ``(bucket ≤ T, slot < pop[bucket])`` but are scattered in
    a sparse [256, K_CAP] grid (~58 real per bucket in K_CAP slots). Instead of
    scanning the wide grid, we build per-bucket compact offsets and, for each of
    the ~count_at_T (≈ k) compact positions, ``searchsorted`` the owning bucket
    and gather just that one slot. So we touch only ~k items/row, not 256·K_CAP.
    Then a single ``torch.topk`` over the ~k-wide compact buffer (buckets < T are
    all kept; only the boundary == T competes, which topk realizes for free
    because bucket-<-T scores < boundary scores). Returns ``(vals, idxs)``
    ``(R, k)``, vals = shifted score ascending.
    """
    R, NB = bcount.shape
    device = bcount.device
    cum = bcount.cumsum(dim=-1)  # (R, NB)
    ge_k = cum >= k
    T = torch.where(
        ge_k.any(dim=-1),
        ge_k.int().argmax(dim=-1),
        torch.full((R,), NB - 1, device=device),
    )  # (R,)

    bk = torch.arange(NB, device=device)
    # stored items per bucket (≤ K_CAP), only for buckets ≤ T
    pop = torch.where(
        bk[None, :] <= T[:, None], bcount.clamp(max=K_CAP), torch.zeros_like(bcount)
    ).to(torch.int64)
    end = pop.cumsum(dim=1)  # (R, NB) inclusive
    offsets = end - pop  # (R, NB) exclusive start
    total = end[:, -1]  # (R,) ≈ count_at_T ≥ k

    W = max(int(total.max().item()), k)
    j = (
        torch.arange(W, device=device, dtype=torch.int64)[None, :]
        .expand(R, W)
        .contiguous()
    )
    valid = j < total[:, None]
    bucket_of = torch.searchsorted(end, j, right=True).clamp_(max=NB - 1)  # (R, W)
    slot_of = j - offsets.gather(1, bucket_of)
    gidx = (bucket_of * K_CAP + slot_of).clamp_(0, NB * K_CAP - 1)

    flat_v = cand_val.reshape(R, NB * K_CAP)
    flat_i = cand_idx.reshape(R, NB * K_CAP)
    packed_v = torch.where(valid, flat_v.gather(1, gidx), torch.inf)
    packed_i = flat_i.gather(1, gidx)

    topv, topp = torch.topk(packed_v, k=k, dim=-1, largest=False, sorted=True)
    topi = torch.gather(packed_i, 1, topp)
    return topv, topi


@torch.no_grad()
def flash_knn_triton_flashlargek(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    return_distances: bool = True,
):
    """Fused (single-matmul) large-K top-K, exact via bucket-partitioned buffer.

    ``x: (N, D)`` queries, ``c: (M, D)`` corpus. Queries are processed in
    chunks of ``BN`` so the ``[CHUNK, 256, k]`` buffer is independent of N.
    Returns ``(vals, idxs)`` (vals = squared L2, ascending) or just ``idxs``.
    """
    assert x.is_cuda and c.is_cuda and x.ndim == 2 and c.ndim == 2
    N, D = x.shape
    M, Dc = c.shape
    assert D == Dc, "x and c must agree on feature dim"
    assert 1 <= k <= M, f"k must be in [1, M={M}], got {k}"
    device = x.device

    c = c.contiguous()
    csq = _get_or_compute_csq(c)  # (M,) fp32
    bucket_origin, bucket_inv_delta = _sample_range_per_row(x, c, csq, k)

    cfg = _heuristic_flashlargek_config(N=N, M=M, D=D)
    BN, BM, D_INNER = cfg["BN"], cfg["BM"], cfg["D_INNER"]
    CHUNK = BN
    K_CAP = k  # k slots / bucket

    out_idxs = torch.empty((N, k), device=device, dtype=torch.int32)
    out_shift = (
        torch.empty((N, k), device=device, dtype=torch.float32)
        if return_distances
        else None
    )

    # reusable per-chunk scratch.
    bcount = torch.empty((CHUNK, _NUM_BUCKETS), device=device, dtype=torch.int32)
    threshold = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    cand_val = torch.empty(
        (CHUNK, _NUM_BUCKETS, K_CAP), device=device, dtype=torch.float32
    )
    cand_idx = torch.empty(
        (CHUNK, _NUM_BUCKETS, K_CAP), device=device, dtype=torch.int32
    )

    verbose = os.environ.get("FLASHLARGEK_VERBOSE") == "1"
    grid_m = math.ceil(M / BM)

    for q0 in range(0, N, CHUNK):
        q1 = min(q0 + CHUNK, N)
        R = q1 - q0
        xb = x[q0:q1].contiguous()  # (R, D)
        ob = bucket_origin[q0:q1]  # (R,)
        ib = bucket_inv_delta[q0:q1]

        bc = bcount[:R]
        th = threshold[:R]
        cv = cand_val[:R]
        ci = cand_idx[:R]
        bc.zero_()
        th.fill_(_NUM_BUCKETS - 1)
        # cand_val/cand_idx need no reset: the gap-free gather only reads
        # written slots (slot < pop), never stale/uninitialised ones.

        grid = (grid_m, math.ceil(R / BN))
        _flashlargek_fused_kernel[grid](
            xb,
            c,
            csq,
            ob,
            ib,
            bc,
            th,
            cv,
            ci,
            xb.stride(0),
            xb.stride(1),
            c.stride(0),
            c.stride(1),
            csq.stride(0),
            ob.stride(0),
            bc.stride(0),
            bc.stride(1),
            th.stride(0),
            cv.stride(0),
            cv.stride(1),
            cv.stride(2),
            ci.stride(0),
            ci.stride(1),
            ci.stride(2),
            N=R,
            M=M,
            D=D,
            K=k,
            K_CAP=K_CAP,
            BN=BN,
            BM=BM,
            D_INNER=D_INNER,
            NUM_BUCKETS=_NUM_BUCKETS,
            NUM_STAGES_PIPE=cfg["num_stages_pipe"],
            num_warps=cfg["num_warps"],
        )

        topv, topi = _gather_topk_chunk(bc, cv, ci, k, K_CAP)
        out_idxs[q0:q1] = topi
        if return_distances:
            out_shift[q0:q1] = topv

        if verbose:
            cum = bc.cumsum(-1)
            T = torch.where(
                (cum >= k).any(-1),
                (cum >= k).int().argmax(-1),
                torch.full((R,), _NUM_BUCKETS - 1, device=device),
            )
            cb_below = torch.where(
                T > 0,
                cum.gather(-1, (T.long() - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
                torch.zeros_like(T),
            )
            overflow = (
                (bc >= K_CAP)
                & (torch.arange(_NUM_BUCKETS, device=device)[None, :] < T[:, None])
            ).sum()
            print(
                f"[flashlargek-bp] q0={q0} R={R} k={k} "
                f"T avg={T.float().mean():.1f} count_below avg={cb_below.float().mean():.0f} "
                f"below_T_overflow_buckets={int(overflow)}"
            )

    if not return_distances:
        return out_idxs
    x_sq = (x.float() * x.float()).sum(dim=-1)  # (N,)
    out_vals = (out_shift + x_sq.unsqueeze(-1)).clamp_min_(0.0)
    return out_vals.float(), out_idxs

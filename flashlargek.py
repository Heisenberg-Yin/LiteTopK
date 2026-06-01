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

_NUM_BUCKETS = 256
_BUCKET_EPS = 1.0e-20


# ────────────────────────── fused pass ─────────────────────────────────


@triton.jit
def _flashlargek_fused_kernel(
    x_ptr,
    c_ptr,
    origin_ptr,
    inv_delta_ptr,
    bcount_ptr,
    threshold_ptr,
    wcount_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    stride_x_n,
    stride_x_d,
    stride_c_m,
    stride_c_d,
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
    CNT_EVERY: tl.constexpr = 1,  # recompute the gate every ~CNT_EVERY cumulative writebacks
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

    # int32 addressing (host asserts every offset stays < 2**31)
    n_offs = pid_n * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    m_offs = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offs < M

    # ── score = ‖c‖² − 2⟨x,c⟩ : single corpus pass. ‖c‖² is accumulated from the
    # SAME c tiles already loaded for the matmul (no separate corpus read). ──
    if D_INNER >= D:
        d_offs = tl.arange(0, D_INNER)
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
        c_f = c_tile.to(tl.float32)
        csq = tl.sum(c_f * c_f, axis=1)  # (BM,)
    else:
        cross = tl.zeros([BN, BM], dtype=tl.float32)
        csq = tl.zeros([BM], dtype=tl.float32)
        for d_start in range(0, D, D_INNER):
            d_offs = d_start + tl.arange(0, D_INNER)
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
            c_f = c_sub.to(tl.float32)
            csq += tl.sum(c_f * c_f, axis=1)

    score = csq[None, :] - 2.0 * cross

    origin = tl.load(origin_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    inv_delta = tl.load(inv_delta_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)

    bucket_f = (score - origin[:, None]) * inv_delta[:, None]
    bucket_raw = bucket_f.to(tl.int32)
    in_range = bucket_raw < NUM_BUCKETS
    bucket_i = tl.minimum(tl.maximum(bucket_raw, 0), NUM_BUCKETS - 1)

    valid = n_mask[:, None] & m_mask[None, :] & in_range

    # histogram + slot counter (one atomic): slot = pre-increment value
    bc_off = n_offs[:, None] * stride_bc_n + bucket_i * stride_bc_k
    slot = tl.atomic_add(bcount_ptr + bc_off, 1, sem="relaxed", mask=valid)

    # streaming threshold (count-triggered). Every program reads the current
    # global gate; it recomputes & tightens the gate (cumsum over the partial
    # histogram + atomic_min) only when the cumulative writeback count crosses a
    # CNT_EVERY (~k/2) boundary. A staler gate is still ≥ the final T, so every
    # true candidate is still written → correctness holds, at the cost of a
    # little more writeback. This amortizes the per-program cumsum (~7% at bs≥64).
    th_off = n_offs * stride_t_n
    gate = tl.load(
        threshold_ptr + th_off, mask=n_mask, other=NUM_BUCKETS - 1
    ).to(tl.int32)
    wmask = valid & (bucket_i <= gate[:, None]) & (slot < K_CAP)
    my_wr = tl.sum(wmask.to(tl.int32))
    prev = tl.atomic_add(wcount_ptr, my_wr)
    if (prev // CNT_EVERY) != ((prev + my_wr) // CNT_EVERY):
        bkt = tl.arange(0, NUM_BUCKETS)
        hrow = tl.load(
            bcount_ptr
            + n_offs[:, None] * stride_bc_n
            + bkt[None, :] * stride_bc_k,
            mask=n_mask[:, None],
            other=0,
        ).to(
            tl.int32
        )  # (BN, NUM_BUCKETS)
        cum = tl.cumsum(hrow, axis=1)
        reached = cum >= K
        cand_b = tl.where(reached, bkt[None, :], NUM_BUCKETS)
        my_thr = tl.minimum(tl.min(cand_b, axis=1), NUM_BUCKETS - 1).to(tl.int32)
        tl.atomic_min(  # tighten the global gate for later programs
            threshold_ptr + th_off, my_thr, sem="relaxed", mask=n_mask
        )
        gate = tl.minimum(gate, my_thr)

    # writeback into the per-(row, bucket) region (k slots/bucket)
    store_mask = valid & (bucket_i <= gate[:, None]) & (slot < K_CAP)
    cv_off = (
        n_offs[:, None] * stride_cv_n + bucket_i * stride_cv_k + slot * stride_cv_s
    )
    ci_off = (
        n_offs[:, None] * stride_ci_n + bucket_i * stride_ci_k + slot * stride_ci_s
    )
    tl.store(cand_val_ptr + cv_off, score, mask=store_mask)
    tl.store(cand_idx_ptr + ci_off, m_offs[None, :].to(tl.int32), mask=store_mask)


# ────────────────────────── host driver ────────────────────────────────


def _heuristic_flashlargek_config(*, N: int, M: int, D: int) -> dict:
    """Shape-only heuristic covering query counts N = 1 .. 1024 (and beyond —
    the host streams queries in chunks of BN, so any N is handled by looping).

    ``BN`` (the query tile) is a power of two, **floored at 16** (tensor-core
    ``tl.dot`` needs the M/N/K dims ≥ 16) and **capped at 64** (the streaming
    threshold pass holds a ``[BN, 256]`` cumsum tile per program; larger BN
    blows shared memory). So N=1..16→16, 17..32→32, ≥33 (incl. 64/256/1024)→64.
    """
    if N <= 16:
        BN = 16
    elif N <= 32:
        BN = 32
    else:
        BN = 64  # N = 64, 256, 1024, ... all chunk into 64-query tiles
    BM = 64 if D >= 256 else 128
    D_INNER = _next_pow2(D) if D <= 64 else 64
    return dict(BN=BN, BM=BM, D_INNER=D_INNER, num_warps=8, num_stages_pipe=2)


@torch.no_grad()
def _sample_range_per_row(
    x: torch.Tensor,
    c: torch.Tensor,
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

    cs = c[:S, :].float()
    csq_sample = (cs * cs).sum(dim=-1)  # (S,) — computed on the slice only
    cross = torch.matmul(x.float(), cs.t())  # (N, S)
    s_sample = csq_sample.unsqueeze(0) - 2.0 * cross

    # We only need two order statistics per row: the minimum (range floor) and
    # the k_eff-th smallest (range ceiling) — never a full ordering.
    if k_eff < S:
        # the k_eff smallest values; their min is the global min and their max
        # is the k_eff-th smallest. sorted=False: order inside the set is unused.
        vals = torch.topk(s_sample, k=k_eff, dim=-1, largest=False, sorted=False).values
        min_score = vals.amin(dim=-1)
        max_score = vals.amax(dim=-1)
    else:  # k_eff == S: the whole sample is the top-k → just its min and max
        min_score = s_sample.amin(dim=-1)
        max_score = s_sample.amax(dim=-1)
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

    topv, topp = torch.topk(packed_v, k=k, dim=-1, largest=False, sorted=False)
    topi = torch.gather(packed_i, 1, topp)
    return topv, topi


def _gather_topk_chunk_boundary(
    bcount: torch.Tensor,
    cand_val: torch.Tensor,
    cand_idx: torch.Tensor,
    k: int,
    K_CAP: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same result as :func:`_gather_topk_chunk`, but the final selection runs
    only on the **boundary bucket T**, not on the whole ~k compact buffer.

    Buckets < T are 'definitely in' (their cumsum < k) and need no selection —
    they are collected directly. Only bucket T competes for the last
    ``k − count_below`` slots, so the ``topk`` is over bucket T's population
    (~k/T_buckets, ≪ k at large k) instead of ~k.
    """
    R, NB = bcount.shape
    device = bcount.device
    rows = torch.arange(R, device=device)
    cum = bcount.cumsum(dim=-1)
    ge_k = cum >= k
    T = torch.where(
        ge_k.any(dim=-1), ge_k.int().argmax(dim=-1),
        torch.full((R,), NB - 1, device=device),
    )
    # count_below = items in buckets strictly < T (all kept)
    below_cum = cum.gather(1, (T - 1).clamp_min(0)[:, None]).squeeze(1)
    count_below = torch.where(T > 0, below_cum, torch.zeros_like(T)).clamp_(max=k).to(torch.int64)
    need = k - count_below  # ≥ 1, taken from bucket T

    bk = torch.arange(NB, device=device)
    # ── collect buckets < T compactly (gap-free, like the base gather) ──
    pop = torch.where(bk[None, :] < T[:, None], bcount.clamp(max=K_CAP), torch.zeros_like(bcount)).to(torch.int64)
    end = pop.cumsum(dim=1)
    offsets = end - pop
    Wb = max(int(count_below.max().item()), 1)
    j = torch.arange(Wb, device=device, dtype=torch.int64)[None, :].expand(R, Wb)
    bucket_of = torch.searchsorted(end, j, right=True).clamp_(max=NB - 1)
    slot_of = j - offsets.gather(1, bucket_of)
    gidx = (bucket_of * K_CAP + slot_of).clamp_(0, NB * K_CAP - 1)
    flat_v = cand_val.reshape(R, NB * K_CAP)
    flat_i = cand_idx.reshape(R, NB * K_CAP)
    below_v = flat_v.gather(1, gidx)
    below_i = flat_i.gather(1, gidx)

    # ── bucket T: extract its slots and topk only `need` of them ──
    bt_pop = bcount.gather(1, T[:, None]).squeeze(1).clamp_(max=K_CAP).to(torch.int64)
    WT = max(int(bt_pop.max().item()), 1)
    bt_v = cand_val[rows, T][:, :WT]
    bt_i = cand_idx[rows, T][:, :WT]
    bt_v = torch.where(torch.arange(WT, device=device)[None, :] < bt_pop[:, None], bt_v, torch.inf)
    kT = max(int(need.max().item()), 1)
    selv, selp = torch.topk(bt_v, k=kT, dim=-1, largest=False, sorted=True)
    seli = bt_i.gather(1, selp)

    # ── assemble [R, k]: below items, then the selected bucket-T items ──
    p = torch.arange(k, device=device)[None, :]
    is_below = p < count_below[:, None]
    bi = p.clamp(max=Wb - 1).expand(R, k)
    ti = (p - count_below[:, None]).clamp_(0, kT - 1)
    out_v = torch.where(is_below, below_v.gather(1, bi), selv.gather(1, ti))
    out_i = torch.where(is_below, below_i.gather(1, bi), seli.gather(1, ti))
    return out_v, out_i


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

    # Pick the launch config and run the int32-offset guard FIRST — before any
    # GPU work (sampling / matmul) — so an out-of-range shape fails fast.
    cfg = _heuristic_flashlargek_config(N=N, M=M, D=D)
    BN, BM, D_INNER = cfg["BN"], cfg["BM"], cfg["D_INNER"]
    CHUNK = BN
    K_CAP = k  # k slots / bucket (exact: buckets < T hold < k items → never overflow)
    # int32 addressing: every tensor offset must stay < 2**31. Largest are the
    # corpus (M·D) and the cand buffer (BN·256·K_CAP). Holds for M ≲ 2.79M at
    # D=768 and k ≲ 131072 at BN=64.
    assert M * D < 2**31 and BN * _NUM_BUCKETS * K_CAP < 2**31, (
        f"int32 offsets would overflow (M*D={M * D}, "
        f"BN*256*K_CAP={BN * _NUM_BUCKETS * K_CAP}); restore int64 addressing"
    )

    c = c.contiguous()
    bucket_origin, bucket_inv_delta = _sample_range_per_row(x, c, k)

    out_idxs = torch.empty((N, k), device=device, dtype=torch.int32)
    out_shift = (
        torch.empty((N, k), device=device, dtype=torch.float32)
        if return_distances
        else None
    )

    # reusable per-chunk scratch.
    bcount = torch.empty((CHUNK, _NUM_BUCKETS), device=device, dtype=torch.int32)
    threshold = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    wcount = torch.zeros((1,), device=device, dtype=torch.int32)  # count-mode global counter
    cand_val = torch.empty(
        (CHUNK, _NUM_BUCKETS, K_CAP), device=device, dtype=torch.float32
    )
    cand_idx = torch.empty(
        (CHUNK, _NUM_BUCKETS, K_CAP), device=device, dtype=torch.int32
    )

    verbose = os.environ.get("FLASHLARGEK_VERBOSE") == "1"
    # recompute the streaming threshold every ~k/2 cumulative writebacks
    # (auto-scales with k; ~7% faster at bs≥64, no recall change). Env overrides.
    cnt_every = int(os.environ.get("FLASHLARGEK_CNT_EVERY", str(max(1, k // 2))))
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
        wcount.zero_()
        # cand_val/cand_idx need no reset: the gap-free gather only reads
        # written slots (slot < pop), never stale/uninitialised ones.

        grid = (grid_m, math.ceil(R / BN))
        _flashlargek_fused_kernel[grid](
            xb,
            c,
            ob,
            ib,
            bc,
            th,
            wcount,
            cv,
            ci,
            xb.stride(0),
            xb.stride(1),
            c.stride(0),
            c.stride(1),
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
            CNT_EVERY=cnt_every,
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


# ══════════════════ experimental: flat per-query append buffer ══════════════
# Replaces the [CHUNK, 256, k] bucket-partitioned buffer with a flat per-query
# append buffer [CHUNK, BUF]: items that pass the streaming gate are appended to
# their query's buffer (atomic per-query counter), then a single torch.topk over
# [CHUNK, BUF] selects the k smallest. Exact only when count_at_T ≲ BUF (else the
# buffer overflows and drops candidates). Speed experiment vs the gap-free gather.


@triton.jit
def _flashlargek_flat_kernel(
    x_ptr,
    c_ptr,
    origin_ptr,
    inv_delta_ptr,
    bcount_ptr,
    threshold_ptr,
    wcount_ptr,
    qcount_ptr,
    buf_val_ptr,
    buf_idx_ptr,
    stride_x_n,
    stride_x_d,
    stride_c_m,
    stride_c_d,
    stride_o_n,
    stride_bc_n,
    stride_bc_k,
    stride_t_n,
    stride_q_n,
    stride_bv_n,
    stride_bv_p,
    stride_bi_n,
    stride_bi_p,
    N: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BUF: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    D_INNER: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    NUM_STAGES_PIPE: tl.constexpr = 2,
    CNT_EVERY: tl.constexpr = 1,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_offs = pid_n * BN + tl.arange(0, BN)
    n_mask = n_offs < N
    m_offs = pid_m * BM + tl.arange(0, BM)
    m_mask = m_offs < M

    # score = ‖c‖² − 2⟨x,c⟩ with ‖c‖² fused from the same c tiles (single pass)
    if D_INNER >= D:
        d_offs = tl.arange(0, D_INNER)
        d_mask = d_offs < D
        x_tile = tl.load(
            x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0,
        )
        c_tile = tl.load(
            c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
            mask=m_mask[:, None] & d_mask[None, :], other=0.0,
        )
        cross = tl.dot(x_tile, tl.trans(c_tile)).to(tl.float32)
        c_f = c_tile.to(tl.float32)
        csq = tl.sum(c_f * c_f, axis=1)
    else:
        cross = tl.zeros([BN, BM], dtype=tl.float32)
        csq = tl.zeros([BM], dtype=tl.float32)
        for d_start in range(0, D, D_INNER):
            d_offs = d_start + tl.arange(0, D_INNER)
            d_mask = d_offs < D
            x_sub = tl.load(
                x_ptr + n_offs[:, None] * stride_x_n + d_offs[None, :] * stride_x_d,
                mask=n_mask[:, None] & d_mask[None, :], other=0.0,
            )
            c_sub = tl.load(
                c_ptr + m_offs[:, None] * stride_c_m + d_offs[None, :] * stride_c_d,
                mask=m_mask[:, None] & d_mask[None, :], other=0.0,
            )
            cross += tl.dot(x_sub, tl.trans(c_sub)).to(tl.float32)
            c_f = c_sub.to(tl.float32)
            csq += tl.sum(c_f * c_f, axis=1)

    score = csq[None, :] - 2.0 * cross
    origin = tl.load(origin_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    inv_delta = tl.load(inv_delta_ptr + n_offs * stride_o_n, mask=n_mask, other=0.0)
    bucket_f = (score - origin[:, None]) * inv_delta[:, None]
    bucket_raw = bucket_f.to(tl.int32)
    in_range = bucket_raw < NUM_BUCKETS
    bucket_i = tl.minimum(tl.maximum(bucket_raw, 0), NUM_BUCKETS - 1)
    valid = n_mask[:, None] & m_mask[None, :] & in_range

    # histogram (for the threshold); the returned slot is unused here.
    bc_off = n_offs[:, None] * stride_bc_n + bucket_i * stride_bc_k
    tl.atomic_add(bcount_ptr + bc_off, 1, sem="relaxed", mask=valid)

    # count-triggered streaming threshold (same as the production kernel)
    th_off = n_offs * stride_t_n
    gate = tl.load(threshold_ptr + th_off, mask=n_mask, other=NUM_BUCKETS - 1).to(tl.int32)
    pass_mask = valid & (bucket_i <= gate[:, None])
    my_wr = tl.sum(pass_mask.to(tl.int32))
    prev = tl.atomic_add(wcount_ptr, my_wr)
    if (prev // CNT_EVERY) != ((prev + my_wr) // CNT_EVERY):
        bkt = tl.arange(0, NUM_BUCKETS)
        hrow = tl.load(
            bcount_ptr + n_offs[:, None] * stride_bc_n + bkt[None, :] * stride_bc_k,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        cum = tl.cumsum(hrow, axis=1)
        cand_b = tl.where(cum >= K, bkt[None, :], NUM_BUCKETS)
        my_thr = tl.minimum(tl.min(cand_b, axis=1), NUM_BUCKETS - 1).to(tl.int32)
        tl.atomic_min(threshold_ptr + th_off, my_thr, sem="relaxed", mask=n_mask)
        gate = tl.minimum(gate, my_thr)

    # append passing items to the per-query flat buffer
    store_mask = valid & (bucket_i <= gate[:, None])
    # broadcast the per-query offset to [BN, BM] so the atomic pointer matches
    # store_mask; concurrent same-address adds (same n) each get a distinct slot.
    q_idx = n_offs[:, None] * stride_q_n + m_offs[None, :] * 0
    qpos = tl.atomic_add(qcount_ptr + q_idx, 1, sem="relaxed", mask=store_mask)
    wmask = store_mask & (qpos < BUF)
    bv_off = n_offs[:, None] * stride_bv_n + qpos * stride_bv_p
    bi_off = n_offs[:, None] * stride_bi_n + qpos * stride_bi_p
    tl.store(buf_val_ptr + bv_off, score, mask=wmask)
    tl.store(buf_idx_ptr + bi_off, m_offs[None, :].to(tl.int32), mask=wmask)


@torch.no_grad()
def flash_knn_triton_flashlargek_flat(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    buf_mult: int = 10,
    return_distances: bool = True,
):
    """Experimental flat-buffer variant: per-query append buffer [CHUNK, buf_mult*k]
    + a single torch.topk, instead of the [CHUNK, 256, k] bucket buffer + gap-free
    gather. Exact only when count_at_T ≲ buf_mult*k. Speed experiment."""
    assert x.is_cuda and c.is_cuda and x.ndim == 2 and c.ndim == 2
    N, D = x.shape
    M, Dc = c.shape
    assert D == Dc and 1 <= k <= M
    device = x.device

    cfg = _heuristic_flashlargek_config(N=N, M=M, D=D)
    BN, BM, D_INNER = cfg["BN"], cfg["BM"], cfg["D_INNER"]
    CHUNK = BN
    BUF = min(buf_mult * k, M)  # per-query append slots (10·k)
    assert M * D < 2**31 and BN * BUF < 2**31

    c = c.contiguous()
    bucket_origin, bucket_inv_delta = _sample_range_per_row(x, c, k)
    cnt_every = int(os.environ.get("FLASHLARGEK_CNT_EVERY", str(max(1, k // 2))))

    out_idxs = torch.empty((N, k), device=device, dtype=torch.int32)
    out_shift = torch.empty((N, k), device=device, dtype=torch.float32) if return_distances else None

    bcount = torch.empty((CHUNK, _NUM_BUCKETS), device=device, dtype=torch.int32)
    threshold = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    wcount = torch.zeros((1,), device=device, dtype=torch.int32)
    qcount = torch.empty((CHUNK,), device=device, dtype=torch.int32)
    buf_val = torch.empty((CHUNK, BUF), device=device, dtype=torch.float32)
    buf_idx = torch.empty((CHUNK, BUF), device=device, dtype=torch.int32)

    grid_m = math.ceil(M / BM)
    for q0 in range(0, N, CHUNK):
        q1 = min(q0 + CHUNK, N)
        R = q1 - q0
        xb = x[q0:q1].contiguous()
        ob = bucket_origin[q0:q1]
        ib = bucket_inv_delta[q0:q1]
        bc = bcount[:R]; th = threshold[:R]; qc = qcount[:R]
        bv = buf_val[:R]; bi = buf_idx[:R]
        bc.zero_(); th.fill_(_NUM_BUCKETS - 1); qc.zero_(); wcount.zero_()
        bv.fill_(float("inf"))

        _flashlargek_flat_kernel[(grid_m, math.ceil(R / BN))](
            xb, c, ob, ib, bc, th, wcount, qc, bv, bi,
            xb.stride(0), xb.stride(1), c.stride(0), c.stride(1), ob.stride(0),
            bc.stride(0), bc.stride(1), th.stride(0), qc.stride(0),
            bv.stride(0), bv.stride(1), bi.stride(0), bi.stride(1),
            N=R, M=M, D=D, K=k, BUF=BUF, BN=BN, BM=BM, D_INNER=D_INNER,
            NUM_BUCKETS=_NUM_BUCKETS, NUM_STAGES_PIPE=cfg["num_stages_pipe"],
            CNT_EVERY=cnt_every, num_warps=cfg["num_warps"],
        )

        # only the first qcount[q] slots of each query are real (rest are inf);
        # topk over the widest fill instead of the full BUF avoids scanning the
        # mostly-empty 10·k buffer at large k.
        w = max(int(qc.max().item()), k)
        topv, topp = torch.topk(bv[:, :w], k=k, dim=-1, largest=False, sorted=False)
        out_idxs[q0:q1] = torch.gather(bi[:, :w], 1, topp)
        if return_distances:
            out_shift[q0:q1] = topv

    if not return_distances:
        return out_idxs
    x_sq = (x.float() * x.float()).sum(dim=-1)
    out_vals = (out_shift + x_sq.unsqueeze(-1)).clamp_min_(0.0)
    return out_vals.float(), out_idxs

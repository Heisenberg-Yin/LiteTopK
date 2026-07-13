"""flash-mst: GPU-resident dense Boruvka MST.

Algorithm (per Boruvka iteration, all on GPU):
  1. Per-component argmin (`_per_component_argmin_v2_kernel`): for each row v,
     scan all N columns of MRD once; pack (weight_bits, dst_idx) as int64 with
     +inf for same-component cells; reduce per-row to a single packed best
     edge; atomic_min into OUT_PACKED[parent[v]] (the row's component bucket).
     SAME kernel folds in OUT_SRC[parent[v]] = atomic_min(orig_i) so we get
     the canonical source vertex per component without a separate kernel.
  2. Concurrent union-find (`_concurrent_uf_kernel`): each component decodes
     its packed-int64 best edge, finds roots with bounded path-walk, atomic-
     CAS merges. Successful merges write an MST edge via atomic counter.
  3. Pointer-jumping (`_pointer_jump_kernel`): parent[v] = parent[parent[v]]
     in parallel for ~log iters until paths are flat. Triton kernel replaces
     the `parent = parent[parent.to(int64)]` torch loop (saves N int64 casts).

Persistent state (allocated ONCE at flash_mst entry, reused across rounds):
  - parent[N] int32        — union-find array; doubles as the component label
    for the next round's argmin (after pointer-jumping it is a valid label,
    no torch.unique relabel needed).
  - out_packed[N] int64    — per-CID atomic_min target, sized N (upper bound
    on component count). Reset to +inf each round via Triton kernel.
  - out_src[N]    int32    — per-CID canonical src; reset to MAX_INT32.
  - mst_{src,dst,w}, edge_count — output.

Why no argsort + no torch.unique relabel:
  - The original code argsort-ed by component for cache locality; with
    parent indexed directly the cache hit-rate is the same (parent[i] is
    a single random read per row, then comp_j is a streaming scan which
    L2-caches).
  - `torch.unique` did a sort + inverse to remap parent[] to dense [0, n_C).
    That step cost ~0.13 ms × log_2(N) iters of pointer-jumping. We skip it:
    the argmin kernel uses `parent[i] != parent[j]` directly, which works
    with sparse labels.

Note: I tried a K-min Boruvka variant (build a per-row top-K cross-component
edge table once, reuse for K mini-merges) — measured 1 outer round was enough
to find 99.97% of MST edges at huge, but the K=4 table build is ~3-4× more
compute-heavy per scan than the simple argmin (4× axis-min + masking + bitonic
merge), and a 2nd cleanup round still costs a full N² scan, so it lost to the
simple argmin. Reverted; orchestration cleanup + tile tuning gave the win.

Packed int64 trick: for non-negative fp32 weights, the IEEE-754 bit pattern
is monotonic (same ordering as float). Pack as
    (weight_bits << 32) | dst_idx
so atomic_min on int64 selects the smallest weight (with index as tiebreaker).
"""

import math
import numpy as np
import torch
import triton
import triton.language as tl


# =============================================================================
# Kernel 1: per-component min outgoing edge
# Each program handles BLOCK_S sorted-contiguous points; per-row argmin over
# the row of MRD (masked by component); atomic_min the result into the
# corresponding component slot.
# =============================================================================

@triton.jit
def _per_component_argmin_kernel(
    MRD_ptr,                # (N, N) fp32
    SORTED_IDX_ptr,         # (N,) int32
    COMP_ptr,               # (N,) int32, indexed by orig_idx
    OUT_PACKED_ptr,         # (n_components,) int64, atomic_min target
    N,
    BLOCK_S: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    s_offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < N

    orig_i = tl.load(SORTED_IDX_ptr + s_offs, mask=s_mask, other=0)
    comp_i = tl.load(COMP_ptr + orig_i, mask=s_mask, other=-1)

    # IEEE-754 +inf bit pattern as int64 high word.
    # 0x7F800000 << 32 = 9151314442816847872
    best_packed = tl.full([BLOCK_S], 9151314442816847872, dtype=tl.int64)
    SHIFT32 = tl.cast(4294967296, tl.int64)  # 2^32 as int64
    n_i64 = tl.cast(N, tl.int64)

    for j_start in range(0, N, BLOCK_J):
        j_offs = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_offs < N
        comp_j = tl.load(COMP_ptr + j_offs, mask=j_mask, other=-2)

        row_offs = (orig_i.to(tl.int64)[:, None] * n_i64
                    + j_offs.to(tl.int64)[None, :])
        # Load and cast to fp32 (handles bf16 / fp16 / fp32 storage uniformly)
        mrd_tile = tl.load(MRD_ptr + row_offs,
                           mask=s_mask[:, None] & j_mask[None, :],
                           other=float('inf')).to(tl.float32)

        # Mask same-component
        diff = (comp_j[None, :] != comp_i[:, None]) & j_mask[None, :] & s_mask[:, None]
        mrd_tile = tl.where(diff, mrd_tile, float('inf'))

        # Pack (weight_bits, j_idx) as int64. Use multiplication by 2^32 instead
        # of `<< 32` to avoid int32 overflow warning.
        wbits_u32 = mrd_tile.to(tl.uint32, bitcast=True)
        wbits_i64 = wbits_u32.to(tl.int64)
        j_i64 = j_offs.to(tl.int64)[None, :]
        packed = wbits_i64 * SHIFT32 + j_i64

        tile_best = tl.min(packed, axis=1)
        best_packed = tl.minimum(best_packed, tile_best)

    # Atomic_min into OUT_PACKED[comp_i] for each row in this block
    out_addrs = OUT_PACKED_ptr + comp_i.to(tl.int64)
    tl.atomic_min(out_addrs, best_packed, mask=s_mask)


# =============================================================================
# Kernel 2: write per-component canonical source (one orig_idx per component)
# Run after argsort. For each component C in dense [0, n_components), picks
# the FIRST orig_idx in sorted order (sorted_idx[comp_start[C]]).
# =============================================================================

@triton.jit
def _set_per_component_src_kernel(
    SORTED_IDX_ptr,         # (N,) int32
    COMP_ptr,               # (N,) int32
    OUT_SRC_ptr,            # (n_components,) int32
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    s_offs = pid * BLOCK + tl.arange(0, BLOCK)
    s_mask = s_offs < N
    orig = tl.load(SORTED_IDX_ptr + s_offs, mask=s_mask, other=0)
    comp = tl.load(COMP_ptr + orig, mask=s_mask, other=0)
    # The "first" element of each component is the one whose sorted position
    # is smaller than all others with the same component. Use atomic_min on
    # (sorted_pos, orig) pair — but for simplicity, just use atomic_min on
    # orig (any vertex works as canonical).
    target = OUT_SRC_ptr + comp.to(tl.int64)
    # Use atomic_min so the smallest orig in each component wins
    # (deterministic). orig values are int32 non-negative.
    tl.atomic_min(target, orig, mask=s_mask)


# =============================================================================
# Kernel 3: concurrent union-find merge
# For each component C, decode (weight, dst) from out_packed[C], set src from
# out_src[C], find roots via bounded loop, atomic_cas to merge.
# Successful merges write an MST edge via atomic counter.
# =============================================================================

@triton.jit
def _concurrent_uf_kernel(
    PARENT_ptr,             # (N,) int32, atomic
    OUT_PACKED_ptr,         # (n_components,) int64
    OUT_SRC_ptr,            # (n_components,) int32
    MST_SRC_ptr,            # (N-1,) int32  — output edge sources
    MST_DST_ptr,            # (N-1,) int32  — output edge targets
    MST_W_ptr,              # (N-1,) fp32   — output edge weights
    EDGE_COUNT_ptr,         # (1,) int32 atomic
    n_components,
    MAX_FIND: tl.constexpr,
):
    cid = tl.program_id(0)
    if cid >= n_components:
        return

    packed = tl.load(OUT_PACKED_ptr + cid)
    # Sentinel: high 32 bits = 0x7F800000 (+inf bit pattern) means "no edge found"
    high32 = (packed >> 32).to(tl.int32)
    if high32 >= 0x7F800000:
        return

    weight = high32.to(tl.float32, bitcast=True)
    dst = (packed & 0xFFFFFFFF).to(tl.int32)
    src = tl.load(OUT_SRC_ptr + cid)
    if src < 0 or src == 2147483647:
        return

    # Single-pass: find roots with path-walking, then atomic_cas to merge.
    # If CAS fails (concurrent thread merged hi first), we drop this edge —
    # next Boruvka iter will re-propose. log(N) extra iters total (rare path).
    rs = src
    for _ in tl.static_range(MAX_FIND):
        p = tl.load(PARENT_ptr + rs)
        rs = tl.where(p == rs, rs, p)
    rd = dst
    for _ in tl.static_range(MAX_FIND):
        p = tl.load(PARENT_ptr + rd)
        rd = tl.where(p == rd, rd, p)

    if rs == rd:
        return

    lo = tl.minimum(rs, rd)
    hi = tl.maximum(rs, rd)
    old = tl.atomic_cas(PARENT_ptr + hi, hi, lo)
    if old == hi:
        idx = tl.atomic_add(EDGE_COUNT_ptr, 1)
        tl.store(MST_SRC_ptr + idx, src)
        tl.store(MST_DST_ptr + idx, dst)
        tl.store(MST_W_ptr + idx, weight)


# =============================================================================
# Generic edge-list union-find kernel (used by flash-dbscan CC, etc.).
# Vectorised over BLOCK edges per program. Path-halving compresses the
# tree during find via best-effort store-back. A merge counter lets the
# caller exit as soon as no more edges can merge (a chain-shaped graph
# with diameter ~ N can need 8+ passes).
# =============================================================================

@triton.jit
def _union_edges_v2_kernel(
    PARENT_ptr,         # (N,) int32 atomic
    ROW_ptr,            # (E,) int32 — edge sources
    COL_ptr,            # (E,) int32 — edge targets
    MERGE_COUNTER_ptr,  # (1,) int32 — atomic counter; nonzero ⇒ keep iterating
    n_edges,
    BLOCK: tl.constexpr,
    MAX_FIND: tl.constexpr,
):
    pid = tl.program_id(0)
    e_offs = pid * BLOCK + tl.arange(0, BLOCK)
    e_mask = e_offs < n_edges
    u = tl.load(ROW_ptr + e_offs, mask=e_mask, other=0)
    v = tl.load(COL_ptr + e_offs, mask=e_mask, other=0)

    # find root of u with path-halving: each step does 2 hops
    # (parent[ru] := parent[parent[ru]]) and writes back the compressed
    # parent. Best-effort — concurrent writers to the same ru race, but
    # since parent links are monotonically toward the root the result is
    # always valid (no need for atomic_cas — a stale write only undoes
    # one step of compression, never causes a cycle).
    ru = u
    for _ in tl.static_range(MAX_FIND):
        p = tl.load(PARENT_ptr + ru, mask=e_mask, other=0)
        gp = tl.load(PARENT_ptr + p, mask=e_mask, other=0)
        # only compress if we actually moved up two levels
        compress = (ru != p) & (p != gp) & e_mask
        tl.store(PARENT_ptr + ru, gp, mask=compress)
        ru = tl.where(p == ru, ru, gp)
    rv = v
    for _ in tl.static_range(MAX_FIND):
        p = tl.load(PARENT_ptr + rv, mask=e_mask, other=0)
        gp = tl.load(PARENT_ptr + p, mask=e_mask, other=0)
        compress = (rv != p) & (p != gp) & e_mask
        tl.store(PARENT_ptr + rv, gp, mask=compress)
        rv = tl.where(p == rv, rv, gp)

    diff = (ru != rv) & e_mask
    lo = tl.minimum(ru, rv)
    hi = tl.maximum(ru, rv)
    # atomic_min instead of atomic_cas: parent[hi] := min(parent[hi], lo).
    # Triton 3.6 doesn't support mask= on atomic_cas, but does on atomic_min.
    # Semantics are actually cleaner for UF: parent[v] is monotonically
    # non-increasing under atomic_min, so cycles are impossible by induction
    # (we always have parent[v] ≤ v). "Made progress" = OLD > lo, i.e., we
    # actually decreased parent[hi].
    old = tl.atomic_min(PARENT_ptr + hi, lo, mask=diff)
    progress = (old > lo) & diff
    n_succ = tl.sum(progress.to(tl.int32))
    # one atomic_add per CTA (not per edge) — negligible contention
    if n_succ > 0:
        tl.atomic_add(MERGE_COUNTER_ptr, n_succ)


# =============================================================================
# Triton compaction: replaces `torch.unique(parent, return_inverse=True)`.
# After parent is fully flattened, parent[v] is either v (if v is a root) or
# the root index. Then dense_id_at[v] = (number of roots in [0, v]) - 1 if v
# is a root, otherwise undefined. We compute it via a torch cumsum on the
# is_root bitmap (cheap, one launch) and then gather labels[v] = dense_id[parent[v]].
# =============================================================================

@triton.jit
def _is_root_kernel(
    PARENT_ptr,         # (N,) int32
    IS_ROOT_ptr,        # (N,) int32 (0/1)
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    p = tl.load(PARENT_ptr + offs, mask=mask, other=-1)
    is_root = (p == offs.to(tl.int32)).to(tl.int32)
    tl.store(IS_ROOT_ptr + offs, is_root, mask=mask)


@triton.jit
def _gather_labels_kernel(
    PARENT_ptr,         # (N,) int32
    DENSE_ID_ptr,       # (N,) int32 — dense id for each root, undefined for non-roots
    LABELS_ptr,         # (N,) int32 — output
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    p = tl.load(PARENT_ptr + offs, mask=mask, other=0)
    label = tl.load(DENSE_ID_ptr + p.to(tl.int64), mask=mask, other=0)
    tl.store(LABELS_ptr + offs, label, mask=mask)







# =============================================================================
# Kernel 4: pointer jumping path compression
# parent[v] = parent[parent[v]] in parallel; iterate to convergence.
# =============================================================================

@triton.jit
def _pointer_jump_kernel(
    PARENT_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    v_offs = pid * BLOCK + tl.arange(0, BLOCK)
    v_mask = v_offs < N
    p = tl.load(PARENT_ptr + v_offs, mask=v_mask, other=0)
    # Cast p (int32) to int64 for pointer arithmetic safety on large N
    pp = tl.load(PARENT_ptr + p.to(tl.int64), mask=v_mask, other=0)
    tl.store(PARENT_ptr + v_offs, pp, mask=v_mask)



def flash_cc_from_edges(rows: torch.Tensor, cols: torch.Tensor, N: int,
                         max_find: int = 8, max_passes: int = 16):
    """Connected components on the graph defined by edge list (rows, cols).

    Iterative converging algorithm:
      - Each pass: vectorized union with path-halving + atomic_cas hooks
        (BLOCK=128 edges per program → 4 warps fully utilized vs the old
         scalar 1-edge-per-warp launch).
      - Then 6 calls to `_pointer_jump_kernel` to flatten the parent forest.
      - Read the merge counter; exit as soon as a pass merges 0 edges.
        For diameter-bound dense graphs this usually fires in 1-2 passes;
        chain-shaped graphs (diameter ~N) converge in log_2(N) passes.

    Args:
        rows, cols: (E,) int32 — edge endpoints.
        N: number of vertices.
        max_find: bounded find depth per pass. With path-halving each step
                  doubles the effective depth, so MAX_FIND=8 covers depth 256.
        max_passes: hard ceiling (log2 of any conceivable graph diameter).

    Returns:
        labels: (N,) int32 — CC component id, dense in [0, n_components).
    """
    device = rows.device
    parent = torch.arange(N, dtype=torch.int32, device=device)
    E = rows.shape[0]

    if E > 0:
        BLOCK_EDGE = 128
        BLOCK_JUMP = 1024
        merge_counter = torch.zeros(1, dtype=torch.int32, device=device)
        grid_edge = (triton.cdiv(E, BLOCK_EDGE),)
        grid_jump = (triton.cdiv(N, BLOCK_JUMP),)

        for _ in range(max_passes):
            merge_counter.zero_()
            _union_edges_v2_kernel[grid_edge](
                parent, rows, cols, merge_counter, E,
                BLOCK=BLOCK_EDGE, MAX_FIND=max_find, num_warps=4,
            )
            # Pointer-jump to flatten any remaining short chains. 6 iters
            # cover depth ≤ 2^6 = 64 (post-compression chains are short).
            for _ in range(6):
                _pointer_jump_kernel[grid_jump](
                    parent, N, BLOCK=BLOCK_JUMP, num_warps=4,
                )
            if merge_counter.item() == 0:
                break

    # Compact: roots have parent[v] == v; assign dense [0, n_cc) by index order.
    is_root = torch.empty(N, dtype=torch.int32, device=device)
    grid_compact = (triton.cdiv(N, 1024),)
    _is_root_kernel[grid_compact](parent, is_root, N, BLOCK=1024, num_warps=4)
    # cumsum gives 1-indexed dense IDs at root positions; subtract 1 → 0-indexed.
    # At non-root positions the value is meaningless (we only gather via parent[v]).
    dense_id = torch.cumsum(is_root, dim=0, dtype=torch.int32) - 1
    labels = torch.empty(N, dtype=torch.int32, device=device)
    _gather_labels_kernel[grid_compact](
        parent, dense_id, labels, N, BLOCK=1024, num_warps=4,
    )
    return labels


# =============================================================================
# Kernel 5: per-component argmin with FOLDED canonical-src. Uses
# ``orig_i = pid * BLOCK_S + lane`` directly (same cache hit-rate since
# the column-side scan dominates the L2 traffic). The
# ``OUT_SRC[component] = min(orig_idx)`` atomic_min is folded into the
# same kernel, saving a separate launch per round.
# =============================================================================

@triton.jit
def _per_component_argmin_v2_kernel(
    MRD_ptr,                # (N, N) bf16 or fp32
    PARENT_ptr,             # (N,) int32 — parent[i] is i's component label
    OUT_PACKED_ptr,         # (N,) int64 — atomic_min target (sized to N)
    OUT_SRC_ptr,            # (N,) int32 — atomic_min target for canonical src
    N,
    BLOCK_S: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    s_offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < N

    orig_i = s_offs.to(tl.int64)
    comp_i = tl.load(PARENT_ptr + orig_i, mask=s_mask, other=0)

    # IEEE-754 +inf bit pattern as int64 high word: 0x7F800000 << 32
    best_packed = tl.full([BLOCK_S], 9151314442816847872, dtype=tl.int64)
    SHIFT32 = tl.cast(4294967296, tl.int64)
    n_i64 = tl.cast(N, tl.int64)

    for j_start in range(0, N, BLOCK_J):
        j_offs = j_start + tl.arange(0, BLOCK_J)
        j_mask = j_offs < N
        comp_j = tl.load(PARENT_ptr + j_offs, mask=j_mask, other=0)

        row_offs = orig_i[:, None] * n_i64 + j_offs.to(tl.int64)[None, :]
        mrd_tile = tl.load(MRD_ptr + row_offs,
                           mask=s_mask[:, None] & j_mask[None, :],
                           other=float('inf')).to(tl.float32)

        # Mask same-component
        diff = (comp_j[None, :] != comp_i[:, None]) & j_mask[None, :] & s_mask[:, None]
        mrd_tile = tl.where(diff, mrd_tile, float('inf'))

        # Pack (weight_bits, j_idx) as int64 — the int64 axis-min uses a single
        # comparator per element regardless of (weight, j) since the high bits
        # dominate the ordering. Trying to track (best_w, best_j) as separate
        # fp32+int (single tl.min + tl.argmin) was *slower* in benchmarking on
        # H200 — the int64 reduction actually beats the fused fp32+argmin path.
        wbits_u32 = mrd_tile.to(tl.uint32, bitcast=True)
        wbits_i64 = wbits_u32.to(tl.int64)
        j_i64 = j_offs.to(tl.int64)[None, :]
        packed = wbits_i64 * SHIFT32 + j_i64

        tile_best = tl.min(packed, axis=1)
        best_packed = tl.minimum(best_packed, tile_best)

    # Atomic_min into OUT_PACKED[comp_i] and OUT_SRC[comp_i] (folded)
    out_addrs = OUT_PACKED_ptr + comp_i.to(tl.int64)
    tl.atomic_min(out_addrs, best_packed, mask=s_mask)
    src_addrs = OUT_SRC_ptr + comp_i.to(tl.int64)
    tl.atomic_min(src_addrs, s_offs.to(tl.int32), mask=s_mask)


# =============================================================================
# Kernel 6: reset (N,)-sized int64 buffer to INF and int32 buffer to MAX_INT32
# Faster than torch.full(...) by ~4× (~30us vs ~125us per pair at huge): single
# kernel launch with one pass over both arrays vs two separate fill kernels +
# Python tensor wrapping overhead.
# =============================================================================

@triton.jit
def _reset_buffers_kernel(
    OUT_PACKED_ptr,         # (N,) int64
    OUT_SRC_ptr,            # (N,) int32
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    INF_PACKED = tl.cast(9151314442816847872, tl.int64)  # 0x7F800000 << 32
    MAX_I32 = tl.cast(2147483647, tl.int32)
    tl.store(OUT_PACKED_ptr + offs, INF_PACKED, mask=mask)
    tl.store(OUT_SRC_ptr + offs, MAX_I32, mask=mask)


# =============================================================================
# Python orchestration
# =============================================================================

def _pick_argmin_tile(N: int):
    """Pick (BLOCK_S, BLOCK_J, num_warps, num_stages) for the argmin kernel.

    Tuned on H200 (GPU 4) at bf16 MRD via `algorithms/hdbscan/_bench_mst.py`
    (sub-process to avoid Triton autotune cache leakage between the per-tile
    sweep and the E2E run).

    Best per-N (of 4.8 TB/s peak BW):
      N=20K:  BS=32, BJ=256, nw=4, ns=4 → 0.376 ms (44.4%)
      N=80K:  BS=32, BJ=128, nw=4, ns=4 → 4.625 ms (57.7%)
      N=200K: BS=32, BJ=128, nw=4, ns=3 → 28.08 ms (59.3%)
    Larger BLOCK_J (256+) stalls at huge — the per-CTA register footprint of
    the int64 packed tile (8 bytes × BLOCK_S × BLOCK_J) hurts occupancy and
    spills into shared memory. Use BJ=128 for N >= 50K.
    """
    if N >= 50_000:
        return 32, 128, 4, 3
    return 32, 256, 4, 4


def flash_mst(MRD: torch.Tensor) -> torch.Tensor:
    """GPU-resident dense Boruvka MST on a symmetric (N, N) distance matrix.

    Args:
        MRD: (N, N) fp32 or bf16 — assumed symmetric, non-negative. Pass bf16
             for 2× HBM speedup on the per-iter argmin scan; weight precision
             reduces to ~7 bits but tie-breaking via packed (weight, dst) keeps
             determinism.

    Returns:
        (N-1, 3) fp32 — MST edges as [src, dst, weight] sorted by weight.
        src/dst are integer vertex IDs stored as fp32 (precise for N <= 2^24).
    """
    assert MRD.is_cuda and MRD.dtype in (torch.float32, torch.bfloat16) and MRD.ndim == 2
    N = MRD.shape[0]
    assert MRD.shape[1] == N
    device = MRD.device

    # Persistent state — allocated once, reused across all rounds.
    parent = torch.arange(N, dtype=torch.int32, device=device)
    out_packed = torch.empty(N, dtype=torch.int64, device=device)
    out_src = torch.empty(N, dtype=torch.int32, device=device)

    mst_src = torch.full((N - 1,), -1, dtype=torch.int32, device=device)
    mst_dst = torch.full((N - 1,), -1, dtype=torch.int32, device=device)
    mst_w = torch.zeros(N - 1, dtype=torch.float32, device=device)
    edge_count = torch.zeros(1, dtype=torch.int32, device=device)

    BLOCK_S, BLOCK_J, num_warps_argmin, num_stages_argmin = _pick_argmin_tile(N)
    BLOCK_RESET = 1024
    BLOCK_JUMP = 1024

    # Standard Boruvka iter cap: log2(N) + small slack
    max_iters = max(2, int(math.ceil(math.log2(max(N, 2)))) + 5)
    target_edges = N - 1
    prev_count = 0

    for it in range(max_iters):
        # Reset per-component buffers in one kernel (int64 + int32 fused)
        grid_reset = (triton.cdiv(N, BLOCK_RESET),)
        _reset_buffers_kernel[grid_reset](
            out_packed, out_src, N,
            BLOCK=BLOCK_RESET, num_warps=4,
        )

        # Per-component argmin: folds canonical-src into the kernel
        grid_arg = (triton.cdiv(N, BLOCK_S),)
        _per_component_argmin_v2_kernel[grid_arg](
            MRD, parent, out_packed, out_src, N,
            BLOCK_S=BLOCK_S, BLOCK_J=BLOCK_J,
            num_warps=num_warps_argmin, num_stages=num_stages_argmin,
        )

        # Concurrent union-find: dispatch over [0, N). Non-root CIDs see
        # OUT_PACKED[c] == INF and early-return — cheap (~2 instructions).
        grid_uf = (N,)
        _concurrent_uf_kernel[grid_uf](
            parent, out_packed, out_src,
            mst_src, mst_dst, mst_w, edge_count,
            N,
            MAX_FIND=8, num_warps=1,
        )

        # Pointer-jump until paths flat. Triton kernel (one launch ≈ 30 µs at
        # huge) replaces the original log_2(N)-deep `parent[parent.to(int64)]`
        # torch loop. 6 iters flatten depth ≤ 64 which covers all observed
        # post-CAS chains; the next round's argmin handles any residual depth
        # (a non-flat parent makes parent[i] != parent[j] occasionally true
        # within a component → at worst one wasted edge proposal which the
        # uf_merge rejects).
        grid_jump = (triton.cdiv(N, BLOCK_JUMP),)
        for _ in range(6):
            _pointer_jump_kernel[grid_jump](
                parent, N, BLOCK=BLOCK_JUMP, num_warps=4,
            )

        # Single host sync per iter to check completion. Cost: ~5 µs each;
        # 8 iters at huge = 40 µs total (negligible vs ~30 ms argmin).
        cur_count = int(edge_count.item())
        if cur_count >= target_edges:
            break
        if cur_count == prev_count:
            # No new edges this round — graph is disconnected.
            break
        prev_count = cur_count

    n_edges = int(edge_count.item())
    if n_edges != target_edges:
        raise RuntimeError(
            f"flash_mst: produced {n_edges} edges, expected {target_edges}. "
            f"Graph may be disconnected, or merge kernel hit MAX_FIND."
        )

    # Sort by weight ascending for downstream HDBSCAN tree-building
    sort_idx = torch.argsort(mst_w)
    mst_src = mst_src[sort_idx]
    mst_dst = mst_dst[sort_idx]
    mst_w = mst_w[sort_idx]

    # Pack as (N-1, 3) fp32 for return — int IDs preserved exactly for N<=2^24
    out = torch.stack([mst_src.to(torch.float32),
                        mst_dst.to(torch.float32),
                        mst_w], dim=1)
    return out

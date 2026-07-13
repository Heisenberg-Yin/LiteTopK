# LiteTopK MS MARCO IP Top-K — B200 (Blackwell / SM100) Port

Third task of the B200 migration (after `../dsa` and the TidalDecode port):
batch retrieval over MS MARCO 768-d embeddings. **64 concurrent queries**
(sampled passages) against a shared corpus, `M ∈ {1M, 2M, 4M, 5M}`,
`k ∈ {128, 1024, 4096, 8192}`, exact top-k by inner product.

The scan is the same SM100 kernel as `litetopk_engine/sm100_tidal_marsco.cuh`
(one extension, one kernel) — this task drove three kernel/pipeline
generalizations:

| aspect | mechanism |
|---|---|
| **D=768** | KV tiles are split into 256-wide **D-chunks**; the TMA/UMMA pipeline is chunk-granular and UMMA accumulates chunks into the same TMEM buffer (smem per stage stays 64KB). KV smem release moved to the UMMA warp (`umma_arrive` on the empty barrier = release-on-MMA-retire); the math warps no longer touch KV barriers. |
| **batch 64 over one corpus** | flat-batch mode: 64 rows = 8 groups × QN=8, all groups map to corpus 0 via the kernel's `q_group_size` divisor. |
| **8-way corpus re-read** | each group's CTAs scan the same corpus → 8× traffic. Fixed by **transposing the grid** (groups on `blockIdx.x`, tile stride on `blockIdx.y`): CUDA's x-fastest launch order makes the 8 groups' same-tile CTAs co-resident, so 7 of 8 reads hit L2 (126MB ≫ drift). 1M scan: 1.84 → 0.81 ms. |

## Correctness findings (all fixed, recall ≥ 0.996 on all 16 cells)

1. **Tail-window sampling is not exchangeable on ordered corpora.** MARCO has
   region structure: at M=2M the tail sample's k-th score admitted **30%** of
   the corpus (vs ~2% at 1M/4M), collapsing the cell to 0.42x. Fix:
   `sample_mode=1` — **strided sampling** across the whole corpus; the sample
   only derives the gate (no seed prefill, `qcount`/`bcount` reset), the scan
   covers `[0, M)` so sampled rows re-emit naturally without duplicates.
2. **The fp16 bucket invariant** (the deep one — see
   `litetopk_engine/sm100_tidal_marsco.cuh` emit path): the thr-select is
   brittle on *both* sides of the refreshed threshold — `cnt(bucket<th) < k`
   (else direct-copy overflow) and `cnt(bucket≤th) ≥ k` *in the select's own
   bucket space* (else the mb-select underfills and a whole query row goes
   dark). The scan used to histogram fp32 scores while the select re-derives
   buckets from the stored **fp16** values; at MARCO's score magnitude
   (|score| ~ 70–115, fp16 ULP 0.06–0.125) edge candidates migrate a bucket
   between the two spaces. Fix: **histogram the fp16-rounded value**
   (`__half2float(__float2half(x))`) so both spaces agree bucket-exactly by
   construction. (Loosening th by +1 bucket instead violates the other
   invariant: recall 0.03.) A `NB=64` default keeps bucket width ≥ several
   ULPs (same choice as the h100 marsco bundle), plus a 4-ULP width floor on
   `inv_delta` as defence.

## Measured results (B200, batch 64, fp16 in / fp32 out, forced QN=64, warmup 10 / iters 20)

Baseline = dense `q @ base.T` (cuBLAS) + `torch.topk`. Defaults: **batch =
64** (one QN=64 row group, corpus read exactly once), **fp32 output values**
(fp16 inputs; the whole candidate chain — buffer, select, output — runs in
fp32, which also removes the fp16 bucket-rounding coupling entirely), fused
sample prep, REFRESH=8, `sample = max(16384, 8k)`,
**QN64+BM256+grid296 in every cell** (the QN8 fallback for small-M large-k
was removed — one engine shape everywhere), **SPEC_THREADS=64** (see the
KDA round below).

| M | k=128 | k=1024 | k=4096 | k=8192 |
|---:|---:|---:|---:|---:|
| 1M | 0.497 / **1.72x** | 0.577 / **1.50x** | 0.756 / **1.14x** | 0.864 / **1.13x** |
| 2M | 0.683 / **2.37x** | 0.800 / **2.03x** | 1.064 / **1.53x** | 1.249 / **1.38x** |
| 4M | 1.106 / **3.00x** | 1.243 / **2.69x** | 1.566 / **2.14x** | 1.868 / **1.84x** |
| 5M | 1.319 / **3.20x** | 1.466 / **2.90x** | 1.795 / **2.38x** | 2.146 / **2.03x** |

(cell = LiteTopK ms / speedup, idle GPU, recall >= 0.9958; **16/16 cells >=
1.13x**; 4M/k128 **3.00x**, 5M/k128 **3.20x at batch 64**, 5M/k8192 crosses
**2x**. Measured on the CLEANED code — all rejected experiment paths deleted,
see the cleanup section.)

### KDA optimization round (2026-07-09, ncu-evidence-driven; -4.4% to -12.5% latency per cell)

Workflow: kernel-design-agents + ncu-report-skill (profile -> diagnose -> plan;
evidence in `profile/run01_1m_k8192/`, decisions in `candidates.jsonl`,
`docs/draft.md` + `docs/plan.md`). Three promoted candidates:

1. **C1 — staged compact select** (`flashtopk_compact_thr_kernel<..., STAGED>`):
   ncu showed the old per-warp emit was bound by global atomicAdd L2 round
   trips on 64 row counters (long_scoreboard 59 cyc/issue, DRAM 1.8%, 229K L2
   atomics). Hits now stage in smem (block-local atomics) and flush per
   watermark with ONE global atomic + coalesced stores: compact 101 -> 17.9 µs
   at 1M/k8192. (A pure int4-vectorization attempt WITHOUT staging made it
   85 -> 101 µs — wrong axis, kept as a rejected candidate.)
2. **C1b — register-copy queue counts** (user-proposed): the QN8 warp-queue
   count is warp-uniform (increments = `popc(ballot)`, flush test uniform), so
   32 identical register copies replace the per-round lane0 smem read + shfl
   broadcast + writeback. QN8-only (+8 regs); QN8 probe 1M/k8192 0.887 ms.
3. **C2 — SPEC_THREADS 128 -> 64**: drop the 2 spare refresh warps; refresh is
   inlined into the math warps at the gate-reread points (after the acc-empty
   arrive, so it overlaps the UMMA's next tile; 8 warps split the rows). At
   320 threads setmaxnreg is illegal (not warpgroup-aligned) and unnecessary:
   `__launch_bounds__(320)` gives a 204-reg budget vs 168. ncu: sleeping
   stalls 4.64 -> 1.58 cyc/inst, no_instruction 8.32 -> 7.06. **-4 to -11% on
   every cell of both marsco and tidal**; env `TIDAL_SPEC_THREADS=128` keeps
   the legacy shape.

4. **C5c — in-place bucket-id gate** (user-designed, 2026-07-10; final form
   after a three-variant bake-off): compute the bucket id ONCE right after the
   score (`braw = int((x-o)*inv)`), gate = one integer compare
   `braw <= gate`, and on hit reuse the bucket id and the exact score
   directly — the fp32 path's histogram bucket is the same expression the
   select recomputes from the stored value, so the invariant holds with zero
   extra work (fp16 candidates keep the rounded-value bucket). Two hard-won
   lessons in `candidates.jsonl`: (a) an earlier scaled-Q variant
   (pre-multiplying Q by `-inv_delta` so TMEM emits bucket coordinates,
   1-compare gate) worked but needs a scale kernel + hit-path score recovery;
   (b) moving the hit-path work inside `if (g)` was **13% SLOWER than doing
   it unconditionally** — the predicated body breaks the compiler's
   interleaving of the 8 unrolled FMA→CVT→CMP chains, and this loop lives on
   ILP, not instruction count. The branchless single-bucket version beats
   every prior variant on every cell (1M/k128 0.485-0.491 best readings;
   tidal too).

### C6 — bucket-space candidate storage (user-proposed, 2026-07-12)

Don't store the raw score after computing the bucket id at emission time —
store the bucket-space coordinate `bq=(x-o)*inv` itself (a float, order-
preserving affine transform of x, not the truncated bucket id). The entire
select machinery (compact, boundary radix, hierarchical merge) never needs
to reconstruct x — every comparison/sort it does is order-invariant under an
affine transform — so `flashtopk_compact_thr_kernel`'s per-candidate bucket
recompute drops from a subtract+multiply+cast to a plain cast. Only the
FINAL K outputs per row get converted back (`val = bq*delta + o`), fused
into the existing host-side negate step as one ATen affine over `[R,K]`
elements instead of a per-candidate-scanned recompute.

Verified with an explicit VALUE check (`profile/verify_bucket_space_values.py`)
since `recall()` only compares index sets and would not catch a wrong-VALUE
bug: max abs error 4.5e-4 vs the dense ground truth (pure fp32 rounding).

**Compatibility bug found and fixed along the way**: tidal's default
`sample_mode=0` (tail-window sampling) prefills `buf_val` via the legacy
arch-agnostic `launch_hopper_seed_from_sample_fp16` kernel, which is unaware
of bucket-space storage and always writes raw scores — mixing that with an
unconditionally-bucket-space scan corrupted tidal's recall to 0.22-0.69.
Fixed with a `store_bucket_space` flag (= `strided_sample`) threaded through
the scan kernel and `flashtopk_compact_thr_kernel`, falling back to the
legacy "store x" behavior for `!strided_sample`; the host debucket step is
skipped in that mode too. marsco (always `strided_sample=1`) was never
affected correctness-wise. Measured cost: the scan side's flag is a runtime
bool (not templated) — marsco is ~2-4% slower across the matrix from the
added branch/param (0.497->0.513 ms @1M/k128), within the previously
documented build-to-build variance band; template-izing it to recover this
is a noted follow-up. tidal's recall is fully restored (0.999+).

### Code cleanup (2026-07-10): only the winning paths remain

All rejected experiment code was deleted from the kernel/host/loader —
row-tagged queue (`TIDAL_RTQ`), per-lane emit (`TIDAL_EMIT_PERLANE`),
bcount merge (`TIDAL_MERGE_BCOUNT`), the pre-C5 epilogue and its
`TIDAL_EPILOGUE_C5B` A/B flag, and the `TIDAL_SKIP_EPILOGUE` probe; plus
the superseded `bench_flashlib.py` (L2 version). What ships: C5c in-place
bucket gate, parallel-reservation emit (QN64), warp queue with register
counts (QN8), SPEC=64 with math-warp refresh (`TIDAL_SPEC_THREADS=128`
legacy shape kept), staged compact select. The rejected variants' designs
and measurements live in `candidates.jsonl`; reproduce the main table with
`bench_marsco_b200.py --m <M> --k <k>` (defaults are the final config).

**Carveout pitfall found during cleanup:** deleting the row-tagged queue's
16.6KB smem reservation shrank the QN64 launch request ~210KB -> ~193KB,
which dropped the L1/smem carveout from the 228KB bucket to 196KB — i.e.
~32KB MORE L1 — and made the k=128 column **~5% slower** (1M/k128
0.49 -> 0.52 ms, reproducible; restoring a dummy pad restored the speed).
Mechanism: `th_bucket`/gate re-reads are plain loads; a larger
(non-coherent) L1 keeps those lines longer, so CTAs see staler = looser
thresholds and over-emit; small k is most threshold-sensitive. Fix shipped:
`cudaFuncAttributePreferredSharedMemoryCarveout = MaxShared` at dispatch —
k=128 column back to 0.486-0.497 ms, other cells unchanged.

Also fixed along the way: tidal `bench_b200.py` default `NUM_BUCKETS` 256 -> 64
(NB=256 bucket width < fp16 ULP at layer-0 score scale -> k<=256 direct-lt
recall collapsed to 0.5-0.75; same class as the fp16 bucket invariant).

### flashlib baseline (k = 128 / 1024 only) and figure

Figure: [figures/marsco_baselines_b200.pdf](figures/marsco_baselines_b200.pdf)

flashlib 0.2.0 (PyPI) + nvidia-cutlass-dsl 4.3.5 cp312, vendored under
`_flashlib/` **with two local modifications** (both documented in-source):

1. **METRIC_IP patch** (`triton/insert.py` + `dispatch.py`): flashlib is
   L2-only upstream; MARCO embeddings are not normalized, so its raw result
   set only overlapped the IP ground truth 0.30-0.56. The patch ranks by
   `-x.c` (constexpr `METRIC_IP`, also drops the corpus-norm term) —
   `flash_knn(..., metric="ip")` now measures the same problem:
   **recall 0.9958-0.9989**, same level as LiteTopK.
2. **B200 re-autotune**: the shape heuristic is "derived from a 92-shape
   autotune sweep on H200"; re-running its own brute-force `autotune=True`
   on B200 is worth 1.7-2.6x at k=128. k=1024 sweeps are impractical
   (>60 min per shape without completing — itself evidence).

Final numbers (`bench_flashlib_ip.py`; k=128 autotuned, k=1024 heuristic):

| M | k=128 | k=1024 |
|---:|---:|---:|
| 1M | 3.73 ms | 489.2 ms |
| 2M | 5.56 ms | 734.2 ms |
| 4M | 6.75 ms | 1046.6 ms |
| 5M | 8.41 ms | 1177.5 ms |

- On this task shape flashlib's fused cutedsl/FA3 path **self-gates off**
  (requires D <= 512 and k <= 32; MARSCO is D=768, k >= 128), so its Triton
  `_flash_knn_insert_kernel` is the best available backend.
- Its compiled sm_100a PTX uses **Ampere-generation instructions**
  (`mma.sync.m16n8k16` + `cp.async`, zero tcgen05/TMA): Triton 3.5 only
  lowers `tl.dot` to tcgen05/MMAv5 when the dot's M dim >= 64 (and N >= 16),
  and flashlib's M dim is the query count (8-16 per CTA).
- **tcgen05 port experiment (negative result)**: transposing the dot to
  corpus-major does emit tcgen05 (25 ops, mma.sync -> 0), but measures
  SLOWER (heuristic 5.1 -> 59 ms; autotuned 3.73 -> 3.98 ms at 1M/k128) —
  without TMA descriptors + MMAv5-aware pipelining the kernel is bound by
  its cp.async load pipeline and insert epilogue, not MMA throughput.
  Reverted; giving flashlib the full modern engine would mean rewriting it
  into the same TMA+UMMA warp-specialized design as our CUDA kernel.
- k=1024 remains catastrophic (130x over its own k=128) — the insert-sort
  emit dominates; k=4096/8192 are not measured (per task spec).

### Previous (pre-KDA) forced-QN64 table, for attribution

| M | k=128 | k=1024 | k=4096 | k=8192 |
|---:|---:|---:|---:|---:|
| 1M | 0.540 / 1.59x | 0.638 / 1.35x | 0.843 / 1.02x | 0.992 / 0.97x |
| 2M | 0.780 / 2.07x | 0.924 / 1.75x | 1.238 / 1.32x | 1.459 / 1.18x |
| 4M | 1.179 / 2.81x | 1.368 / 2.44x | 1.847 / 1.81x | 2.193 / 1.56x |
| 5M | 1.434 / 2.94x | 1.634 / 2.60x | 2.143 / 1.99x | 2.541 / 1.71x |

(This forced-QN64 + fp32-out config had already beaten the older per-cell-best
table below — the fp32 emit path and fused sample prep landed after the sweep
that had settled the QN8 fallback: e.g. 5M/k4096 1.99x vs 1.75x.)

### batch 128 reference (fp32 out, QN8 fallback at small-M large-k)

Two QN=64 row groups share the single corpus read via the grid-x lockstep;
ratios rise with batch because the baseline's matmul AND topk double with rows
while LiteTopK's corpus read and fixed stages are shared. 5M/k128 crosses
**3x**; 4M/k128 2.94x exceeds the H100 report's bs=128 numbers (2.86x @4M).

| M | k=128 | k=1024 | k=4096 | k=8192 |
|---:|---:|---:|---:|---:|
| 1M | 0.735 / **1.80x** | 0.998 / **1.33x** | 1.462 / 0.91x | 1.604 / 0.87x |
| 2M | 1.135 / **2.32x** | 1.450 / **1.82x** | 2.600 / **1.02x** | 2.752 / 0.98x |
| 4M | 1.866 / **2.94x** | 2.257 / **2.46x** | 3.255 / **1.71x** | 3.986 / **1.41x** |
| 5M | 2.304 / **3.08x** | 2.733 / **2.62x** | 3.807 / **1.89x** | 4.646 / **1.55x** |

### batch 64 reference (fp16 out, per-cell-best incl. QN8, older defaults)

| M | k=128 | k=1024 | k=4096 | k=8192 |
|---:|---:|---:|---:|---:|
| 1M | 0.564 / 1.51x | 0.695 / 1.24x | 0.863 / 1.00x | 0.953 / 1.01x |
| 2M | 0.834 / 1.93x | 1.014 / 1.60x | 1.444 / 1.13x | 1.540 / 1.12x |
| 4M | 1.319 / 2.51x | 1.561 / 2.14x | 2.099 / 1.59x | 2.470 / 1.39x |
| 5M | 1.586 / 2.66x | 1.870 / 2.27x | 2.444 / 1.75x | 2.877 / 1.51x |

## Fused sample prep (ported from `../dsa/dsa_marsco_v3.cu`)

The v3 insight: the pipeline never needs the exact sample top-K — only
(origin, inv_delta, th, histogram). `tidal_seed_params_kernel` (one block per
row, all state in smem) derives the bucket params from the RAW fp16 sample
scores in two passes (vectorized min/max -> full-range o/inv; smem histogram
-> th = bucket of the K-th best) and zeroes `bcount`/`qcount` in the same
launch. It replaces `neg_ + flash_topk_min + add_sample_idx_offset +
seed_from_sample + 2 memsets` (~8 launches) with ONE. The strided variant
drops v3's pass-3 candidate emission (the full-range scan re-emits sampled
rows, so prefilled seeds would duplicate — the double-count lesson). Bonus:
full-range buckets are wider than the old seed-top-K-range buckets, moving
further from the fp16 ULP constraint. Measured: -7% to -12% per cell; it is
what pushed the last two sub-1.0x cells (1M k4096/k8192) over the line.

## Large-k emit: what was tried

- **Deeper warp queue** (QN8 path, `TIDAL_WARP_QUEUE_CAP`): CAP=48 fits with
  BM=256 and gains ~1%; CAP=64 exceeds smem (234KB). Exhausted lever — flush
  frequency was never the large-k bottleneck.
- **Row-tagged shared queue for QN=64** (`TIDAL_RTQ=1`, default off): 8 rows
  (= one TMEM slice) share a per-warp smem queue slot with the local row id
  packed into idx bits [28,31) — 12.3KB instead of the 100KB per-row layout,
  so it FITS in shared memory. It is correct (recall 0.999) but blocked by a
  different wall: BM=256 runs 384 threads/CTA, so `__launch_bounds__` caps the
  compiled register budget at ~170 (the runtime `warpgroup_reg_alloc<200>`
  cannot exceed what ptxas compiled for); the queue machinery pushes spills
  into the hot loop — 3x kernel regression at every k. To unlock it: shrink
  the specialized warpgroup to 64 threads (TMA+UMMA only, 320 threads -> ~204
  regs), which requires moving the in-scan refresh back onto the math warps
  (DSA-style) — a coherent but separate surgery.

## QN=64 single-pass mode (implemented, opt-in via `TIDAL_FLAT_QN=64`)

The kernel now supports putting all 64 queries in the UMMA N dimension
(TMEM 64x2 accumulators drained in 8-column slices, per-row coefficients in
smem, corpus read ONCE from DRAM instead of relayed 8x through L2). The
engine side works as designed — the dense-epilogue variant reads the corpus
at 5.5 TB/s (0.26 ms @1M vs 0.65 ms for the 8-group relay). But the SPARSE
epilogue does not yet keep up: with the warp queue disabled (it does not fit
smem at 64 rows) the direct-atomic emit pays L2 atomic round-trips per
8-row slice even after parallelizing the reservations across lanes (one
warp-wide atomicAdd + shfl broadcast, worth ~150 µs @1M), and at k >= 1024
the emit volume makes QN=64 clearly slower than the QN=8 path (1M/k8192:
1.80 vs 1.11 ms). Net: ~parity at k=128, behind at large k — so QN=8 stays
the default. To make QN=64 pay: a row-grouped smem queue that fits 64 rows,
and/or software-pipelining the slice reservations across slices.

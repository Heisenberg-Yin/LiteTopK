# KDA Plan — marsco-sm100-scan-opt (from draft.md + run01 evidence)

## Evidence base (profile/run01_1m_k8192)

Scan kernel @1M/k8192 (767µs device = 80% of e2e 961µs; ncu duration 1.257ms under replay):
- Stalls per issued inst (total 26.6): no_instruction 8.32 (31%), long_scoreboard 6.36 (24%),
  sleeping 4.64 (17%), wait 2.60, short_scoreboard 1.76, barrier 1.13.
- DRAM read 15%, tensor pipe 5.8%, issue active 11.3% — latency-bound, nothing saturated.
- Spills present (168 regs + 387K local ops) but only ~0.5% of 81M instructions.
- Occupancy by design: 1 CTA × 384 thr = 12 warps/SM (smem+reg block limit 1).
- Structural: 1M/BM256 = 3906 tiles / 296 CTAs = 13.2 tiles/CTA; 3-stage pipeline fill/drain ~23%.
- Non-scan: compact_thr select 85µs (9%), gather 39µs, sample matmul 19µs, boundary 18µs, seed 13µs.
- Hyperparameter grid EXHAUSTED at this cell (ctas {148,296} × sample {64K,128K,256K} ×
  refresh {4,8}): incumbent (296/65536/8) = best 1.0046ms on GPU2; bigger samples lose to
  their own scoring cost; refresh=4 loses. Kernel changes are the only lever left.

## Candidates (ranked by expected value / risk)

- **C1 (cheap, select-side): vectorize compact_thr sparse branch.** marsco fp32 select passes
  buf_idx≠null, sample_idx=null → scalar fallback today. Both val and idx rows are contiguous
  and 16B-aligned (BUF % VEC == 0). int4-load values; load idx int4s only when the 4/8-element
  group has hits. Target 85µs → <25µs (+6-8% e2e on weak cell). Risk: low (select-only,
  recall must stay ≥0.99; also used by tidal select path — keep fp16 semantics identical).
- **C2 (medium, scan): SPEC warpgroup 128→64** (drop 2 spare warps; move in-scan refresh to
  math warps DSA-style, or keep 1 spare warp within 64): 320 threads → launch_bounds regs
  ~204/thread. Attacks: sleeping 17%, spills, and unlocks C3. Risk: touches warp layout,
  barrier arrival counts, reg alloc/dealloc numbers.
- **C3 (big, scan): row-tagged smem queue for QN64 emit** (previously 3× regression at 168-reg
  ceiling; C2 lifts ceiling). Attacks: scattered stores (17.3/32 B/sector), reservation
  atomics, and emit code footprint (no_instruction).
- **C4 (probe): icache diet** — __noinline__ cold paths / narrower unrolls in math slice loop.
  Only if C2/C3 don't already move no_instruction.

## Validation per candidate

1M/k8192 (weak cell) + 1M/k128 + 5M/k128 + 5M/k8192 corners, WARMUP=10 ITERS=20, idle GPU;
recall ≥ 0.99 each; promote per contract (≥2% win, no >2% regression). Log to candidates.jsonl.

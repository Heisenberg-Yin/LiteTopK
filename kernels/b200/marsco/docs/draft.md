# KDA Draft — MARSCO SM100 scan kernel optimization round

Workflow: kernel-design-agents (KDA) + ncu-report-skill. Profile → Diagnose → Plan; no guessing.

## Task contract

- **Task name**: `marsco-sm100-scan-opt`
- **Objective**: reduce LiteTopK fused end-to-end latency on the MARSCO matrix
  (bs=64, fp16 in / fp32 out, forced QN64+BM256+grid296, REFRESH=8), with priority on
  the weakest cells: 1M/k8192 (0.97x, only losing cell), 1M/k4096 (1.02x), 2M/k8192 (1.18x).
- **Correctness requirements**: recall ≥ 0.99 vs dense cuBLAS+topk on every touched cell.
- **Performance target**: 1M/k8192 ≥ 1.0x; no cell regresses by >2%.
- **Allowed approaches**: CUDA/CuTe changes to `sm100_tidal_marsco.cuh` + host ops;
  no algorithm-level recall changes; tidal task must keep passing (shared kernel).
- **Validation command**: `bench_marsco_b200.py --m <M> --k <k>` in container `simtopk`,
  idle GPU, `WARMUP=10 ITERS=20`; recall printed by the bench.
- **Evaluation command**: same (latency + speedup line), plus 16-cell matrix for promotion.
- **Promotion criteria**: cell latency improves ≥ 2% with recall PASS and no >2%
  regression elsewhere; evidence = benchmark lines + ncu metric delta recorded in
  `profile/<run>/` and `candidates.jsonl`.

## Baseline (validated 2026-07-09, idle GPU 1)

| M | k=128 | k=1024 | k=4096 | k=8192 |
|---:|---:|---:|---:|---:|
| 1M | 0.540 / 1.59x | 0.638 / 1.35x | 0.843 / 1.02x | 0.992 / 0.97x |
| 2M | 0.780 / 2.07x | 0.924 / 1.75x | 1.238 / 1.32x | 1.459 / 1.18x |
| 4M | 1.179 / 2.81x | 1.368 / 2.44x | 1.847 / 1.81x | 2.193 / 1.56x |
| 5M | 1.434 / 2.94x | 1.634 / 2.60x | 2.143 / 1.99x | 2.541 / 1.71x |

All recall ≥ 0.9958. Known anatomy (pre-NCU, from bench-level probes): scan kernel
dominates; large-k cost is emission + select; small-M large-k has least corpus I/O to
amortize fixed stages against.

## Risks / unknowns (to be resolved by profiling)

1. Is the 1M/k8192 scan emission-bound (atomic/store pressure), latency-bound, or
   is the select/epilogue chain the gap? Never NCU-profiled — all prior anatomy was
   wall-clock ablation.
2. How much headroom vs DRAM floor at 5M/k128 (are we already at the corpus-read wall)?
3. Register ceiling (BM256 compiles at ~168 regs, 384 threads): are spills present in
   the hot loop already?
4. Launch-gap overhead share at 1M (CUDA-graph opportunity was estimated ~50µs on tidal).

## Profiling plan (evidence before candidates)

- Container `simtopk`, ncu 2025.1, pick an idle GPU not used by the concurrent
  session (another ncu is live on vllm-prefill/GPU1).
- Shapes: 1M/k8192 (weak cell) and 5M/k128 (strong cell, floor check).
- Recipe 1 (`--set full` + PmSampling) on `regex:sm100_tidal`, skip warmup launches
  (`-s`), then Recipe 3 details page; per-line stalls via `--set source` if needed
  (extension already builds with `-lineinfo`).
- Also capture the full launch timeline (`nsys` or ncu `-c N` over all kernels) to
  size the non-scan stages (seed params, select, gather).

## Candidate directions (to rank AFTER evidence; expected-value guesses only)

- C1: emission-path relief for QN64 large-k (slice-pipelined reservations or
  smem row-grouped staging) — blocked previously by register ceiling; needs stall evidence.
- C2: CUDA graph / launch fusion for the fixed-stage chain (helps small-M most).
- C3: select-kernel work reduction at large k (bucket<th direct copy is output-bound).
- C4: KV pipeline depth / CHUNK_D retune for QN64 (stage budget 32KB — check LG throttle).
- C5: spare-warp refresh cost trim (fixed ~15µs arming cost measured on tidal).

## First concrete steps

1. Smoke-test ncu on an idle GPU (done: blocked by concurrent session; retry on GPU 2/3).
2. Full profile of the two shapes; write `profile/run01_1m_k8192/REPORT.md`.
3. Rank candidates by measured stall/throughput evidence; convert this draft to plan.md.
4. Implement one candidate at a time; validate per contract; log to candidates.jsonl.

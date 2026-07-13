# NCU Report — marsco scan kernel @ 1M/k8192 (weak cell), 2026-07-09

Question: why is 1M/k8192 the only losing cell (0.97x) — where does the scan's
time go when DRAM is only 15% utilized?

## Collection

- Container `simtopk`, ncu 2025.1, GPU 2 (idle; another session was profiling
  on a different GPU — B200 driver allows concurrent sessions per GPU).
- `--set full --section PmSampling --section PmSampling_WarpStates
  -k regex:sm100_tidal -s 3 -c 1` over `bench_marsco_b200.py --m 1000000 --k 8192`
  (WARMUP=3 ITERS=2, so -s 3 catches the first steady-state timing launch).
- Reports in `reports/`, extraction scripts + details page in `analysis/`.
- Kernel: `sm100_tidal_ip<64,768,256,3,148,128,256,float,2>` grid (1,296) x 384.

## Findings (baseline, spec=128)

Torch-profiler breakdown: scan 766.8 µs = 80% of 961 µs device; compact select
85.1 µs = 9%; gather 38.7; sample matmul 18.9; boundary radix 17.6; seed 13.0.

Scan kernel (ncu, duration 1.257 ms under replay):
- Stall mix (cycles per issued inst, total 26.6): **no_instruction 8.32 (31%)**,
  **long_scoreboard 6.36 (24%)**, **sleeping 4.64 (17%)**, wait 2.60,
  short_scoreboard 1.76, barrier 1.13.
- DRAM read 15%, tensor pipe 5.8%, issue active 11.3% — latency-bound, nothing
  saturated. 168 regs/thread + 387K local ld/st (spills present, ~0.5% of 81M inst).
- Occupancy by design: 1 CTA (384 thr = 12 warps)/SM; block limited by smem+regs.
- Structural: 3906 tiles / 296 CTAs = 13.2 tiles/CTA — 3-stage pipeline
  fill/drain is ~23% at this M (amortized at 5M: 66 tiles/CTA).

Compact select kernel (targeted metrics): long_scoreboard 59 cyc/issue,
DRAM 1.78%, 229K L2 atomic sectors — bound by per-warp global atomicAdd
round-trips on 64 row counters, NOT by loads.

## Actions taken (see ../../candidates.jsonl)

- C1 staged compact (smem staging + watermark block-flush): 101 -> 17.9 µs.
- C1b register-copy queue counts (user-proposed, QN8 path).
- C2 SPEC_THREADS 128 -> 64, refresh inlined into math warps: sleeping
  4.64 -> 1.58, no_instruction 8.32 -> 7.06, regs 168 -> 112, local ops -25%;
  scan (ncu replay) 1.257 -> 1.13 ms; every cell -4~-12%.

## Post-state and remaining headroom

Weak cell e2e 0.992 -> 0.879 ms (0.97x -> 1.11x). Remaining gaps, evidence-ranked:
1. no_instruction still ~27% of stall cycles — icache pressure of the fused
   emit/epilogue loop; candidate: code-diet/noinline cold paths (C4, unproven).
2. long_scoreboard ~24% — TMEM drain + emit store latency; candidate: C3
   row-tagged smem queue for QN64, now unblocked by C2's 204-reg budget.
3. 13 tiles/CTA pipeline fill at 1M is inherent to grid 296; grid 148 measured
   worse (hyperparam regrid) — parked.

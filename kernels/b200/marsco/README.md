# LiteTopK B200 marsco (MS MARCO IP top-k) — How to Test & Results

Batch retrieval over MS MARCO 768-d embeddings on B200 (SM100): **64 concurrent
queries** against a shared corpus, `M ∈ {1M, 2M, 4M, 5M}`, `k ∈ {128, 1024,
4096, 8192}`, exact top-k by inner product. Pipeline: sample → gate (bucket
threshold) → SM100 UMMA/TMEM sparse scan → boundary select; recall exact by
construction.

## How to test

Environment (container `simtopk`, torch 2.12 / nvcc):

- **Data**: `MARSCO_DATA=<dir>` holding `base_5m_fp16_768.bin` (5M × 768 fp16
  corpus; the bench slices the first `M` rows). First run converts
  `base_5m.fvecs` to this fp16 `.bin` cache if it is absent.
- **Kernel**: `sm100_litetopk_marsco.cuh` (SM100 UMMA/TMEM scan) +
  `litetopk_sm100_torch.cu` (host binding + seed-prep) + `litetopk_select.cu`
  (boundary select); JIT loader `litetopk_ops.py`.
- **Build env**: `LITETOPK_CUTLASS_INCLUDE=<cutlass>/include`,
  `DSA_DEEP_GEMM_INCLUDE=<deepgemm>/include` (the **legacy** deep_gemm — the
  new DG25 API is incompatible), `TORCH_CUDA_ARCH_LIST=10.0a`.

Run one cell (JIT-builds the extension on first call):

```bash
CUDA_VISIBLE_DEVICES=3 python3 bench_marsco_b200.py --m 1000000 --k 128
```

Baseline = dense `q @ base.T` (cuBLAS) + `torch.topk`. Each cell prints LiteTopK
latency, speedup, and a **full recall check** against the exact dense+topk
result. **Recall ≥ 0.996 on every cell is the red line** (guaranteed by
construction: the sample's k-th score is a safe bound and boundary buckets are
re-scanned exactly).

## Results

B200, batch 64, fp16 in / fp32 out, forced QN=64, warmup 10 / iters 20, idle GPU.
Cell = LiteTopK ms / speedup; recall ≥ 0.9958 on all 16 cells.

| M | k=128 | k=1024 | k=4096 | k=8192 |
|---:|---:|---:|---:|---:|
| 1M | 0.497 / **1.72x** | 0.577 / **1.50x** | 0.756 / **1.14x** | 0.864 / **1.13x** |
| 2M | 0.683 / **2.37x** | 0.800 / **2.03x** | 1.064 / **1.53x** | 1.249 / **1.38x** |
| 4M | 1.106 / **3.00x** | 1.243 / **2.69x** | 1.566 / **2.14x** | 1.868 / **1.84x** |
| 5M | 1.319 / **3.20x** | 1.466 / **2.90x** | 1.795 / **2.38x** | 2.146 / **2.03x** |

16/16 cells ≥ 1.13x; 4M/k128 3.00x, 5M/k128 3.20x, 5M/k8192 crosses 2x.

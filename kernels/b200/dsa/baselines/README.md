# Baselines (vendored)

The two comparison baselines used by the B200 DSA benchmarks
(`threeway_b200.py`, `sweep_b200_real_chunk.py`, `figures/dsa_3way.pdf`),
preserved in-repo so the comparisons stay reproducible without external paths.

## 1. `deepgemm_patched_official/` — the "official" baseline (LiteTopK up to **3.39x** vs this)

The official DeepSeek **DeepGEMM** package (SM100/tcgen05 JIT kernels) with the
one-line SGLang #19529 patch in
`deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh`:

```cpp
// original: constexpr uint32_t kNumWeightsInReg = 52;   // requires num_heads >= 52
constexpr uint32_t kNumWeightsInReg = (kNumHeads < 52) ? (kNumHeads / 4 * 4) : 52;
```

Without the patch the stock kernel static-asserts and fails to compile for the
GLM-5 indexer shape (`num_heads=32`, `head_dim=128`, repro model
`zai-org/GLM-5-FP8`). The patched kernel produces bit-identical logits to our
dense kernel (top-k recall 100%).

Baseline pipeline: `deep_gemm.fp8_mqa_logits` materializes the full `[Q, S]`
logits to HBM (32 GB at 1M/chunk 8192), then `torch.topk` reads them back.
`threeway_b200.py` picks this copy up automatically (it prepends this directory
to `sys.path`); to use it elsewhere put this directory first on `PYTHONPATH`.

## 2. `dense/` — the "ideal dense" baseline (LiteTopK up to **2.73x** vs this)

Frozen snapshot of our own KV-split dense fp8 scoring kernel — the **same**
sources as `../dsa_marsco.cu` / `../sm100_dsa_marsco.cuh`, built with
`-DDENSE_WRITE` (write every logit densely, no sparse gate/queue) + `torch.topk`.
It shares the tile/pipeline optimizations with LiteTopK, so it is a *stronger*
baseline than the official kernel (better tile/epilogue, KV-split scheduling).

The live benchmarks build the dense baseline from the parent-directory sources
(so both sides always share the same kernel code); this snapshot pins the exact
sources behind the published 2.73x numbers. To benchmark this frozen copy
explicitly, point the JIT build at it, e.g.
`load(sources=["baselines/dense/dsa_marsco.cu"], extra_cuda_cflags=[..., "-DDENSE_WRITE"])`
(quoted includes resolve to this directory first).

## Headline comparison (real GLM-5 caches, K=2048, recall 100% everywhere)

| seq/chunk | official (ms) | dense (ms) | LiteTopK (ms) | vs official | vs dense |
|---:|---:|---:|---:|---:|---:|
| 1M / 8192 | 182.89 | 147.00 | 53.94 | **3.39x** | **2.73x** |

Full tables: `../threeway_b200.json`, `../figure_b200_realchunk.json`,
figures in `../figures/`.

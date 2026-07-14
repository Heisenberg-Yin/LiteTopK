# LiteTopK B200 DSA — How to Test & Results

Benchmark workspace for the GLM-5.2 vLLM prefill sparse top-k indexer (the
LiteTopK v3 kernel), scored against the official dense `mqa_logits` + `top_k`
path on real GLM-5 KV.

## How to test

Environment (container `vllm-prefill`, `/opt/vllm-venv/bin/python`):

- **Data**: `/data/dsa_caches/glm5_{256k,512k,768k,1m}_realtext_chunk8192.safetensors`
  (real text, K=2048, Q up to 8192).
- **Kernel**: `dsa_litetopk.cu` + `sm100_dsa_litetopk.cuh`; module
  `glm5_prefill/litetopk_vllm/litetopk_indexer.py`.

Run the main scoreboard — Q=8192, four scales (256K/512K/768K/1M), official vs
ours with a full recall check:

```bash
CUDA_VISIBLE_DEVICES=3 python bench_q8192.py
```

It builds the GATE4 bucket-gate kernel (the shipped default), warms up, then
reports per-cell latency + speedup vs the official path. **Recall is checked at
100% on every cell** (against the exact official dense+topk result — anything
below 100% is a failure).

## Results

### Kernel (Q=8192, real GLM-5 KV, recall 100%, reproduced ×2)

| | 256K | 512K | 768K | 1M |
|---|---|---|---|---|
| ours (ms) | 12.61–12.70 | 23.64–23.66 | 33.93–34.06 | 43.53–43.75 |
| vs official | 1.05x | 1.13x | 1.18–1.20x | 1.23–1.24x |

### Full-model E2E (78L GLM-5.2-FP8, TP8+EP, 8×B200, chunk 8192, prefill wall seconds)

Prefix caching off

| | 512K | 768K | 1M |
|---|---|---|---|
| official default (B=512MB) | 55.2 | 122.3 | 236.3 |
| **ours hybrid** | **46.66** | **81.82** | **124.41** |

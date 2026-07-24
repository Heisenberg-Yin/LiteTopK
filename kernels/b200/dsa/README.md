# LiteTopK B200 DSA — How to Test & Results

Benchmark workspace for the GLM-5.2 vLLM prefill sparse top-k indexer (the
LiteTopK v3 kernel), scored against the official dense `mqa_logits` + `top_k`
path on real GLM-5 KV.

**`bench_q8192.py`** is the one and only way to reproduce it: it builds
`dsa_litetopk.cu` + `sm100_dsa_litetopk.cuh` (the same source vLLM's
production op compiles) and scores it inside the full vLLM stack. 1M:
**1.27–1.28x**.

(A second, standalone repro used to live here — a prebuilt `.so` from a
separately-tuned kernel copy that briefly ran ahead of this one, 1.25–1.26x
@1M. Its tuning wins were backported below and it's since been removed;
`bench_q8192.py` alone is now faster, so there's no reason to keep both.)


## How to test

Environment (container `glm5-prefill`, `/opt/vllm-venv/bin/python`):

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

## Results (Q=8192, real GLM-5 KV, recall 100%, reproduced ×3 clean rebuilds)

| | 256K | 512K | 768K | 1M |
|---|---|---|---|---|
| ours (ms) | 11.74–11.93 | 21.92–22.04 | 31.80–31.94 | 41.79–41.95 |
| vs official | 1.13x | 1.21–1.22x | 1.25–1.28x | 1.27–1.28x |

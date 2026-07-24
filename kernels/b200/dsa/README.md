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

## 2026-07-24 backport: TMEM double buffering

`sm100_dsa_litetopk.cuh` was missing four tuning wins that the (now-removed)
standalone kernel copy already had: TMEM double buffering
(`DSA_UMMA_STAGES=2`), 4-row TMEM loads (`DSA_TMEM_ROWS=4`, was 2), the 240/24
register split (was 232/40), and `DSA_REFRESH_STRIDE=2` (was 16) — the four
are coupled, not independent (see the comment on `DSA_REFRESH_STRIDE`).
Backporting them lifted the kernel from 1.23–1.24x to **1.27–1.28x @1M**,
recall still 100%, reproduced across 3 clean rebuilds.

The one non-obvious catch: doubling the UMMA buffer count in the kernel also
doubles the barrier count in `dsa_litetopk.cu`'s `compute_smem_bytes()` (the
**host-side** dynamic-shared-memory sizing passed to `cudaFuncSetAttribute`)
— missing that update compiles fine but is an illegal memory access at launch
(the kernel's internal barrier/TMEM-pointer offsets run past whatever smem
was actually allocated).

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

### Full-model E2E (78L GLM-5.2-FP8, TP8+EP, 8×B200, chunk 8192, prefill wall seconds)

Prefix caching off. **Predates the 2026-07-24 backport above — not yet
re-measured with it**, so treat as a lower bound, not current.

| | 512K | 768K | 1M |
|---|---|---|---|
| official default (B=512MB) | 55.2 | 122.3 | 236.3 |
| **ours hybrid** | **46.66** | **81.82** | **124.41** |

## bench_whole_dsa.py — indexer + attention together

Whole DSA = the fused indexer top-k above **plus** the sparse MLA attention
that consumes its output. Compares three combinations on the same off-shelf
cache (real GLM-5.2 layer-0 indexer inputs + real absorbed MLA q/kv), all fed
the identical real `topk_idx`, so only the attention kernel varies:

- **vLLM**: dense indexer + stock trtllm sparse MLA (`flashinfer`)
- **LiteTopK**: this repo's fused indexer (the 1.27–1.28x kernel above) + stock
  trtllm sparse MLA
- **LiteTopK+LiteDSA**: same indexer + `flashinfer_port/litedsa.so` (grouped
  masked MLA, G=16 union-dedup attention)

Self-contained to this repo — the indexer goes through `litetopk_indexer.py`
(same call `bench_q8192.py` times) and LiteDSA goes through
`flashinfer_port/litedsa.so` directly via `tvm_ffi`. No external vLLM fork
needed (unlike the version of this script that used to live under
`repro_1p26x/`, which required a separately-built `vllm-lt-venv`).

```bash
CUDA_VISIBLE_DEVICES=3 TAGS="768k 1m" python bench_whole_dsa.py
```

### Results (2026-07-24, idx recall 99.98%, attn max abs diff vs stock ≤ 0.0084)

| S | idx: vLLM / LiteTopK (ms) | attn: stock / LiteDSA (ms) | whole: vLLM / LiteTopK / LiteTopK+LiteDSA (ms) | speedup |
|---|---|---|---|---|
| 768K | 39.5–39.8 / 32.0–32.1 | 1.55 / 0.47 | 41.0–41.4 / 33.5–33.7 / 32.4–32.6 | **1.26–1.27x** |
| 1M | 53.6–53.7 / 42.3–42.4 | 1.78–1.79 / 0.98–0.99 | 55.4–55.5 / 44.1–44.2 / 43.3–43.4 | **1.28x** |

LiteDSA's grouped attention is itself ~3.3–3.6x faster than stock trtllm sparse
MLA at this shape (0.47–0.99ms vs 1.55–1.79ms) from the union-dedup, but at
Q=8192/K=2048 the indexer dominates whole-DSA latency (attention is only
~2–4% of the total either way) — so LiteTopK+LiteDSA's edge over plain
LiteTopK here is modest (~1.03x), and the bulk of the whole-DSA win over
stock vLLM comes from the indexer, not the attention swap.

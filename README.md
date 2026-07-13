# LiteTopK — Reproduction Package

Code to reproduce every experimental result in the paper
([`litetopk_arxiv.pdf`](litetopk_arxiv.pdf)). 

LiteTopK is a fused SM100 (B200 / Blackwell) top‑k kernel with three
deployments, and this package reproduces one figure for each:

- **MARSCO retrieval** — batched exact inner‑product top‑k over MS MARCO.
- **DSA indexer** — the sparse top‑k indexer for GLM‑5.2 vLLM prefill.
- **GLM‑5.2 end‑to‑end** — the full 78‑layer model on 8×B200 via a vLLM hook.

Every result is verified for correctness: **100% recall** on the DSA and
end‑to‑end paths, **≥0.99 recall** on MARSCO.

## Repository layout

```
kernels/b200/            SM100 (B200) kernels + benchmarks
  dsa/                   DSA LiteTopK: dsa_marsco{,_v2,_v3}.cu + sm100_*.cuh,
                         sweep_b200_real_chunk.py (16-cell real-chunk figure),
                         test_dsa.py, threeway_b200.py, baselines/
    kda_bench/           bench_q8192.py — the 4-shape (256K/512K/768K/1M)
                         Q=8192 scoreboard vs official on real KV; drives the
                         DSA kernel via glm5_prefill/litetopk_vllm.
  marsco/                MS MARCO batch retrieval: bench_marsco_b200.py,
                         sweep_marsco_b200.py, bench_flashlib_ip.py,
                         litetopk_engine/ (kernel + JIT loader),
                         _flashlib/ (vendored flashlib 0.2.0 + METRIC_IP patch),
                         docs/, profile/ (reports + code only)
kernels/h100/            H100 originals (dsa / marsco);
                         gen_dsa_caches.py lives in h100/dsa
vllm/                    FULL vLLM v0.23.0 source tree (@0fc695f, .git stripped)
                         with our changes applied. Only two files differ from
                         official:
                           model_executor/layers/sparse_attn_indexer.py  (hook)
                           v1/attention/backends/mla/indexer.py           (env)
vllm_patches/            Unified diffs of those two files vs official v0.23.0.
glm5_prefill/            GLM-5.2 E2E harness: run_prefill.py + run_e2e_best.sh
                         (the single fastest measured config),
                         build_fp8_truncated.py (16L/1L shadow builder),
                         litetopk_vllm/ (the indexer module the vLLM hook loads)
deepgemm_patch/          1-file patch to deep_gemm (SGLang #19529 class fix)
                         for GLM-5's 32-head shape.
```

Per-directory documentation:
[kernels/b200/dsa/kda_bench](kernels/b200/dsa/kda_bench/README.md) ·
[kernels/b200/marsco](kernels/b200/marsco/README.md) ·
[vllm_patches](vllm_patches/README.md) ·
[deepgemm_patch](deepgemm_patch/README.md)

---


## Configuration — placeholder paths

This package is anonymized: absolute paths are generic placeholders. Point
them at your machine by editing the placeholder or (for the load-bearing
ones) via environment variables.

| Placeholder | Env var | What it is |
|---|---|---|
| `/opt/simtopk_repro` | — (edit) | this package's root |
| `/opt/simtopk_repro/kernels/b200/dsa` | `LITETOPK_DSA_DIR` | DSA kernel sources the module JIT-builds |
| `/opt/simtopk_repro/glm5_prefill/litetopk_vllm` | `LITETOPK_MODULE_DIR` | the indexer module the vLLM hook imports |
| `/opt/glm5_prefill_test/DeepGEMM` | `DEEPGEMM_DIR` | DeepGEMM 2.5 source (E2E) |
| `/opt/venvs/deepgemm/.../deep_gemm/include` | `DEEPGEMM_INCLUDE` | deep_gemm headers |
| `/opt/cutlass/include` | `CUTLASS_INCLUDE` | CUTLASS 4.2.1 headers |
| `/models/glm5-fp8-official` | `MODEL` | GLM-5.2-FP8 weights |
| `/models/glm5`, `/data/dsa_caches` | `DSA_CACHE_DIR` | GLM-5 caches / chunk caches |
| `/data` (marsco) | `MARSCO_DATA` | MS MARCO fvecs |
| container `vllm-prefill`, `simtopk` | — (edit `docker exec`) | your containers |

> **Note:** the vLLM hook hardcodes the module path
> `/opt/simtopk_repro/glm5_prefill/litetopk_vllm` into `sys.path`. When
> reproducing elsewhere, point it at `glm5_prefill/litetopk_vllm`
> (see [vllm_patches/README.md](vllm_patches/README.md)).

---

## Third-party dependencies (not vendored)

- **DeepGEMM @ 891d57b** (vLLM-pinned): clone and build from source
  (`git clone https://github.com/deepseek-ai/DeepGEMM && git checkout 891d57b`,
  or vllm's `tools/install_deepgemm.sh`). CUDA 12.8 works at this commit;
  later HEADs need >12.8. No local modifications.
- **nvidia-cutlass-dsl 4.3.5 (cp312)** — needed next to
  `kernels/b200/marsco/_flashlib/flashlib/`; obtain via
  `pip download nvidia-cutlass-dsl==4.3.5 --python-version 3.12` and extract.
  (Our patched `flashlib` — the METRIC_IP patch, no upstream — **is** vendored.)
- **vLLM v0.23.0** — full modified source vendored at [`vllm/`](vllm/).
  Deployment used the official wheel with the two modified files overwritten
  in site-packages (fastest); building `vllm/` from source also works.
- The removed `kernels/b200/dsa/baselines/deepgemm_patched_official/` was a
  full deep_gemm package copy whose only delta is
  `deepgemm_patch/sm100_fp8_mqa_logits.cuh` — rebuild it by copying the venv's
  deep_gemm package and overwriting that one header.

---

## Data (excluded — locations + generators)

- **DSA GLM-5 chunked caches**: `~/glm5/dsa_caches`
  (`glm5_{tag}_realtext_chunk{N}.safetensors`) — regenerate with
  `kernels/h100/dsa/gen_dsa_caches.py` (needs GLM-5 fp8 shard 1 + tokenizer
  in `~/glm5/`).
- **MS MARCO corpus** (marsco benches): see
  [kernels/b200/marsco/README.md](kernels/b200/marsco/README.md).
- **GLM-5.2 checkpoints**: official FP8 repo `zai-org/GLM-5.2-FP8`
  (`~/glm5-fp8-official`, 141 shards); truncated 16L/1L shadows built by
  `glm5_prefill/build_fp8_truncated.py`.
- **KDA ref-topk caches** (`refcache_*.pt`): auto-regenerated by the benches.

---

## Reproducing the experiments

Numbers below are fresh runs on the 8×B200 host; treat them as the target a
successful reproduction should land near. Small deltas (a few %) are normal —
fresh-run vs idle-GPU-best variance (during these runs GPU 0 carried a
~1.1 GB neighbor allocation throughout). Recall and the qualitative pattern
reproduce exactly.

### 1. MARSCO retrieval

Container `simtopk`, GPU 0, full 16-cell grid, batch 64, fp32-out, defaults.

```bash
kernels/b200/marsco/bench_marsco_b200.py     # single 16-cell grid
kernels/b200/marsco/sweep_marsco_b200.py     # full sweep
kernels/b200/marsco/bench_flashlib_ip.py     # flashlib baseline
```

Each cell = LiteTopK ms / speedup vs cuBLAS dense+top‑k. Recall
0.9958–0.9990 on every cell (red line ≥0.9958) — all PASS.

The speedup pattern — grows with M, shrinks with k — is the headline. The op
label prints `LiteTopK SM100 fused`.

Final matrix + methodology: [kernels/b200/marsco/README.md](kernels/b200/marsco/README.md)
and its `docs/`.

### 2. DSA indexer kernel — Q=8192 scoreboard

Container `vllm-prefill`, real GLM-5 chunk caches, Q=8192.

```bash
VLLM_LITETOPK_PREP_TILE=0 kernels/b200/dsa/kda_bench/bench_q8192.py
```

```bash
VLLM_LITETOPK_PREP_TILE=0 VLLM_LITETOPK_XFLAGS=DSA_BUCKET_GATE4=1 \
  kernels/b200/dsa/kda_bench/bench_q8192.py
# and force the prefix plan: try_chunk(..., strided_plan=False)
```

Methodology (forced sampling modes, ref caching, recall red line):
[kernels/b200/dsa/kda_bench/README.md](kernels/b200/dsa/kda_bench/README.md).

### 3. DSA kernel — 16-cell real-chunk figure

Container `simtopk`.

```bash
kernels/b200/dsa/sweep_b200_real_chunk.py    # ~4 min → figure_b200_realchunk.json
```

Scripts that need the official deep_gemm reference (`test_dsa.py`,
`threeway_b200.py`) must prepend the patched deep_gemm package dir to
`PYTHONPATH` — see [deepgemm_patch/README.md](deepgemm_patch/README.md).

### 4. GLM-5.2 end-to-end prefill

Container `vllm-prefill`, full 78L GLM-5.2-FP8, TP8+EP, 8×B200.

```bash
# install vLLM with the two modified files (see vllm/ + vllm_patches/README.md), then:
glm5_prefill/run_e2e_best.sh
```

`run_e2e_best.sh` is the single fastest measured config

Reproduces within ~1%. A `killed` task status printed *after* the results is
normal — it is vLLM's SIGTERM engine teardown (`PREFILL-EXIT=0`), not a run
failure.

---

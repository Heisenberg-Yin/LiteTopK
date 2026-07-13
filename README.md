# LiteTopK — Reproduction Package (code only)

All code needed to reproduce our experimental results. **No data / weights /
logs / profiles / plotting code are included** — data locations and
generation scripts are documented below.

## Configuration — placeholder paths

This package is anonymized: absolute paths are generic placeholders. Point
them at your machine either by editing the placeholder or (for the
load-bearing ones) via environment variables:

| placeholder | env var | what it is |
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

Snapshot date: 2026-07-12. Source trees:
`~/github_simtopk`, `~/glm5_prefill_test`, `~/simtopk_kda`,
`~/deepgemm_patched`, container `vllm-prefill:/opt/vllm-venv`.

## Layout

```
kernels/b200/            SM100 (B200) kernels + benches
  dsa/                   DSA LiteTopK: dsa_marsco{,_v2,_v3}.cu + sm100_*.cuh,
                         sweeps (sweep_b200_real_chunk.py = main 16-cell E2E),
                         test_dsa.py, threeway_b200.py, baselines/
    kda_bench/           bench_q8192.py — the main 4-shape (256K/512K/768K/1M)
                         Q=8192 scoreboard vs official, real KV; drives the
                         DSA kernel through glm5_prefill/litetopk_vllm. README.md
                         = bench methodology + the honest result tables.
  marsco/                MS MARCO batch retrieval: bench_marsco_b200.py,
                         sweep_marsco_b200.py, bench_flashlib_ip.py,
                         litetopk_engine/ (the LiteTopK kernel + JIT loader,
                         formerly under tidaldecode/ — compile verified from
                         this location), _flashlib/flashlib/ (vendored
                         flashlib 0.2.0 WITH our METRIC_IP patch), docs/,
                         profile/ (reports+code only)
kernels/h100/            H100 originals (dsa / marsco);
                         gen_dsa_caches.py lives in h100/dsa
vllm/                    FULL vLLM v0.23.0 source tree (@0fc695f, .git
                         stripped) WITH our modifications applied. Only two
                         files differ from official (verified by full-tree
                         diff of the deployed container venv):
                         vllm/model_executor/layers/sparse_attn_indexer.py
                         (LiteTopK hook) and
                         vllm/v1/attention/backends/mla/indexer.py
                         (VLLM_LITETOPK_RUNTIME_LOGITS_MB).
vllm_patches/            Unified diffs of those two files vs official
                         v0.23.0, for quick review / patching an installed
                         wheel. See its README.
glm5_prefill/            GLM-5.2 E2E harness: run_prefill.py +
                         run_e2e_best.sh (the ONE fastest measured config),
                         build_fp8_truncated.py (16L/1L shadow builder),
                         litetopk_vllm/ (the LiteTopK indexer module loaded by
                         the vLLM hook, + recall verification)
deepgemm_patch/          1-file patch to deep_gemm (SGLang #19529 class fix)
                         for GLM-5's 32-head shape. See its README.
```

Removed relative to the working trees: the TidalDecode experiment (benches,
official baseline, debug tools — only its engine sources survive, relocated
to `kernels/b200/marsco/litetopk_engine/`), all figure/plotting scripts, the
non-benchmark parts of the `simtopk_kda` workspace (diagnostics, contracts,
result records), and all experimental glm5_prefill run/patch scripts (only
the fastest config survives as `run_e2e_best.sh`).

## Environments

Two docker containers on the 8xB200 host:

- **`simtopk`** — standalone kernel benches (kernels/).
  Python: `PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages /usr/bin/python3.12`
  (torch + deep_gemm SM100).
  CUTLASS 4.2.1 headers: `/opt/cutlass/include`;
  deep_gemm headers from the venv site-packages; gencode
  `arch=compute_100a,code=sm_100a`. JIT caches under container `/tmp/build_*`.
- **`vllm-prefill`** (image `glm5-vllm:snap`) — vLLM E2E (glm5_prefill/).
  venv `/opt/vllm-venv`: vllm 0.23.0 wheel + torch 2.11+cu130 +
  DeepGEMM 2.5.0 built from source at the vLLM-pinned commit.

## Third-party dependencies NOT vendored here

- **DeepGEMM @ 891d57b** (vLLM-pinned): clone and build from source
  (`git clone https://github.com/deepseek-ai/DeepGEMM && git checkout 891d57b`,
  or vllm's `tools/install_deepgemm.sh`). CUDA 12.8 works at this commit;
  later HEADs need >12.8. No local modifications.
- **nvidia-cutlass-dsl 4.3.5 (cp312)** — needed next to
  `kernels/b200/marsco/_flashlib/flashlib/`; obtain via
  `pip download nvidia-cutlass-dsl==4.3.5 --python-version 3.12` and extract.
  Our patched `flashlib` package IS vendored (the METRIC_IP patch has no
  upstream).
- **vLLM v0.23.0** — the full modified source tree is vendored at `vllm/`.
  Deployment used the official wheel with the two modified files overwritten
  in site-packages (fastest); building `vllm/` from source also works.
- The removed `kernels/b200/dsa/baselines/deepgemm_patched_official/` was a
  full deep_gemm package copy whose only delta is
  `deepgemm_patch/sm100_fp8_mqa_logits.cuh` — rebuild it by copying the venv's
  deep_gemm package and overwriting that one header.

## Data (excluded; locations + generators)

- DSA GLM-5 chunked caches: `~/glm5/dsa_caches`
  (`glm5_{tag}_realtext_chunk{N}.safetensors`) — regenerate with
  `kernels/h100/dsa/gen_dsa_caches.py` (needs GLM-5 fp8 shard 1 + tokenizer
  in `~/glm5/`).
- MS MARCO corpus for marsco benches: see `kernels/b200/marsco/README.md`.
- GLM-5.2 checkpoints: official FP8 repo `zai-org/GLM-5.2-FP8`
  (`~/glm5-fp8-official`, 141 shards); truncated 16L/1L shadows built by
  `glm5_prefill/build_fp8_truncated.py`.
- KDA ref-topk caches (`refcache_*.pt`) are auto-regenerated by the benches.

## Reproducing the headline results

1. **DSA kernel (16-cell real-chunk figure)** — container `simtopk`:
   `kernels/b200/dsa/sweep_b200_real_chunk.py` (~4 min) →
   `figure_b200_realchunk.json`.
   Scripts needing the official deep_gemm reference (test_dsa.py,
   threeway_b200.py) must prepend the patched deep_gemm package dir to
   PYTHONPATH (see `deepgemm_patch/README.md`).
2. **MARSCO** — `kernels/b200/marsco/bench_marsco_b200.py` /
   `sweep_marsco_b200.py`; flashlib baseline `bench_flashlib_ip.py`.
   Final matrix + methodology in its README.md and docs/.
3. **vLLM E2E (GLM-5.2 full 78L model, 8xB200)** — install vLLM with our two
   modified files (see `vllm/` + `vllm_patches/README.md`), then run
   `glm5_prefill/run_e2e_best.sh` (the fastest measured config; the
   header documents the numbers and the tuned-official baseline =
   VLLM_LITETOPK=0). Honest hybrid results (2026-07-13): 512K 46.66s (+4.6%),
   768K 81.82s (+8.6%), 1M 124.41s (+12.8%) vs tuned official; +18/+50/+90%
   vs default-budget official.
4. **DSA kernel ablations / stage decompositions** —
   `kernels/b200/dsa/kda_bench/README.md` documents the methodology (forced
   sampling modes, ref caching, red lines) and each tool's purpose.

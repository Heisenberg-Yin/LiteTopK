# LiteTopK — Reproduction Package (code only)

B200 (SM100) LiteTopK sparse top-k: the GLM-5.2 DSA prefill indexer and the MS
MARCO inner-product retrieval kernel. Code only — no data / weights / logs.

**Each subfolder has its own README with how to build, run, and reproduce its
results. This top-level file only says what each subfolder is.**

## Subfolders

| folder | what it is |
|---|---|
| `kernels/b200/dsa/` | **DSA LiteTopK indexer kernel** (`dsa_litetopk.cu` + `sm100_dsa_litetopk.cuh`) + `bench_q8192.py` — the 4-shape (256K/512K/768K/1M) Q=8192 scoreboard vs the official `mqa_logits` + `top_k` path on real GLM-5 KV. Built/driven through `glm5_prefill/litetopk_vllm`. |
| `kernels/h100/` | **H100 (SM90) port** of the DSA V3 kernel (WGMMA scoring loop, same epilogue/host ABI) + a per-feature support matrix of which B200 attempts carry to Hopper. Compile-verified for sm_90a; NOT yet run (no H100 on this host). |
| `kernels/b200/marsco/` | **MS MARCO IP top-k kernel** (`sm100_litetopk_marsco.cuh` + `litetopk_sm100_torch.cu` + `litetopk_select.cu`; JIT loader `litetopk_ops.py`) + `bench_marsco_b200.py` — batch-64 retrieval over 768-d embeddings, M∈{1,2,4,5}M, k∈{128..8192}. |
| `glm5_prefill/` | **GLM-5.2 full-model vLLM E2E harness**: `run_prefill.py` / `run_e2e_best.sh` (the fastest measured config), `build_fp8_truncated.py` (16L/1L shadow-model builder), and `litetopk_vllm/` — the LiteTopK indexer module the vLLM hook imports (+ recall verification). |
| `vllm_patches/` | Unified diffs of the two vLLM v0.23.0 files we modify — `.../layers/sparse_attn_indexer.py` (the LiteTopK hook) and `.../mla/indexer.py` (runtime logits-budget knob) — vs official v0.23.0 (verified against a clean `git clone` of `vllm-project/vllm` @0fc695f). Apply with `patch -p0` to an installed wheel or a fresh checkout. |
| `deepgemm_patch/` | 1-file patch to deep_gemm (SGLang #19529 fix) enabling GLM-5's 32-head shape. |

## Environments (2 docker containers on the 8×B200 host)

- **`simtopk`** — standalone kernel benches (`kernels/`).
- **`vllm-prefill`** / **`glm5-prefill`** — vLLM E2E (`glm5_prefill/`).

Exact build/run commands, env vars, and result tables live in each subfolder's
README.

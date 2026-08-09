# LiteTopK / LiteDSA

This repository is the canonical CUDA source and benchmark workspace for the
SM100 DSA optimizations integrated in
`/data01/home/ziqi.yin/vllm-v026`.

The current qualified scope is:

- GLM-5.2: FP8 LiteTopK indexer and grouped FP8 LiteDSA attention;
- DeepSeek-V4-Flash: FP4 LiteTopK indexer and BF16 head-packed C128A
  attention; and
- B200 MS MARCO top-k kernels under `kernels/b200/marsco/`.

Model weights, prompt data, container images, cache dumps, generated shared
objects, and E2E result logs are machine-local artifacts rather than source.

## Layout

| Path | Purpose |
|---|---|
| [`kernels/b200/dsa/`](kernels/b200/dsa/README.md) | Canonical LiteTopK, generic FP8 LiteDSA, and DSV4 packed-attention sources |
| [`e2e/`](e2e/README.md) | Reproducible GLM-5.2 and DSV4 four-arm vLLM benchmark |
| `kernels/b200/marsco/` | B200 MS MARCO top-k implementation and benchmark |

The former repository-local `glm5_prefill/` tree has been removed; `e2e/` is
the only maintained benchmark entry point. The separately mounted
`/data01/home/ziqi.yin/glm5_prefill_test` directory is used only for machine-local
build artifacts, caches, and DeepGEMM, not as benchmark source of truth.

## Source-of-truth mapping

Production vLLM uses vendored copies. The following files must remain
byte-identical to their canonical counterparts.

| Canonical source | vLLM source |
|---|---|
| `kernels/b200/dsa/{dsa_litetopk.cu,sm100_dsa_litetopk.cuh,dense_topk_litetopk.cuh}` | `vllm-v026/csrc/libtorch_stable/attention/dsa/latest/` |
| `kernels/b200/dsa/litedsa_attention_sm100.cuh` | `vllm-v026/csrc/libtorch_stable/attention/dsa/vendor_fmla/sm100/prefill/sparse/fwd/head128_fp8/phase1.cuh` |
| `kernels/b200/dsa/litedsa_union.cuh` | `vllm-v026/csrc/libtorch_stable/attention/dsa/include/flashinfer/litedsa/litedsa_union.cuh` |
| `kernels/b200/dsa/{litedsa_attention_sm100_dsv4.cuh,litedsa_dsv4.cu,litedsa_dsv4_atoms.cuh,litedsa_dsv4_binding.cu}` | `vllm-v026/csrc/libtorch_stable/attention/dsa/dsv4_packed/` |

The generic `litedsa.cu` translation units intentionally differ. The
canonical file exports TVM-FFI operations for standalone iteration; the vLLM
file registers the same kernels through the stable Torch ABI.

Current LiteTopK source ID: `edd074998b45`.

## Kernel roles

### LiteTopK

The indexer kernel scores the long suffix and emits a conservative candidate
set, then an exact selector returns the requested top-k. The same source
contains the qualified H32/BQ4 FP8 path used by GLM-5.2 and the H64/BQ2 FP4
path used by DSV4. Dense crossover, overflow fallback, status checks, and the
8192/8128 query-tail paths are production behavior and are retained.

### Generic FP8 LiteDSA

For GLM-5.2, adjacent query tokens are grouped so their local heads fill one
128-row SM100 attention tile. A union kernel deduplicates their top-k KV
indices, while a query-major membership mask preserves each token's exact
attention set. The standalone module exports:

- `union_qm`
- `masked_mla_fp8`

Build it with:

```bash
cd /data01/home/ziqi.yin/litetopk/kernels/b200/dsa
PYTHON_BIN=/opt/vllm-venv/bin/python \
CUDA_HOME=/usr/local/cuda \
LITEDSA_SO=/tmp/litedsa_fp8_build/litedsa.so \
./build_litedsa.sh
```

The script discovers TVM-FFI, FlashInfer, CUTLASS, Python, and the PyTorch C++
ABI from the selected interpreter, builds `sm_100a`, and atomically publishes
the requested output file.

### DSV4 packed attention

DSV4 C128A layers attend a compressed prefix plus a contiguous sliding
window. The packed BF16 kernel combines `128 / local_heads` adjacent tokens
into one 128-row tile and represents each query with two exact ranges. It is
independent of the FP4 LiteTopK indexer used by C4A layers.

The runtime-default artifact remains in vLLM at
`csrc/libtorch_stable/attention/dsa/dsv4_packed/dsa_dsv4.so`. Rebuild it with
the colocated `build_dsv4.sh`; rerun `build_probe_dsv4_smem.sh` after every
shared-memory layout change.

## End-to-end benchmark

The launcher runs four orthogonal arms so an attention-only speedup is not
misreported as a LiteTopK+attention combination:

| Arm | LiteTopK | GLM attention | DSV4 attention |
|---|---:|---|---|
| `raw` | off | stock | stock |
| `litetopk` | on | stock | stock |
| `litedsa` | off | grouped FP8 LiteDSA | packed BF16 C128A |
| `combo` | on | grouped FP8 LiteDSA | packed BF16 C128A |

Run the qualified 1M configurations with:

```bash
cd /data01/home/ziqi.yin/litetopk/e2e

MODEL_FAMILY=glm5.2 \
VLLM_LITEDSA_SO=/data01/home/ziqi.yin/litetopk/kernels/b200/dsa/.codex_variants/litedsa_cuda13_b264c21c9ce1_cuda13.so \
REPEATS=3 WARMUPS=1 ./run_e2e.sh

MODEL_FAMILY=dsv4 REPEATS=3 WARMUPS=1 ./run_e2e.sh
```

GLM defaults to TP8; DSV4 defaults to TP4 with expert parallel enabled across
the four workers. Both use 1,048,512 input tokens, chunk 8192, FP8 attention
KV, MTP, and `async_scheduling=true`.

The runner records source/SO hashes and actual arm switches. Attention arms
must emit a post-dispatch marker, and the four-way comparator requires matched
benchmark configuration plus identical warmup and trial token IDs. Results
are written below `e2e/results/`, which is gitignored.

### Qualified 1M results (2026-08-09)

All values are medians of three trials after one warmup, with asynchronous
scheduling and expert parallel enabled.

| Model | raw | LiteTopK | LiteDSA | combo | raw/TopK | raw/DSA | raw/combo |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.2 TP8+EP8 | 152.331 s | 118.668 s | 143.955 s | 109.343 s | 1.284x | 1.058x | **1.393x** |
| DSV4 TP4+EP4 | 60.007 s | 55.779 s | 52.397 s | 48.185 s | 1.076x | 1.145x | **1.245x** |

The GLM FP8 path now has one compiled policy: warm-started ring refresh, its
daemon, and 2048 ns pacing are always active and have no runtime switches.
The fixed-policy source (`f75dfed60674`) reproduced the combo at 109.763 s.
See [`e2e/README.md`](e2e/README.md) for the exact arm definitions,
candidate/status gates, artifact hashes, and DSV4 mixed CUDA 12/13 caveat.

## Cleanup policy

Only code with no build entry, Python binding, runtime caller, or retained
microbenchmark is removed during source cleanup. Shape fallbacks, exact
selectors, status/overflow handling, architecture guards, FP8/FP4 variants,
and model-specific paths are not treated as dead merely because the current
E2E matrix does not exercise every branch.

Generated Ninja state, `.bak`/`.works*` snapshots, and stale experimental
translation units are not canonical source. Build artifacts should live in a
gitignored artifact directory or `/tmp` and be selected explicitly by path
and hash for reproducible measurements.

The FlashMLA-derived attention sources retain their upstream license in
`kernels/b200/dsa/LICENSE.deepseek-flashmla` and the corresponding vendored
vLLM directories.

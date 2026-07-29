# LiteTopK

LiteTopK provides CUDA implementations of sparse top-k selection for:

- the GLM-5 DSA prefill indexer; and
- 768-dimensional MS MARCO inner-product retrieval.

The repository contains CUDA/Python source, vLLM patches, documentation, and
one prebuilt LiteDSA shared object. It does not include model weights,
benchmark datasets, recorded DSA caches, container images, or benchmark logs.

## Repository layout

| Path | Purpose |
|---|---|
| [`kernels/b200/dsa/`](kernels/b200/dsa/README.md) | B200 (SM100a) DSA indexer, recorded-cache benchmark, and Whole-DSA benchmark |
| [`kernels/b200/dsa/flashinfer_port/`](kernels/b200/dsa/flashinfer_port/README.md) | LiteDSA grouped sparse-attention source and prebuilt module |
| [`kernels/b200/marsco/`](kernels/b200/marsco/README.md) | B200 MS MARCO inner-product top-k |
| [`kernels/h100/`](kernels/h100/README.md) | H100 (SM90a) source and compile check; the DSA runtime path is not validated here |
| [`glm5_prefill/`](glm5_prefill/README.md) | Truncated-model builder, vLLM prefill runner, E2E launcher, and Python adapters |
| [`vllm_patches/`](vllm_patches/README.md) | Python patches for vLLM v0.23.0, commit `0fc695f` |
| [`deepgemm_patch/`](deepgemm_patch/README.md) | Compatibility patch for older DeepGEMM releases; DeepGEMM 2.5 does not need it |

## Requirements

### B200 DSA and GLM prefill

- NVIDIA B200 (SM100a) and a CUDA toolkit with NVCC;
- Python with CUDA-enabled PyTorch;
- DeepGEMM 2.5 and its CUTLASS submodule;
- vLLM 0.23.0 with the required
  [repository patches](vllm_patches/README.md);
- `safetensors`, FlashInfer, and TVM-FFI; and
- a writable PyTorch extension build directory.

The commands below use the `glm5-prefill` container and
`/opt/vllm-venv/bin/python`. On another system, provide an equivalent
environment and adjust the container and Python paths.

### MS MARCO

- B200 for the SM100 implementation, or H100 for the SM90 implementation;
- CUDA-enabled PyTorch and NumPy;
- compatible CUTLASS and DeepGEMM include directories; and
- the MS MARCO query and corpus files described below.

## External inputs

These inputs are not distributed with the repository:

| Input | Required by | Required contents |
|---|---|---|
| `glm5_{256k,512k,768k,1m}_realtext_chunk8192.safetensors` | B200 DSA benchmark | `q_index`, `q_index_scale`, `idx_k_cache`, `idx_k_scale`, `gate_w` |
| The same DSA caches with attention data | Whole-DSA | the fields above plus `topk_idx`, `mla_q`, `mla_kv`, and `metadata["mla"]` |
| GLM-5 FP8 checkpoint | vLLM prefill | a checkpoint readable by the patched vLLM environment |
| Tokenizer/configuration assets | truncated-model builder | files consumed by `build_fp8_truncated.py` |
| Parquet prompt corpus | vLLM prefill | a string column named `text` |
| MS MARCO data directory | retrieval benchmark | `query.fvecs` and either `base_5m_fp16_768.bin` or writable `base_5m.fvecs` |

There are no scripts in this repository that download these assets or create
the recorded DSA caches.

## Build behavior

The B200 DSA and MS MARCO PyTorch extensions are JIT-compiled on the first
operator call. Running the corresponding benchmark is therefore also the
build check. Set the documented include-directory variables before the first
run.

`kernels/b200/dsa/flashinfer_port/litedsa.so` is a prebuilt SM100,
architecture- and ABI-specific TVM-FFI module. Rebuild it when the CUDA
toolchain, TVM-FFI ABI, C++ ABI, or target architecture differs from the
environment that produced it. Its source and build requirements are
documented in the [LiteDSA guide](kernels/b200/dsa/flashinfer_port/README.md).

## Reproduce the B200 DSA indexer benchmark

The repository must be visible inside the container at the same path stored
in `REPO_DIR`.

```bash
export REPO_DIR=/data01/home/ziqi.yin/litetopk_github_clone
export DEEPGEMM_DIR=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM

docker exec \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e PATH=/opt/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  -e DEEPGEMM_DIR="$DEEPGEMM_DIR" \
  -e DSA_CACHE_DIR=/data/dsa_caches \
  -e LITETOPK_MODULE_DIR="$REPO_DIR/glm5_prefill/litetopk_vllm" \
  -e LITETOPK_DSA_DIR="$REPO_DIR/kernels/b200/dsa" \
  -w "$REPO_DIR/kernels/b200/dsa" \
  glm5-prefill /opt/vllm-venv/bin/python bench_q8192.py
```

`bench_q8192.py` processes the 256K, 512K, 768K, and 1M cache files and
compares the Q=8192 LiteTopK result with the official dense indexer.

### Latest validated B200 result

Measured on July 29, 2026 on one NVIDIA B200 with `Q=8192`, `K=2048`, the
recorded GLM-5 caches, and the fixed numerical-FP16 specialization in this
repository. The dense baseline is DeepGEMM `fp8_fp4_mqa_logits` followed by
vLLM `top_k_per_row_prefill`.

| KV length | Official dense | LiteTopK | Speedup | Set-overlap recall |
|---:|---:|---:|---:|---:|
| 256K | 12.83 ms | 10.55 ms | 1.22x | 99.95% |
| 512K | 26.12 ms | 20.37 ms | 1.28x | 99.95% |
| 768K | 39.83 ms | 29.56 ms | 1.35x | 99.95% |
| 1M | 52.56 ms | 39.30 ms | 1.34x | 99.95% |

The benchmark uses five warm-up calls, then 20 timed calls at 256K/512K and
eight at 768K/1M. LiteTopK consumes a precomputed fixed hot carry, so its timer
covers the current indexer call but excludes construction of the carry for the
next call. These are indexer-only observations from one validation run, not
Whole-DSA or full-model results and not fixed pass criteria. Re-run the command
above when changing the GPU, toolchain, input caches, or kernel configuration.

## Reproduce Whole-DSA

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e PATH=/opt/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  -e DEEPGEMM_DIR="$DEEPGEMM_DIR" \
  -e TAGS="768k 1m" \
  -e DSA_CACHE_DIR=/data/dsa_caches \
  -e LITETOPK_MODULE_DIR="$REPO_DIR/glm5_prefill/litetopk_vllm" \
  -e LITETOPK_DSA_DIR="$REPO_DIR/kernels/b200/dsa" \
  -e LITEDSA_SO="$REPO_DIR/kernels/b200/dsa/flashinfer_port/litedsa.so" \
  -w "$REPO_DIR/kernels/b200/dsa" \
  glm5-prefill /opt/vllm-venv/bin/python bench_whole_dsa.py
```

Whole-DSA separately times the indexer and sparse MLA attention and then adds
the two times. It is not a fused kernel. Both attention paths consume the
recorded cache's `topk_idx`, rather than the indices produced by the current
LiteTopK invocation, so the attention implementations receive identical
inputs.

## Run GLM-5 end-to-end prefill

First apply the appropriate [vLLM patches](vllm_patches/README.md). Then run
the host-side launcher:

```bash
cd "$REPO_DIR/glm5_prefill"

MODEL=/models/glm5-fp8-16l \
DEEPGEMM_DIR="$DEEPGEMM_DIR" \
DEVICES=0,1,2,3 \
TP=4 \
LENGTHS="262144 524288" \
./run_e2e.sh
```

`MODEL` and all other data paths must be visible inside the container. See
the [prefill guide](glm5_prefill/README.md) for constructing a truncated
checkpoint, running stock vLLM, and enabling the optional LiteDSA attention
hook.

## Run the B200 MS MARCO benchmark

```bash
cd "$REPO_DIR/kernels/b200/marsco"

export MARSCO_DATA=/path/to/marsco
export LITETOPK_CUTLASS_INCLUDE=/path/to/cutlass/include
export DSA_DEEP_GEMM_INCLUDE=/path/to/DeepGEMM/deep_gemm/include
export TORCH_CUDA_ARCH_LIST=10.0a

CUDA_VISIBLE_DEVICES=0 \
python3 bench_marsco_b200.py --m 1000000 --k 128
```

`--m` is required and must be a multiple of 64. The first call builds the
SM100 PyTorch extension. The complete argument list and Python API are in the
[B200 MS MARCO guide](kernels/b200/marsco/README.md).

## H100 status

The repository includes SM90a DSA and MS MARCO source. The DSA source has a
documented [`sm_90a` compile check](kernels/h100/README.md), but this
repository does not provide a validated H100 DSA runtime path. The
[H100 MS MARCO guide](kernels/h100/marsco/README.md) documents the runtime
command that must be executed on an H100 host.

## Reproduction criteria

A successful reproduction must satisfy the criteria for the selected entry:

- **B200 DSA indexer:** all four recorded-cache shapes complete; index overlap
  recall is at least `99.9%` for every shape; and there is no extension build
  failure, fallback, candidate overflow, CUDA error, or invalid
  boundary-metadata failure.
- **Numerical FP16 behavior:** exact source-bucket classification is retained,
  but ties inside the boundary bucket are selected using FP16 scores. Indices
  tied at that boundary need not be byte-identical to the FP32 reference.
- **Whole-DSA:** every requested tag completes without CUDA or module-loading
  errors, and the reported `idx%` and `attn_maxdiff` values are recorded.
  `bench_whole_dsa.py` does not impose pass thresholds for those two fields.
- **MS MARCO:** the benchmark prints `RESULT: PASS`, which requires
  set-overlap recall of at least `0.99`.
- **End-to-end prefill:** the log confirms that the LiteTopK extension loaded
  and that eligible calls did not silently use the dense fallback.

Reported latency and speedup values describe only the current run. This
repository does not define fixed performance numbers as reproduction
criteria.

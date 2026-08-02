# LiteTopK

LiteTopK provides B200 CUDA implementations of sparse top-k selection for:

- the GLM-5/LongCat DSA prefill indexer; and
- 768-dimensional MS MARCO inner-product retrieval.

Model weights, benchmark datasets, recorded DSA caches, container images, and
benchmark logs are external.

## Repository layout

| Path | Purpose |
|---|---|
| [`kernels/b200/dsa/`](kernels/b200/dsa/) | B200 DSA indexer and LiteDSA grouped sparse-attention implementation |
| [`kernels/b200/marsco/`](kernels/b200/marsco/) | B200 MS MARCO inner-product top-k implementation and benchmark |
| [`glm5_prefill/`](glm5_prefill/README.md) | GLM-5.2/LongCat native-vLLM end-to-end A/B harness |

### `kernels/b200/dsa/`

| File | Purpose |
|---|---|
| `dsa_litetopk.cu` | LiteTopK PyTorch CUDA extension entry |
| `sm100_dsa_litetopk.cuh` | B200/SM100 fused score, gate, histogram and emit kernel |
| `dense_topk_litetopk.cuh` | Exact short-sequence dense top-k selector |
| `litedsa_jit_binding.cu` | LiteDSA TVM-FFI compilation unit (includes `litedsa.cu`) |
| `litedsa.cu` | LiteDSA union and sparse-attention launch wrapper |
| `litedsa_attention_sm100.cuh` | Self-contained SM100 FP8 sparse-attention kernel and helpers |
| `litedsa_union.cuh` | Union, membership bitmap and physical-index kernels |
| `build_litedsa.ninja.repro` | Current-machine `sm_100a` build template |
| `LICENSE.deepseek-flashmla` | Upstream FlashMLA MIT license |

`litedsa_attention_sm100.cuh` is derived from
[`deepseek-ai/FlashMLA`](https://github.com/deepseek-ai/FlashMLA) commit
`9241ae3` with local FP8 attention and membership modifications.

### `kernels/b200/marsco/`

| File | Purpose |
|---|---|
| `bench_marsco_b200.py` | Dense/LiteTopK benchmark with recall and CUDA-event timing |
| `litetopk_ops.py` | JIT loader and public `fused_ip_sparse_b200` API |
| `litetopk_sm100_torch.cu` | PyTorch registration, sampling, scan and final selection |
| `sm100_litetopk_marsco.cuh` | B200 UMMA/TMEM scan kernel |
| `litetopk_select.cu`, `litetopk_select.h`, `litetopk_topk.h` | Candidate compaction and boundary selection |

## Results (B200, 2026-08-02, kernel source `8ac06eb50e45`)

### DSA kernel-level A/B

One B200, recorded GLM DSA caches, `Q=8192`, `K=2048`; medians over three
independent runs.

| KV length | Dense (ms) | LiteTopK (ms) | Paired speedup median | Recall median |
|---:|---:|---:|---:|---:|
| 262,144 | 13.0469 | 10.6117 | 1.2266x | 99.9977% |
| 524,288 | 25.8997 | 20.4716 | 1.2652x | 99.9980% |
| 786,432 | 39.2471 | 30.0235 | 1.3084x | 99.9978% |
| 1,048,576 | 51.8106 | 40.1796 | 1.2929x | 99.9978% |

### Native-vLLM end-to-end prefill A/B

Eight B200s, TP=8, chunk size 8192, FP8 KV cache, one untimed warmup, median
of two trials; both arms produced exactly matching generated tokens.

| Model | Input tokens | MTP | Dense (s) | LiteTopK (s) | Median speedup |
|---|---:|---:|---:|---:|---:|
| GLM-5.2 | 1,048,512 | 5 | 151.140 | 120.513 | 1.2541x |
| LongCat | 974,848 | 3 | 114.114 | 89.546 | 1.2744x |

The harness writes machine-readable results to
`glm5_prefill/results/<run-id>/summary.json` (gitignored; the runs above are
`20260802-rerun-glm5.2` and `20260802-rerun-longcat` on the benchmark host).

## Reproducing

### End-to-end prefill A/B

Requires: B200s (`sm_100a`), CUDA PyTorch, DeepGEMM 2.5 with its CUTLASS
submodule, `safetensors`/FlashInfer/TVM-FFI, a vLLM tree with the native
LiteTopK integration, a GLM-5.2 or LongCat checkpoint, and a Parquet prompt
corpus with a `text` column. The commands below use the `glm5-prefill`
container; adjust paths for another environment.

```bash
export REPO_DIR=/data01/home/ziqi.yin/litetopk
export DEEPGEMM_DIR=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM
export LITETOPK_VLLM_SRC=/data01/home/ziqi.yin/vllm-litetopk-longcat

cd "$REPO_DIR/glm5_prefill"
MODEL_FAMILY=glm5.2 REPEATS=2 ./run_e2e.sh
MODEL_FAMILY=longcat REPEATS=2 ./run_e2e.sh
```

The harness runs the dense vLLM indexer first, then the same workload with
LiteTopK, and rejects mismatched A/B configurations or generated tokens. See
the [prefill harness notes](glm5_prefill/README.md) for model defaults and
result layout.

### MS MARCO benchmark

Requires: one B200, CUDA PyTorch and NumPy, CUTLASS and DeepGEMM headers, and
a `MARSCO_DATA` directory containing `query.fvecs` and either
`base_5m_fp16_768.bin` or `base_5m.fvecs`.

```bash
cd "$REPO_DIR/kernels/b200/marsco"

export MARSCO_DATA=/path/to/marsco
export LITETOPK_CUTLASS_INCLUDE=/path/to/cutlass/include
export DSA_DEEP_GEMM_INCLUDE=/path/to/DeepGEMM/deep_gemm/include
export TORCH_CUDA_ARCH_LIST=10.0a

# Full corpus-size / top-k matrix:
for m in 1000000 2000000 4000000 5000000; do
  for k in 128 1024 4096 8192; do
    CUDA_VISIBLE_DEVICES=0 \
      python3 bench_marsco_b200.py --m "$m" --k "$k"
  done
done
```

The first call JIT-builds the extension. The benchmark reports set-overlap
recall plus dense and LiteTopK CUDA-event latency, and prints `RESULT: PASS`
when recall is at least `0.99`.

### Build LiteDSA (`litedsa.so`)

The runtime loads a prebuilt LiteDSA module that is not checked in
(architecture-, toolchain- and TVM-FFI-ABI-specific); build it with the
template below. `build_litedsa.ninja.repro` contains current-machine absolute
paths — adjust after migration.

```bash
docker exec glm5-prefill bash -lc '
  mkdir -p /tmp/litedsa_build /tmp/litedsa_ninja
  cd /tmp/litedsa_ninja
  /opt/vllm-venv/bin/ninja \
    -f /data01/home/ziqi.yin/litetopk/kernels/b200/dsa/build_litedsa.ninja.repro
'
```

The output is `/tmp/litedsa_build/dsa_indexer.so`; verify it exports
`union_qm` and `masked_mla_fp8` before replacing `litedsa.so`.

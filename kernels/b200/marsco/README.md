# B200 MS MARCO inner-product top-k

This directory contains a PyTorch CUDA extension for batched inner-product
selection over 768-dimensional MS MARCO embeddings on B200 (`sm_100a`).
`bench_marsco_b200.py` compares the extension with dense matrix
multiplication followed by `torch.topk`.

## Files

| File | Purpose |
|---|---|
| `bench_marsco_b200.py` | Loads MS MARCO vectors, runs both implementations, and reports recall and CUDA-event latency. |
| `litetopk_ops.py` | JIT loader and public Python function `fused_ip_sparse_b200`. |
| `litetopk_sm100_torch.cu` | PyTorch operator registration, sampling setup, scan launch, and final selection. |
| `sm100_litetopk_marsco.cuh` | B200 UMMA/TMEM scan kernel. |
| `litetopk_select.cu`, `litetopk_select.h`, `litetopk_topk.h` | Candidate compaction and boundary selection. |

## Requirements

- An NVIDIA B200 GPU and a CUDA toolkit with `nvcc` support for `sm_100a`.
- Python with PyTorch CUDA support and NumPy.
- CUTLASS headers. Set `LITETOPK_CUTLASS_INCLUDE` to the directory that
  contains `cute/arch/mma_sm100_umma.hpp`.
- DeepGEMM headers. Set `DSA_DEEP_GEMM_INCLUDE` to the directory that
  contains `deep_gemm/common/sm100_utils.cuh` and the device helpers
  `swap`, `get_lane_idx`, and `ld_shared` in
  `deep_gemm/common/utils.cuh`.
- A writable `MARSCO_DATA` directory containing:

  - `query.fvecs`;
  - either `base_5m_fp16_768.bin`, or `base_5m.fvecs` from which the benchmark
    can create the fp16 cache.

The data, PyTorch environment, CUTLASS, and DeepGEMM are external
dependencies; they are not included in this repository.

## Run

Set the paths for your environment, then run one benchmark cell:

```bash
export MARSCO_DATA=/path/to/marsco
export LITETOPK_CUTLASS_INCLUDE=/path/to/cutlass/include
export DSA_DEEP_GEMM_INCLUDE=/path/to/DeepGEMM/deep_gemm/include
export TORCH_CUDA_ARCH_LIST=10.0a

CUDA_VISIBLE_DEVICES=0 \
python3 bench_marsco_b200.py --m 1000000 --k 128
```

To run the complete corpus-size/top-k grid:

```bash
for m in 1000000 2000000 4000000 5000000; do
  for k in 128 1024 4096 8192; do
    CUDA_VISIBLE_DEVICES=0 \
      python3 bench_marsco_b200.py --m "$m" --k "$k"
  done
done
```

On the current shared host, the equivalent container command is:

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e MARSCO_DATA=/data01/home/ziqi.yin/github_simtopk/data/marsco \
  -e LITETOPK_CUTLASS_INCLUDE=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM/third-party/cutlass/include \
  -e DSA_DEEP_GEMM_INCLUDE=/data01/home/ziqi.yin/deepgemm_patched/deep_gemm/include \
  -w /data01/home/ziqi.yin/litetopk_github_clone/kernels/b200/marsco \
  simtopk \
  python3 bench_marsco_b200.py --m 1000000 --k 128
```

The first operator call builds `litetopk_marsco_ext` in PyTorch's extension
cache. The build fixes `LITETOPK_KV_STAGES=6` and
`LITETOPK_WARP_QUEUE_CAP=32`; these are not runtime tuning variables.
Set `FLASHTOPK_BUILD_VERBOSE=1` to show the JIT build commands.

The benchmark prints:

1. the GPU and effective benchmark arguments;
2. set-overlap recall against the selected dense baseline;
3. average CUDA-event time for both paths;
4. `RESULT: PASS` when recall is at least `0.99`, otherwise
   `RESULT: RECALL_LOW`.

## Benchmark arguments

| Argument | Default | Meaning |
|---|---:|---|
| `--m` | required | Number of corpus rows; must be a multiple of 64. |
| `--k` | `K` or `128` | Number of returned indices. |
| `--bs` | `BS` or `64` | Number of query vectors read from `query.fvecs`. |
| `--sample-size` | `SAMPLE_SIZE` or auto | Number of strided calibration rows. |
| `--num-buckets` | `NUM_BUCKETS` or `64` | Number of score buckets. |
| `--refresh` | `REFRESH` or auto (`8`) | Threshold refresh interval. |
| `--num-ctas-x` | `NUM_CTAS_X` or auto | Corpus-grid width. |
| `--warmup` | `WARMUP` or `10` | Untimed iterations. |
| `--iters` | `ITERS` or `20` | Timed iterations. |

`BASELINE=torch` uses `q @ base.T` followed by `torch.topk`.
`BASELINE=engine_dense` uses the extension's dense-score helper followed by
`torch.topk`. `OUT_FP32=1` requests fp32 output values. The benchmark fixes
`LITETOPK_FLAT_QN=64`, `LITETOPK_BM=256`, and `num_ctas_x=296` unless the
caller supplies explicit alternatives.

## Python API

```python
from litetopk_ops import fused_ip_sparse_b200

values, indices = fused_ip_sparse_b200(
    q,                 # [B, D], fp16 CUDA tensor
    k_cache,           # [1, M, D], fp16 CUDA tensor
    k=128,
    sample_mode=1,     # the only supported sampling mode
)
```

`B` must be greater than 8 and divisible by the selected query tile; `M` must
be a multiple of 64. The function returns selected values and int32 corpus
indices. Inputs are made contiguous by the Python wrapper.

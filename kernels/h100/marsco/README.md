# H100 MS MARCO inner-product top-k

This directory contains the H100 (`sm_90a`) counterpart of the MS MARCO
PyTorch extension. It uses an SM90 WGMMA scan and the same strided-sampling
and boundary-selection contract as the B200 entry.

## Requirements

- A real NVIDIA H100-class GPU for execution and correctness testing.
- A CUDA toolkit with `nvcc` support for `sm_90a`.
- Python with PyTorch CUDA support and NumPy.
- `MARSCO_DATA` containing `query.fvecs` and either
  `base_5m_fp16_768.bin` or the writable source `base_5m.fvecs`.
- CUTLASS and DeepGEMM include directories:

  - `LITETOPK_CUTLASS_INCLUDE` must contain the SM90 CuTe headers;
  - `DSA_DEEP_GEMM_INCLUDE` must contain `deep_gemm/mma/sm90.cuh`.

These dependencies and the MS MARCO data are not included in this
repository. This host has B200 GPUs only, so this entry cannot be runtime
validated here.

## Run on an H100 host

```bash
cd /path/to/litetopk_github_clone/kernels/h100/marsco

export MARSCO_DATA=/path/to/marsco
export LITETOPK_CUTLASS_INCLUDE=/path/to/cutlass/include
export DSA_DEEP_GEMM_INCLUDE=/path/to/DeepGEMM/deep_gemm/include
export TORCH_CUDA_ARCH_LIST=9.0a

CUDA_VISIBLE_DEVICES=0 \
python3 bench_marsco_h100.py --m 1000000 --k 128
```

The first operator call builds `litetopk_marsco_sm90_ext` in PyTorch's
extension cache. The build fixes `LITETOPK_KV_STAGES=6` and
`LITETOPK_WARP_QUEUE_CAP=32`. The H100 host wrapper supports BM=128.
Set `FLASHTOPK_BUILD_VERBOSE=1` to display JIT build commands.

The benchmark prints the effective arguments, set-overlap recall against the
dense reference, CUDA-event timings, and `RESULT: PASS` when recall is at
least `0.99`.

The command-line arguments and data conversion behavior match the B200
benchmark documented in `../../b200/marsco/README.md`. The public Python
function is `fused_ip_sparse_h100`; it accepts fp16 CUDA tensors `q[B,D]`
and `k_cache[1,M,D]`, requires `B > 8`, `M % 64 == 0`, and
`sample_mode=1`, and returns values plus int32 corpus indices.

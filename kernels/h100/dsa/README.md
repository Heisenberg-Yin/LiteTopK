# DSA LiteTopK

This directory contains the fastest DSA top-k implementation used in the current
benchmark: GLM-5.2-FP8 DSA ReLU-MQA scoring with KV-split scheduling and fused
sparse sample/write selection.

## Files

| file | purpose |
|---|---|
| `sm90_dsa_marsco.cuh` | SM90 fp8 WGMMA scoring kernel with KV-split + sparse candidate epilogue |
| `dsa_marsco.cu` | PyTorch/CUDA binding, launch wrapper, compact radix top-k select |
| `test_dsa.py` | single-run correctness/latency check |
| `bench_3way.py` | A/B/O benchmark: stock deep_gemm, ideal KV-split dense, fused sparse |
| `sweep_chunks.py` | benchmark over chunk sizes 1024/2048/4096/8192 and seq lengths 256k/512k/768k/1m |
| `gen_dsa_caches.py` | regenerate GLM-5.2-FP8 real-text DSA caches for arbitrary chunk size |

## Quick Run

```bash
cd /workspace/project/src/dsa
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST=9.0a

# Check one cache, default CHUNK=4096
python3 test_dsa.py 256k

# 3-way benchmark
CHUNK=4096 python3 bench_3way.py 256k 512k 768k 1m

# Full chunk sweep
python3 sweep_chunks.py
```

The scripts expect caches under `/workspace/project/dsa_caches/` and DeepGEMM
under `/workspace/project/build/src/hopper/dsa/DeepGEMM_reduced_topk`.

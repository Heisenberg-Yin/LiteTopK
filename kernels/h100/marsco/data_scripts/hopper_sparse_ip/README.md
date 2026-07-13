# Hopper Sparse IP Top-K on MS MARCO

This folder contains the Hopper sparse inner-product top-k implementation used
for the MS MARCO speedup report.

## Contents

```text
hopper_sparse_ip/
├── hopper_topk_select.cu       # Hopper WGMMA sparse score kernels + select launchers
├── hopper_topk_torch.cu        # torch custom-op registration / dispatch
├── hopper_topk_select_mb.cu    # multi-block select kernels
├── hopper_topk.h
├── litetopk_select.h
├── hopper_ops.py               # JIT loader
├── bs_common.py
├── bs1_sparse.py
├── bs8_sparse.py
├── bs32_sparse.py
├── bs64_sparse.py
├── bs128_sparse.py
└── report_speedup_vs_torch.py  # report script
```

The implementation compares against:

```python
torch.topk(q @ base.T, k)
```

on real MS MARCO `.fvecs` embeddings.

## Environment

Use the `litetopk2` container or an equivalent H100 CUDA environment with:

- NVIDIA H100 / sm_90a
- CUDA nvcc available
- PyTorch with CUDA
- CUTLASS headers, either at the default FlashInfer path or via
  `HOPPER_CUTLASS_INCLUDE`

Inside the existing container:

```bash
docker exec -it litetopk2 bash
cd /workspace/project/src/marsco/data_scripts/hopper_sparse_ip
```

Use a fresh `TORCH_EXTENSIONS_DIR` after changing CUDA code or compile flags.

## Data

The default data paths are resolved relative to this folder:

```text
../../../data/marsco/base_5m.fvecs
../../../data/marsco/query.fvecs
```

Expected shape:

- `base_5m.fvecs`: 5,000,000 x 768 corpus vectors
- `query.fvecs`: query vectors, 768-d

## Reproduce The Speedup Report

Run the default report:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_EXTENSIONS_DIR=/tmp/hopper_sparse_ip_report \
python3 report_speedup_vs_torch.py \
  --bs-list 1,8,32,64,128 \
  --m-list 262144,1000000,4000000 \
  --k-list 10,100,1000 \
  --warmup 10 --iters 30 --reps 3
```

The script prints:

```text
   bs         N      k    torch_ms   sparse_ms   speedup   recall
-----------------------------------------------------------------
```

`speedup = torch_ms / sparse_ms`; values greater than 1 mean the sparse Hopper
path is faster.  `recall` is index recall@k versus torch.

Representative H100 results:

| bs | N | k=10 | k=100 | k=1000 |
|---:|---:|---:|---:|---:|
| 1 | 262k | 0.81x | 0.53x | 0.65x |
| 1 | 1M | 1.05x | 0.74x | 1.02x |
| 1 | 4M | 1.06x | 0.94x | 1.04x |
| 8 | 262k | 0.90x | 0.79x | 0.66x |
| 8 | 1M | 1.09x | 1.02x | 0.99x |
| 8 | 4M | 1.22x | 1.20x | 1.18x |
| 32 | 262k | 0.92x | 0.86x | 0.83x |
| 32 | 1M | 1.39x | 1.35x | 1.34x |
| 32 | 4M | 1.63x | 1.62x | 1.63x |
| 64 | 262k | 1.06x | 0.98x | 0.82x |
| 64 | 1M | 1.79x | 1.75x | 1.70x |
| 64 | 4M | 2.17x | 2.16x | 2.15x |
| 128 | 262k | 1.32x | 1.26x | 1.06x |
| 128 | 1M | 2.20x | 2.17x | 2.09x |
| 128 | 4M | 2.86x | 2.86x | 2.86x |

Interpretation:

- Larger batch and larger corpus size give higher speedup.
- Small corpus size (256k) can be slower because sample/select overhead is not
  yet amortized.
- Recall in the measured report was >= 0.99 for all listed points.

## Optional m64n8 Register-Row Queue

`hopper_topk_select.cu` contains two write-out strategies for the m64n8 full
sparse kernel (used by bs <= 8):

```bash
HOPPER_M8_REG_ROW_QUEUE=0  # default: per-lane atomicAdd
HOPPER_M8_REG_ROW_QUEUE=1  # tidal-style 8-row register queue
```

Default is `0` because it is best or tied for normal k.  The register-row queue
can help only for very large k, where atomic contention in the m64n8 tile becomes
visible.

Example large-k comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_EXTENSIONS_DIR=/tmp/hopper_sparse_ip_atomic \
HOPPER_M8_REG_ROW_QUEUE=0 \
python3 report_speedup_vs_torch.py \
  --bs-list 8 --m-list 1000000,4000000 --k-list 8192,16384

CUDA_VISIBLE_DEVICES=0 \
TORCH_EXTENSIONS_DIR=/tmp/hopper_sparse_ip_regrow \
HOPPER_M8_REG_ROW_QUEUE=1 \
python3 report_speedup_vs_torch.py \
  --bs-list 8 --m-list 1000000,4000000 --k-list 8192,16384
```

Measured pattern:

- bs=8, k=8192: register-row is about 1-2% faster.
- bs=8, k=16384: register-row is about 4-7% faster.
- bs=32 should stay on the default atomic path; 32-row queues were measured to
  be slower, especially at large k.

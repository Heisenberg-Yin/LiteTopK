# MS MARCO / fvecs LiteTopK Reproduction

This directory contains two reproducible MS MARCO top-k benchmark paths:

```text
1. L2 LiteTopK reproduction
   src/litetopk_fused.py + src/litetopk_select.cu

2. Hopper sparse IP top-k report
   data_scripts/hopper_sparse_ip/
```

Both use `.fvecs` data under `../data/marsco`.

## 1. Directory Layout

```text
marsco/
├── bench_all.py
├── src/                         # original LiteTopK L2/IP CUDA path
├── data_scripts/
│   └── hopper_sparse_ip/         # Hopper sparse IP report code
├── test_fused.py
└── README.md

../data/marsco/
├── base.fvecs                    # 1,000,000 x 768 corpus vectors
├── base_5m.fvecs                 # 5,000,000 x 768 corpus vectors
└── query.fvecs                   # query vectors, 768-d
```

## 2. Environment

Use the `litetopk2` container or an equivalent H100 CUDA environment.

```bash
docker exec -it litetopk2 bash
cd /workspace/project/src/marsco
export CUDA_VISIBLE_DEVICES=0
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

Use a fresh `TORCH_EXTENSIONS_DIR` after changing CUDA code or compile flags.

## 3. Reproduce L2 Speedup

This is the original MS MARCO L2 KNN reproduction in `bench_all.py`.

```bash
cd /workspace/project/src/marsco
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_marsco_l2

python3 bench_all.py \
  --base ../data/marsco/base.fvecs \
  --query ../data/marsco/query.fvecs \
  --metric l2 \
  --dtype fp16 \
  --methods ours,torch \
  --num-base 1000000 \
  --num-queries 1000 \
  --k 10,100,1024 \
  --warmup 5 --iters 20
```

Representative H100 fp16 L2 median latency:

| batch | M | k | torch ms | ours ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1000 | 1,000,000 | 10 | ~14.94 | ~10.61 | ~1.41x |
| 1000 | 1,000,000 | 100 | ~14.88 | ~10.63 | ~1.40x |
| 1000 | 1,000,000 | 1024 | ~14.85 | ~10.93 | ~1.36x |

Check the `recall` column in the summary.

## 4. Reproduce Hopper Sparse IP Speedup Report

This is the Hopper WGMMA sparse inner-product top-k path used for the
batch/N/k report. It compares:

```text
torch:  torch.topk(q @ base.T, k)
ours:   data_scripts/hopper_sparse_ip sparse fused path
```

Run the default report directly from `marsco/`:

```bash
cd /workspace/project/src/marsco
export TORCH_EXTENSIONS_DIR=/tmp/hopper_sparse_ip_report

python3 data_scripts/hopper_sparse_ip/report_speedup_vs_torch.py \
  --bs-list 1,8,32,64,128 \
  --m-list 262144,1000000,4000000 \
  --k-list 10,100,1000 \
  --warmup 10 --iters 30 --reps 3
```

The script uses:

```text
../data/marsco/base_5m.fvecs
../data/marsco/query.fvecs
```

and prints:

```text
   bs         N      k    torch_ms   sparse_ms   speedup   recall
-----------------------------------------------------------------
```

`speedup = torch_ms / sparse_ms`; values greater than 1 mean the Hopper sparse
path is faster. `recall` is index recall@k versus torch.

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

Main pattern:

- Larger batch and larger corpus size give higher speedup.
- Small corpus size (256k) can be slower because sample/select overhead is not
  amortized.
- Recall in the measured report was >= 0.99 for all listed points.

## 5. Optional m64n8 Register-Row Queue

For bs <= 8, `data_scripts/hopper_sparse_ip/hopper_topk_select.cu` contains two
write-out strategies for the m64n8 full sparse kernel:

```bash
HOPPER_M8_REG_ROW_QUEUE=0  # default: per-lane atomicAdd
HOPPER_M8_REG_ROW_QUEUE=1  # tidal-style 8-row register queue
```

Default is `0`, because it is best or tied for normal k. The register-row queue
can help only for very large k, where atomic contention in the m64n8 tile becomes
visible.

```bash
cd /workspace/project/src/marsco

export TORCH_EXTENSIONS_DIR=/tmp/hopper_sparse_ip_atomic
HOPPER_M8_REG_ROW_QUEUE=0 python3 data_scripts/hopper_sparse_ip/report_speedup_vs_torch.py \
  --bs-list 8 --m-list 1000000,4000000 --k-list 8192,16384

export TORCH_EXTENSIONS_DIR=/tmp/hopper_sparse_ip_regrow
HOPPER_M8_REG_ROW_QUEUE=1 python3 data_scripts/hopper_sparse_ip/report_speedup_vs_torch.py \
  --bs-list 8 --m-list 1000000,4000000 --k-list 8192,16384
```

Measured pattern:

- bs=8, k=8192: register-row is about 1-2% faster.
- bs=8, k=16384: register-row is about 4-7% faster.
- bs=32 should stay on the default atomic path; 32-row queues were measured to
  be slower, especially at large k.

## 6. 5M Corpus Variant

Use `base_5m.fvecs` for larger-corpus tests:

```bash
cd /workspace/project/src/marsco
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_marsco_l2_5m

python3 bench_all.py \
  --base ../data/marsco/base_5m.fvecs \
  --query ../data/marsco/query.fvecs \
  --metric l2 \
  --dtype fp16 \
  --methods ours,torch \
  --num-base 5000000 \
  --num-queries 1000 \
  --k 10,100,1024 \
  --warmup 5 --iters 20
```

## 7. Notes

- `query.fvecs` has finite query rows. Some scripts tile queries if a larger
  batch is requested; the Hopper sparse IP report uses the first `bs` rows.
- Keep `torch` enabled because recall and speedup are measured against it.
- `faiss`, `raft`, and `flashlib` are optional baselines and may be skipped if
  their packages are not installed.
- L2 and IP are different tasks; do not compare their speedups directly.


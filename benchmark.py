"""Standalone benchmark for efficientlargek (a self-contained copy of flashlib-2's
large-K fused KNN kernel).

Loads the real ``base.fvecs`` corpus, runs ``flash_knn_triton_flashlargek`` on it,
reports timing + peak memory, and checks recall@k against an exact brute-force
top-k on a subset of queries (so the test needs nothing but this folder + the
data file).

Usage:
    python efficientlargek/benchmark.py                       # defaults: M=1M, N=10000, k=10000
    python efficientlargek/benchmark.py --num-base 200000 --num-queries 1000 --k 5000
    python efficientlargek/benchmark.py --recall-queries 128  # brute-force this many for recall
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# make the flat local imports (flashlargek / _common / _row_norm) resolve
# regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from flashlargek import flash_knn_triton_flashlargek

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BASE = os.path.join(os.path.dirname(_HERE), "base.fvecs")


def read_fvecs(filename: str, limit=None) -> np.ndarray:
    """Minimal .fvecs reader (each vector: int32 dim header + dim float32s)."""
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)
    if data.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(data[0])
    rec = dim + 1
    n = data.size // rec
    arr = data.reshape(n, rec)[:, 1:].view(np.float32)
    if limit is not None:
        arr = arr[:limit]
    return np.ascontiguousarray(arr)


def _time_cuda(fn, warmup: int, iters: int, label: str):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    last = None
    for i in range(iters):
        starts[i].record()
        last = fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    print(
        f"[{label}] iters={iters}  mean={times.mean():.3f} ms  "
        f"median={np.median(times):.3f} ms  min={times.min():.3f} ms  max={times.max():.3f} ms"
    )
    return last, times


@torch.no_grad()
def _exact_recall(x_sub, c, idxs_sub, k):
    """Brute-force exact squared-L2 top-k for the subset, return mean recall@k."""
    d2 = torch.cdist(x_sub.float(), c.float()).pow(2)  # (S, M)
    ref = torch.topk(d2, k, dim=-1, largest=False).indices
    got = idxs_sub.long()
    rec = 0.0
    for i in range(got.shape[0]):
        rec += len(set(got[i].tolist()) & set(ref[i].tolist())) / k
    return rec / got.shape[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--num-base", type=int, default=1_000_000, help="corpus rows M")
    p.add_argument("--num-queries", type=int, default=10_000, help="query rows N")
    p.add_argument("--k", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--recall-queries", type=int, default=64,
                   help="brute-force this many queries for an exact recall check (0 to skip)")
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    base_np = read_fvecs(args.base, limit=args.num_base)
    print(f"read {base_np.shape} from {args.base} in {time.time()-t0:.1f}s")
    X = torch.from_numpy(base_np).to(device)                 # (M, D)
    M, D = X.shape
    N = min(args.num_queries, M)
    k = min(args.k, M)
    Q = X[:N].contiguous()                                   # queries = first N rows
    x = Q.unsqueeze(0)                                        # (1, N, D)
    c = X.unsqueeze(0)                                        # (1, M, D)
    print(f"M={M} N={N} D={D} k={k}  X={X.element_size()*X.numel()/1e9:.2f} GB")

    fn = lambda: flash_knn_triton_flashlargek(x, c, k, return_distances=True)

    print("warming up (triton compile/autotune) ...")
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    (vals, idxs), times = _time_cuda(fn, warmup=0, iters=args.iters, label="efficientlargek")
    print(f"peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if args.recall_queries > 0:
        S = min(args.recall_queries, N)
        rec = _exact_recall(Q[:S], X, idxs[0, :S], k)
        print(f"recall@{k} (exact brute-force on first {S} queries) = {rec:.5f}")


if __name__ == "__main__":
    main()

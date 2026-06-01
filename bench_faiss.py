"""Disabled benchmark entry point.

Use ``benchmark.py`` as the single active benchmark script. The previous
baseline implementation is commented below for reference.
"""

'''
"""Baseline 4 — faiss-GPU ``knn_gpu`` (brute-force exact top-K).

faiss-GPU has no public API to top-k a *precomputed* score matrix (unlike
``raft.select_k``); its only brute-force entry point is ``faiss.knn_gpu``, which
does the matmul **and** the selection itself. So here faiss owns the whole KNN
(there is no separate ``torch.matmul``). With ``metric=METRIC_L2`` it returns
true squared-L2 distances + indices — same I/O units as the other benches.

Note: faiss-GPU k-selection is capped at **k ≤ 2048** (warp/block-select
kernels); larger k raises an error and is skipped.

Run with the python10 env (faiss-gpu-cu12 installed there):
    /mnt/hdd/yinziqi/miniconda3/envs/python10/bin/python \\
        efficientlargek/bench_faiss.py --batch-size 64 --k 2048
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss
import faiss.contrib.torch_utils  # noqa: F401  (enables torch-tensor args)
import torch

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, _time_cuda, read_fvecs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--query", default=_DEFAULT_QUERY)
    p.add_argument("--num-base", type=int, default=1_000_000, help="corpus rows M")
    p.add_argument("--batch-size", type=int, default=512, help="queries per call")
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  faiss {faiss.__version__}")

    X = torch.from_numpy(read_fvecs(args.base, limit=args.num_base)).to(device).float()
    Q = torch.from_numpy(read_fvecs(args.query)).to(device).float()
    M, D = X.shape
    B = args.batch_size
    k = min(args.k, M)
    x = Q[torch.arange(B, device=device) % Q.shape[0]].contiguous()  # tile to fill bs
    c = X.contiguous()  # corpus
    assert x.shape[1] == D, f"query dim {x.shape[1]} != corpus dim {D}"
    print(f"M={M} D={D} batch_size={B} (of {Q.shape[0]} queries) k={k}")

    res = faiss.StandardGpuResources()

    # knn_gpu(res, xq, xb, k, metric) -> (distances, indices); METRIC_L2 gives
    # true squared-L2 distances. faiss does the matmul + selection internally.
    def fn():
        Dout, Iout = faiss.knn_gpu(res, x, c, k, metric=faiss.METRIC_L2)
        return Dout, Iout

    torch.cuda.reset_peak_memory_stats()
    _time_cuda(
        fn,
        warmup=args.warmup,
        iters=args.iters,
        label=f"faiss.knn_gpu B={B} M={M} k={k}",
    )
    print(f"peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
'''

"""Unified isolated sweep (one process, no autotune methods → no cross-shape
contamination): torch / raft / faiss / ours over bs ∈ {1,8,64,256,1024} ×
k ∈ {100,512,2048,4096,10000,40000}. Queries tiled to fill bs. Run ALONE on
the GPU. flashlib is measured separately (per-config isolated)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils  # noqa: F401
from pylibraft.common import DeviceResources

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, read_fvecs
from bench_torch_topk import torch_matmul_topk
from bench_raft_topk import matmul_raft_topk
from flashlargek import flash_knn_triton_flashlargek

device = "cuda"
X = torch.from_numpy(read_fvecs(_DEFAULT_BASE, limit=1_000_000)).to(device).float().contiguous()
Q = torch.from_numpy(read_fvecs(_DEFAULT_QUERY)).to(device).float()
M, D = X.shape
BATCHES = [1, 8, 64, 256, 1024]
KS = [100, 512, 2048, 4096, 10000, 40000]
handle = DeviceResources()
res = faiss.StandardGpuResources()


def timeit(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return float(np.mean([a.elapsed_time(b) for a, b in zip(s, e)]))


print("bs,k,torch_ms,raft_ms,faiss_ms,ours_ms", flush=True)
for bs in BATCHES:
    x = Q[torch.arange(bs, device=device) % Q.shape[0]].contiguous()  # tile to fill bs
    for k in KS:
        kk = min(k, M)
        t_torch = timeit(lambda: torch_matmul_topk(x, X, kk))
        t_raft = timeit(lambda: matmul_raft_topk(x, X, kk, handle))
        t_ours = timeit(lambda: flash_knn_triton_flashlargek(x, X, kk, return_distances=True))
        if kk <= 2048:
            t_faiss = f"{timeit(lambda: faiss.knn_gpu(res, x, X, kk, metric=faiss.METRIC_L2)):.2f}"
        else:
            t_faiss = "NA"  # faiss-GPU k-selection cap
        print(f"{bs},{kk},{t_torch:.2f},{t_raft:.2f},{t_faiss},{t_ours:.2f}", flush=True)
print("DONE", flush=True)

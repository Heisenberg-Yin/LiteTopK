"""faiss-GPU knn_gpu sweep (k ≤ 2048; faiss GPU k-selection cap).
Loads corpus once, prints B,k,faiss_ms for all batches × k≤2048."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss
import faiss.contrib.torch_utils  # noqa: F401
import numpy as np
import torch

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, read_fvecs

device = "cuda"
X = torch.from_numpy(read_fvecs(_DEFAULT_BASE, limit=1_000_000)).to(device).float().contiguous()
Q = torch.from_numpy(read_fvecs(_DEFAULT_QUERY)).to(device).float()
M, D = X.shape
KS = [10, 32, 64, 128, 256, 512, 1024, 2048]  # faiss GPU caps at 2048
BATCHES = [4, 8, 16, 32, 64, 128]
res = faiss.StandardGpuResources()


def timeit(fn, warmup=5, iters=100):
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


for B in BATCHES:
    x = Q[:B].contiguous()
    for k in KS:
        kk = min(k, M)
        t = timeit(lambda: faiss.knn_gpu(res, x, X, kk, metric=faiss.METRIC_L2))
        print(f"{B},{kk},{t:.2f}", flush=True)
print("DONE", flush=True)

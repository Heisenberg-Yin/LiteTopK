"""Re-measure ours (flashlargek) after int32 addressing + sample min/max.
Loads corpus once, prints B,k,ms for all batches × k."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, read_fvecs
from flashlargek import flash_knn_triton_flashlargek

device = "cuda"
X = torch.from_numpy(read_fvecs(_DEFAULT_BASE, limit=1_000_000)).to(device).float().contiguous()
Q = torch.from_numpy(read_fvecs(_DEFAULT_QUERY)).to(device).float()
M, D = X.shape
KS = [10, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10000, 20000, 40000]
BATCHES = [4, 8, 16, 32, 64, 128]


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

        t = timeit(lambda: flash_knn_triton_flashlargek(x, X, kk, return_distances=True))
        print(f"{B},{kk},{t:.2f}", flush=True)
print("DONE", flush=True)

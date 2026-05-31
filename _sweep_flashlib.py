"""flashlib.flash_knn steady-state sweep (M=1M, batches × k).

flash_knn autotunes per (shape, k) on the FIRST call (slow for large k); warmup
absorbs it, then we measure steady-state. Loads corpus once, prints B,k,ms.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flashlib"))

import numpy as np
import torch

from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, read_fvecs
from flashlib import flash_knn

device = "cuda"
X = torch.from_numpy(read_fvecs(_DEFAULT_BASE, limit=1_000_000)).to(device).float().contiguous()
Q = torch.from_numpy(read_fvecs(_DEFAULT_QUERY)).to(device).float()
M, D = X.shape
KS = [10, 32, 64, 128, 256, 512, 1024]  # k≥2048: OutOfResources; k≥10000: autotune hangs
BATCHES = [4, 8, 16, 32, 64, 128]


def timeit(fn, warmup=8, iters=50):
    for _ in range(warmup):  # absorbs per-(shape,k) autotune + compile
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
        try:
            t = timeit(lambda: flash_knn(x, X, kk, return_distances=True))
            print(f"{B},{kk},{t:.2f}", flush=True)
        except Exception as ex:
            print(f"{B},{kk},ERR:{type(ex).__name__}", flush=True)
print("DONE", flush=True)

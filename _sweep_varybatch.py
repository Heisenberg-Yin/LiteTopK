"""One-off sweep: M=1M, batch ∈ {8,16,32,128}, k swept, 3 methods.
Loads the corpus once, writes latency_varybatch_m1m.csv."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from pylibraft.common import DeviceResources

from _row_norm import _csq_cache
from bench_raft_topk import matmul_raft_topk
from bench_torch_topk import torch_matmul_topk
from benchmark import _DEFAULT_BASE, _DEFAULT_QUERY, read_fvecs
from flashlargek import flash_knn_triton_flashlargek

device = "cuda"
X = torch.from_numpy(read_fvecs(_DEFAULT_BASE, limit=1_000_000)).to(device).float()
Q = torch.from_numpy(read_fvecs(_DEFAULT_QUERY)).to(device).float()
M, D = X.shape
KS = [10, 32, 64, 128, 256, 512, 1024, 2048, 4096, 10000, 20000, 40000]
BATCHES = [8, 16, 32, 128]
handle = DeviceResources()


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


rows = []
for B in BATCHES:
    x = Q[:B].contiguous()
    for k in KS:
        kk = min(k, M)
        t = timeit(lambda: torch_matmul_topk(x, X, kk))
        od = torch.empty(B, kk, device=device, dtype=torch.float32)
        oi = torch.empty(B, kk, device=device, dtype=torch.int64)
        r = timeit(lambda: matmul_raft_topk(x, X, kk, handle, od, oi))

        def ours():
            _csq_cache.clear()
            return flash_knn_triton_flashlargek(x, X, kk, return_distances=True)

        o = timeit(ours)
        rows.append((B, M, D, kk, t, r, o))
        print(f"B={B} k={kk}: torch={t:.2f} raft={r:.2f} ours={o:.2f}", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_varybatch_m1m.csv")
with open(out, "w") as f:
    f.write("batch_size,M,D,k,torch_ms,raft_ms,efficienttopk_ms\n")
    for B, M_, D_, kk, t, r, o in rows:
        f.write(f"{B},{M_},{D_},{kk},{t:.2f},{r:.2f},{o:.2f}\n")
print("WROTE", out, flush=True)

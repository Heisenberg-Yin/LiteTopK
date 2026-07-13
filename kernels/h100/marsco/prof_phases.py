import os, sys, torch, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_HERE), "data", "marsco")
sys.path.insert(0, os.path.join(_HERE, "src"))
from litetopk_fused import fused_knn_topk_ip

def read_fvecs(p, limit=None):
    a = np.fromfile(p, dtype=np.int32); d = a[0]
    a = a.reshape(-1, d + 1)
    if limit: a = a[:limit]
    return torch.from_numpy(a[:, 1:].copy().view(np.float32))

M, N, k = 1_000_000, 256, 100
X = read_fvecs(os.path.join(_DATA_DIR, "base.fvecs"), M).cuda().bfloat16()
Q = read_fvecs(os.path.join(_DATA_DIR, "query.fvecs"), N).cuda().bfloat16()

for _ in range(5):
    fused_knn_topk_ip(Q, X, k, return_distances=False)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(10):
        fused_knn_topk_ip(Q, X, k, return_distances=False)
    torch.cuda.synchronize()

evts = [e for e in prof.key_averages() if e.device_time_total > 0]
evts.sort(key=lambda e: -e.device_time_total)
total = sum(e.device_time_total for e in evts)
print(f"total CUDA time/iter = {total/10/1000:.3f} ms  (N={N} M={M} k={k} bf16 IP)")
print(f"{'kernel':<55} {'ms/iter':>9} {'pct':>6}")
for e in evts[:15]:
    print(f"{e.key[:54]:<55} {e.device_time_total/10/1000:>9.3f} {100*e.device_time_total/total:>5.1f}%")

# ---- baseline: pure torch Q@X.T GEMM alone, and GEMM+topk ----
Xf = X.t().contiguous()  # [D, M] for Q@X.T
def gemm_only():
    return Q @ Xf
def gemm_topk():
    s = Q @ Xf
    return torch.topk(s, k, dim=-1, largest=True).indices
for _ in range(5):
    gemm_topk()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p2:
    for _ in range(10):
        gemm_only()
    torch.cuda.synchronize()
t_gemm = sum(e.device_time_total for e in p2.key_averages())/10/1000
with profile(activities=[ProfilerActivity.CUDA]) as p3:
    for _ in range(10):
        gemm_topk()
    torch.cuda.synchronize()
t_gemmtopk = sum(e.device_time_total for e in p3.key_averages())/10/1000
print(f"\n[torch baseline] pure Q@X.T GEMM = {t_gemm:.3f} ms   GEMM+topk = {t_gemmtopk:.3f} ms")
print(f"[ours] _flat_kernel = {evts[0].device_time_total/10/1000:.3f} ms")
print(f"=> ours flat does GEMM + bucket + atomic-hist + buf-writeback in one pass")

"""Per-kernel time breakdown of one fused-op iteration via torch profiler."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../litetopk_engine"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
import torch
from bench_marsco_b200 import load_base_fp16, read_fvecs, DATA
from tidal_hopper_ops import fused_ip_gqa_sparse_b200
import numpy as np

M = int(os.environ.get("M", 1000000))
K = int(os.environ.get("K", 8192))
BS = 64
os.environ.setdefault("TIDAL_FLAT_QN", "64")
os.environ.setdefault("TIDAL_BM", "256")

base = load_base_fp16(M)
qv, d = read_fvecs(os.path.join(DATA, "query.fvecs"), max_rows=BS)
q = torch.from_numpy(qv.astype(np.float16)).cuda().contiguous()
kv3 = base.unsqueeze(0)
sample = min(M, max(16384, min(8 * K, 262144)))

def ours():
    return fused_ip_gqa_sparse_b200(
        q, kv3, K, num_buckets=64, sample_size=sample,
        refresh_every=8, num_ctas_x=296, sample_mode=1, out_fp32=True)[1]

for _ in range(5):
    ours()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        ours()
    torch.cuda.synchronize()

ka = prof.key_averages()
rows = [(e.key, e.device_time_total / 10.0, e.count // 10) for e in ka if e.device_time_total > 0]
rows.sort(key=lambda r: -r[1])
tot = sum(r[1] for r in rows)
print(f"M={M} K={K} bs={BS} sample={sample}  total device us/iter: {tot:.1f}")
for name, us, cnt in rows:
    print(f"  {us:8.1f} us  x{cnt:<3d} {name[:110]}")

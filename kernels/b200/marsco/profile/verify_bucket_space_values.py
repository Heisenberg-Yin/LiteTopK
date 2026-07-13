"""Explicit VALUE correctness check for the bucket-space candidate storage change.

bench_marsco_b200.py's recall() only intersects INDEX sets -- it would not catch
a bug where out_val holds the wrong number (e.g. un-converted bucket-space bq
instead of the real IP score). This script checks the actual returned scores
against the dense ground truth for the same indices.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
import torch

from bench_marsco_b200 import load_base_fp16, read_fvecs, DATA
from tidal_hopper_ops import fused_ip_gqa_sparse_b200

M = int(os.environ.get("M", 1000000))
K = int(os.environ.get("K", 128))
BS = 64

base = load_base_fp16(M)
qv, d = read_fvecs(os.path.join(DATA, "query.fvecs"), max_rows=BS)
q = torch.from_numpy(qv.astype(np.float16)).cuda().contiguous()
kv3 = base.unsqueeze(0)
sample = min(M, max(16384, 8 * K))

val, idx = fused_ip_gqa_sparse_b200(
    q, kv3, K, num_buckets=64, sample_size=sample,
    refresh_every=8, num_ctas_x=296, sample_mode=1, out_fp32=True)

dense = (q.float() @ base.float().t())
true_val = torch.gather(dense, 1, idx.long())

diff = (val - true_val).abs()
rel = diff / true_val.abs().clamp_min(1e-6)
print(f"M={M} K={K}: max_abs_err={diff.max().item():.6g}  "
      f"mean_abs_err={diff.mean().item():.6g}  max_rel_err={rel.max().item():.6g}")
print(f"val sample:  {val[0, :5].tolist()}")
print(f"true sample: {true_val[0, :5].tolist()}")
assert diff.max().item() < 0.05, "VALUE MISMATCH -- bucket-space conversion is broken"
print("PASS: output values match dense ground truth for the returned indices")

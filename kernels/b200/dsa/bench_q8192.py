#!/usr/bin/env python3
"""Benchmark the Q=8192 LiteTopK indexer on recorded GLM-5 KV caches.

The LiteTopK timer receives a precomputed hot-position sample through
``hot_prev`` and sets ``_carry_io=False``. It therefore covers the current
indexing call but not construction of the carry consumed by the next call.

Set ``DSA_CACHE_DIR`` to the directory containing the four cache files.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODULE_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "glm5_prefill", "litetopk_vllm"))
sys.path.insert(
    0, os.environ.get("LITETOPK_MODULE_DIR", _DEFAULT_MODULE_DIR))
os.environ.setdefault("VLLM_LITETOPK_MIN_S", "0")
# Keep the full calibration-score matrix so this script times one fixed path.
os.environ.setdefault("VLLM_LITETOPK_PREP_TILE", "0")
# Each shape initializes the carried-position sample from the dense reference
# before timing the LiteTopK path.
os.environ.setdefault("VLLM_LITETOPK_HOTONLY", "1")
import litetopk_indexer  # noqa: E402

from safetensors import safe_open  # noqa: E402
import deep_gemm  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

K, Q = 2048, 8192
CACHE_DIR = os.environ.get("DSA_CACHE_DIR", "/data/dsa_caches")


def evt(fn, w=5, n=20):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


print(f"{'S':>5} {'Q':>5} {'dense':>9} {'litetopk':>9} {'speedup':>8} {'recall':>8}")
for tag in ["256k", "512k", "768k", "1m"]:
    p = os.path.join(
        CACHE_DIR, f"glm5_{tag}_realtext_chunk8192.safetensors")
    T = {}
    with safe_open(p, "pt", device="cuda") as f:
        for kk in f.keys():
            T[kk] = f.get_tensor(kk)
    q = T["q_index"][:Q].contiguous()
    qs = T["q_index_scale"].squeeze(-1)[:Q]
    kf = T["idx_k_cache"][0].contiguous()
    ksc = T["idx_k_scale"][0, :, 0].contiguous()
    w = (T["gate_w"][:Q] * qs * (q.shape[2] ** -0.5)).contiguous().float()
    S = kf.shape[0]
    rows = torch.arange(Q, device="cuda", dtype=torch.int32)
    ks = torch.zeros(Q, dtype=torch.int32, device="cuda")
    ke = (S - Q + 1 + rows).contiguous()
    out = torch.full((Q, K), -1, dtype=torch.int32, device="cuda")
    outo = torch.full((Q, K), -1, dtype=torch.int32, device="cuda")

    def dense_reference():
        lg = deep_gemm.fp8_fp4_mqa_logits((q, None), (kf, ksc), w, ks, ke, clean_logits=False)
        ops.top_k_per_row_prefill(lg, ks, ke, outo, Q, lg.stride(0), lg.stride(1), K)

    hot_key = f"bench_{tag}"

    # Initialize and read back a fixed carried-position sample for the timer.
    dense_reference()
    torch.cuda.synchronize()
    litetopk_indexer.stash_carry(hot_key, outo, S)
    torch.cuda.synchronize()
    carry = litetopk_indexer._HOT_CARRY[(str(q.device), hot_key)][0]

    def litetopk_run():
        # Use the fixed sample without generating the next call's carry.
        assert litetopk_indexer.try_chunk(q, kf, ksc, w, ks, ke, out, K,
                                          hot_prev=carry, _carry_io=False)

    n = 20 if tag in ("256k", "512k") else 8
    t_litetopk = evt(litetopk_run, n=n)
    t_dense = evt(dense_reference, n=n)
    dense_reference()
    torch.cuda.synchronize()
    a = outo.long().sort(dim=1).values
    b = out.long().sort(dim=1).values
    pp = torch.searchsorted(a, b).clamp(max=K - 1)
    rec = 100.0 * (torch.gather(a, 1, pp) == b).float().mean().item()
    print(f"{tag:>5} {Q:>5} {t_dense:>9.2f} {t_litetopk:>9.2f} "
          f"{t_dense / t_litetopk:>7.2f}x {rec:>7.2f}%",
          flush=True)
    del T, q, kf, ksc, w, out, outo
    torch.cuda.empty_cache()

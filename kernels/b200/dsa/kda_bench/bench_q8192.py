#!/usr/bin/env python3
"""Merged-chunk shape: Q=8192 vs official, real KV, all four S."""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("LITETOPK_MODULE_DIR", "/opt/simtopk_repro/glm5_prefill/litetopk_vllm"))
os.environ.setdefault("VLLM_LITETOPK_MIN_S", "0")
# Record-mode defaults: GATE4 = the bucket-gate kernel the hot-start
# deployment builds (~0.9ms faster at 1M, lifts 256k->1.07x/512k->1.14x);
# PREP_TILE=0 = full slog (the record's memory-vs-speed choice, ~0.2ms).
os.environ.setdefault("VLLM_LITETOPK_XFLAGS", "DSA_BUCKET_GATE4=1")
os.environ.setdefault("VLLM_LITETOPK_PREP_TILE", "0")
# This is the COLD GATE4 kernel figure (deterministic, no carry). The
# hot-start method numbers are in REPRODUCTION.md (measured with per-size
# carry keys). Module now defaults HOTONLY=1, so opt out here.
os.environ.setdefault("VLLM_LITETOPK_HOTONLY", "0")
import litetopk_indexer  # noqa: E402

from safetensors import safe_open  # noqa: E402
import deep_gemm  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

K, Q = 2048, 8192


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


print(f"{'S':>5} {'Q':>5} {'official':>9} {'ours':>9} {'speedup':>8} {'recall':>8}")
for tag in ["256k", "512k", "768k", "1m"]:
    p = f"/data/dsa_caches/glm5_{tag}_realtext_chunk8192.safetensors"
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

    def official():
        lg = deep_gemm.fp8_fp4_mqa_logits((q, None), (kf, ksc), w, ks, ke, clean_logits=False)
        ops.top_k_per_row_prefill(lg, ks, ke, outo, Q, lg.stride(0), lg.stride(1), K)

    def ours():
        assert litetopk_indexer.try_chunk(q, kf, ksc, w, ks, ke, out, K)

    n = 20 if tag in ("256k", "512k") else 8
    t_ours = evt(ours, n=n)
    t_off = evt(official, n=n)
    official()
    torch.cuda.synchronize()
    a = outo.long().sort(dim=1).values
    b = out.long().sort(dim=1).values
    pp = torch.searchsorted(a, b).clamp(max=K - 1)
    rec = 100.0 * (torch.gather(a, 1, pp) == b).float().mean().item()
    print(f"{tag:>5} {Q:>5} {t_off:>9.2f} {t_ours:>9.2f} {t_off / t_ours:>7.2f}x {rec:>7.2f}%",
          flush=True)
    del T, q, kf, ksc, w, out, outo
    torch.cuda.empty_cache()

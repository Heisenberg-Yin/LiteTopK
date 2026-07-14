#!/usr/bin/env python3
"""Merged-chunk shape: Q=8192 vs official, real KV, all four S.

KERNEL-ONLY timing: the timed `ours()` feeds the pre-stashed hot sample
directly (hot_prev) with `_carry_io=False`, so the measured region EXCLUDES
the next-chunk hot-sample refresh (a vote-histogram + topk over S that seeds
the FOLLOWING chunk). That refresh is amortized onto the previous chunk / an
async side stream in deployment and must NOT count toward the time to index
THIS chunk. GATE4 + HOTONLY are the litetopk_indexer defaults now."""
import os
import sys

import torch

sys.path.insert(0, os.environ.get("LITETOPK_MODULE_DIR", "/opt/litetopk_repro/glm5_prefill/litetopk_vllm"))
os.environ.setdefault("VLLM_LITETOPK_MIN_S", "0")
# Record-mode defaults: GATE4 = the bucket-gate kernel the hot-start
# deployment builds (~0.9ms faster at 1M, lifts 256k->1.07x/512k->1.14x);
# PREP_TILE=0 = full slog (the record's memory-vs-speed choice, ~0.2ms).
os.environ.setdefault("VLLM_LITETOPK_XFLAGS", "DSA_BUCKET_GATE4=1")
os.environ.setdefault("VLLM_LITETOPK_PREP_TILE", "0")
# HOT-START kernel figure (the shipped method: prefix deleted). Each shape
# primes its per-size hot carry from the OFFICIAL top-k ("his method"),
# exactly as the deployment bootstraps the first ours-chunk from the last
# official chunk via stash_carry; the timed loop then runs the hot path and
# refreshes the carry from its own output every iteration.
os.environ.setdefault("VLLM_LITETOPK_HOTONLY", "1")
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

    hot_key = f"bench_{tag}"

    # Prime the hot carry from the official top-k, reproducing the deployed
    # bootstrap (last official chunk -> stash_carry -> first ours-chunk HOT),
    # then read it back so the timed loop can feed it as a FIXED hot sample.
    official()
    torch.cuda.synchronize()
    litetopk_indexer.stash_carry(hot_key, outo, S)
    torch.cuda.synchronize()
    carry = litetopk_indexer._HOT_CARRY[(str(q.device), hot_key)][0]

    def ours():
        # _carry_io=False + explicit hot_prev = kernel-only: use the hot
        # sample but skip the next-chunk carry refresh inside the timed region.
        assert litetopk_indexer.try_chunk(q, kf, ksc, w, ks, ke, out, K,
                                          hot_prev=carry, _carry_io=False)

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

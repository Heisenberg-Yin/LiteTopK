#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-way benchmark for the B200 (SM100) DSA fused sparse top-k kernel.

Compares, on the same B200, at ~100% recall vs the exact deep_gemm top-k:
  A  stock          : deep_gemm.fp8_mqa_logits (SM100, no KV-split) + torch.topk
  B  ideal KV-split : our KV-split DENSE scoring (-DDENSE_WRITE) + torch.topk
  O  ours (fused)   : KV-split scoring + fused sparse gate/select

A->B isolates the (generic) KV-split scheduling win; B->O isolates our fused
sparse-selection win (no dense logits materialization, no million-wide torch.topk).

Usage:
    python3 bench_3way.py [256k 512k 768k 1m]
Env knobs: K(=2048) SAMPLE(=65536) NB(=256) REFRESH(=64) WARMUP(=8) ITERS(=30)
           CHUNK(=4096) and the DSA_* path overrides from b200_config.
"""
import os, sys, torch
from torch.utils.cpp_extension import load

import b200_config as cfg
cfg.prepare_env()
import deep_gemm
from safetensors import safe_open


def build():
    os.makedirs("/tmp/build_sparse_b200", exist_ok=True); os.makedirs("/tmp/build_dense_b200", exist_ok=True)
    sparse = load(name="github_dsa_b200_sparse", sources=[cfg.SRC], extra_include_paths=cfg.INC,
                  extra_cuda_cflags=cfg.FLAGS, build_directory="/tmp/build_sparse_b200",
                  extra_ldflags=["-lcuda"], verbose=False)
    dense = load(name="github_dsa_b200_dense", sources=[cfg.SRC], extra_include_paths=cfg.INC,
                 extra_cuda_cflags=cfg.FLAGS + ["-DDENSE_WRITE"], build_directory="/tmp/build_dense_b200",
                 extra_ldflags=["-lcuda"], verbose=False)
    return sparse, dense


WARMUP = int(os.environ.get("WARMUP", "8")); ITERS = int(os.environ.get("ITERS", "30"))
def evt(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS


def run(tag, sparse, dense):
    K = int(os.environ.get("K", "2048")); Ssample = int(os.environ.get("SAMPLE", "65536"))
    NB = int(os.environ.get("NB", "256")); refresh = int(os.environ.get("REFRESH", "64"))
    T = {}
    with safe_open(cfg.cache_path(tag), "pt", device="cuda") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
    q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
    kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
    gw = T["gate_w"]; Q, H, D = q.shape; S = kf.shape[0]
    w = (gw * qs * (D ** -0.5)).contiguous().float()
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke_full = torch.full((Q,), S, dtype=torch.int32, device="cuda")
    lg_ref = deep_gemm.fp8_mqa_logits(q, (kf, ksc), w, ks0, ke_full, clean_logits=True, max_seqlen_k=0).contiguous()
    ref = lg_ref.topk(K, dim=-1).indices.long(); refs, _ = ref.sort(dim=-1)
    def recall(idx):
        h = 0
        for i in range(Q):
            p = torch.searchsorted(refs[i], idx[i]).clamp(max=K - 1); h += (refs[i][p] == idx[i]).sum().item()
        return 100 * h / (Q * K)

    # A: stock deep_gemm (no KV-split) + torch.topk
    def A_e2e():
        lg = deep_gemm.fp8_mqa_logits(q, (kf, ksc), w, ks0, ke_full, clean_logits=True, max_seqlen_k=0)
        return lg.topk(K, dim=-1)
    tA = evt(A_e2e)

    # B: ideal KV-split DENSE scoring + torch.topk
    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda")
    th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    def B_score():
        return dense.mqa_logits_dsa_marsco(q, kf, ksc, w, ks0, ke_full, o0, i1, th0, sv0, si0, NB, S, K, 0, -1)
    cv, _, _ = B_score(); vb, ib = cv.topk(K, dim=-1); rB = recall(ib.long())
    tBs = evt(B_score); tB = evt(lambda: B_score()[0].topk(K, dim=-1))

    # O: ours fused sparse (head sample + KV-split + fused select)
    head = min(Ssample, S); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous(); Ss = ksamp.shape[0]
    ke_s = torch.full((Q,), Ss, dtype=torch.int32, device="cuda")
    ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    col_full = pos.to(torch.int32).view(1, -1).expand(Q, -1).contiguous()
    cnt_full = torch.full((Q,), head, dtype=torch.int32, device="cuda")

    def O_prep():
        sl = deep_gemm.fp8_mqa_logits(q, (ksamp, kss), w, ks0, ke_s, clean_logits=True, max_seqlen_k=0).contiguous()
        xs = -sl; o2 = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
        inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
        sv, si = sparse.compact_topk_min_idx_marsco(xs, col_full, cnt_full, K)
        sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(dim=1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        return o2, inv2, th2, sv, si

    prep = O_prep()
    def O_score_from_prep(prep):
        o2, inv2, th2, sv, si = prep
        return sparse.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th2, sv, si, NB, 1, K, refresh, -1)
    def O_full_from_prep(prep):
        cv2, ci2, cc2 = O_score_from_prep(prep)
        o2, inv2, th2, sv, si = prep
        return sparse.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K)
    def O_e2e():
        return O_full_from_prep(O_prep())

    ov, oi = O_e2e(); torch.cuda.synchronize()
    rO = recall(oi.long())
    tOprep = evt(O_prep)
    tOs = evt(lambda: O_score_from_prep(prep))
    tOmain_sel = evt(lambda: O_full_from_prep(prep))
    tOsel = tOmain_sel - tOs
    tO = evt(O_e2e)
    return dict(Q=Q, S=S, K=K, tA=tA, tBs=tBs, tB=tB, rB=rB, tOprep=tOprep, tOs=tOs, tOsel=tOsel, tO=tO, rO=rO)


if __name__ == "__main__":
    sparse, dense = build()
    tags = sys.argv[1:] or ["256k", "512k", "768k", "1m"]
    rows = []
    for tag in tags:
        r = run(tag, sparse, dense); rows.append((tag, r))
        print(f"[{tag}] done recall_O={r['rO']:.3f}% recall_B={r['rB']:.3f}%", flush=True)
    print("\n================ 3-WAY BENCHMARK (B200 SM100, K=2048) ================")
    print(f"{'seq':>5} {'A_stock':>9} {'B_ideal':>9} {'O_ours':>9} {'O/A':>6} {'O/B':>6}  (ms, e2e)")
    for tag, r in rows:
        print(f"{tag:>5} {r['tA']:>9.3f} {r['tB']:>9.3f} {r['tO']:>9.3f} {r['tA']/r['tO']:>5.2f}x {r['tB']/r['tO']:>5.2f}x")
    print("\nBreakdown (ms):")
    print(f"{'seq':>5} {'B_score':>8} {'O_sample':>9} {'O_score':>8} {'O_select':>9}")
    for tag, r in rows:
        print(f"{tag:>5} {r['tBs']:>8.3f} {r['tOprep']:>9.3f} {r['tOs']:>8.3f} {r['tOsel']:>9.3f}")

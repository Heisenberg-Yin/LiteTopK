#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B200 (SM100) DSA speedup benchmark on the REAL GLM-5 DSA caches, WITHOUT
depending on deep_gemm.

Why no deep_gemm: the shipped `deep_gemm.fp8_mqa_logits` SM100 kernel only
supports kHeadDim=64 (DeepSeek MLA); it static-asserts on GLM DSA's kHeadDim=128
(`kNumWeightsInReg <= kNumHeads` fails). So the baseline here is our OWN KV-split
DENSE fp8 scoring kernel (`-DDENSE_WRITE`) + torch.topk -- exactly the "DSA"
ideal baseline used in the H100 figures -- and the LiteTopK path is our fused
sparse scan + radix select. The exact top-k reference (for recall) also comes
from the dense kernel, so both share identical scoring numerics.

For each real cache glm5_{tag}_realtext.safetensors (Q=64, H=32, D=128), the row
of KV tokens S in {256k,512k,768k,1m}, it times:
  DSA     = dense KV-split fp8 logits + torch.topk           (baseline)
  LiteTopK = sample seed + fused sparse scan + thr radix select (ours)
and prints latency + speedup + recall, and dumps a JSON the plotting script uses.

Run inside the `simtopk` container:
  PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
  DSA_CACHE_DIR=/data/dsa CHUNK=0 \
  /usr/bin/python3.12 bench_b200_nodg.py 256k 512k 768k 1m
"""
import os, sys, json, torch
from torch.utils.cpp_extension import load

import b200_config as cfg
cfg.prepare_env()
from safetensors import safe_open


def build():
    os.makedirs("/tmp/build_sparse_b200", exist_ok=True); os.makedirs("/tmp/build_dense_b200", exist_ok=True)
    sp = load(name="github_dsa_b200_sparse", sources=[cfg.SRC], extra_include_paths=cfg.INC, extra_cuda_cflags=cfg.FLAGS,
              build_directory="/tmp/build_sparse_b200", extra_ldflags=["-lcuda"], verbose=False)
    dn = load(name="github_dsa_b200_dense", sources=[cfg.SRC], extra_include_paths=cfg.INC, extra_cuda_cflags=cfg.FLAGS + ["-DDENSE_WRITE"],
              build_directory="/tmp/build_dense_b200", extra_ldflags=["-lcuda"], verbose=False)
    return sp, dn


WARMUP = int(os.environ.get("WARMUP", "8")); ITERS = int(os.environ.get("ITERS", "30"))
def evt(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS


def dense_scores(dn, q, kf, ksc, w, ks0, ke_full, o0, i1, th0, sv0, si0, NB, S, K):
    # DENSE_WRITE kernel: writes the full [Q,S] logits into cand_val.
    cv, _, _ = dn.mqa_logits_dsa_marsco(q, kf, ksc, w, ks0, ke_full, o0, i1, th0, sv0, si0, NB, S, K, 0, -1)
    return cv


def run(tag, sp, dn):
    K = int(os.environ.get("K", "2048")); NB = int(os.environ.get("NB", "256"))
    refresh = int(os.environ.get("REFRESH", "64")); Ssample = int(os.environ.get("SAMPLE", "65536"))
    T = {}
    with safe_open(cfg.cache_path(tag), "pt", device="cuda") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
    q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
    kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
    gw = T["gate_w"]; Q, H, D = q.shape; S = kf.shape[0]
    w = (gw * qs * (D ** -0.5)).contiguous().float()
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke_full = torch.full((Q,), S, dtype=torch.int32, device="cuda")
    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda")
    th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")

    # exact reference top-k (dense kernel scoring + torch.topk)
    cvref = dense_scores(dn, q, kf, ksc, w, ks0, ke_full, o0, i1, th0, sv0, si0, NB, S, K)
    ref = cvref.topk(K, dim=-1).indices.long(); refs, _ = ref.sort(dim=-1); del cvref
    def recall(idx):
        h = 0
        for i in range(Q):
            p = torch.searchsorted(refs[i], idx[i]).clamp(max=K - 1); h += (refs[i][p] == idx[i]).sum().item()
        return 100 * h / (Q * K)

    # DSA baseline: dense scoring + torch.topk (end-to-end)
    def DSA_e2e():
        cv = dense_scores(dn, q, kf, ksc, w, ks0, ke_full, o0, i1, th0, sv0, si0, NB, S, K)
        return cv.topk(K, dim=-1)
    tDSA = evt(DSA_e2e)

    # LiteTopK ours: HEAD sample -> seed buckets/threshold -> fused sparse scan -> thr select.
    head = min(Ssample, S); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous(); Ss = ksamp.shape[0]
    ke_s = torch.full((Q,), Ss, dtype=torch.int32, device="cuda")
    ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    col_full = pos.to(torch.int32).view(1, -1).expand(Q, -1).contiguous()
    cnt_full = torch.full((Q,), head, dtype=torch.int32, device="cuda")

    def O_prep():
        # sample stage: dense-score the head sample, define buckets + threshold + seed.
        sl = dense_scores(dn, q, ksamp, kss, w, ks0, ke_s, o0, i1, th0, sv0, si0, NB, Ss, K)
        xs = -sl; o2 = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
        inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
        sv, si = sp.compact_topk_min_idx_marsco(xs, col_full, cnt_full, K)
        sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(dim=1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        return o2, inv2, th2, sv, si

    prep = O_prep()
    def O_score(p):
        o2, inv2, th2, sv, si = p
        return sp.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th2, sv, si, NB, 1, K, refresh, -1)
    def O_full(p):
        cv2, ci2, cc2 = O_score(p); o2, inv2, th2, sv, si = p
        return sp.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K)
    def O_e2e():
        return O_full(O_prep())

    ov, oi = O_e2e(); torch.cuda.synchronize(); rO = recall(oi.long())
    tprep = evt(O_prep); tscore = evt(lambda: O_score(prep)); tfull = evt(lambda: O_full(prep))
    tO = evt(O_e2e)
    return dict(Q=Q, S=S, K=K, tDSA=tDSA, tO=tO, tprep=tprep, tscore=tscore, tsel=tfull - tscore, rO=rO)


if __name__ == "__main__":
    sp, dn = build()
    tags = sys.argv[1:] or ["256k", "512k", "768k", "1m"]
    rows = []
    for tag in tags:
        r = run(tag, sp, dn); rows.append((tag, r))
        print(f"[{tag}] Q={r['Q']} S={r['S']} DSA={r['tDSA']:.3f}ms LiteTopK={r['tO']:.3f}ms "
              f"speedup={r['tDSA']/r['tO']:.2f}x recall={r['rO']:.3f}%", flush=True)
    out = {tag: r for tag, r in rows}
    jpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_b200_results.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=========== DSA E2E (B200 SM100, real GLM-5 caches, K=%d) ===========" % rows[0][1]["K"])
    print(f"{'seq':>5} {'Q':>4} {'DSA(ms)':>9} {'LiteTopK(ms)':>12} {'speedup':>8} {'recall':>8}")
    for tag, r in rows:
        print(f"{tag:>5} {r['Q']:>4} {r['tDSA']:>9.3f} {r['tO']:>12.3f} {r['tDSA']/r['tO']:>7.2f}x {r['rO']:>7.3f}%")
    print("saved:", jpath)

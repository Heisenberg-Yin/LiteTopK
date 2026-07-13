#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B200 DSA speedup sweep for the V2 kernel (DeepGEMM-2.5-based scoring loop),
IDENTICAL harness to sweep_b200_real_chunk.py: real chunked GLM-5 caches
(Q == chunk), baseline "DSA" = our KV-split dense fp8 kernel + torch.topk,
ours "LiteTopK" = sample seed + fused sparse scan + thr radix select — only the
sparse extension is built from dsa_marsco_v2.cu.

Run inside the `simtopk` container:
  PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
  DSA_CACHE_DIR=/data/dsa_caches \
  /usr/bin/python3.12 sweep_b200_v2.py
"""
import os, sys, json, torch
from torch.utils.cpp_extension import load

import b200_config as cfg
cfg.prepare_env()
from safetensors import safe_open

CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "1024 2048 4096 8192").split()]
TAGS = os.environ.get("TAGS", "256k 512k 768k 1m").split()
K = int(os.environ.get("K", "2048")); NB = int(os.environ.get("NB", "256"))
REFRESH = int(os.environ.get("REFRESH", "64")); SAMPLE = int(os.environ.get("SAMPLE", "65536"))
WARMUP = int(os.environ.get("WARMUP", "5")); ITERS = int(os.environ.get("ITERS", "20"))

DG25 = os.environ.get("DSA_DG25_ROOT", "/opt/glm5_prefill_test/DeepGEMM")


def build():
    os.makedirs("/tmp/build_v2_sparse_b200", exist_ok=True); os.makedirs("/tmp/build_dense_b200", exist_ok=True)
    sp = load(name="github_dsa_b200_v2_sparse", sources=[os.path.join(cfg.DSA_DIR, "dsa_marsco_v2.cu")],
              extra_include_paths=[cfg.DSA_DIR,
                                   os.path.join(DG25, "deep_gemm/include"),
                                   os.path.join(DG25, "third-party/cutlass/include")],
              extra_cuda_cflags=cfg.FLAGS,
              build_directory="/tmp/build_v2_sparse_b200", extra_ldflags=["-lcuda"], verbose=False)
    dn = load(name="github_dsa_b200_dense", sources=[cfg.SRC], extra_include_paths=cfg.INC, extra_cuda_cflags=cfg.FLAGS + ["-DDENSE_WRITE"],
              build_directory="/tmp/build_dense_b200", extra_ldflags=["-lcuda"], verbose=False)
    return sp, dn


def evt(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS


def real_chunk_path(tag, chunk):
    d = os.environ.get("DSA_CACHE_DIR", "/data/dsa_caches")
    return os.path.join(d, f"glm5_{tag}_realtext_chunk{chunk}.safetensors")


def load_cache(tag, chunk):
    T = {}
    with safe_open(real_chunk_path(tag, chunk), "pt", device="cuda") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
    q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
    kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
    gw = T["gate_w"]; Q, H, D = q.shape; S = kf.shape[0]
    w = (gw * qs * (D ** -0.5)).contiguous().float()
    return q, kf, ksc, w, Q, H, D, S


def dense_scores(dn, q, kf, ksc, w, S):
    Q = q.shape[0]
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke = torch.full((Q,), kf.shape[0], dtype=torch.int32, device="cuda")
    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda"); th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    cv, _, _ = dn.mqa_logits_dsa_marsco(q, kf, ksc, w, ks0, ke, o0, i1, th0, sv0, si0, NB, S, K, 0, -1)
    return cv


def run_cell(sp, dn, q, kf, ksc, w, S):
    Q = q.shape[0]; Smax = kf.shape[0]
    ke_full = torch.full((Q,), Smax, dtype=torch.int32, device="cuda")
    cvref = dense_scores(dn, q, kf, ksc, w, Smax)
    ref = cvref.topk(K, dim=-1).indices.long(); refs, _ = ref.sort(dim=-1); del cvref
    def recall(idx):
        p = torch.searchsorted(refs, idx).clamp(max=K - 1)
        return 100.0 * (torch.gather(refs, 1, p) == idx).sum().item() / (Q * K)

    def DSA_e2e():
        cv = dense_scores(dn, q, kf, ksc, w, Smax); cv.topk(K, dim=-1); del cv
    tDSA = evt(DSA_e2e)

    head = min(SAMPLE, Smax); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous()
    ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    col_full = pos.to(torch.int32).view(1, -1).expand(Q, -1).contiguous()
    cnt_full = torch.full((Q,), head, dtype=torch.int32, device="cuda")

    def O_prep():
        sl = dense_scores(dn, q, ksamp, kss, w, head)
        xs = -sl; o2 = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
        inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
        sv, si = sp.compact_topk_min_idx_marsco(xs, col_full, cnt_full, K)
        sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(dim=1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        return o2, inv2, th2, sv, si

    def O_e2e():
        o2, inv2, th2, sv, si = O_prep()
        cv2, ci2, cc2 = sp.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th2, sv, si, NB, 1, K, REFRESH, -1)
        return sp.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K)

    ov, oi = O_e2e(); torch.cuda.synchronize(); rO = recall(oi.long())
    tO = evt(O_e2e)
    return tDSA, tO, rO


def main():
    sp, dn = build()
    results = {}
    for tag in TAGS:
        results[tag] = {}
        for chunk in CHUNKS:
            p = real_chunk_path(tag, chunk)
            if not os.path.exists(p):
                print(f"[{tag} chunk={chunk}] MISSING {p}", flush=True); continue
            try:
                q, kf, ksc, w, Q, H, D, S = load_cache(tag, chunk)
                tDSA, tO, rO = run_cell(sp, dn, q, kf, ksc, w, S)
                results[tag][str(chunk)] = dict(DSA=tDSA, LiteTopK=tO, speedup=tDSA / tO, recall=rO, Q=Q, S=S)
                print(f"[{tag} chunk={chunk:>4}] Q={Q} S={S} DSA={tDSA:.3f}ms LiteTopK_v2={tO:.3f}ms "
                      f"speedup={tDSA/tO:.2f}x recall={rO:.3f}%", flush=True)
                del q, kf, ksc, w
            except Exception as ex:
                results[tag][str(chunk)] = dict(error=str(ex)[:140])
                print(f"[{tag} chunk={chunk}] ERR {str(ex)[:140]}", flush=True)
            torch.cuda.empty_cache()

    jpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure_b200_v2.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2)
    print("\n===== DSA E2E V2 (B200 SM100, REAL chunked GLM-5, Q=chunk, K=%d) =====" % K)
    print(f"{'seq':>5} {'chunk':>6} {'Q':>6} {'DSA(ms)':>9} {'LiteTopK_v2(ms)':>14} {'speedup':>8} {'recall':>8}")
    for tag in TAGS:
        for chunk in CHUNKS:
            c = results[tag].get(str(chunk), {})
            if "DSA" in c:
                print(f"{tag:>5} {chunk:>6} {c['Q']:>6} {c['DSA']:>9.3f} {c['LiteTopK']:>14.3f} {c['speedup']:>7.2f}x {c['recall']:>7.2f}%")
    print("saved:", jpath)


if __name__ == "__main__":
    main()

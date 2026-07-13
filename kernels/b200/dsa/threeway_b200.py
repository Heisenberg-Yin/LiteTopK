#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-way DSA comparison on the REAL chunked GLM-5 caches (Q=chunk), B200 SM100:

  A  official_patched : deepseek DeepGEMM sm100_fp8_mqa_logits (official kernel,
                        one-line SGLang #19529 patch so num_heads=32 compiles)
                        producing the full [Q,S] logits, then torch.topk.
  B  dense_ours        : our own KV-split DENSE fp8 scoring kernel + torch.topk
                        (the "ideal dense" baseline; same kernel as LiteTopK with
                        -DDENSE_WRITE).
  O  litetopk_ours      : our fused sparse scan + radix select.

All three return the same top-k INDICES; recall is set-overlap vs the exact
reference (torch relu-MQA + topk). All should be ~100%.

The PATCHED deep_gemm is vendored in-repo at `baselines/deepgemm_patched_official`
and is preferred automatically; the sglang site-packages is still needed for torch:
  PYTHONPATH=/opt/venvs/deepgemm/lib/python3.12/site-packages \
  DSA_CACHE_DIR=/data/dsa_caches \
  /usr/bin/python3.12 threeway_b200.py
"""
import os, sys, json, torch
sys.path.insert(0, "/opt/simtopk_src/b200/dsa")
# Prefer the in-repo patched deep_gemm (official baseline, SGLang #19529 patch).
_PATCHED_DG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "baselines", "deepgemm_patched_official")
if os.path.isdir(os.path.join(_PATCHED_DG, "deep_gemm")):
    sys.path.insert(0, _PATCHED_DG)
from torch.utils.cpp_extension import load
import b200_config as cfg
cfg.prepare_env()
import deep_gemm
from safetensors import safe_open

assert "deepgemm_patched" in deep_gemm.__file__, \
    f"expected patched deep_gemm, got {deep_gemm.__file__}"

CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "1024 2048 4096 8192").split()]
TAGS = os.environ.get("TAGS", "256k 512k 768k 1m").split()
K = int(os.environ.get("K", "2048")); NB = int(os.environ.get("NB", "256"))
REFRESH = int(os.environ.get("REFRESH", "64")); SAMPLE = int(os.environ.get("SAMPLE", "65536"))
WARMUP = int(os.environ.get("WARMUP", "5")); ITERS = int(os.environ.get("ITERS", "20"))


def build():
    os.makedirs("/tmp/build_sparse_b200", exist_ok=True); os.makedirs("/tmp/build_dense_b200", exist_ok=True)
    sp = load(name="github_dsa_b200_sparse", sources=[cfg.SRC], extra_include_paths=cfg.INC, extra_cuda_cflags=cfg.FLAGS,
              build_directory="/tmp/build_sparse_b200", extra_ldflags=["-lcuda"], verbose=False)
    dn = load(name="github_dsa_b200_dense", sources=[cfg.SRC], extra_include_paths=cfg.INC, extra_cuda_cflags=cfg.FLAGS + ["-DDENSE_WRITE"],
              build_directory="/tmp/build_dense_b200", extra_ldflags=["-lcuda"], verbose=False)
    return sp, dn


def evt(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS


def path(tag, chunk):
    d = os.environ.get("DSA_CACHE_DIR", "/data/dsa_caches")
    return os.path.join(d, f"glm5_{tag}_realtext_chunk{chunk}.safetensors")


def dense_scores(dn, q, kf, ksc, w, S):
    Q = q.shape[0]
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke = torch.full((Q,), kf.shape[0], dtype=torch.int32, device="cuda")
    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda"); th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    cv, _, _ = dn.mqa_logits_dsa_marsco(q, kf, ksc, w, ks0, ke, o0, i1, th0, sv0, si0, NB, S, K, 0, -1)
    return cv


def run_cell(sp, dn, q, kf, ksc, w, S):
    Q = q.shape[0]; Smax = kf.shape[0]
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke_full = torch.full((Q,), Smax, dtype=torch.int32, device="cuda")

    # exact reference (dense kernel + topk; official patched matches it bit-exactly)
    cvref = dense_scores(dn, q, kf, ksc, w, Smax); ref = cvref.topk(K, -1).indices.long(); refs, _ = ref.sort(-1); del cvref
    def recall(idx):
        p = torch.searchsorted(refs, idx).clamp(max=K - 1)
        return 100.0 * (torch.gather(refs, 1, p) == idx).sum().item() / (Q * K)

    # A: official patched deep_gemm full logits + topk
    def A():
        lg = deep_gemm.fp8_mqa_logits(q, (kf, ksc), w, ks0, ke_full, clean_logits=True, max_seqlen_k=0)
        return lg.topk(K, dim=-1)
    _, aidx = A(); rA = recall(aidx.long()); tA = evt(A)

    # B: our dense + topk
    def B():
        cv = dense_scores(dn, q, kf, ksc, w, Smax); cv.topk(K, -1); del cv
    tB = evt(B)

    # O: LiteTopK
    head = min(SAMPLE, Smax); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous()
    ks_scan = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    col = pos.to(torch.int32).view(1, -1).expand(Q, -1).contiguous(); cnt = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    def O():
        sl = dense_scores(dn, q, ksamp, kss, w, head); xs = -sl; o2 = xs.min(1).values.contiguous()
        inv2 = ((NB - 1) / (xs.max(1).values - o2).clamp_min(1e-9)).contiguous()
        sv, si = sp.compact_topk_min_idx_marsco(xs, col, cnt, K); sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        cv2, ci2, cc2 = sp.mqa_logits_dsa_marsco(q, kf, ksc, w, ks_scan, ke_full, o2, inv2, th2, sv, si, NB, 1, K, REFRESH, -1)
        return sp.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K)
    _, oidx = O(); rO = recall(oidx.long()); tO = evt(O)
    return dict(tA=tA, rA=rA, tB=tB, tO=tO, rO=rO, Q=Q, S=S)


def main():
    sp, dn = build()
    results = {}
    for tag in TAGS:
        results[tag] = {}
        for chunk in CHUNKS:
            if not os.path.exists(path(tag, chunk)):
                print(f"[{tag} {chunk}] MISSING", flush=True); continue
            T = {}
            with safe_open(path(tag, chunk), "pt", device="cuda") as f:
                for k in f.keys(): T[k] = f.get_tensor(k)
            q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
            kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
            gw = T["gate_w"]; Q, H, D = q.shape; S = kf.shape[0]; w = (gw * qs * (D ** -0.5)).contiguous().float()
            try:
                r = run_cell(sp, dn, q, kf, ksc, w, S)
                results[tag][str(chunk)] = r
                print(f"[{tag} {chunk:>4}] official={r['tA']:.2f}ms(r={r['rA']:.1f}) dense={r['tB']:.2f}ms "
                      f"LiteTopK={r['tO']:.2f}ms(r={r['rO']:.1f}) | O/official={r['tA']/r['tO']:.2f}x O/dense={r['tB']/r['tO']:.2f}x", flush=True)
            except Exception as ex:
                print(f"[{tag} {chunk}] ERR {str(ex)[:120]}", flush=True)
            del T, q, kf, ksc, w; torch.cuda.empty_cache()
    jp = "/opt/simtopk_src/b200/dsa/threeway_b200.json"
    json.dump(results, open(jp, "w"), indent=2)
    print("\n===== 3-WAY (B200 SM100, real GLM-5 Q=chunk, K=%d) =====" % K)
    print(f"{'seq':>5} {'chunk':>6} {'official(ms)':>12} {'dense(ms)':>10} {'LiteTopK(ms)':>12} {'O/official':>11} {'O/dense':>8}")
    for tag in TAGS:
        for chunk in CHUNKS:
            c = results[tag].get(str(chunk))
            if c:
                print(f"{tag:>5} {chunk:>6} {c['tA']:>12.2f} {c['tB']:>10.2f} {c['tO']:>12.2f} "
                      f"{c['tA']/c['tO']:>10.2f}x {c['tB']/c['tO']:>7.2f}x")
    print("saved:", jp)


if __name__ == "__main__":
    main()

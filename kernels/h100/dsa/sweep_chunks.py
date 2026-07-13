#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep the 3-way DSA benchmark over (chunk_size x seq_len).

For each cache glm5_{size}_realtext_chunk{chunk}.safetensors, times:
  A stock deep_gemm+torch.topk, B ideal KV-split dense+torch.topk, O ours fused,
and reports O/A, O/B speedups plus our recall vs exact deep_gemm topk.
"""
import os, sys, torch
from torch.utils.cpp_extension import load

DG = "/workspace/project/build/src/hopper/dsa/DeepGEMM_reduced_topk"
sys.path.insert(0, DG)
import deep_gemm
from safetensors import safe_open

DSA_DIR = "/workspace/project/src/dsa"
SRC = DSA_DIR + "/dsa_marsco.cu"
INC = [DSA_DIR, DG + "/deep_gemm/include", DG + "/third-party/cutlass/include"]
FLAGS = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"]
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

def build():
    os.makedirs("/tmp/build_sparse", exist_ok=True); os.makedirs("/tmp/build_dense", exist_ok=True)
    sp = load(name="github_dsa_sparse", sources=[SRC], extra_include_paths=INC, extra_cuda_cflags=FLAGS,
              build_directory="/tmp/build_sparse", extra_ldflags=["-lcuda"], verbose=False)
    dn = load(name="github_dsa_dense", sources=[SRC], extra_include_paths=INC, extra_cuda_cflags=FLAGS + ["-DDENSE_WRITE"],
              build_directory="/tmp/build_dense", extra_ldflags=["-lcuda"], verbose=False)
    return sp, dn

WARMUP = int(os.environ.get("WARMUP", "5")); ITERS = int(os.environ.get("ITERS", "15"))
def evt(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(ITERS): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / ITERS

def run(tag, chunk, sp, dn):
    K = int(os.environ.get("K", "2048")); NB = int(os.environ.get("NB", "256"))
    refresh = int(os.environ.get("REFRESH", "64"))
    T = {}
    with safe_open(f"/workspace/project/dsa_caches/glm5_{tag}_realtext_chunk{chunk}.safetensors", "pt", device="cuda") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
    q = T["q_index"].contiguous(); qs = T["q_index_scale"].squeeze(-1)
    kf = T["idx_k_cache"][0].contiguous(); ksc = T["idx_k_scale"][0, :, 0].contiguous()
    gw = T["gate_w"]; Q, H, D = q.shape; S = kf.shape[0]
    w = (gw * qs * (D ** -0.5)).contiguous().float()
    # SAMPLE default scales with S (S/6..S/8 sweet spot); allow override
    Ssample = int(os.environ.get("SAMPLE", str(min(131072, S))))
    ks0 = torch.zeros(Q, dtype=torch.int32, device="cuda"); ke_full = torch.full((Q,), S, dtype=torch.int32, device="cuda")
    RBref = (lambda raw: min(Q, ((Q + ((Q + raw - 1)//raw) - 1)//((Q + raw - 1)//raw) + 3) & ~3))(max(256, (1 << 30) // (S * 4)))
    ref = torch.empty(Q, K, dtype=torch.long, device="cuda")
    for r0 in range(0, Q, RBref):
        r1 = min(r0 + RBref, Q)
        lg = deep_gemm.fp8_mqa_logits(q[r0:r1].contiguous(), (kf, ksc), w[r0:r1].contiguous(),
                                      ks0[r0:r1], ke_full[r0:r1], clean_logits=True, max_seqlen_k=0)
        ref[r0:r1] = lg.topk(K, dim=-1).indices.long(); del lg
    refs, _ = ref.sort(dim=-1)
    def recall(idx):
        h = 0
        for i in range(Q):
            p = torch.searchsorted(refs[i], idx[i]).clamp(max=K - 1); h += (refs[i][p] == idx[i]).sum().item()
        return 100 * h / (Q * K)

    def A_e2e():
        # Row-batched so the dense [Q,S] logits buffer stays bounded (baseline
        # would chunk queries the same way for a large prefill chunk).
        RB = (lambda raw: min(Q, ((Q + ((Q + raw - 1)//raw) - 1)//((Q + raw - 1)//raw) + 3) & ~3))(max(256, (1 << 30) // (S * 4)))  # ~1GB; multiple of BLOCK_Q=4
        for r0 in range(0, Q, RB):
            r1 = min(r0 + RB, Q)
            lg = deep_gemm.fp8_mqa_logits(q[r0:r1].contiguous(), (kf, ksc), w[r0:r1].contiguous(),
                                          ks0[r0:r1], ke_full[r0:r1], clean_logits=True, max_seqlen_k=0)
            lg.topk(K, dim=-1)
            del lg
    tA = evt(A_e2e)

    o0 = torch.zeros(Q, device="cuda"); i1 = torch.ones(Q, device="cuda")
    th0 = torch.zeros(Q, dtype=torch.int32, device="cuda")
    sv0 = torch.zeros(Q, 1, device="cuda"); si0 = torch.zeros(Q, 1, dtype=torch.int32, device="cuda")
    def B_score():
        RB = (lambda raw: min(Q, ((Q + ((Q + raw - 1)//raw) - 1)//((Q + raw - 1)//raw) + 3) & ~3))(max(256, (1 << 30) // (S * 4)))
        for r0 in range(0, Q, RB):
            r1 = min(r0 + RB, Q)
            cv, _, _ = dn.mqa_logits_dsa_marsco(q[r0:r1].contiguous(), kf, ksc, w[r0:r1].contiguous(),
                                                ks0[r0:r1], ke_full[r0:r1], o0[r0:r1], i1[r0:r1], th0[r0:r1],
                                                sv0[r0:r1], si0[r0:r1], NB, S, K, 0, -1)
            cv.topk(K, dim=-1); del cv
    tB = evt(B_score)

    head = min(Ssample, S); pos = torch.arange(0, head, device="cuda", dtype=torch.long)
    ksamp = kf[pos].contiguous(); kss = ksc[pos].contiguous(); Ss = ksamp.shape[0]
    ks_scan_full = torch.full((Q,), head, dtype=torch.int32, device="cuda")
    RBo = (lambda raw: min(Q, ((Q + ((Q + raw - 1)//raw) - 1)//((Q + raw - 1)//raw) + 3) & ~3))(max(256, (1 << 30) // (S * 4)))
    # Sample-stage top-k-min via the radix-select CUDA kernel (replaces torch.topk).
    col_full = pos.to(torch.int32).view(1, -1).expand(RBo, -1).contiguous()
    cnt_full = torch.full((RBo,), head, dtype=torch.int32, device="cuda")

    def O_prep_block(r0, r1):
        # E2E sample stage: score the head sample, build seed candidates and threshold.
        ke_s = torch.full((r1 - r0,), Ss, dtype=torch.int32, device="cuda")
        sl = deep_gemm.fp8_mqa_logits(q[r0:r1].contiguous(), (ksamp, kss), w[r0:r1].contiguous(),
                                      ks0[r0:r1], ke_s, clean_logits=True, max_seqlen_k=0).contiguous()
        xs = -sl; o2 = xs.min(dim=1).values.contiguous(); hi = xs.max(dim=1).values
        inv2 = ((NB - 1) / (hi - o2).clamp_min(1e-20)).contiguous()
        sv, si = sp.compact_topk_min_idx_marsco(xs, col_full[:r1 - r0], cnt_full[:r1 - r0], K)
        sv = sv.contiguous(); si = si.contiguous()
        th2 = ((sv.max(dim=1).values - o2) * inv2).floor().clamp(0, NB - 1).to(torch.int32).contiguous()
        return o2, inv2, th2, sv, si

    def O_prep_all():
        return {r0: O_prep_block(r0, min(r0 + RBo, Q)) for r0 in range(0, Q, RBo)}

    # Static prep is used only for breakdown timing. The E2E O timing below
    # recomputes sample/prep every iteration.
    prep_static = O_prep_all()

    def O_score_from_prep(prep):
        for r0 in range(0, Q, RBo):
            r1 = min(r0 + RBo, Q); o2, inv2, th2, sv, si = prep[r0]
            sp.mqa_logits_dsa_marsco(q[r0:r1].contiguous(), kf, ksc, w[r0:r1].contiguous(),
                                     ks_scan_full[r0:r1], ke_full[r0:r1], o2, inv2, th2, sv, si, NB, 1, K, refresh, -1)

    def O_full_from_prep(prep):
        oi_all = torch.empty(Q, K, dtype=torch.int32, device="cuda")
        for r0 in range(0, Q, RBo):
            r1 = min(r0 + RBo, Q); o2, inv2, th2, sv, si = prep[r0]
            cv2, ci2, cc2 = sp.mqa_logits_dsa_marsco(q[r0:r1].contiguous(), kf, ksc, w[r0:r1].contiguous(),
                                                     ks_scan_full[r0:r1], ke_full[r0:r1], o2, inv2, th2, sv, si, NB, 1, K, refresh, -1)
            # th2 is tightened in-place by the scan; reuse it for boundary-bucket select.
            _, oi = sp.compact_topk_min_thr_marsco(cv2, ci2, cc2, o2, inv2, th2, NB, K); oi_all[r0:r1] = oi; del cv2, ci2, cc2
        return oi_all

    def O_e2e():
        prep = O_prep_all()
        return O_full_from_prep(prep)

    oi = O_e2e(); torch.cuda.synchronize(); rO = recall(oi.long())
    tOprep = evt(O_prep_all)
    tOs = evt(lambda: O_score_from_prep(prep_static))
    tOmain_sel = evt(lambda: O_full_from_prep(prep_static))
    tOsel = tOmain_sel - tOs
    tO = evt(O_e2e)
    del T, q, kf, ksc, ref, refs; torch.cuda.empty_cache()
    return dict(Q=Q, S=S, tA=tA, tB=tB, tO=tO, tOprep=tOprep, tOs=tOs, tOsel=tOsel, rO=rO)

if __name__ == "__main__":
    sp, dn = build()
    chunks = [int(x) for x in os.environ.get("CHUNKS", "1024 2048 4096 8192").split()]
    tags = sys.argv[1:] or ["256k", "512k", "768k", "1m"]
    rows = []
    for chunk in chunks:
        for tag in tags:
            try:
                r = run(tag, chunk, sp, dn); rows.append((chunk, tag, r))
                print(f"[chunk{chunk} {tag}] O/A={r['tA']/r['tO']:.2f}x O/B={r['tB']/r['tO']:.2f}x recall={r['rO']:.2f}%", flush=True)
            except Exception as ex:
                print(f"[chunk{chunk} {tag}] ERR {ex}", flush=True)
    print("\n================== DSA FUSED E2E SPEEDUP SWEEP (H100, K=2048) ==================")
    print(f"{'chunk':>6} {'seq':>5} {'Q':>6} {'A_stock':>9} {'B_e2e':>9} {'O_e2e':>9} {'O/A':>6} {'O/B':>6} {'recall':>7}")
    for chunk, tag, r in rows:
        print(f"{chunk:>6} {tag:>5} {r['Q']:>6} {r['tA']:>9.3f} {r['tB']:>9.3f} {r['tO']:>9.3f} "
              f"{r['tA']/r['tO']:>5.2f}x {r['tB']/r['tO']:>5.2f}x {r['rO']:>6.2f}%")
    print("\nBreakdown (ms):  O_sample=sample logits/topk/threshold, O_score=main fused scan, O_select=compact radix-select")
    print(f"{'chunk':>6} {'seq':>5} {'O_sample':>9} {'O_score':>8} {'O_select':>9}")
    for chunk, tag, r in rows:
        print(f"{chunk:>6} {tag:>5} {r['tOprep']:>9.3f} {r['tOs']:>8.3f} {r['tOsel']:>9.3f}")

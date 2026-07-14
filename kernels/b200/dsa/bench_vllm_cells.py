"""vLLM production indexer kernel (deep_gemm fp8 logits + top_k_per_row) per
(seq_len, chunk) cell, on the same real chunked caches as the dsa.pdf figure."""
import torch
from safetensors import safe_open
import deep_gemm
from vllm import _custom_ops as ops

K = 2048
for tag in ("256k", "512k", "768k", "1m"):
    for Q in (1024, 2048, 4096, 8192):
        p = (f"/data/dsa_caches/"
             f"glm5_{tag}_realtext_chunk{Q}.safetensors")
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
        ks = torch.zeros(Q, dtype=torch.int32, device="cuda")
        ke = (S - Q + 1 + torch.arange(Q, device="cuda", dtype=torch.int32)).contiguous()
        out = torch.full((Q, K), -1, dtype=torch.int32, device="cuda")

        def fn():
            lg = deep_gemm.fp8_fp4_mqa_logits((q, None), (kf, ksc), w, ks, ke,
                                              clean_logits=False)
            ops.top_k_per_row_prefill(lg, ks, ke, out, Q,
                                      lg.stride(0), lg.stride(1), K)

        n = 12 if S <= 524288 else 8
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        print(f"VLLM {tag} chunk{Q}: {s.elapsed_time(e)/n:.3f} ms", flush=True)
        del T, q, kf, ksc, w
        torch.cuda.empty_cache()
print("VLLMCELLS-DONE", flush=True)

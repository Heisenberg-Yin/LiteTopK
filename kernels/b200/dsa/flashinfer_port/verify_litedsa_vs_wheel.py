"""Correctness check: litetopk_repro's standalone litedsa.so (tvm_ffi port) vs
the wheel's torch.ops._C.litedsa_union_qm/litedsa_masked_mla_fp8 (known-good,
already end-to-end verified against the stock reference earlier today). Same
kernel source, different binding layer -- outputs should match near-exactly."""
import os, json
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")
import torch, tvm_ffi
from safetensors import safe_open

DEV = "cuda"
K, G, BLOCK_SIZE = 2048, 16, 64
S = 1048576
Q = G

import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
m = tvm_ffi.load_module(_os.environ.get("LITEDSA_SO", _os.path.join(_here, "litedsa.so")))

T = {}
with safe_open(f"/data/dsa_caches/glm5_1m_realtext_chunk8192.safetensors", "pt", device="cuda") as f:
    for k in f.keys():
        T[k] = f.get_tensor(k)
meta = json.loads(dict(safe_open(f"/data/dsa_caches/glm5_1m_realtext_chunk8192.safetensors", "pt", device="cuda").metadata())["mla"])

mla_kv_full = T["mla_kv"].contiguous()
mla_q_full = T["mla_q"][:Q].contiguous()
topk_full = T["topk_idx"][:Q].contiguous().int()
n_rank_heads = meta["n_rank_heads"]
bmm1, bmm2 = meta["bmm1_scale"], meta["bmm2_scale"]

nblk = -(-S // BLOCK_SIZE)
kv_paged = torch.zeros(nblk * BLOCK_SIZE, mla_kv_full.shape[-1], dtype=mla_kv_full.dtype, device=DEV)
kv_paged[:S] = mla_kv_full
kv_paged = kv_paged.view(nblk, BLOCK_SIZE, -1)
blk_tbl = torch.arange(nblk, device=DEV, dtype=torch.int32)[None, :]
req_id = torch.zeros(Q, dtype=torch.int32, device=DEV)

# --- wheel (known-good) ---
from vllm.model_executor.layers.litedsa import litedsa_masked_mqa
wheel_out = litedsa_masked_mqa(mla_q_full, kv_paged, topk_full, req_id, blk_tbl, BLOCK_SIZE,
                               n_rank_heads, bmm1, bmm2, G, version=101)

# --- standalone repro .so, calling the two primitives directly ---
cap = G * K
u_phys = torch.empty(1, cap, dtype=torch.int32, device=DEV)
counts = torch.empty(1, dtype=torch.int32, device=DEV)
memb_qm = torch.empty(1, G, cap // 32, dtype=torch.int32, device=DEV)
req_pg = req_id[::G].contiguous()  # [ng]=1
m.union_qm(topk_full.contiguous(), u_phys, counts, memb_qm, req_pg, blk_tbl, BLOCK_SIZE, 1 << 20)

qp = mla_q_full.view(1, 128, mla_q_full.shape[2]).contiguous()
out = torch.empty(1, 128, 512, dtype=torch.bfloat16, device=DEV)
max_logits = torch.empty(1, 128, dtype=torch.float32, device=DEV)
lse = torch.empty(1, 128, dtype=torch.float32, device=DEV)
m.masked_mla_fp8(qp, kv_paged.view(-1, 1, kv_paged.shape[-1]), u_phys.view(1, 1, -1),
                 bmm1, bmm2, counts, memb_qm, n_rank_heads, out, max_logits, lse)
repro_out = out.view(Q, n_rank_heads, 512)

diff = (wheel_out.float() - repro_out.float()).abs()
print(f"max abs diff: {diff.max().item():.6f}   mean abs diff: {diff.mean().item():.6f}")
ok = diff.max().item() < 0.01
print("PASS: repro .so matches wheel exactly" if ok else "FAIL: mismatch")

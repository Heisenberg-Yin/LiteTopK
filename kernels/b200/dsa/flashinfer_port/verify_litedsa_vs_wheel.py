#!/usr/bin/env python3
"""Compare the standalone LiteDSA TVM-FFI module with an external reference.

The reference is ``vllm.model_executor.layers.litedsa.litedsa_masked_mqa``.
That module is not part of stock vLLM 0.23. Run this script in an environment
that provides the compatible reference wheel or checkout described in
``README.md``.
"""

import json
import os

import torch
import tvm_ffi
from safetensors import safe_open

DEVICE = "cuda"
TOPK = 2048
GROUP_SIZE = 16
BLOCK_SIZE = 64
SEQ_SPACE = 1 << 20

HERE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.environ.get("LITEDSA_SO", os.path.join(HERE, "litedsa.so"))
DATA_PATH = os.environ.get(
    "LITETOPK_DSA_CACHE",
    "/data/dsa_caches/glm5_1m_realtext_chunk8192.safetensors",
)
ATOL = float(os.environ.get("LITEDSA_ATOL", "0.01"))


def load_inputs():
    tensors = {}
    with safe_open(DATA_PATH, framework="pt", device=DEVICE) as data:
        metadata = dict(data.metadata() or {})
        for name in data.keys():
            tensors[name] = data.get_tensor(name)
    if "mla" not in metadata:
        raise RuntimeError(f"{DATA_PATH} does not contain the required 'mla' metadata")
    return tensors, json.loads(metadata["mla"])


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("the bundled LiteDSA module requires an SM100 GPU")
    if not os.path.isfile(SO_PATH):
        raise FileNotFoundError(f"LiteDSA module not found: {SO_PATH}")
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"verification cache not found: {DATA_PATH}")

    try:
        from vllm.model_executor.layers.litedsa import litedsa_masked_mqa
    except ImportError as exc:
        raise RuntimeError(
            "the external LiteDSA reference module is unavailable; use the "
            "compatible reference vLLM wheel or checkout documented in README.md"
        ) from exc

    module = tvm_ffi.load_module(SO_PATH)
    tensors, mla = load_inputs()

    mla_kv = tensors["mla_kv"].contiguous()
    mla_q = tensors["mla_q"][:GROUP_SIZE].contiguous()
    topk_indices = tensors["topk_idx"][:GROUP_SIZE].contiguous().int()
    seq_len = mla_kv.shape[0]
    num_heads = mla["n_rank_heads"]
    bmm1_scale = mla["bmm1_scale"]
    bmm2_scale = mla["bmm2_scale"]

    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_paged = torch.zeros(
        num_blocks * BLOCK_SIZE,
        mla_kv.shape[-1],
        dtype=mla_kv.dtype,
        device=DEVICE,
    )
    kv_paged[:seq_len] = mla_kv
    kv_paged = kv_paged.view(num_blocks, BLOCK_SIZE, -1)
    block_table = torch.arange(
        num_blocks, device=DEVICE, dtype=torch.int32
    ).unsqueeze(0)
    request_ids = torch.zeros(GROUP_SIZE, dtype=torch.int32, device=DEVICE)

    reference = litedsa_masked_mqa(
        mla_q,
        kv_paged,
        topk_indices,
        request_ids,
        block_table,
        BLOCK_SIZE,
        num_heads,
        bmm1_scale,
        bmm2_scale,
        GROUP_SIZE,
        version=1,
    )

    capacity = GROUP_SIZE * TOPK
    union_indices = torch.empty(1, capacity, dtype=torch.int32, device=DEVICE)
    counts = torch.empty(1, dtype=torch.int32, device=DEVICE)
    membership = torch.empty(
        1,
        GROUP_SIZE,
        capacity // 32,
        dtype=torch.int32,
        device=DEVICE,
    )
    module.union_qm(
        topk_indices,
        union_indices,
        counts,
        membership,
        request_ids[::GROUP_SIZE].contiguous(),
        block_table,
        BLOCK_SIZE,
        SEQ_SPACE,
    )

    packed_q = mla_q.view(1, 128, mla_q.shape[2]).contiguous()
    output = torch.empty(1, 128, 512, dtype=torch.bfloat16, device=DEVICE)
    max_logits = torch.empty(1, 128, dtype=torch.float32, device=DEVICE)
    lse = torch.empty(1, 128, dtype=torch.float32, device=DEVICE)
    module.masked_mla_fp8(
        packed_q,
        kv_paged.view(-1, 1, kv_paged.shape[-1]),
        union_indices.view(1, 1, capacity),
        bmm1_scale,
        bmm2_scale,
        counts,
        membership,
        num_heads,
        output,
        max_logits,
        lse,
    )

    standalone = output.view(GROUP_SIZE, num_heads, 512)
    diff = (reference.float() - standalone.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max abs diff: {max_diff:.6f}")
    print(f"mean abs diff: {mean_diff:.6f}")
    if max_diff > ATOL:
        print(f"FAIL: max abs diff exceeds atol={ATOL}")
        raise SystemExit(1)
    print(f"PASS: outputs agree within atol={ATOL}")


if __name__ == "__main__":
    main()

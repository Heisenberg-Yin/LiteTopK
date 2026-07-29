#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM adapter for the B200 LiteDSA grouped sparse-attention kernel.

Adjacent query tokens are packed into a 128-row attention call over the union
of their top-k lists. A per-query membership bitmap preserves each token's
original sparse attend set. The module loads ``litedsa.so`` through
``tvm_ffi`` and reuses union/membership buffers while the sparse-index version
is unchanged.

``VLLM_LITETOPK_LITEDSA=1`` enables the patched vLLM hook. ``LITEDSA_SO`` may
override the repository-relative shared-library path.
"""
import functools
import os

import torch

_SEQ_SPACE = 1 << 20  # position-bitmap span (max 1M logical positions)

_here = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SO = os.path.join(
    _here, "..", "..", "kernels", "b200", "dsa", "flashinfer_port", "litedsa.so"
)

_MOD = None
_LOAD_FAILED = False


def _mod():
    global _MOD, _LOAD_FAILED
    if _MOD is None and not _LOAD_FAILED:
        try:
            import tvm_ffi

            so_path = os.environ.get("LITEDSA_SO", _DEFAULT_SO)
            _MOD = tvm_ffi.load_module(so_path)
        except Exception:  # noqa: BLE001
            _LOAD_FAILED = True
            raise
    return _MOD


@functools.cache
def litedsa_available() -> bool:
    """True iff the SM100 grouped-attention kernel is loadable."""
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        return _mod() is not None
    except Exception:  # noqa: BLE001
        return False


# single-slot version cache: [key, payload, hits, misses]
_CACHE: list = [None, None, 0, 0]
# persistent overwrite-write buffers per (ng, group_size, cap): (u_phys, counts, memb_qm)
_BUFS: dict = {}


def _build_or_reuse(
    topk_indices: torch.Tensor,
    num_tokens: int,
    group_size: int,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    version: int,
):
    ng = num_tokens // group_size
    cap = group_size * topk_indices.shape[1]
    key = (version, num_tokens, group_size, cap)
    if _CACHE[0] == key:
        _CACHE[2] += 1
        return _CACHE[1]
    buf_key = (ng, group_size, cap)
    bufs = _BUFS.get(buf_key)
    if bufs is None:
        dev = topk_indices.device
        bufs = _BUFS[buf_key] = (
            torch.empty(ng, cap, dtype=torch.int32, device=dev),
            torch.empty(ng, dtype=torch.int32, device=dev),
            torch.empty(ng, group_size, cap // 32, dtype=torch.int32, device=dev),
        )
    u_phys, counts, memb_qm = bufs
    _mod().union_qm(
        topk_indices.contiguous(),
        u_phys,
        counts,
        memb_qm,
        req_id_per_token[:num_tokens:group_size].contiguous(),
        block_table.contiguous(),
        block_size,
        _SEQ_SPACE,
    )
    _CACHE[0], _CACHE[1] = key, bufs
    _CACHE[3] += 1
    return bufs


def litedsa_masked_mqa(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    req_id_per_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_heads: int,
    bmm1_scale: float,
    bmm2_scale: float,
    group_size: int,
    version: int,
) -> torch.Tensor:
    """Grouped masked sparse attention over the group union.

    ``q``: [T, num_heads, 576] fp8_e4m3 (T % group_size == 0,
    num_heads * group_size == 128). Returns bf16 [T, num_heads, 512].
    """
    num_tokens = q.shape[0]
    ng = num_tokens // group_size
    u_phys, counts, memb_qm = _build_or_reuse(
        topk_indices,
        num_tokens,
        group_size,
        req_id_per_token,
        block_table,
        block_size,
        version,
    )
    qp = q.view(ng, 128, q.shape[2]).contiguous()
    out = torch.empty(ng, 128, 512, dtype=torch.bfloat16, device=q.device)
    max_logits = torch.empty(ng, 128, dtype=torch.float32, device=q.device)
    lse = torch.empty(ng, 128, dtype=torch.float32, device=q.device)
    _mod().masked_mla_fp8(
        qp,
        kv_cache.view(-1, 1, kv_cache.shape[-1]),
        u_phys.view(ng, 1, -1),
        bmm1_scale,
        bmm2_scale,
        counts,
        memb_qm,
        num_heads,
        out,
        max_logits,
        lse,
    )
    return out.view(num_tokens, num_heads, 512)

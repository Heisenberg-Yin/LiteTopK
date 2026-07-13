# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# ---- LiteTopK fused sparse indexer hook (VLLM_LITETOPK=1) ---------------------
_LITETOPK_MOD = None
_LITETOPK_OFF = os.environ.get("VLLM_LITETOPK", "0") != "1"


def _litetopk_try(q, k, k_scale, w, ks, ke, out_idx, topk,
                 num_reqs=None, ke_min_hint=None) -> bool:
    global _LITETOPK_MOD, _LITETOPK_OFF
    if _LITETOPK_OFF:
        return False
    if _LITETOPK_MOD is None:
        try:
            import sys as _sys
            _p = os.environ.get("LITETOPK_MODULE_DIR", "/opt/simtopk_repro/glm5_prefill/litetopk_vllm")
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
            import litetopk_indexer as _LITETOPK_MOD_  # noqa: N813
            _LITETOPK_MOD = _LITETOPK_MOD_
            logger.info("LiteTopK indexer hook enabled (min_s=%d, sample=%d)",
                        _LITETOPK_MOD.MIN_S, _LITETOPK_MOD.SAMPLE)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiteTopK hook import failed, disabled: %s", e)
            _LITETOPK_OFF = True
            return False
    return _LITETOPK_MOD.try_chunk(q, k, k_scale, w, ks, ke, out_idx, topk,
                                  num_reqs=num_reqs, ke_min_hint=ke_min_hint)


def _litetopk_merge_wanted(total_seq_lens: int) -> bool:
    """True when the module is loaded, merge mode is on and S >= MIN_S."""
    global _LITETOPK_MOD, _LITETOPK_OFF
    if _LITETOPK_OFF:
        return False
    if _LITETOPK_MOD is None:
        try:
            import sys as _sys
            _p = os.environ.get("LITETOPK_MODULE_DIR", "/opt/simtopk_repro/glm5_prefill/litetopk_vllm")
            if _p not in _sys.path:
                _sys.path.insert(0, _p)
            import litetopk_indexer as _LITETOPK_MOD_  # noqa: N813
            _LITETOPK_MOD = _LITETOPK_MOD_
        except Exception as e:  # noqa: BLE001
            logger.warning("LiteTopK hook import failed, disabled: %s", e)
            _LITETOPK_OFF = True
            return False
    return bool(getattr(_LITETOPK_MOD, "MERGE", False)
                and total_seq_lens >= _LITETOPK_MOD.MIN_S)


_LITETOPK_SIDE = None


def _litetopk_try_merged(q, k, k_scale, w, out_idx, topk, S, Qtot,
                        probe_k=None, probe_scale=None, gather_event=None,
                        strided_plan=None, pre_compacted=False,
                        hot_key=None) -> bool:
    if _LITETOPK_OFF or _LITETOPK_MOD is None:
        return False
    try:
        return _LITETOPK_MOD.try_merged(q, k, k_scale, w, out_idx, topk, S, Qtot,
                                       probe_k=probe_k, probe_scale=probe_scale,
                                       gather_event=gather_event,
                                       strided_plan=strided_plan,
                                       pre_compacted=pre_compacted,
                                       hot_key=hot_key)
    except Exception as e:  # noqa: BLE001
        print(f"[litetopk] merged fallback due to: {e}", flush=True)
        return False


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        merged_done = False
        _chunks = prefill_metadata.chunks
        if (
            not use_fp4_cache
            and q_scale is None
            and len(_chunks) > 1
            and all(c.num_reqs == 1 for c in _chunks)
            and _litetopk_merge_wanted(_chunks[-1].total_seq_lens)
        ):
            # LiteTopK merged mode with gather/sample overlap: probe pages are
            # gathered straight from the PAGED cache on the main stream (the
            # threshold probe only needs 256 spread pages), the full workspace
            # gather runs CONCURRENTLY on a side stream, and the scan waits on
            # its event inside the module. Official's monolithic dense GEMM
            # cannot start before the full gather; our sample can.
            global _LITETOPK_SIDE
            _S = _chunks[-1].total_seq_lens
            _t0, _t1 = _chunks[0].token_start, _chunks[-1].token_end
            _Qtot = _t1 - _t0
            _pk = _psc = _gev = None
            _dev = q_quant.device
            _ke_min = _S - _Qtot + 1
            # mode + probe size from the module policy (certificate-strided
            # for S>=400K with a 64K probe; prefix below), decided BEFORE
            # gathering: strided -> gather the workspace COMPACTED (probe
            # pages skipped at gather time = zero re-copy)
            try:
                _strided, _ps = _LITETOPK_MOD.plan_sampling(_Qtot, _ke_min)
            except Exception:  # noqa: BLE001
                _strided, _ps = False, int(getattr(_LITETOPK_MOD, "SSAMPLE", 16384))
            _npage = _ps // 64
            _span_pg = (_ke_min - topk_tokens) // 64
            _pstp = max(_span_pg // _npage, 1) if _span_pg > 2 * _npage else 0
            _strided = _strided and _pstp >= 2
            # HOTONLY: hot indices are original coordinates -> workspace must
            # be gathered FULL (no probe-page compaction) and no external
            # probe is needed (module-side PAGE_PROBE covers cold start).
            _hot = (bool(getattr(_LITETOPK_MOD, "HOTONLY", False))
                    and int(getattr(_LITETOPK_MOD, "HOTSAMPLE", 0)) > 0)
            _compacted = False
            if _LITETOPK_SIDE is None:
                _LITETOPK_SIDE = torch.cuda.Stream()
            # big gather kicked FIRST (nothing consumes it until the scan)
            _LITETOPK_SIDE.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_LITETOPK_SIDE):
                if _strided and not _hot:
                    try:
                        _nblk = (_S + 63) // 64
                        _probe_pg = torch.arange(_npage, device=_dev,
                                                 dtype=torch.long) * _pstp
                        _keep = torch.ones(_nblk, dtype=torch.bool, device=_dev)
                        _keep[_probe_pg] = False
                        _kbt = _chunks[-1].block_table[0, :_nblk][_keep].view(1, -1)
                        _Sk0 = _S - _npage * 64
                        _cu = torch.tensor([0, _Sk0], dtype=torch.int32,
                                           device=_dev)
                        ops.cp_gather_indexer_k_quant_cache(
                            kv_cache, k_quant_full[:_Sk0], k_scale_full[:_Sk0],
                            _kbt, _cu)
                        _compacted = True
                    except Exception:  # noqa: BLE001
                        _compacted = False
                if not _compacted:
                    for chunk in _chunks:
                        if not chunk.skip_kv_gather:
                            ops.cp_gather_indexer_k_quant_cache(
                                kv_cache,
                                k_quant_full[: chunk.total_seq_lens],
                                k_scale_full[: chunk.total_seq_lens],
                                chunk.block_table,
                                chunk.cu_seq_lens,
                            )
                _gev = torch.cuda.Event()
                _gev.record()
            # probe pages on the MAIN stream, concurrent with the big gather
            if _strided and not _hot:
                try:
                    _pids = (torch.arange(_npage, device=_dev, dtype=torch.long)
                             * _pstp).view(1, -1)
                    _pbt = _chunks[-1].block_table[:1].long().gather(1, _pids) \
                        .to(_chunks[-1].block_table.dtype)
                    _pk = torch.empty(_ps, k_quant_full.shape[-1],
                                      dtype=k_quant_full.dtype, device=_dev)
                    _psc_raw = torch.empty(_ps, k_scale_full.shape[-1],
                                           dtype=k_scale_full.dtype, device=_dev)
                    _pcu = torch.tensor([0, _ps], dtype=torch.int32, device=_dev)
                    ops.cp_gather_indexer_k_quant_cache(
                        kv_cache, _pk, _psc_raw, _pbt, _pcu)
                    _psc = _psc_raw.view(torch.float32).squeeze(-1)
                except Exception:  # noqa: BLE001
                    _pk = _psc = None
            if _compacted and _pk is None:
                # compacted workspace requires the external probe (its pages
                # are absent from the workspace); without it, regather full
                _LITETOPK_SIDE.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(_LITETOPK_SIDE):
                    for chunk in _chunks:
                        if not chunk.skip_kv_gather:
                            ops.cp_gather_indexer_k_quant_cache(
                                kv_cache,
                                k_quant_full[: chunk.total_seq_lens],
                                k_scale_full[: chunk.total_seq_lens],
                                chunk.block_table,
                                chunk.cu_seq_lens,
                            )
                    _gev = torch.cuda.Event()
                    _gev.record()
                _compacted = False
            _Sk = _S - _npage * 64 if _compacted else _S
            merged_done = _litetopk_try_merged(
                q_quant[_t0:_t1],
                k_quant_full[:_Sk],
                k_scale_full[:_Sk].view(torch.float32).squeeze(-1),
                weights[_t0:_t1],
                topk_indices_buffer[_t0:_t1, :topk_tokens],
                topk_tokens,
                _S,
                _Qtot,
                probe_k=_pk,
                probe_scale=_psc,
                gather_event=_gev,
                strided_plan=_strided,
                pre_compacted=_compacted,
                hot_key=str(k_cache_prefix),
            )
            if not merged_done and _gev is not None:
                # fallback path re-gathers on the main stream; order it after
                # the side-stream writes to the same workspace
                torch.cuda.current_stream().wait_event(_gev)
        for chunk in ([] if merged_done else prefill_metadata.chunks):
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            # ---- LiteTopK fused sparse top-k (VLLM_LITETOPK=1) --------------
            # Skips materializing the full [Q, S] logits for long contexts;
            # falls through to the official path on any unmet assumption.
            if (
                not use_fp4_cache
                and q_scale_slice is None
                and _litetopk_try(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices_buffer[
                        chunk.token_start : chunk.token_end, :topk_tokens
                    ],
                    topk_tokens,
                    # CPU-side hints (avoid .item() device syncs in the hook):
                    # single-request causal chunk => ks all 0 and
                    # ke_min = total_seq_lens - chunk_query_len + 1.
                    num_reqs=chunk.num_reqs,
                    ke_min_hint=chunk.total_seq_lens
                    - (chunk.token_end - chunk.token_start)
                    + 1,
                )
            ):
                continue
            if current_platform.is_xpu():
                if q_scale_slice is not None:
                    raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            # LiteTopK: on the LAST official chunk before the merged band,
            # seed the layer's hot carry from this topk (the boundary
            # merged call then runs hot -- no cold start).
            if (_LITETOPK_MOD is not None and not _LITETOPK_OFF
                    and getattr(_LITETOPK_MOD, "HOTONLY", False)):
                _S_here = chunk.total_seq_lens
                if _LITETOPK_MOD.MIN_S - 32768 <= _S_here < _LITETOPK_MOD.MIN_S:
                    try:
                        _LITETOPK_MOD.stash_carry(
                            str(k_cache_prefix), topk_indices, _S_here)
                    except Exception:  # noqa: BLE001
                        pass

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )

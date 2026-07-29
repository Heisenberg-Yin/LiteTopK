// LiteTopK DSA scoring kernel V3 = "hybrid":
//   * scoring loop of DeepGEMM 2.5 (commit 891d57b, the vLLM-pinned version):
//     per-q-block weights held in REGISTERS, per-row 32-element TMEM loads
//     with early UMMA release, tight tcgen05 fencing -- the generation that
//     makes the official kernel fast at large Q;
//   * scheduling of our V1: NON-persistent KV-split. blockIdx.x = q-block,
//     blockIdx.y = KV split window. This is what keeps all 148 SMs busy on the
//     tiny-Q chunks vLLM actually produces at long context (its 512MB logits
//     budget shrinks Q to ~128 at S=1M, where a persistent grid would idle
//     116/148 SMs);
//   * LiteTopK sparse epilogue (batched-vote emit, strided gate reload,
//     warp-local candidate queues) and the spare-warp threshold-refresh daemon
//     (V1 semantics: one q-block per CTA, fixed rows).
//
// Ragged Q (vLLM chunks) handled by forcing an empty KV range on padded rows.

#pragma once

#include "dsa_litetopk_config.cuh"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace dsa_litetopk {

using namespace deep_gemm;

#ifndef DSA_WARP_QUEUE_CAP
#define DSA_WARP_QUEUE_CAP 64
#endif

#ifndef DSA_REFRESH_STRIDE
// KV blocks between progress signals in the legacy refresh path.
#define DSA_REFRESH_STRIDE 2
#endif

#ifndef DSA_GATE_STRIDE
#ifdef DSA_SPARSE_REFRESH
// The shared Gate4 mailbox is cheap to consume, but checking it more often
// does not make the daemon produce thresholds sooner. Keep the math path to
// one LDS.128 every 64 KV blocks.
#define DSA_GATE_STRIDE 64
#else
#define DSA_GATE_STRIDE 16
#endif
#endif
#ifndef DSA_MATH_REGS
#define DSA_MATH_REGS 240
#endif
#ifndef DSA_SPEC_REGS
#define DSA_SPEC_REGS 24
#endif
#ifndef DSA_TMEM_ROWS
// Rows retired per tcgen05.ld. Four rows pull the whole BLOCK_Q accumulator
// in one load. The 240/24 math/specialized register split keeps the larger
// accumulator resident without spilling.
#define DSA_TMEM_ROWS 4
#endif
#ifndef DSA_UMMA_STAGES
// Accumulator buffers per math warp group.
#define DSA_UMMA_STAGES 2
#endif

#ifdef DSA_CANDIDATE_U16
// Six-byte global candidate record for the current <=1M-token contract.
//
// The exact layouts keep the low 20 bits of cand_idx as the KV index. The
// remaining 12 index bits and the 16 cand_val bits carry a score encoding.
//
// The default is a 28-bit monotonic code. Values below bucket 1 collapse
// because they bypass score selection when the certified threshold is >=1.
// DSA_CANDIDATE_U16_BUCKET_RESIDUAL instead stores:
//   cand_val[15:0] = FP32 bits[15:0]
//   cand_idx[27:20] = exact integer bucket
//   cand_idx[31:28] = FP32 bits[19:16]
// For a certified threshold >=8, the boundary bucket's exponent and upper
// mantissa bits are fixed by that bucket, so its remaining 20 bits preserve
// the exact FP32 order. Both layouts halve the global value slab without
// quantizing the live boundary bucket.
//
// DSA_CANDIDATE_U16_BUCKET8_FRAC8 is an explicitly approximate alternative:
// cand_val stores {bucket:8, floor(fraction*256):8}, while cand_idx remains a
// full 32-bit KV index. It preserves threshold bucket classification exactly
// but can create ties within the boundary bucket.
//
// DSA_CANDIDATE_FP16_NUMERIC stores the IEEE FP16 score bits in cand_val and
// the exact source bucket in cand_idx[27:20]. Keeping the bucket separately
// prevents FP16 rounding at an integer edge from invalidating threshold
// metadata; ranking inside that bucket is intentionally FP16-approximate.
using CandidateValue = uint16_t;
constexpr uint32_t kCandidateIndexBits = 20;
constexpr uint32_t kCandidateIndexMask =
    (1u << kCandidateIndexBits) - 1u;

CUTLASS_DEVICE uint32_t candidate_pack_index(
        const uint32_t payload, const uint32_t kv_index) {
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
    (void)payload;
    return kv_index;
#else
    return (kv_index & kCandidateIndexMask) |
           ((payload >> 16) << kCandidateIndexBits);
#endif
}

CUTLASS_DEVICE uint32_t candidate_score_code(const float bq) {
    return bq >= 1.0f
               ? (__float_as_uint(bq) - 0x3f800000u + 2u)
               : 1u;
}

CUTLASS_DEVICE uint32_t candidate_bucket8_frac8_code(const float bq) {
    // For finite Gate4 hits, bq < 256. Incrementing the FP32 exponent by
    // eight is exactly bq*256, and the u16 conversion truncates positive
    // values while clamping negative bucket-0 tails to zero.
    const uint32_t scaled_bits =
        __float_as_uint(bq) + 0x04000000u;
    const float scaled = __uint_as_float(scaled_bits);
    uint16_t code;
    asm volatile(
        "cvt.rzi.u16.f32 %0, %1;"
        : "=h"(code)
        : "f"(scaled));
    return static_cast<uint32_t>(code);
}

CUTLASS_DEVICE uint16_t candidate_fp16_bits(const float value) {
    uint16_t bits;
    asm volatile(
        "cvt.rn.f16.f32 %0, %1;"
        : "=h"(bits)
        : "f"(value));
    return bits;
}

CUTLASS_DEVICE float candidate_fp16_decode(const uint16_t bits) {
    float value;
    asm volatile(
        "cvt.f32.f16 %0, %1;"
        : "=f"(value)
        : "h"(bits));
    return value;
}

CUTLASS_DEVICE uint32_t candidate_load_score_code(
        const CandidateValue value, const int32_t packed_idx) {
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
    (void)packed_idx;
    return static_cast<uint32_t>(value);
#else
    return (static_cast<uint32_t>(packed_idx) >>
            kCandidateIndexBits) << 16 |
           static_cast<uint32_t>(value);
#endif
}

#if defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) || \
    defined(DSA_CANDIDATE_FP16_NUMERIC)
CUTLASS_DEVICE uint32_t candidate_load_bucket(
        const int32_t packed_idx) {
    return (static_cast<uint32_t>(packed_idx) >> 20) &
           0xffu;
}

CUTLASS_DEVICE uint32_t candidate_load_residual(
        const CandidateValue value, const int32_t packed_idx) {
    return (static_cast<uint32_t>(packed_idx) >> 28) << 16 |
           static_cast<uint32_t>(value);
}
#endif

CUTLASS_DEVICE float candidate_decode_score(
        const CandidateValue value, const int32_t packed_idx) {
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
    (void)packed_idx;
    const uint32_t code = static_cast<uint32_t>(value);
    return static_cast<float>(code >> 8) +
           static_cast<float>(code & 0xffu) * (1.0f / 256.0f);
#elif defined(DSA_CANDIDATE_FP16_NUMERIC)
    (void)packed_idx;
    return candidate_fp16_decode(value);
#elif defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL)
    const uint32_t bucket =
        candidate_load_bucket(packed_idx);
    const uint32_t residual =
        candidate_load_residual(value, packed_idx);
    return __uint_as_float(
        __float_as_uint(static_cast<float>(bucket)) |
        residual);
#else
    const uint32_t code =
        candidate_load_score_code(value, packed_idx);
    return code > 1u
               ? __uint_as_float(
                     code - 2u + 0x3f800000u)
               : 0.0f;
#endif
}

CUTLASS_DEVICE int32_t candidate_decode_index(
        const int32_t packed_idx) {
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
    return packed_idx;
#else
    return static_cast<int32_t>(
        static_cast<uint32_t>(packed_idx) &
        kCandidateIndexMask);
#endif
}

CUTLASS_DEVICE void store_candidate(
        CandidateValue* value_dst, int32_t* index_dst,
        const float bq, const uint32_t kv_index) {
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
    const uint32_t payload =
        candidate_bucket8_frac8_code(bq);
#elif defined(DSA_CANDIDATE_FP16_NUMERIC)
    const int braw = static_cast<int>(bq);
    const uint32_t bucket =
        static_cast<uint32_t>(max(braw, 0));
    const uint32_t payload =
        static_cast<uint32_t>(candidate_fp16_bits(bq)) |
        (bucket << 16);
#elif defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL)
    const int braw = static_cast<int>(bq);
    const uint32_t bucket =
        static_cast<uint32_t>(max(braw, 0));
    const uint32_t bits = __float_as_uint(bq);
    const uint32_t payload =
        (bits & 0xffffu) |
        (bucket << 16) |
        ((bits & 0xf0000u) << 8);
#else
    const uint32_t code = candidate_score_code(bq);
    const uint32_t payload = code;
#endif
    __stcs(value_dst, static_cast<CandidateValue>(payload));
    __stcs(
        index_dst,
        static_cast<int32_t>(
            candidate_pack_index(payload, kv_index)));
}

CUTLASS_DEVICE void store_candidate_payload(
        CandidateValue* value_dst, int32_t* index_dst,
        const uint32_t payload, const uint32_t kv_index) {
    __stcs(value_dst, static_cast<CandidateValue>(payload));
    __stcs(
        index_dst,
        static_cast<int32_t>(
            candidate_pack_index(payload, kv_index)));
}
#else
using CandidateValue = float;

CUTLASS_DEVICE float candidate_decode_score(
        const CandidateValue value, const int32_t) {
    return value;
}

CUTLASS_DEVICE int32_t candidate_decode_index(
        const int32_t index) {
    return index;
}

CUTLASS_DEVICE void store_candidate(
        CandidateValue* value_dst, int32_t* index_dst,
        const float value, const uint32_t kv_index) {
    __stcs(value_dst, value);
    __stcs(index_dst, static_cast<int32_t>(kv_index));
}
#endif

CUTLASS_DEVICE uint32_t candidate_record_payload(
        const float value) {
#if defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8)
    return candidate_bucket8_frac8_code(value);
#elif defined(DSA_CANDIDATE_FP16_NUMERIC)
    const int braw = static_cast<int>(value);
    const uint32_t bucket =
        static_cast<uint32_t>(max(braw, 0));
    return static_cast<uint32_t>(candidate_fp16_bits(value)) |
           (bucket << 16);
#elif defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL)
    const int braw = static_cast<int>(value);
    const uint32_t bucket =
        static_cast<uint32_t>(max(braw, 0));
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0xffffu) |
           (bucket << 16) |
           ((bits & 0xf0000u) << 8);
#elif defined(DSA_CANDIDATE_U16) && \
    defined(DSA_CANDIDATE_U16_LOCAL_CODE)
    return candidate_score_code(value);
#else
    return __float_as_uint(value);
#endif
}

CUTLASS_DEVICE uint32_t candidate_record_payload(
        const float value, const uint32_t bucket) {
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
    (void)bucket;
    return candidate_bucket8_frac8_code(value);
#elif defined(DSA_CANDIDATE_FP16_NUMERIC)
    return static_cast<uint32_t>(candidate_fp16_bits(value)) |
           (bucket << 16);
#elif defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL)
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0xffffu) |
           (bucket << 16) |
           ((bits & 0xf0000u) << 8);
#else
    (void)bucket;
    return candidate_record_payload(value);
#endif
}

CUTLASS_DEVICE void store_candidate_record(
        CandidateValue* value_dst, int32_t* index_dst,
        const uint32_t payload, const uint32_t kv_index) {
#if defined(DSA_CANDIDATE_U16_LOCAL_CODE) || \
    defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) || \
    defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) || \
    defined(DSA_CANDIDATE_FP16_NUMERIC)
    store_candidate_payload(
        value_dst, index_dst, payload, kv_index);
#else
    store_candidate(
        value_dst, index_dst,
        __uint_as_float(payload), kv_index);
#endif
}

#define DSA_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define DSA_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))
#define DSA_ST_CANDIDATE_PAIR(vdst, idst, v, i) \
    dsa_litetopk::store_candidate(&(vdst), &(idst), (v), (i))
#define DSA_ST_CANDIDATE_RECORD(vdst, idst, v, i) \
    dsa_litetopk::store_candidate_record(&(vdst), &(idst), (v), (i))

#ifdef DSA_CHUNKED_EMIT
#if !defined(DSA_EMIT_CHUNK_BLOCKS) || !defined(DSA_EMIT_LANE_SLOTS)
#error "fixed LiteTopK emit geometry is missing"
#endif
#if DSA_EMIT_CHUNK_BLOCKS < 1 || DSA_EMIT_LANE_SLOTS < 1
#error "invalid DSA_CHUNKED_EMIT geometry"
#endif
#ifdef DSA_CANDIDATE_FP16_LOCAL32
#if DSA_EMIT_LANE_SLOTS > 18 || DSA_EMIT_CHUNK_BLOCKS > 256
#error "FP16 local32 supports at most 18 slots and 256 KV blocks"
#endif
#else
#if DSA_EMIT_LANE_SLOTS > 9
#error "64-bit local records support at most 9 slots"
#endif
#endif
#if defined(DSA_HIST_DEFER) || defined(DSA_PERSIST) || defined(DSA_EMIT_NULL)
#error "DSA_CHUNKED_EMIT requires the non-persistent producer-fed histogram path"
#endif
#endif
#if defined(DSA_CHUNKED_FLUSH_DIRECT) && !defined(DSA_CHUNKED_EMIT)
#error "DSA_CHUNKED_FLUSH_DIRECT requires DSA_CHUNKED_EMIT"
#endif
#if defined(DSA_CANDIDATE_U16) && \
    (!defined(DSA_BUCKET_GATE4) || \
     !defined(DSA_CHUNKED_FLUSH_DIRECT) || \
     !defined(DSA_INPLACE_BOUNDARY_SELECT))
#error "DSA_CANDIDATE_U16 requires Gate4, flush-direct, and the in-place selector"
#endif
#if defined(DSA_CANDIDATE_U16_LOCAL_CODE) && \
    !defined(DSA_CANDIDATE_U16)
#error "DSA_CANDIDATE_U16_LOCAL_CODE requires DSA_CANDIDATE_U16"
#endif
#if defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) && \
    !defined(DSA_CANDIDATE_U16)
#error "DSA_CANDIDATE_U16_BUCKET_RESIDUAL requires DSA_CANDIDATE_U16"
#endif
#if defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) && \
    !defined(DSA_CANDIDATE_U16)
#error "DSA_CANDIDATE_U16_BUCKET8_FRAC8 requires DSA_CANDIDATE_U16"
#endif
#if defined(DSA_CANDIDATE_FP16_NUMERIC) && \
    !defined(DSA_CANDIDATE_U16)
#error "DSA_CANDIDATE_FP16_NUMERIC requires DSA_CANDIDATE_U16"
#endif
#if defined(DSA_CANDIDATE_FP16_LOCAL32) && \
    !defined(DSA_CANDIDATE_FP16_NUMERIC)
#error "DSA_CANDIDATE_FP16_LOCAL32 requires DSA_CANDIDATE_FP16_NUMERIC"
#endif
#if defined(DSA_CANDIDATE_FP16_LOCAL32) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX)
#error "DSA_CANDIDATE_FP16_LOCAL32 requires packed-prefix direct flush"
#endif
#if defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) && \
    defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL)
#error "choose only one DSA_CANDIDATE_U16 bucket encoding"
#endif
#if defined(DSA_CANDIDATE_FP16_NUMERIC) && \
    (defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) || \
     defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL))
#error "DSA_CANDIDATE_FP16_NUMERIC is a separate candidate encoding"
#endif
#if defined(DSA_CHUNKED_FLUSH_DIRECT_CTA) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
#error "DSA_CHUNKED_FLUSH_DIRECT_CTA requires DSA_CHUNKED_FLUSH_DIRECT"
#endif
#if defined(DSA_CHUNKED_FLUSH_DIRECT_PARROWS) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
#error "DSA_CHUNKED_FLUSH_DIRECT_PARROWS requires DSA_CHUNKED_FLUSH_DIRECT"
#endif
#if defined(DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
#error "DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX requires DSA_CHUNKED_FLUSH_DIRECT"
#endif
#if (defined(DSA_CHUNKED_FLUSH_DIRECT_PARROWS) + \
     defined(DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX) + \
     defined(DSA_CHUNKED_FLUSH_DIRECT_CTA)) > 1
#error "choose one flush-direct aggregation mode"
#endif
#if defined(DSA_PACKEDPREFIX_BYTEFAST) || \
    defined(DSA_CHUNKED_FLUSH_DIRECT_WG) || \
    defined(DSA_CHUNKED_FLUSH_DIRECT_ROWGROUPS) || \
    defined(DSA_CHUNKED_EMIT_INLINE)
#error "unsupported emit specialization"
#endif
#if defined(DSA_STATIC_GATE) && \
    (!defined(DSA_CHUNKED_EMIT) || !defined(DSA_BUCKET_GATE4))
#error "DSA_STATIC_GATE requires DSA_CHUNKED_EMIT and DSA_BUCKET_GATE4"
#endif
#if defined(DSA_SPARSE_REFRESH) && \
    (!defined(DSA_CHUNKED_EMIT) || !defined(DSA_BUCKET_GATE4))
#error "DSA_SPARSE_REFRESH requires DSA_CHUNKED_EMIT and DSA_BUCKET_GATE4"
#endif
#if defined(DSA_STATIC_GATE) && defined(DSA_SPARSE_REFRESH)
#error "DSA_STATIC_GATE and DSA_SPARSE_REFRESH are mutually exclusive"
#endif
#if defined(DSA_INPLACE_BOUNDARY_SELECT) && \
    !defined(DSA_SPARSE_REFRESH)
#error "DSA_INPLACE_BOUNDARY_SELECT requires DSA_SPARSE_REFRESH"
#endif
#if defined(DSA_BOUNDARY_FLOAT_CLASSIFY) && \
    !defined(DSA_BUCKET_GATE4)
#error "DSA_BOUNDARY_FLOAT_CLASSIFY requires DSA_BUCKET_GATE4"
#endif
#if defined(DSA_SPARSE_REFRESH_REDUX_SKIP) && \
    !defined(DSA_SPARSE_REFRESH)
#error "DSA_SPARSE_REFRESH_REDUX_SKIP requires DSA_SPARSE_REFRESH"
#endif
#if defined(DSA_SPARSE_REFRESH_ADAPTIVE_SLEEP) && \
    !defined(DSA_SPARSE_REFRESH)
#error "DSA_SPARSE_REFRESH_ADAPTIVE_SLEEP requires DSA_SPARSE_REFRESH"
#endif
#if defined(DSA_SPARSE_REFRESH_ZERO_BASE) && \
    (!defined(DSA_SPARSE_REFRESH) || \
     !defined(DSA_CANDIDATE_U16))
#error "DSA_SPARSE_REFRESH_ZERO_BASE requires sparse refresh and the hot-only U16 contract"
#endif
#if defined(DSA_SPARSE_REFRESH_DEFERRED_FEED) && \
    (!defined(DSA_SPARSE_REFRESH) || \
     !defined(DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX))
#error "DSA_SPARSE_REFRESH_DEFERRED_FEED requires sparse refresh and packed-prefix flush-direct"
#endif
#if defined(DSA_SPARSE_REFRESH_DEFERRED_FEED) && \
    defined(DSA_CANDIDATE_U16)
#error "DSA_SPARSE_REFRESH_DEFERRED_FEED does not support packed U16 candidates"
#endif
#if defined(DSA_SPARSE_REFRESH_DEFERRED_FEED) && defined(DSA_HIST_DEFER)
#error "choose only one deferred histogram implementation"
#endif
#ifdef DSA_SPARSE_REFRESH
#ifndef DSA_SPARSE_REFRESH_NS
#define DSA_SPARSE_REFRESH_NS 512
#endif
#if DSA_SPARSE_REFRESH_NS < 1
#error "DSA_SPARSE_REFRESH_NS must be positive"
#endif
#ifndef DSA_SPARSE_REFRESH_IDLE_NS
#define DSA_SPARSE_REFRESH_IDLE_NS 2048
#endif
#if DSA_SPARSE_REFRESH_IDLE_NS < DSA_SPARSE_REFRESH_NS
#error "DSA_SPARSE_REFRESH_IDLE_NS must be >= DSA_SPARSE_REFRESH_NS"
#endif
#endif
template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          bool kSingleKVSplit = false>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_dsa_litetopk(const uint32_t seq_len, const uint32_t seq_len_kv,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         const float* __restrict__ origin,     // [seq_len]
                         const float* __restrict__ inv_delta,  // [seq_len]
                         int32_t* __restrict__ th_bucket,      // [seq_len]
                         int32_t* __restrict__ bcount,         // [seq_len, num_buckets]
                         const uint32_t num_buckets,
                         const uint32_t topk,
                         const uint32_t refresh_every,
                         const uint32_t num_kv_splits,
                         const uint32_t probe_group,    // compacted-space group size
                                                        // (pstp-1)*64; 0 = no probe
                                                        // compaction (identity map)
                         const uint64_t probe_magic,    // ceil(2^42/probe_group):
                                                        // exact div via mul-shift
                         const uint32_t probe_add_max,  // npage*64 cap for the map
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
                         uint64_t* __restrict__ emit_workspace,
#endif
                         CandidateValue* __restrict__ cand_val, // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_idx,       // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_cnt,       // [seq_len]
                         const uint32_t cand_cap,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    constexpr uint32_t kNumUmmaStages = DSA_UMMA_STAGES;
    constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;
#ifdef DSA_CHUNKED_EMIT
    DG_STATIC_ASSERT(BLOCK_Q == 4 && kNumMathWarps == 8,
                     "DSA_CHUNKED_EMIT requires BLOCK_Q=4 and 8 math warps");
#ifdef DSA_CANDIDATE_FP16_LOCAL32
    DG_STATIC_ASSERT(
        BLOCK_KV == 256,
        "FP16 local32 infers the KV tile coordinate from its owner lane");
#endif
#endif
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
    const uint32_t emit_total_blocks =
        math::ceil_div(seq_len_kv, BLOCK_KV);
    const uint32_t emit_blocks_per_split =
        math::ceil_div(emit_total_blocks, num_kv_splits);
    const uint32_t emit_chunks_per_split =
        math::ceil_div(emit_blocks_per_split,
                       static_cast<uint32_t>(DSA_EMIT_CHUNK_BLOCKS));
    const uint64_t emit_num_chunks =
        static_cast<uint64_t>(num_q_blocks) * num_kv_splits *
        emit_chunks_per_split * kNumMathWarps;
    const uint64_t emit_record_count =
        emit_num_chunks * BLOCK_Q * DSA_EMIT_LANE_SLOTS * 32u;
    auto emit_counts =
        reinterpret_cast<uint32_t*>(emit_workspace + emit_record_count);
#endif

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    extern __shared__ __align__(512) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumUmmaBuffers;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumUmmaBuffers + i); });

    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumUmmaBuffers * 2);
    auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
    auto kv_progress_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 2);
#ifdef DSA_PERSIST
    auto done_qb_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 3);
    auto ack_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 4);  // [2]
    auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 8);
#else
    auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
#endif
#ifdef DSA_SPARSE_REFRESH
    // Chunked emit does not use the legacy warp-queue count words. Reuse four
    // of them as a CTA-local Gate4 mailbox. Values are positive float edge
    // bit-patterns, so unsigned atomicMin is exactly a monotonic gate tighten.
    auto sparse_gate_bits = reinterpret_cast<uint32_t*>(warpq_count);
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
    // The next four legacy count words publish the contiguous candidate
    // prefix whose payload stores are complete. The daemon advances its own
    // last-consumed cursor and may catch up across multiple flush windows.
    auto sparse_safe_end =
        reinterpret_cast<int32_t*>(warpq_count + BLOCK_Q);
    // Daemon-owned consumption cursor. It boots from the seed frontier and
    // only the owning spare warp advances it, so a late daemon start cannot
    // skip a math flush that already moved sparse_safe_end.
    auto sparse_last_safe =
        reinterpret_cast<int32_t*>(warpq_count + 2 * BLOCK_Q);
#endif
#endif
#ifdef DSA_CHUNKED_FLUSH_DIRECT_CTA
    // The first four legacy count words stay reserved for the sparse-refresh
    // mailbox. The remaining 112 bytes hold all 8x4 warp prefixes plus four
    // row bases, so CTA aggregation adds no dynamic shared-memory allocation.
    auto flush_warp_prefix =
        reinterpret_cast<uint16_t*>(warpq_count + BLOCK_Q);
    auto flush_row_base = reinterpret_cast<int32_t*>(
        flush_warp_prefix + kNumMathWarps * BLOCK_Q);
#endif
#ifdef DSA_CHUNKED_EMIT
#ifdef DSA_CANDIDATE_FP16_LOCAL32
    auto emit_smem_records = reinterpret_cast<uint32_t*>(
        warpq_count + kNumMathWarps * BLOCK_Q);
    auto smem_hist = reinterpret_cast<int32_t*>(
        emit_smem_records +
        kNumMathWarps * BLOCK_Q * DSA_EMIT_LANE_SLOTS * 32u);
#else
    auto emit_smem_records = reinterpret_cast<uint64_t*>(
        warpq_count + kNumMathWarps * BLOCK_Q);
    auto smem_hist = reinterpret_cast<int32_t*>(
        emit_smem_records +
        kNumMathWarps * BLOCK_Q * DSA_EMIT_LANE_SLOTS * 32u);
#endif
#else
    auto warpq_val = reinterpret_cast<float*>(warpq_count + kNumMathWarps * BLOCK_Q);
    auto warpq_idx = reinterpret_cast<int32_t*>(warpq_val + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);
    // Per-CTA refresh histogram (BLOCK_Q x num_buckets). When this CTA is the
    // ONLY scanner of its rows (num_kv_splits == 1, i.e. all large-Q shapes),
    // the per-candidate histogram feed goes to smem instead of RED.GLOBAL:
    // cheaper atomic, no 64-bit address math, no L2 pressure. The daemon then
    // reads global bcount (seed counts) + this smem part. Counts and totals
    // are identical to the global path, so thresholds and recall are
    // unchanged; a racing read can only UNDERcount -> looser gate -> safe.
    auto smem_hist = reinterpret_cast<int32_t*>(warpq_idx + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);
#endif
#ifdef DSA_HIST_DEFER
#if defined(DSA_EMIT_NULL) || defined(DSA_PERSIST)
#error "DSA_HIST_DEFER: only the default queue/drain path is instrumented"
#endif
    // Deferred-feed watermark: per row-local slot, the count of candidates
    // whose PAYLOAD is fence-visible in the global cand buffers. Drains
    // publish (threadfence_block + smem atomic); the daemon batch-feeds the
    // histogram from cand_val[prev..safe). Reading less = undercount =
    // looser gate = recall-safe.
    auto smem_safe = reinterpret_cast<int32_t*>(smem_hist + BLOCK_Q * 256);
#endif

    DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        *scan_done_flag = 0;
        *kv_progress_ptr = 0;
#ifdef DSA_PERSIST
        *done_qb_ptr = 0;
        ack_ptr[0] = 0;
        ack_ptr[1] = 0;
#endif
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumUmmaBuffers; ++ i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
#ifdef DSA_SPARSE_REFRESH
    // Sparse refresh is a compile-time mode and its host contract guarantees
    // refresh_every > 0. A single split owns each row, so its complete
    // histogram can live in shared memory.
    constexpr bool hist_in_smem = kSingleKVSplit;
#else
    const bool hist_in_smem = (num_kv_splits == 1) &&
                              (refresh_every > 0) &&
                              (refresh_every != 0x7fffffff);
#endif
    if (hist_in_smem) {
        for (uint32_t idx = threadIdx.x;
             idx < BLOCK_Q * num_buckets;
             idx += blockDim.x) {
#ifdef DSA_SPARSE_REFRESH
            // Copy the exact-subset base once. The daemon can then refresh
            // entirely from shared memory instead of rereading global bcount
            // on every pass. Scan hits are added to this same histogram.
#ifdef DSA_SPARSE_REFRESH_ZERO_BASE
            // The hot-only FP16 contract emits no sample seeds. The scan
            // revisits the complete KV range, so seed_prep intentionally
            // supplies an all-zero base histogram. Initialize it locally and
            // avoid reading Q * NB zero words from L2 at CTA startup.
            smem_hist[idx] = 0;
#else
            const uint32_t local_row = idx / num_buckets;
            const uint32_t bucket = idx - local_row * num_buckets;
            const uint32_t row =
                static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + local_row;
            smem_hist[idx] =
                row < seq_len
                    ? __ldcg(
                          bcount +
                          static_cast<uint64_t>(row) * num_buckets +
                          bucket)
                    : 0;
#endif
#else
            smem_hist[idx] = 0;
#endif
        }
#ifdef DSA_HIST_DEFER
        if (threadIdx.x < BLOCK_Q) smem_safe[threadIdx.x] = 0;
#endif
    }
#ifdef DSA_SPARSE_REFRESH
    if (threadIdx.x < BLOCK_Q) {
        const uint32_t row =
            min(static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + threadIdx.x,
                seq_len - 1);
        const int gate = __ldcg(th_bucket + row);
        ptx::st_shared(
            sparse_gate_bits + threadIdx.x,
            __float_as_uint(static_cast<float>(gate + 1)));
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
        const int seed_end =
            min(max(__ldcg(cand_cnt + row), 0),
                static_cast<int>(cand_cap));
        sparse_last_safe[threadIdx.x] = seed_end;
        sparse_safe_end[threadIdx.x] = seed_end;
#endif
    }
#endif
    __syncthreads();

    constexpr uint32_t kNumSpecializedRegisters = DSA_SPEC_REGS;
    constexpr uint32_t kNumMathRegisters = DSA_MATH_REGS;

    // V1 KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
    // contiguous KV sub-window. Split boundaries are BLOCK_KV-aligned.
#ifndef DSA_PERSIST
    const uint32_t block_q_idx = blockIdx.x;
#else
    // DSA_PERSIST: each role loops q-blocks with a static stride, shadowing
    // block_q_idx so the body text is unchanged. One boot per CTA (TMEM,
    // barriers, pipelines); barrier phases run on GLOBAL counters.
    #define DSA_QB_LOOP \
        for (uint32_t block_q_idx = blockIdx.x, qb_it = 0; \
             block_q_idx < num_q_blocks; block_q_idx += gridDim.x, ++ qb_it)
#endif
    const uint32_t kv_split = blockIdx.y;
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t block_q_idx) -> cute::tuple<uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            if (block_q_idx * BLOCK_Q + i >= seq_len) {
                // Padded row of a ragged final q-block: empty, aggregation-neutral.
                seq_k_start[i] = seq_len_kv;
                seq_k_end[i] = 0;
            }
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        const uint32_t total_blocks = math::ceil_div(seq_len_kv, BLOCK_KV);
        const uint32_t blocks_per_split = math::ceil_div(total_blocks, num_kv_splits);
        const uint32_t split_lo = kv_split * blocks_per_split * BLOCK_KV;
        const uint32_t split_hi = min((kv_split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv);
        start = start / 4 * 4;  // TMA alignment for SF KV
        if (start < split_lo) start = split_lo;
        if (end > split_hi) end = split_hi;
        const uint32_t nkv = (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
        return {start, nkv};
    };

    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_block_idx % kNumKVStages, (kv_block_idx / kNumKVStages) & 1};
    };

    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        if (cute::elect_one_sync()) {
#ifdef DSA_PERSIST
            uint32_t kvt = 0;
            DSA_QB_LOOP {
            empty_q_barriers[0]->wait((qb_it & 1) ^ 1);
#else
            if (block_q_idx < num_q_blocks) {
#endif
            // Q + weights once for this q-block.
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[0], smem_q[0], 0, block_q_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[0], smem_weights[0], 0, block_q_idx * BLOCK_Q);
            full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);

            CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
#ifdef DSA_PERSIST
                CUTE_TIE_DECL(get_kv_pipeline(kvt ++), kv_stage_idx, kv_phase);
#else
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
#endif
                empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                        smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                          smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

#ifdef DSA_PERSIST
        uint32_t kvt_base = 0;
        DSA_QB_LOOP {
            CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
            full_q_barriers[0]->wait(qb_it & 1);

            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                const uint32_t kvg = kvt_base + kv_block_idx;
                CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);
#else
        if (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
            full_q_barriers[0]->wait(0);

            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                const uint32_t kvg = kv_block_idx;
                CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);
#endif

                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
                // Round-robin over kNumUmmaStages accumulators. A stage is
                // reused every kNumUmmaStages tiles, so its phase toggles at
                // that rate, not every tile.
                const uint32_t umma_stage = kvg % kNumUmmaStages;
                const uint32_t umma_phase = (kvg / kNumUmmaStages) & 1;
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    const uint32_t buf = i * kNumUmmaStages + umma_stage;
                    empty_umma_barriers[buf]->wait(umma_phase ^ 1);
                    ptx::tcgen05_after_thread_sync();
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[0], 0, k * UMMA_K);
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, buf * UMMA_N, k, runtime_instr_desc);
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[buf]));
                }
            }
            empty_q_barriers[0]->arrive();
#ifdef DSA_PERSIST
            kvt_base += num_kv_blocks;
#endif
        }
    } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
        // Spare-warp threshold-refresh daemon. Keeping refresh off the math
        // warps avoids placing histogram scan latency on their critical path.
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

#ifdef DSA_SPARSE_REFRESH
        constexpr bool in_scan_refresh = true;
#else
        const bool in_scan_refresh =
            refresh_every > 0 && refresh_every != 0x7fffffff;
#endif
#ifdef DSA_PERSIST
        if (in_scan_refresh) {
            const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
            DSA_QB_LOOP {
#else
        if (in_scan_refresh && block_q_idx < num_q_blocks) {
            const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
#endif
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
            const auto drain_sparse_hist = [&](
                    const uint32_t r) {
                if constexpr (!kSingleKVSplit) return;
                const uint32_t row =
                    block_q_idx * BLOCK_Q + r;
                if (row >= seq_len) return;
                int safe = lane_idx == 0
                    ? atomicAdd(sparse_safe_end + r, 0)
                    : 0;
                safe = __shfl_sync(0xffffffffu, safe, 0);
                int prev = lane_idx == 0
                    ? sparse_last_safe[r]
                    : 0;
                prev = __shfl_sync(0xffffffffu, prev, 0);
                if (safe <= prev) return;
                // Acquire the payload epoch published after the producer-side
                // block fences. This warp is not a participant in the math
                // barriers, so its own block-scope fence is required here.
                __threadfence_block();
                const float* cv =
                    cand_val +
                    static_cast<uint64_t>(row) * cand_cap;
                for (int j = prev + static_cast<int>(lane_idx);
                     j < safe; j += 32) {
                    const int braw =
                        static_cast<int>(__ldcg(cv + j));
                    const int b =
                        min(max(braw, 0),
                            static_cast<int>(num_buckets) - 1);
                    atomicAdd(
                        smem_hist + r * num_buckets + b,
                        1);
                }
                __syncwarp();
                if (lane_idx == 0)
                    sparse_last_safe[r] = safe;
            };
#endif
            const auto refresh_row = [&](
                    const uint32_t row,
                    const bool publish_boundary_counts) {
                if (row >= seq_len) return false;
#ifdef DSA_SPARSE_REFRESH
                const uint32_t local_row =
                    row - block_q_idx * BLOCK_Q;
#endif
                const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
                const int32_t* srow = smem_hist + (row - block_q_idx * BLOCK_Q) * num_buckets;
#ifdef DSA_SPARSE_REFRESH
                const int current_gate = min(
                    max(static_cast<int>(__uint_as_float(
                            ptx::ld_shared(
                                sparse_gate_bits + local_row))) -
                            1,
                        0),
                    static_cast<int>(num_buckets) - 1);
                // Only buckets strictly below the published gate can tighten
                // it. If they contain fewer than topk entries, keep the
                // current gate and avoid scanning the dead upper tail.
                const uint32_t search_buckets =
                    static_cast<uint32_t>(current_gate);
                int found = current_gate;
#else
                const uint32_t search_buckets = num_buckets;
                int found = static_cast<int>(num_buckets) - 1;
#endif
                int carry = 0;
                int found_lt = 0;
                int found_eq = 0;
                bool done = false;
                for (uint32_t base = 0;
                     base < search_buckets && !done;
                     base += 32) {
                    uint32_t b = base + lane_idx;
                    int v = 0;
                    if (b < search_buckets) {
#ifdef DSA_SPARSE_REFRESH
                        // In the single-split fast path smem already contains
                        // the initial global histogram plus every scan hit.
                        if constexpr (kSingleKVSplit)
                            v = srow[b];
                        else
                            v = brow[b];
#else
                        v = brow[b];
                        if (hist_in_smem) v += srow[b];
#endif
                    }
#ifdef DSA_SPARSE_REFRESH_REDUX_SKIP
                    // Histogram entries are nonnegative. Most 32-bucket
                    // groups remain strictly below K, so reject them with
                    // one REDUX and reserve the five dependent prefix
                    // shuffles for the single group that crosses K.
                    const int group_sum =
                        __reduce_add_sync(0xffffffffu, v);
                    if (carry + group_sum <
                        static_cast<int>(topk)) {
                        carry += group_sum;
                        continue;
                    }
#endif
                    int prefix = v;
                    #pragma unroll
                    for (int off = 1; off < 32; off <<= 1) {
                        int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
                        if (static_cast<int>(lane_idx) >= off) prefix += nsh;
                    }
                    int incl = carry + prefix;
                    bool hit = (b < search_buckets) &&
                               (incl >= static_cast<int>(topk)) &&
                               (incl - v < static_cast<int>(topk));
                    unsigned hm = __ballot_sync(0xffffffffu, hit);
                    if (hm) {
                        const int hit_lane = __ffs(hm) - 1;
                        found = static_cast<int>(base) + hit_lane;
                        found_lt = __shfl_sync(
                            0xffffffffu, incl - v, hit_lane);
                        found_eq = __shfl_sync(
                            0xffffffffu, v, hit_lane);
                        done = true;
                    } else {
                        carry +=
                            __shfl_sync(0xffffffffu, prefix, 31);
                    }
                }
#ifdef DSA_SPARSE_REFRESH
                if (!done) {
                    // No lower bucket reached K, so the current gate remains
                    // the boundary. `carry` is exactly count(bucket<gate).
                    found_lt = carry;
                    if constexpr (kSingleKVSplit)
                        found_eq = srow[found];
                    else
                        found_eq = brow[found];
                }
#endif
                if (lane_idx == 0) {
#ifdef DSA_SPARSE_REFRESH
                    if constexpr (kSingleKVSplit) {
                        // One CTA owns this row. Avoid an unchanged global
                        // atomic on every daemon pass; publish only a genuine
                        // tightening. The shared atomic makes the mailbox race
                        // well-defined, while a stale math-warp read remains
                        // conservatively loose.
                        const uint32_t edge = __float_as_uint(
                            static_cast<float>(found + 1));
#ifdef DSA_SPARSE_REFRESH_REDUX_SKIP
                        // We only search buckets below current_gate, so a
                        // crossing group proves a strict tightening. Each row
                        // has exactly one daemon-warp owner.
                        if (done) {
                            th_bucket[row] = found;
                            atomicMin(
                                sparse_gate_bits + local_row,
                                edge);
                        }
#else
                        const uint32_t old_edge = ptx::ld_shared(
                            sparse_gate_bits + local_row);
                        if (edge < old_edge) {
                            th_bucket[row] = found;
                            atomicMin(sparse_gate_bits + local_row, edge);
                        }
#endif
                        if (publish_boundary_counts &&
                            num_buckets >= 3) {
                            // The final daemon pass already has the exact
                            // histogram boundary. Reuse the now-dead bcount
                            // row as selector metadata and avoid another
                            // candidate scan. A negative complemented
                            // threshold is an unambiguous validity tag:
                            // ordinary histogram entries are nonnegative.
                            int32_t* meta =
                                bcount +
                                static_cast<uint64_t>(row) * num_buckets;
                            meta[0] = ~found;
                            meta[1] = found_lt;
                            meta[2] = found_eq;
                        }
                    } else {
                        // Multiple KV splits contribute to global bcount.
                        // atomicMin returns the cross-CTA threshold so this
                        // CTA's local mailbox never becomes tighter than it.
                        const int old_gate =
                            atomicMin(th_bucket + row, found);
                        const int tight_gate = min(old_gate, found);
                        atomicMin(
                            sparse_gate_bits + local_row,
                            __float_as_uint(
                                static_cast<float>(tight_gate + 1)));
                    }
#else
                    if (found < th_bucket[row]) th_bucket[row] = found;
#endif
                }
                return done;
            };
#ifdef DSA_PERSIST
            // per-qb: poll progress; on math's done signal, final refresh,
            // then ZERO this qb's hist rows and ack so math may emit the
            // next qb (daemon-side zeroing avoids a math<->daemon race).
            int last_prog = 0;
            while (true) {
                const int done = (*done_qb_ptr > static_cast<int>(qb_it));
                const int prog = *kv_progress_ptr;
                if (prog > last_prog) {
                    for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                        refresh_row(block_q_idx * BLOCK_Q + r, false);
                    last_prog = prog;
                } else if (done) {
                    for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                        refresh_row(block_q_idx * BLOCK_Q + r, true);
                    if (hist_in_smem) {
                        for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                            for (uint32_t b = lane_idx; b < num_buckets; b += 32)
                                smem_hist[r * num_buckets + b] = 0;
                    }
                    __threadfence_block();
                    if (lane_idx == 0) ack_ptr[spare_id] = static_cast<int>(qb_it) + 1;
                    break;
                } else {
                    __nanosleep(256);
                }
            }
            }
        }
#else
#ifdef DSA_SPARSE_REFRESH
            while (*scan_done_flag == 0) {
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
                drain_sparse_hist(spare_id);
                drain_sparse_hist(spare_id + 2);
#endif
#ifdef DSA_SPARSE_REFRESH_ADAPTIVE_SLEEP
                bool tightened = false;
                for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                    tightened |= refresh_row(
                        block_q_idx * BLOCK_Q + r, false);
                __nanosleep(
                    tightened
                        ? DSA_SPARSE_REFRESH_NS
                        : DSA_SPARSE_REFRESH_IDLE_NS);
#else
                for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                    refresh_row(block_q_idx * BLOCK_Q + r, false);
                __nanosleep(DSA_SPARSE_REFRESH_NS);
#endif
            }
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
            drain_sparse_hist(spare_id);
            drain_sparse_hist(spare_id + 2);
#endif
            for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                refresh_row(block_q_idx * BLOCK_Q + r, true);
#else
            int last_prog = 0;
#ifdef DSA_HIST_DEFER
            // Batch histogram feed: pull fenced candidates from the global
            // cand buffer and bucket them here, so the math warps' insert
            // path carries no per-hit feed. dh_prev boots from the seed
            // count (cand_cnt after seed_prep); a racing early drain can
            // only make us SKIP candidates (undercount = looser = safe).
            float dh_o[2], dh_inv[2];
            int dh_prev[2];
            for (uint32_t r = spare_id, li = 0; r < BLOCK_Q; r += 2, ++li) {
                const uint32_t row = min(block_q_idx * BLOCK_Q + r, seq_len - 1);
                dh_o[li] = origin[row];
                dh_inv[li] = inv_delta[row];
                dh_prev[li] = cand_cnt[row];
            }
            const auto drain_hist = [&](const uint32_t r, const uint32_t li) {
                const uint32_t row = block_q_idx * BLOCK_Q + r;
                if (row >= seq_len || !hist_in_smem) return;
                const int safe = *reinterpret_cast<volatile int32_t*>(smem_safe + r);
                const int p = dh_prev[li];
                if (safe <= p) return;
                const float* cv = cand_val + static_cast<uint64_t>(row) * cand_cap;
                for (int j = p + static_cast<int>(lane_idx); j < safe; j += 32) {
                    const float x = cv[j];
                    int braw = static_cast<int>((x - dh_o[li]) * dh_inv[li]);
                    int b = braw < 0 ? 0
                        : (braw > static_cast<int>(num_buckets) - 1
                           ? static_cast<int>(num_buckets) - 1 : braw);
                    atomicAdd(smem_hist + r * num_buckets + b, 1);
                }
                dh_prev[li] = safe;
            };
#endif
            while (true) {
                const int done = *scan_done_flag;
                const int prog = *kv_progress_ptr;
                if (prog > last_prog) {
                    for (uint32_t r = spare_id, li = 0; r < BLOCK_Q; r += 2, ++li) {
#ifdef DSA_HIST_DEFER
                        drain_hist(r, li);
#endif
                        refresh_row(block_q_idx * BLOCK_Q + r, false);
                    }
                    last_prog = prog;
                } else if (done) {
                    for (uint32_t r = spare_id, li = 0; r < BLOCK_Q; r += 2, ++li) {
#ifdef DSA_HIST_DEFER
                        drain_hist(r, li);
#endif
                        refresh_row(block_q_idx * BLOCK_Q + r, true);
                    }
                    break;
                } else {
                    __nanosleep(256);
                }
            }
#endif
        }
#endif
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_thread_idx = warp_idx * 32 + lane_idx;

        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64 or N == 128, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::conditional_t<N == 64,
                    cute::SM100_TMEM_LOAD_32dp32b64x,
                    cute::SM100_TMEM_LOAD_32dp32b128x>>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

#if defined(DSA_BUCKET_GATE2) && defined(DSA_BUCKET_GATE)
#error "DSA_BUCKET_GATE2: incompatible variant combination"
#endif
#if defined(DSA_BUCKET_GATE4) && (defined(DSA_BUCKET_GATE) || defined(DSA_BUCKET_GATE2) \
    || defined(DSA_PERSIST))
#error "DSA_BUCKET_GATE4: incompatible variant combination"
#endif
        // Bucket comparisons use the affine score space. Raw float bit
        // patterns are nonuniform across exponent ranges and cannot preserve
        // this threshold contract.
        float weights[BLOCK_Q][kNumHeads];
        float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
        uint32_t kstart_reg[BLOCK_Q], kspan_reg[BLOCK_Q];  // unsigned range-check trick
#if !defined(DSA_STATIC_GATE) && !defined(DSA_SPARSE_REFRESH)
        int gate_reg[BLOCK_Q];
#endif
        const unsigned FULL = 0xffffffffu;

#ifdef DSA_PERSIST
        uint32_t kvt_base = 0;
        DSA_QB_LOOP {
            CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
            full_q_barriers[0]->wait(qb_it & 1);
            // previous qb's hist must be zeroed (daemon acks after its final
            // refresh); only relevant when the smem histogram is live.
            if (hist_in_smem && qb_it > 0) {
                while (ack_ptr[0] < static_cast<int>(qb_it) ||
                       ack_ptr[1] < static_cast<int>(qb_it))
                    __nanosleep(128);
            }
#else
        if (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
            full_q_barriers[0]->wait(0);
#endif

            // Weights into registers (once per CTA -- the 2.5-generation win).
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[0] + i * kNumHeads + j);
            }
            // Queue fill counts are warp-uniform: every lane tracks them
            // redundantly in registers, so the hot emit path needs no shared
            // bookkeeping or shuffle broadcast.
#ifdef DSA_CHUNKED_EMIT
            // Four private 8-bit counts. They reset at every fixed-size chunk,
            // bounding the address dispersion of direct global record stores.
            uint32_t emit_lane_counts = 0;
#else
            int qn_reg[BLOCK_Q];
#endif
#ifdef DSA_EMIT_NULL
            uint32_t emit_sink = 0;  // keeps the gate live without emit code
#endif
#ifdef DSA_GATE_NULL
            float gate_sink = 0.f;   // keeps scoring live without the gate
#endif
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
                o_reg[i] = origin[rq];
                inv_reg[i] = inv_delta[rq];
#ifdef DSA_BUCKET_GATE
                // bucket-space affine: braw = (x - o)*inv with x = -v folds to
                // one FFMA per column: braw = f2i(fmaf(v, -inv, -o*inv)).
                // Reuse the slots (zero new registers, the 232-reg law):
                // inv_reg stores -inv, vth_reg stores c0 = -o*inv.
                inv_reg[i] = -inv_reg[i];
                vth_reg[i] = o_reg[i] * inv_reg[i];
#endif
#ifdef DSA_BUCKET_GATE2
                // braw = (x - o)*inv = fmaf(x, +inv, -o*inv): inv stays,
                // vth_reg repurposed to -o*inv.
                vth_reg[i] = -o_reg[i] * inv_reg[i];
#endif
#ifdef DSA_BUCKET_GATE4
                // GATE4 (user's final form): bucket-space FLOAT end to end.
                // bq = fmaf(scale_kv, sum', c0) -- form-identical to the sign
                // gate; gate = INT compare of bq's BITS vs edge float(g+1)
                // bits (edge >= 1 > 0, so ALL negative bq bit-patterns
                // compare below it and pass: no sign flip needed). cand_val
                // stores bq itself (affine preserves order; select runs in
                // bucket space, indices are the only output -- the exact
                // score is never reconstructed). vth_reg = c0; o_reg is
                // repurposed at consume time to hold the edge float.
                vth_reg[i] = -o_reg[i] * inv_reg[i];
                o_reg[i] = 0.0f;  // gate closed until the first consume
#endif
#if !defined(DSA_STATIC_GATE) && !defined(DSA_SPARSE_REFRESH)
                gate_reg[i] = cute::numeric_limits<int32_t>::max();
#endif
#ifndef DSA_CHUNKED_EMIT
                qn_reg[i] = 0;
#endif
                kstart_reg[i] = seq_k_start[i];
                kspan_reg[i] = seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
            }
#ifdef DSA_BUCKET_GATE4
            // Fold -inv into the register weights: the whole ReLU-weighted
            // chain then accumulates directly in bucket units. 128 FMULs
            // once per qb, amortized over thousands of kv blocks.
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] *= -inv_reg[i];
            }
#endif
            // Interior-block bounds (warp-uniform): a kv block fully inside
            // every row's [ks, ke) needs no per-element range checks.
            uint32_t rs_max = 0, re_min = 0xffffffffu;
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                rs_max = max(rs_max, kstart_reg[i]);
                re_min = min(re_min, kstart_reg[i] + kspan_reg[i]);
            }
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
            // Counts for actual chunks are published at their boundary. Clear
            // only padding chunks so the compactor can scan a rectangular
            // workspace without memset-ing the multi-GiB record allocation.
            const uint32_t emit_actual_chunks =
                math::ceil_div(num_kv_blocks,
                               static_cast<uint32_t>(
                                   DSA_EMIT_CHUNK_BLOCKS));
            for (uint32_t chunk = emit_actual_chunks;
                 chunk < emit_chunks_per_split; ++ chunk) {
                const uint64_t chunk_linear =
                    ((static_cast<uint64_t>(block_q_idx) *
                          num_kv_splits +
                      kv_split) *
                         emit_chunks_per_split +
                     chunk) *
                        kNumMathWarps +
                    warp_idx;
                emit_counts[chunk_linear * 32u + lane_idx] = 0;
            }
#endif

#ifdef DSA_SPARSE_REFRESH
            // The initial exact-subset edge was copied to the shared mailbox
            // before the CTA-wide boot barrier. Thereafter the spare-warp
            // daemon may only decrease it. A racing/stale shared load is a
            // looser edge and therefore recall-safe.
            const float4 initial_gate = ptx::ld_shared(
                reinterpret_cast<const float4*>(sparse_gate_bits));
            o_reg[0] = initial_gate.x;
            o_reg[1] = initial_gate.y;
            o_reg[2] = initial_gate.z;
            o_reg[3] = initial_gate.w;
#elif defined(DSA_STATIC_GATE)
            // An exact-subset threshold is a looser, recall-safe admission
            // edge. Consume it once and keep it in registers for the scan.
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const int g = __ldcg(
                    th_bucket +
                    min(block_q_idx * BLOCK_Q + i, seq_len - 1));
                o_reg[i] = static_cast<float>(g + 1);
            }
#else
            // Gate PREFETCH: th_bucket lives in global and is tightened
            // concurrently by the refresh; loading it at the consume point
            // stalls the warp on the LDG->ISETP->BRA chain (NCU: ~27% of all
            // stall samples, long_scoreboard on the reload branches). Instead
            // consume the value fetched one window earlier and issue the next
            // window's load right after -- GATE_STRIDE blocks of latency
            // cover. A one-window-stale gate is recall-safe: refresh only
            // TIGHTENS th, so staleness admits extra candidates, never drops.
            int th_pf[BLOCK_Q];
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i)
                th_pf[i] = __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + i, seq_len - 1));
#endif

            // Drain a (warp,row) queue segment to the global candidate
            // buffer. The probe index mapping lives HERE, not on the insert
            // path: it used to run warp-wide per ballot group (~4 predicated
            // instructions dragged by a single hit); at drain time 32 lanes
            // retire DSA_WARP_QUEUE_CAP entries in parallel, so it costs
            // ~1/30th in issue slots and is semantically identical.
#if !defined(DSA_EMIT_NULL) && !defined(DSA_CHUNKED_EMIT)
            const auto drain_queue = [&](const uint32_t i, const uint32_t row_q,
                                         const uint32_t queue_base, const int qn,
                                         const int base) {
                const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
#ifndef DSA_DRAIN_NULL   // diagnostic build: price the drain payload ceiling
                for (int t = static_cast<int>(lane_idx); t < qn; t += 32) {
                    const float x = warpq_val[queue_base + t];
                    uint32_t kvo = static_cast<uint32_t>(warpq_idx[queue_base + t]);
                    if (probe_group != 0) {
                        // compacted -> original position (probe pages were
                        // excluded from the workspace; the probe itself
                        // seeds them); exact c/probe_group via magic mul-shift
                        const uint32_t sup = (uint32_t)(((uint64_t)kvo * probe_magic) >> 42);
                        kvo += min((sup + 1) * 64u, probe_add_max);
                    }
                    const int w = base + t;
                    if (w < static_cast<int>(cand_cap)) {
                        DSA_ST_CAND_VAL(cand_val[out_base + w], x);
                        DSA_ST_CAND_IDX(cand_idx[out_base + w], static_cast<int32_t>(kvo));
                    }
                }
#ifdef DSA_HIST_DEFER
                // Publish the fenced watermark: every lane's STGs above must
                // be block-visible before lane 0 advances smem_safe, so the
                // daemon never reads unwritten payload. Clipped entries
                // (w >= cap) are excluded (never written, never counted).
                if (hist_in_smem) {
                    const int wlen = min(qn, max(static_cast<int>(cand_cap) - base, 0));
                    __threadfence_block();
                    __syncwarp(FULL);
                    if (wlen > 0 && lane_idx == 0)
                        atomicAdd(smem_safe + i, wlen);
                }
#endif
#else
                (void)out_base;  // timing-only: counts kept, payload skipped
#endif
            };
#endif  // !DSA_EMIT_NULL && !DSA_CHUNKED_EMIT

            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
#ifdef DSA_PERSIST
                const uint32_t kvg = kvt_base + kv_block_idx;
#else
                const uint32_t kvg = kv_block_idx;
#endif
                CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);

#ifdef DSA_SPARSE_REFRESH
                if (kv_block_idx != 0 &&
                    (kv_block_idx % DSA_GATE_STRIDE) == 0) {
                    const float4 gate = ptx::ld_shared(
                        reinterpret_cast<const float4*>(
                            sparse_gate_bits));
                    o_reg[0] = gate.x;
                    o_reg[1] = gate.y;
                    o_reg[2] = gate.z;
                    o_reg[3] = gate.w;
                }
#endif
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

#if !defined(DSA_STATIC_GATE) && \
    !defined(DSA_SPARSE_REFRESH)
                if ((kv_block_idx % DSA_GATE_STRIDE) == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const int g = th_pf[i];
                        if (g != gate_reg[i]) {
                            gate_reg[i] = g;
#if defined(DSA_BUCKET_GATE4)
                            // edge = float(g+1): exact for small ints, no
                            // division. The gate compares BITS against it.
                            o_reg[i] = static_cast<float>(g + 1);
#elif defined(DSA_BUCKET_GATE) || defined(DSA_BUCKET_GATE2)
                            // bucket gate compares braw <= gate_reg directly:
                            // nothing to precompute (no edge float, no div).
#else
                            // SIGN GATE (default): stores +th_x; the hot path computes
                            // t = fmaf(scale, sums, th_x) = v - C and tests the
                            // SIGN BIT (ISETP on the INT pipe); the FSETP and
                            // the final FMUL are algebraically absorbed.
                            vth_reg[i] = o_reg[i] + static_cast<float>(g + 1) / inv_reg[i];
#endif
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i)
                        th_pf[i] = __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + i, seq_len - 1));
                }
#endif

                float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

                const uint32_t umma_buf = warpgroup_idx * kNumUmmaStages + (kvg % kNumUmmaStages);
                const auto tmem_start = umma_buf * UMMA_N;
                full_umma_barriers[umma_buf]->wait((kvg / kNumUmmaStages) & 1);
                ptx::tcgen05_after_thread_sync();

                empty_kv_barriers[kv_stage_idx]->arrive();

                const auto kv_offset = kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

                uint32_t pass_bits = 0;
                float v_row[BLOCK_Q];
#ifdef DSA_BUCKET_GATE2
                int braw_row[BLOCK_Q];  // gate-time bucket ids, reused on hits
#endif

                // P1: row-PAIR TMEM loads (32dp32b64x): half the tcgen05.ld
                // instructions and half the fences on the governor loop; the
                // UMMA release also moves one row earlier.
                // Interior-block gate elision: one warp-uniform branch per
                // block picks a loop body WITHOUT the per-element range
                // checks (SASS: saves 4x VIADD + 4x ISETP per column) for the
                // >99% of blocks fully inside every row's [ks, ke).
                DG_STATIC_ASSERT(BLOCK_Q % DSA_TMEM_ROWS == 0,
                                 "BLOCK_Q must divide evenly into TMEM load groups");
#if defined(DSA_GATE_NULL)
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float v = scale_kv * (sum.x + sum.y);                            \
                        v_row[i] = v;                                                          \
                        gate_sink += v;
#elif defined(DSA_BUCKET_GATE4)
                // GATE4: column cost IDENTICAL to the sign gate (FFMA whose
                // addend is c0 instead of th_x, ISETP on bits instead of the
                // sign). NaN bq maps to a large positive pattern -> DROPPED
                // (old FSETP semantics; recall check is the arbiter).
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float bq = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);            \
                        v_row[i] = bq;                                                         \
                        bool g = __float_as_int(bq) < __float_as_int(o_reg[i]);                \
                        if constexpr (RC)                                                      \
                            g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]);            \
                        pass_bits |= g ? (1u << i) : 0u;
#elif defined(DSA_BUCKET_GATE2)
                // BUCKET GATE v2 (A/B): bucket id computed BEFORE the gate and
                // KEPT (braw_row) so hits pay ZERO bucket math; v_row holds x
                // directly (negation folded into the FMUL operand modifier),
                // so hits also skip the x reconstruction. Column cost:
                // FMUL + FFMA + F2I + ISETP (4) vs sign-gate's 2; the bet
                // being tested is whether zeroing the hit-side work repays it.
                // +4 int registers of liveness (braw_row) on a 232-saturated
                // budget -- spills are part of the verdict.
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float x = __fmul_rn(-scale_kv, sum.x + sum.y);                   \
                        v_row[i] = x;                                                          \
                        braw_row[i] = static_cast<int>(fmaf(x, inv_reg[i], vth_reg[i]));       \
                        bool g = braw_row[i] <= gate_reg[i];                                   \
                        if constexpr (RC)                                                      \
                            g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]);            \
                        pass_bits |= g ? (1u << i) : 0u;
#elif defined(DSA_BUCKET_GATE)
                // BUCKET GATE (A/B): quantize the score straight to its
                // histogram bucket and compare bucket ids on the INT pipe.
                // Gate and hist feed share ONE FFMA+F2I formula -> consistent
                // by construction (no float-edge vs quantization seam).
                // Cost: +FMUL+F2I per column vs the sign gate (its FFMA
                // absorbed the scale multiply). F2I(NaN)=0 -> NaN admitted
                // (loose-safe). RISK: prep's sample histogram buckets with
                // FSUB+FMUL; a 1-ULP FFMA divergence on an exact boundary
                // value could tighten the gate past a counted sample column
                // -- the recall check is the arbiter.
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float v = scale_kv * (sum.x + sum.y);                            \
                        v_row[i] = v;                                                          \
                        const int braw_g = static_cast<int>(fmaf(v, inv_reg[i], vth_reg[i]));  \
                        bool g = braw_g <= gate_reg[i];                                        \
                        if constexpr (RC)                                                      \
                            g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]);            \
                        pass_bits |= g ? (1u << i) : 0u;
#else
                // t = v - C in one FFMA; hit <=> sign bit clear. +0/NaN admit
                // (loose-safe; the old FSETP DROPPED NaN scores). True x is
                // reconstructed only on the hit path: x = th_x - t.
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float v = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);             \
                        v_row[i] = v;                                                          \
                        bool g = __float_as_int(v) >= 0;                                       \
                        if constexpr (RC)                                                      \
                            g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]);            \
                        pass_bits |= g ? (1u << i) : 0u;
#endif
                const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
                const bool interior = (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
                #define DSA_SCORE_ROWS(RANGE_CHECK)                                            \
                _Pragma("unroll")                                                              \
                for (uint32_t pr = 0; pr < BLOCK_Q / DSA_TMEM_ROWS; ++ pr) {                   \
                    float accum2[kNumHeads * DSA_TMEM_ROWS];                                   \
                    tmem_load(cute::Int<kNumHeads * DSA_TMEM_ROWS>{},                          \
                              tmem_start + pr * DSA_TMEM_ROWS * kNumHeads, accum2);            \
                    if (pr == BLOCK_Q / DSA_TMEM_ROWS - 1) {                                   \
                        ptx::tcgen05_before_thread_sync();                                     \
                        empty_umma_barriers[umma_buf]->arrive();                               \
                    }                                                                          \
                    _Pragma("unroll")                                                          \
                    for (uint32_t k = 0; k < DSA_TMEM_ROWS; ++ k) {                            \
                        const uint32_t i = pr * DSA_TMEM_ROWS + k;                             \
                        const float* accum = accum2 + k * kNumHeads;                           \
                        auto sum_0 = make_float2(0, 0);                                        \
                        auto sum_1 = make_float2(0, 0);                                        \
                        const auto transform = [&](const uint32_t& j, const float2& sum) {     \
                            auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));  \
                            auto b = make_float2(weights[i][j], weights[i][j + 1]);            \
                            return __ffma2_rn(a, b, sum);                                      \
                        };                                                                     \
                        _Pragma("unroll")                                                      \
                        for (uint32_t j = 0; j < kNumHeads; j += 4) {                          \
                            sum_0 = transform(j, sum_0);                                       \
                            sum_1 = transform(j + 2, sum_1);                                   \
                        }                                                                      \
                        auto sum = __fadd2_rn(sum_0, sum_1);                                   \
                        DSA_SCORE_GATE(i, RANGE_CHECK)                                         \
                    }                                                                          \
                }
                if (interior) { DSA_SCORE_ROWS(false) } else { DSA_SCORE_ROWS(true) }
                #undef DSA_SCORE_ROWS
                #undef DSA_SCORE_GATE
#ifdef DSA_GATE_NULL
                if (__float_as_uint(gate_sink) == 0x13371337u) *scan_done_flag = 3;
#endif

                // redux pruning: inside an active block, one redux.sync.or
                // gives the warp-wide union of hit rows, so the ballot (and
                // its queue bookkeeping) runs only for rows that actually
                // have hits (~1.3 of BLOCK_Q=4 at production density). The
                // cheap VOTE.ANY stays as the outer gate: redux costs more
                // than a vote and must not run on the ~40% inactive blocks.
                // Branches are warp-uniform (no divergence around collectives).
#ifdef DSA_CHUNKED_EMIT
                // Direct per-lane log. Counts reset every chunk, so lanes with
                // different hit histories remain within a small, cache-local
                // group of slot planes. There are no warp collectives, shared
                // queues, or returning atomics on the normal path.
                if (pass_bits != 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        if ((pass_bits >> i) & 1u) {
#if defined(DSA_BUCKET_GATE4)
                            const float x = v_row[i];
#elif defined(DSA_BUCKET_GATE2)
                            const float x = v_row[i];
#elif defined(DSA_BUCKET_GATE)
                            const float x = -v_row[i];
#else
                            const float x = vth_reg[i] - v_row[i];
#endif
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
                            // The packed code gives both the exact bucket used
                            // by sparse refresh and the approximate fractional
                            // rank written to the candidate log.
                            const uint32_t candidate_packed_payload =
                                candidate_bucket8_frac8_code(x);
                            const uint32_t candidate_bucket =
                                candidate_packed_payload >> 8;
                            const int candidate_braw =
                                static_cast<int>(candidate_bucket);
#elif defined(DSA_CANDIDATE_FP16_LOCAL32)
                            // One 32-bit local record keeps FP16 score bits,
                            // the exact source bucket, and the block offset
                            // inside a <=256-block window. The writer lane
                            // already identifies the low eight KV bits.
                            const uint32_t candidate_half =
                                static_cast<uint32_t>(
                                    candidate_fp16_bits(x));
                            const int candidate_braw =
                                static_cast<int>(x);
                            const uint32_t candidate_bucket =
                                static_cast<uint32_t>(
                                    max(candidate_braw, 0));
                            const uint32_t candidate_packed_payload =
                                candidate_half |
                                (candidate_bucket << 16);
#elif defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) || \
    defined(DSA_CANDIDATE_FP16_NUMERIC)
                            // Sparse refresh needs this exact bucket for the
                            // SMEM histogram below.  Compute it once and also
                            // use it to form the packed score payload.
                            const int candidate_braw =
                                static_cast<int>(x);
                            const uint32_t candidate_bucket =
                                static_cast<uint32_t>(
                                    max(candidate_braw, 0));
                            const uint32_t candidate_packed_payload =
                                candidate_record_payload(
                                    x, candidate_bucket);
#endif
                            const uint32_t count =
                                (emit_lane_counts >> (i * 8)) & 0xffu;
                            if (count < DSA_EMIT_LANE_SLOTS) {
                                const uint32_t pos =
                                    ((warp_idx * BLOCK_Q + i) *
                                         DSA_EMIT_LANE_SLOTS +
                                     count) *
                                        32u +
                                    lane_idx;
                                const uint32_t record_payload =
#if defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) || \
    defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) || \
    defined(DSA_CANDIDATE_FP16_NUMERIC)
                                    candidate_packed_payload;
#else
                                    candidate_record_payload(x);
#endif
#ifdef DSA_CANDIDATE_FP16_LOCAL32
                                const uint32_t local_block =
                                    kv_block_idx %
                                    DSA_EMIT_CHUNK_BLOCKS;
                                const uint32_t record =
                                    (record_payload & 0xffffu) |
                                    (local_block << 16) |
                                    (candidate_bucket << 24);
#else
                                const uint64_t record =
                                    record_payload |
                                    (static_cast<uint64_t>(
                                         kv_offset) << 32);
#endif
                                emit_smem_records[pos] = record;
                                emit_lane_counts += 1u << (i * 8);
                            } else {
                                // A skewed lane can overflow its small local
                                // quota without losing candidates. This slow
                                // path writes directly to the final buffer.
                                const uint32_t row_q =
                                    block_q_idx * BLOCK_Q + i;
                                const int out =
                                    atomicAdd(cand_cnt + row_q, 1);
                                if (out < static_cast<int>(cand_cap)) {
                                    uint32_t kvo = kv_offset;
                                    if (probe_group != 0) {
                                        const uint32_t sup =
                                            static_cast<uint32_t>(
                                                (static_cast<uint64_t>(kvo) *
                                                 probe_magic) >>
                                                42);
                                        kvo += min((sup + 1) * 64u,
                                                   probe_add_max);
                                    }
                                    const uint64_t out_base =
                                        static_cast<uint64_t>(row_q) *
                                        cand_cap;
                                    DSA_ST_CANDIDATE_RECORD(
                                        cand_val[out_base + out],
                                        cand_idx[out_base + out],
#if defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) || \
    defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) || \
    defined(DSA_CANDIDATE_FP16_NUMERIC)
                                        candidate_packed_payload,
#else
                                        candidate_record_payload(x),
#endif
                                        kvo);
                                }
                            }

#ifndef DSA_STATIC_GATE
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
                            if constexpr (!kSingleKVSplit) {
#endif
#ifdef DSA_SPARSE_REFRESH
                            {
#else
                            if (refresh_every > 0) {
#endif
#if defined(DSA_BUCKET_GATE4)
#if defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) || \
    defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) || \
    defined(DSA_CANDIDATE_FP16_NUMERIC)
                                int braw = candidate_braw;
#else
                                int braw = static_cast<int>(v_row[i]);
#endif
#elif defined(DSA_BUCKET_GATE2)
                                int braw = braw_row[i];
#elif defined(DSA_BUCKET_GATE)
                                int braw = static_cast<int>(
                                    fmaf(v_row[i], inv_reg[i], vth_reg[i]));
#else
                                int braw = static_cast<int>(
                                    (x - o_reg[i]) * inv_reg[i]);
#endif
#if defined(DSA_BUCKET_GATE4)
                                // A passing positive bq is below the float
                                // edge (gate + 1), hence already < num_buckets.
#if defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) || \
    defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) || \
    defined(DSA_CANDIDATE_FP16_NUMERIC)
                                int b =
                                    static_cast<int>(candidate_bucket);
#else
                                int b = max(braw, 0);
#endif
#else
                                int b = braw < 0 ? 0 :
                                    (braw >
                                             static_cast<int>(num_buckets) - 1
                                         ? static_cast<int>(num_buckets) - 1
                                         : braw);
#endif
                                const uint32_t row_q =
                                    block_q_idx * BLOCK_Q + i;
#ifdef DSA_SPARSE_REFRESH
                                if constexpr (kSingleKVSplit) {
#else
                                if (hist_in_smem) {
#endif
                                    atomicAdd(
                                        smem_hist + i * num_buckets + b, 1);
                                } else {
                                    atomicAdd(
                                        &bcount[
                                            static_cast<uint64_t>(row_q) *
                                                num_buckets +
                                            b],
                                        1);
                                }
                            }
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
                            }
#endif
#endif
                        }
                    }
                }

                if (((kv_block_idx + 1) %
                         DSA_EMIT_CHUNK_BLOCKS) == 0 ||
                    kv_block_idx + 1 == num_kv_blocks) {
#ifdef DSA_CHUNKED_FLUSH_DIRECT_CTA
                    // First publish each warp's four totals. All 256 math
                    // threads participate in both named barriers; producer
                    // warps are deliberately excluded.
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        const uint32_t total =
                            __reduce_add_sync(FULL, count);
                        if (lane_idx == 0) {
                            flush_warp_prefix[
                                i * kNumMathWarps + warp_idx] =
                                    static_cast<uint16_t>(total);
                        }
                    }
                    cutlass::arch::NamedBarrier(
                        kNumMathThreads, 0).sync();

                    // Four lanes reserve the four rows concurrently. Replace
                    // totals with exclusive warp prefixes for the copy phase.
                    if (warp_idx == 0 && lane_idx < BLOCK_Q) {
                        const uint32_t i = lane_idx;
                        uint32_t cta_total = 0;
                        #pragma unroll
                        for (uint32_t w = 0;
                             w < kNumMathWarps; ++ w) {
                            const uint32_t total =
                                flush_warp_prefix[
                                    i * kNumMathWarps + w];
                            flush_warp_prefix[
                                i * kNumMathWarps + w] =
                                    static_cast<uint16_t>(cta_total);
                            cta_total += total;
                        }
                        const uint32_t row_q =
                            block_q_idx * BLOCK_Q + i;
                        flush_row_base[i] =
                            row_q < seq_len && cta_total != 0
                                ? atomicAdd(
                                      cand_cnt + row_q,
                                      static_cast<int>(cta_total))
                                : 0;
                    }
                    cutlass::arch::NamedBarrier(
                        kNumMathThreads, 0).sync();

                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        uint32_t inclusive = count;
                        #pragma unroll
                        for (int delta = 1;
                             delta < 32; delta <<= 1) {
                            const uint32_t other =
                                __shfl_up_sync(
                                    FULL, inclusive, delta);
                            if (lane_idx >=
                                static_cast<uint32_t>(delta))
                                inclusive += other;
                        }
                        const uint32_t offset =
                            static_cast<uint32_t>(
                                flush_warp_prefix[
                                    i * kNumMathWarps + warp_idx]) +
                            inclusive - count;
                        const uint32_t row_q =
                            block_q_idx * BLOCK_Q + i;
                        const int out_base = flush_row_base[i];
                        const int copy_count =
                            row_q < seq_len
                                ? min(
                                      static_cast<int>(count),
                                      max(
                                          static_cast<int>(cand_cap) -
                                              out_base -
                                              static_cast<int>(offset),
                                          0))
                                : 0;
                        for (int slot = 0;
                             slot < copy_count; ++ slot) {
                            const uint32_t local_pos =
                                ((warp_idx * BLOCK_Q + i) *
                                     DSA_EMIT_LANE_SLOTS +
                                 slot) *
                                    32u +
                                lane_idx;
                            const uint64_t record =
                                emit_smem_records[local_pos];
                            const int out =
                                out_base +
                                static_cast<int>(offset) + slot;
                            const uint64_t out_pos =
                                static_cast<uint64_t>(row_q) *
                                    cand_cap +
                                out;
                            const uint32_t record_payload =
                                static_cast<uint32_t>(record);
                            uint32_t kvo =
                                static_cast<uint32_t>(record >> 32);
                            if (probe_group != 0) {
                                const uint32_t sup =
                                    static_cast<uint32_t>(
                                        (static_cast<uint64_t>(kvo) *
                                         probe_magic) >>
                                        42);
                                kvo += min((sup + 1) * 64u,
                                           probe_add_max);
                            }
                            DSA_ST_CANDIDATE_RECORD(
                                cand_val[out_pos],
                                cand_idx[out_pos],
                                record_payload,
                                kvo);
                        }
                    }
                    emit_lane_counts = 0;
#elif defined(DSA_CHUNKED_FLUSH_DIRECT)
                    // Reserve one contiguous final-buffer segment per
                    // (source warp, row, chunk), then have each lane copy its
                    // private records into that segment. This keeps all
                    // collectives and the returning global atomic off the
                    // ordinary-hit path while deleting the rectangular global
                    // chunk workspace and the post-scan compactor.
#ifdef DSA_CHUNKED_FLUSH_DIRECT_PACKEDPREFIX
                    // Keep the four row reservations parallel and early so
                    // their L2 round trips overlap the prefix work. Replace
                    // four independent 32-lane scans (20 shuffles) with two
                    // scans whose 16-bit fields carry two rows each (10
                    // shuffles). A row prefix is at most 32*18=576, so fields
                    // cannot carry into one another.
                    int my_row_base = 0;
                    uint32_t my_row_total = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        const uint32_t total =
                            __reduce_add_sync(FULL, count);
                        if (lane_idx == i) my_row_total = total;
                    }
                    if (lane_idx < BLOCK_Q) {
                        const uint32_t row_q =
                            block_q_idx * BLOCK_Q + lane_idx;
                        if (row_q < seq_len && my_row_total != 0) {
                            my_row_base = atomicAdd(
                                cand_cnt + row_q,
                                static_cast<int>(my_row_total));
                        }
                    }

                    const uint32_t count0 =
                        emit_lane_counts & 0xffu;
                    const uint32_t count1 =
                        (emit_lane_counts >> 8) & 0xffu;
                    const uint32_t count2 =
                        (emit_lane_counts >> 16) & 0xffu;
                    const uint32_t count3 =
                        emit_lane_counts >> 24;
                    uint32_t inclusive01 =
                        count0 | (count1 << 16);
                    uint32_t inclusive23 =
                        count2 | (count3 << 16);
                    #pragma unroll
                    for (int delta = 1; delta < 32; delta <<= 1) {
                        const uint32_t other01 =
                            __shfl_up_sync(
                                FULL, inclusive01, delta);
                        const uint32_t other23 =
                            __shfl_up_sync(
                                FULL, inclusive23, delta);
                        if (lane_idx >=
                            static_cast<uint32_t>(delta)) {
                            inclusive01 += other01;
                            inclusive23 += other23;
                        }
                    }

#ifdef DSA_CANDIDATE_FP16_LOCAL32
                    const uint32_t emit_window_kv_base =
                        kv_start +
                        (kv_block_idx / DSA_EMIT_CHUNK_BLOCKS) *
                            DSA_EMIT_CHUNK_BLOCKS * BLOCK_KV;
#endif
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        const uint32_t inclusive =
                            i < 2
                                ? ((inclusive01 >>
                                    ((i & 1u) * 16)) &
                                   0xffffu)
                                : ((inclusive23 >>
                                    ((i & 1u) * 16)) &
                                   0xffffu);
                        const uint32_t offset = inclusive - count;
                        const uint32_t row_q =
                            block_q_idx * BLOCK_Q + i;
                        const int out_base = __shfl_sync(
                            FULL, my_row_base, i);
                        const int copy_count =
                            row_q < seq_len
                                ? min(
                                      static_cast<int>(count),
                                      max(
                                          static_cast<int>(cand_cap) -
                                              out_base -
                                              static_cast<int>(offset),
                                          0))
                                : 0;
                        for (int slot = 0;
                             slot < copy_count; ++ slot) {
                            const uint32_t local_pos =
                                ((warp_idx * BLOCK_Q + i) *
                                     DSA_EMIT_LANE_SLOTS +
                                 slot) *
                                    32u +
                                lane_idx;
#ifdef DSA_CANDIDATE_FP16_LOCAL32
                            const uint32_t record =
                                emit_smem_records[local_pos];
#else
                            const uint64_t record =
                                emit_smem_records[local_pos];
#endif
                            const int out =
                                out_base +
                                static_cast<int>(offset) + slot;
                            const uint64_t out_pos =
                                static_cast<uint64_t>(row_q) *
                                    cand_cap +
                                out;
#ifdef DSA_CANDIDATE_FP16_LOCAL32
                            const uint32_t half_bits =
                                record & 0xffffu;
                            const uint32_t bucket =
                                record >> 24;
                            const uint32_t record_payload =
                                half_bits |
                                (bucket << 16);
                            uint32_t kvo =
                                emit_window_kv_base +
                                (((record >> 16) & 0xffu) *
                                     BLOCK_KV) +
                                math_thread_idx;
#else
                            const uint32_t record_payload =
                                static_cast<uint32_t>(record);
                            uint32_t kvo =
                                static_cast<uint32_t>(record >> 32);
#endif
                            if (probe_group != 0) {
                                const uint32_t sup =
                                    static_cast<uint32_t>(
                                        (static_cast<uint64_t>(kvo) *
                                         probe_magic) >>
                                        42);
                                kvo += min((sup + 1) * 64u,
                                           probe_add_max);
                            }
                            DSA_ST_CANDIDATE_RECORD(
                                cand_val[out_pos],
                                cand_idx[out_pos],
                                record_payload,
                                kvo);
                        }
                    }
#ifdef DSA_SPARSE_REFRESH_DEFERRED_FEED
                    if constexpr (kSingleKVSplit) {
                        // Each writer first makes its candidate payload stores
                        // block-visible. The first barrier closes the current
                        // reservation epoch; the second prevents any math warp
                        // from reserving the next epoch before warp 0 snapshots
                        // cand_cnt and publishes the contiguous safe frontier.
                        __threadfence_block();
                        cutlass::arch::NamedBarrier(
                            kNumMathThreads, 1).sync();
                        if (warp_idx == 0 &&
                            lane_idx < BLOCK_Q) {
                            const uint32_t row_q =
                                block_q_idx * BLOCK_Q +
                                lane_idx;
                            const int safe =
                                row_q < seq_len
                                    ? min(
                                          max(
                                              __ldcg(
                                                  cand_cnt +
                                                  row_q),
                                              0),
                                          static_cast<int>(
                                              cand_cap))
                                    : 0;
                            atomicMax(
                                sparse_safe_end + lane_idx,
                                safe);
                        }
                        cutlass::arch::NamedBarrier(
                            kNumMathThreads, 1).sync();
                    }
#endif
                    emit_lane_counts = 0;
#elif defined(DSA_CHUNKED_FLUSH_DIRECT_PARROWS)
                    // Compute all four totals first, then let lanes 0..3 issue
                    // the four independent row reservations in one warp
                    // instruction. Broadcast the returned bases only after
                    // every request is in flight, hiding most of the otherwise
                    // serial L2 round trips without a CTA-wide barrier.
                    uint32_t my_row_total = 0;
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        const uint32_t total =
                            __reduce_add_sync(FULL, count);
                        if (lane_idx == i) my_row_total = total;
                    }
                    int my_row_base = 0;
                    if (lane_idx < BLOCK_Q) {
                        const uint32_t row_q =
                            block_q_idx * BLOCK_Q + lane_idx;
                        if (row_q < seq_len && my_row_total != 0) {
                            my_row_base = atomicAdd(
                                cand_cnt + row_q,
                                static_cast<int>(my_row_total));
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        uint32_t inclusive = count;
                        #pragma unroll
                        for (int delta = 1;
                             delta < 32; delta <<= 1) {
                            const uint32_t other =
                                __shfl_up_sync(
                                    FULL, inclusive, delta);
                            if (lane_idx >=
                                static_cast<uint32_t>(delta))
                                inclusive += other;
                        }
                        const uint32_t offset = inclusive - count;
                        const uint32_t row_q =
                            block_q_idx * BLOCK_Q + i;
                        const int out_base = __shfl_sync(
                            FULL, my_row_base, i);
                        const int copy_count =
                            row_q < seq_len
                                ? min(
                                      static_cast<int>(count),
                                      max(
                                          static_cast<int>(cand_cap) -
                                              out_base -
                                              static_cast<int>(offset),
                                          0))
                                : 0;
                        for (int slot = 0;
                             slot < copy_count; ++ slot) {
                            const uint32_t local_pos =
                                ((warp_idx * BLOCK_Q + i) *
                                     DSA_EMIT_LANE_SLOTS +
                                 slot) *
                                    32u +
                                lane_idx;
                            const uint64_t record =
                                emit_smem_records[local_pos];
                            const int out =
                                out_base +
                                static_cast<int>(offset) + slot;
                            const uint64_t out_pos =
                                static_cast<uint64_t>(row_q) *
                                    cand_cap +
                                out;
                            const uint32_t record_payload =
                                static_cast<uint32_t>(record);
                            uint32_t kvo =
                                static_cast<uint32_t>(record >> 32);
                            if (probe_group != 0) {
                                const uint32_t sup =
                                    static_cast<uint32_t>(
                                        (static_cast<uint64_t>(kvo) *
                                         probe_magic) >>
                                        42);
                                kvo += min((sup + 1) * 64u,
                                           probe_add_max);
                            }
                            DSA_ST_CANDIDATE_RECORD(
                                cand_val[out_pos],
                                cand_idx[out_pos],
                                record_payload,
                                kvo);
                        }
                    }
                    emit_lane_counts = 0;
#else
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        uint32_t inclusive = count;
                        #pragma unroll
                        for (int delta = 1; delta < 32; delta <<= 1) {
                            const uint32_t other =
                                __shfl_up_sync(FULL, inclusive, delta);
                            if (lane_idx >= static_cast<uint32_t>(delta))
                                inclusive += other;
                        }
                        const uint32_t total =
                            __shfl_sync(FULL, inclusive, 31);
                        const uint32_t offset = inclusive - count;
                        const uint32_t row_q =
                            block_q_idx * BLOCK_Q + i;
                        if (row_q < seq_len && total != 0) {
                            int out_base = 0;
                            if (lane_idx == 0)
                                out_base = atomicAdd(
                                    cand_cnt + row_q,
                                    static_cast<int>(total));
                            out_base =
                                __shfl_sync(FULL, out_base, 0);
                            const int copy_count = min(
                                static_cast<int>(count),
                                max(static_cast<int>(cand_cap) -
                                        out_base -
                                        static_cast<int>(offset),
                                    0));
                            for (int slot = 0;
                                 slot < copy_count; ++ slot) {
                                const uint32_t local_pos =
                                    ((warp_idx * BLOCK_Q + i) *
                                         DSA_EMIT_LANE_SLOTS +
                                     slot) *
                                        32u +
                                    lane_idx;
                                const uint64_t record =
                                    emit_smem_records[local_pos];
                                const int out =
                                    out_base +
                                    static_cast<int>(offset) + slot;
                                const uint64_t out_pos =
                                    static_cast<uint64_t>(row_q) *
                                        cand_cap +
                                    out;
                                const uint32_t record_payload =
                                    static_cast<uint32_t>(record);
                                uint32_t kvo =
                                    static_cast<uint32_t>(record >> 32);
                                if (probe_group != 0) {
                                    const uint32_t sup =
                                        static_cast<uint32_t>(
                                            (static_cast<uint64_t>(kvo) *
                                             probe_magic) >>
                                            42);
                                    kvo += min((sup + 1) * 64u,
                                               probe_add_max);
                                }
                                DSA_ST_CANDIDATE_RECORD(
                                    cand_val[out_pos],
                                    cand_idx[out_pos],
                                    record_payload,
                                    kvo);
                            }
                        }
                    }
                    emit_lane_counts = 0;
#endif
#else
                    const uint32_t emit_chunk =
                        kv_block_idx / DSA_EMIT_CHUNK_BLOCKS;
                    const uint64_t chunk_linear =
                        ((static_cast<uint64_t>(block_q_idx) *
                              num_kv_splits +
                          kv_split) *
                             emit_chunks_per_split +
                         emit_chunk) *
                            kNumMathWarps +
                        warp_idx;
                    const uint64_t record_base =
                        chunk_linear *
                        (BLOCK_Q * DSA_EMIT_LANE_SLOTS * 32u);
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const uint32_t count =
                            (emit_lane_counts >> (i * 8)) & 0xffu;
                        for (uint32_t slot = 0; slot < count; ++ slot) {
                            const uint32_t local_pos =
                                ((warp_idx * BLOCK_Q + i) *
                                     DSA_EMIT_LANE_SLOTS +
                                 slot) *
                                    32u +
                                lane_idx;
                            const uint64_t global_pos =
                                record_base +
                                (i * DSA_EMIT_LANE_SLOTS + slot) * 32u +
                                lane_idx;
                            const uint64_t record =
                                emit_smem_records[local_pos];
                            __stcs(
                                reinterpret_cast<unsigned long long*>(
                                    emit_workspace + global_pos),
                                static_cast<unsigned long long>(
                                    record));
                        }
                    }
                    emit_counts[chunk_linear * 32u + lane_idx] =
                        emit_lane_counts;
                    emit_lane_counts = 0;
#endif
                }
#elif defined(DSA_EMIT_NULL)
                emit_sink += pass_bits;      // gate + vote priced, zero emit
#else
                if (__any_sync(FULL, pass_bits)) {
                    const uint32_t rows_union = __reduce_or_sync(FULL, pass_bits);
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        if (rows_union & (1u << i)) {
                            const bool g = (pass_bits >> i) & 1u;
                            const unsigned m = __ballot_sync(FULL, g);
                            // Emit through the shared warp queue with a
                            // warp-uniform register fill count. The returning
                            // global reservation happens only when draining.
                            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                            const int cnt = __popc(m);
                            const uint32_t queue_base = (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
                            int qn = qn_reg[i];
                            if (qn + cnt > static_cast<int>(DSA_WARP_QUEUE_CAP)) {
                                int base = 0;
                                if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
                                base = __shfl_sync(FULL, base, 0);
                                drain_queue(i, row_q, queue_base, qn, base);
                                qn = 0;
                                __syncwarp(FULL);  // queue slots reusable
                            }
                            if (g) {
                                // Defer probe index mapping to drain_queue, but
                                // feed the histogram immediately so threshold
                                // tightening does not lag behind the queue.
#if defined(DSA_BUCKET_GATE4)
                                const float x = v_row[i];   // bucket-space value IS the payload
#elif defined(DSA_BUCKET_GATE2)
                                const float x = v_row[i];   // v_row holds x directly
#elif defined(DSA_BUCKET_GATE)
                                const float x = -v_row[i];  // v is the raw scaled score here
#else
                                const float x = vth_reg[i] - v_row[i];  // x = -v = th_x - t
#endif
                                const unsigned below = (1u << lane_idx) - 1u;
                                const int pos = qn + __popc(m & below);
                                warpq_val[queue_base + pos] = x;
                                warpq_idx[queue_base + pos] = static_cast<int32_t>(kv_offset);
                                if (refresh_every > 0) {
#if defined(DSA_BUCKET_GATE4)
                                    // one F2I off the stored bucket float --
                                    // bucket-identical to the gate.
                                    int braw = static_cast<int>(v_row[i]);
#elif defined(DSA_BUCKET_GATE2)
                                    // ZERO hit-side bucket math: reuse the
                                    // gate-time braw.
                                    int braw = braw_row[i];
#elif defined(DSA_BUCKET_GATE)
                                    // SAME FFMA as the gate: feed and gate are
                                    // bucket-identical by construction.
                                    int braw = static_cast<int>(fmaf(v_row[i], inv_reg[i], vth_reg[i]));
#else
                                    int braw = static_cast<int>((x - o_reg[i]) * inv_reg[i]);
#endif
                                    int b = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                                    if (hist_in_smem) {
#ifndef DSA_HIST_DEFER
                                        atomicAdd(smem_hist + i * num_buckets + b, 1);
#endif
                                        // (DEFER: daemon batch-feeds from the
                                        // cand buffer; b's chain dead-codes.)
                                    } else {
                                        atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], 1);
                                    }
                                }
                            }
                            qn_reg[i] = qn + cnt;
                        }
                    }
                }
#endif  // emit alternatives

#if !defined(DSA_STATIC_GATE) && !defined(DSA_SPARSE_REFRESH)
                if (threadIdx.x == 0 && ((kv_block_idx + 1) % DSA_REFRESH_STRIDE) == 0) {
                    __threadfence_block();
                    *kv_progress_ptr = static_cast<int>(kvg + 1);
                }
#endif
#ifdef DSA_EMIT_NULL
                if (emit_sink == 0x13371337u) *scan_done_flag = 2;  // unprovable sink
#endif
            }

#ifdef DSA_CHUNKED_EMIT
            // Every actual chunk was published at its final KV block.
#elif !defined(DSA_EMIT_NULL)
            // Flush this CTA's warp queues (counts live in qn_reg).
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                const int qn = qn_reg[i];
                if (row_q < seq_len && qn > 0) {
                    const uint32_t queue_base = (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
                    int base = 0;
                    if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
                    base = __shfl_sync(FULL, base, 0);
                    drain_queue(i, row_q, queue_base, qn, base);
                }
            }

#endif  // emit flush
#ifdef DSA_PERSIST
            // per-qb epilogue: all math warps done -> signal daemon (final
            // refresh + hist zero + ack), release the q stage, next block.
            cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
            if (threadIdx.x == 0) {
                __threadfence_block();
                *done_qb_ptr = static_cast<int>(qb_it) + 1;
            }
            empty_q_barriers[0]->arrive();
            kvt_base += num_kv_blocks;
        }
#else
            empty_q_barriers[0]->arrive();
        }
#endif

        // Signal the refresh daemon, then free tensor memory.
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (threadIdx.x == 0) {
            __threadfence_block();
            *scan_done_flag = 1;
        }
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace dsa_litetopk

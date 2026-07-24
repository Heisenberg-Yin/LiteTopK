// LiteTopK DSA scoring kernel V3 — H100 (SM90) port of sm100_dsa_litetopk.cuh.
//
// What carries over from the B200 V3 "hybrid" unchanged (arch-agnostic):
//   * V1 NON-persistent KV-split scheduling: blockIdx.x = q-block,
//     blockIdx.y = KV split window (fills all 132 SMs on tiny-Q vLLM chunks);
//   * LiteTopK sparse epilogue: strided gate reload (GATE_STRIDE prefetch),
//     warp-local candidate queues with REGISTER fill counts (qn_reg),
//     redux.sync row-union pruning, batched-vote emit, streaming cand stores;
//   * spare-warp threshold-refresh daemon (2 idle producer-warpgroup warps,
//     progress polling + __nanosleep) + per-CTA smem histogram feed;
//   * probe compaction (magic mul-shift) deferred to drain; ragged-Q padded
//     rows forced to an empty KV range; interior-block range-check elision;
//   * setmaxnreg register redistribution (Hopper-native feature).
//
// What is REPLACED (no SM90 equivalent for the Blackwell features):
//   * tcgen05.mma (UMMA) + dedicated UMMA-issue warp  ->  WGMMA (m64) issued
//     by the math warpgroups themselves (register accumulators, commit/wait
//     fencing). The full_umma/empty_umma barrier pair disappears.
//   * TMEM accumulators + per-row 32dp32b tcgen05.ld  ->  wgmma register
//     fragments. The B200 "1 math thread = 1 KV row" mapping becomes the
//     deep_gemm SM90 layout: each thread holds 2 KV rows (lane/4 and
//     lane/4+8 within its warp's 16 rows) with its own kNumHeads/4 column
//     subset; the head reduction needs a quad shfl_xor, after which the 4
//     lanes of a quad hold the SAME two row scores -> emit elects lane%4==0
//     representatives (8+8 candidates per warp ballot instead of 32).
//   * "early UMMA release" / tight tcgen05 fencing  ->  warpgroup_wait<0>
//     (all-at-once; SM90 cannot release the accumulator per row-pair).
//
// Scoring math (WGMMA + per-head fmaxf(.,0)*weight + weighted sum + per-KV
// scale + quad reduction) is taken VERBATIM from
// deep_gemm/impls/sm90_fp8_mqa_logits.cuh (DeepGEMM 2.5, the vLLM-pinned
// version) — the same generation discipline as the B200 V3 kernel.
//
// Supported gate variants: default (negated-threshold compare) and
// DSA_BUCKET_GATE4 (bucket-space float end-to-end, the shipped B200 default).
// The B200 A/B leftovers (BUCKET_GATE1-3, LANE/DIRECT emit, BULK_DRAIN,
// PERSIST, HIST_DEFER, *_NULL diagnostics) are NOT ported — see the #error
// guards below and kernels/h100/README.md for why.

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace dsa_litetopk {

using namespace deep_gemm;

#if defined(DSA_PERSIST) || defined(DSA_EMIT_NULL) \
    || defined(DSA_GATE_NULL) || defined(DSA_HIST_DEFER) || defined(DSA_BUCKET_GATE) \
    || defined(DSA_BUCKET_GATE2)
#error "SM90 port: only the default gate and DSA_BUCKET_GATE4 are ported (see kernels/h100/README.md)"
#endif

#ifndef DSA_WARP_QUEUE_CAP
#define DSA_WARP_QUEUE_CAP 64
#endif

#ifndef DSA_REFRESH_STRIDE
#define DSA_REFRESH_STRIDE 16
#endif

#ifndef DSA_GATE_STRIDE
#define DSA_GATE_STRIDE 16
#endif

#define DSA_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define DSA_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm90_dsa_litetopk(const uint32_t seq_len, const uint32_t seq_len_kv,
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
                         float* __restrict__ cand_val,         // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_idx,       // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_cnt,       // [seq_len]
                         const uint32_t cand_cap,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(BLOCK_KV == kNumMathWarpGroups * WGMMA::M, "BLOCK_KV must be 64 rows per math warpgroup");
    DG_STATIC_ASSERT(kNumQStages == 1, "V1 scheduling holds one q-block per CTA");

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Layout mirrors the B200 kernel: q stages, weights, kv stages, kv scales,
    // barriers, aux slots, warp queues, per-CTA refresh histogram.
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
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });

    // Aux slot 0 kept reserved so the host smem accounting matches the B200
    // layout shape (it held the TMEM pointer there; SM90 has no TMEM).
    auto aux_ptr = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2);
    auto scan_done_flag = reinterpret_cast<volatile int*>(aux_ptr + 1);
    auto kv_progress_ptr = reinterpret_cast<volatile int*>(aux_ptr + 2);
    auto warpq_count = reinterpret_cast<int32_t*>(aux_ptr + 4);
    auto warpq_val = reinterpret_cast<float*>(warpq_count + kNumMathWarps * BLOCK_Q);
    auto warpq_idx = reinterpret_cast<int32_t*>(warpq_val + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);
    // Per-CTA refresh histogram (BLOCK_Q x num_buckets) — same single-scanner
    // optimization as B200: when num_kv_splits == 1 the per-candidate
    // histogram feed goes to smem instead of RED.GLOBAL. A racing read can
    // only UNDERcount -> looser gate -> recall-safe.
    auto smem_hist = reinterpret_cast<int32_t*>(warpq_idx + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);

    DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64, "Invalid threads");
    if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        *scan_done_flag = 0;
        *kv_progress_ptr = 0;
        cutlass::arch::fence_barrier_init();
    }
    const bool hist_in_smem = (num_kv_splits == 1) && (refresh_every > 0) &&
                              (refresh_every != 0x7fffffff);
    if (hist_in_smem) {
        for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets; idx += blockDim.x)
            smem_hist[idx] = 0;
    }
    __syncthreads();

#ifdef DSA_REGS_224
    // SM90 budget: math warpgroups hold the WGMMA accumulator in REGISTERS
    // (kNumAccum = BLOCK_Q*kNumHeads/2 per thread), unlike B200 where it
    // lives in TMEM — so "spend everything on math" tops out lower here.
    // 224x256 + 32x128 = 61440 (4096 slack).
    constexpr uint32_t kNumSpecializedRegisters = 32;
    constexpr uint32_t kNumMathRegisters = 224;
#else
    // Default: 184/40 (51 200 total) — accum(64) + weights(8/q-row) +
    // epilogue state fit without spills; bump via DSA_REGS_224 if NCU says
    // otherwise on real hardware.
    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 184;
#endif

    // V1 KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
    // contiguous KV sub-window. Split boundaries are BLOCK_KV-aligned.
    const uint32_t block_q_idx = blockIdx.x;
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

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        if (cute::elect_one_sync()) {
            if (block_q_idx < num_q_blocks) {
            // Q + weights once for this q-block.
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[0], smem_q[0], 0, block_q_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[0], smem_weights[0], 0, block_q_idx * BLOCK_Q);
            full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);

            CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                        smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                          smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1 or warp_idx == kSpecWarpStart + 2) {
        // Spare-warp threshold-refresh daemon — verbatim B200 V1 semantics
        // (fixed rows, free-running, progress polling). On B200 these were
        // spec warps 2/3 (warp 1 was the UMMA issuer); with WGMMA issued by
        // the math warpgroups the daemon moves up to warps 1/2.
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        const bool in_scan_refresh = (refresh_every > 0 && refresh_every != 0x7fffffff);
        if (in_scan_refresh && block_q_idx < num_q_blocks) {
            const uint32_t spare_id = warp_idx - (kSpecWarpStart + 1);  // 0 or 1
            const auto refresh_row = [&](const uint32_t row) {
                if (row >= seq_len) return;
                const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
                const int32_t* srow = smem_hist + (row - block_q_idx * BLOCK_Q) * num_buckets;
                int carry = 0;
                int found = static_cast<int>(num_buckets) - 1;
                bool done = false;
                for (uint32_t base = 0; base < num_buckets && !done; base += 32) {
                    uint32_t b = base + lane_idx;
                    int v = (b < num_buckets) ? brow[b] : 0;
                    if (hist_in_smem && b < num_buckets) v += srow[b];
                    int prefix = v;
                    #pragma unroll
                    for (int off = 1; off < 32; off <<= 1) {
                        int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
                        if (static_cast<int>(lane_idx) >= off) prefix += nsh;
                    }
                    int incl = carry + prefix;
                    bool hit = (b < num_buckets) && (incl >= static_cast<int>(topk)) &&
                               (incl - v < static_cast<int>(topk));
                    unsigned hm = __ballot_sync(0xffffffffu, hit);
                    if (hm) { found = static_cast<int>(base) + (__ffs(hm) - 1); done = true; }
                    else    { carry += __shfl_sync(0xffffffffu, prefix, 31); }
                }
                if (lane_idx == 0 && found < th_bucket[row]) th_bucket[row] = found;
            };
            int last_prog = 0;
            while (true) {
                const int done = *scan_done_flag;
                const int prog = *kv_progress_ptr;
                if (prog > last_prog) {
                    for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                        refresh_row(block_q_idx * BLOCK_Q + r);
                    last_prog = prog;
                } else if (done) {
                    for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                        refresh_row(block_q_idx * BLOCK_Q + r);
                    break;
                } else {
                    __nanosleep(256);
                }
            }
        }
    } else if (warp_idx >= kSpecWarpStart + 3) {
        // Unused producer-warpgroup warp.
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // deep_gemm SM90 fragment geometry: warp owns 16 KV rows; each thread
        // holds rows (lane/4) and (lane/4 + 8) of them, with the kNumHeads/4
        // head columns selected by lane%4. After the quad shfl_xor reduction
        // the 4 lanes of a quad carry the SAME two row scores.
        const auto warpgroup_idx = warp_idx / 4;
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto warp_offset = warp_idx * 16;
        const auto v_0_offset = lane_idx / 4 + 0;
        const auto v_1_offset = lane_idx / 4 + 8;
        const bool lane_emit = ((lane_idx & 3) == 0);

        float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
        uint32_t kstart_reg[BLOCK_Q], kspan_reg[BLOCK_Q];  // unsigned range-check trick
        int gate_reg[BLOCK_Q];
        const unsigned FULL = 0xffffffffu;

        if (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
            full_q_barriers[0]->wait(0);

            // Weights into registers (once per CTA): only this thread's
            // kNumHeads/4 column subset — cheaper than the B200 full row.
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[0] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }
            // Queue fill counts are warp-uniform: every lane tracks them
            // redundantly in registers (qn_reg) — no smem bookkeeping, no
            // shfl broadcast on the hot emit path (the B200 V3 win).
            int qn_reg[BLOCK_Q];
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
                o_reg[i] = origin[rq];
                inv_reg[i] = inv_delta[rq];
#ifdef DSA_BUCKET_GATE4
                // GATE4 (shipped B200 default): bucket-space FLOAT end to end.
                // bq = v' + c0 where v' accumulates with -inv folded into the
                // register weights; gate = INT compare of bq's bits vs edge
                // float(g+1) bits. cand_val stores bq itself. vth_reg = c0 =
                // -o*inv; o_reg is repurposed to hold the edge float.
                vth_reg[i] = -o_reg[i] * inv_reg[i];
                o_reg[i] = 0.0f;  // gate closed until the first consume
#endif
                gate_reg[i] = cute::numeric_limits<int32_t>::max();
                qn_reg[i] = 0;
                kstart_reg[i] = seq_k_start[i];
                kspan_reg[i] = seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
            }
#ifdef DSA_BUCKET_GATE4
            // Fold -inv into the register weights: the whole ReLU-weighted
            // chain then accumulates directly in bucket units.
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
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

            // Gate PREFETCH (B200 V3): consume the th_bucket value fetched one
            // GATE_STRIDE window earlier; issue the next window's load right
            // after. One-window-stale is recall-safe (refresh only TIGHTENS).
            int th_pf[BLOCK_Q];
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i)
                th_pf[i] = __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + i, seq_len - 1));

            // Drain a (warp,row) queue segment to the global candidate
            // buffer; probe index mapping lives HERE (32 lanes retire the
            // queue in parallel), not on the insert path.
            const auto drain_queue = [&](const uint32_t row_q,
                                         const uint32_t queue_base, const int qn,
                                         const int base) {
                const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                for (int t = static_cast<int>(lane_idx); t < qn; t += 32) {
                    const float x = warpq_val[queue_base + t];
                    uint32_t kvo = static_cast<uint32_t>(warpq_idx[queue_base + t]);
                    if (probe_group != 0) {
                        // compacted -> original position; exact c/probe_group
                        // via magic mul-shift
                        const uint32_t sup = (uint32_t)(((uint64_t)kvo * probe_magic) >> 42);
                        kvo += min((sup + 1) * 64u, probe_add_max);
                    }
                    const int w = base + t;
                    if (w < static_cast<int>(cand_cap)) {
                        DSA_ST_CAND_VAL(cand_val[out_base + w], x);
                        DSA_ST_CAND_IDX(cand_idx[out_base + w], static_cast<int32_t>(kvo));
                    }
                }
            };

            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                if ((kv_block_idx % DSA_GATE_STRIDE) == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const int g = th_pf[i];
                        if (g != gate_reg[i]) {
                            gate_reg[i] = g;
#ifdef DSA_BUCKET_GATE4
                            // edge = float(g+1): exact for small ints.
                            o_reg[i] = static_cast<float>(g + 1);
#else
                            // Negated threshold: emit <=> x < o+(g+1)/inv
                            // <=> v > vth (v is the positive reduced score,
                            // one FSETP on the hot path; the B200 FFMA sign
                            // trick has nothing to absorb here because the
                            // quad reduction already produced the final v).
                            vth_reg[i] = -(o_reg[i] + static_cast<float>(g + 1) / inv_reg[i]);
#endif
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i)
                        th_pf[i] = __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + i, seq_len - 1));
                }

                float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                // WGMMA scoring — verbatim deep_gemm sm90_fp8_mqa_logits.
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[0] + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over heads + gate, all rows first (mirrors the B200
                // two-phase structure so collectives stay out of the reduce).
                const auto kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

                uint32_t p0_bits = 0, p1_bits = 0;
                float x0_row[BLOCK_Q], x1_row[BLOCK_Q];

#ifdef DSA_BUCKET_GATE4
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float bq0 = v_0 + vth_reg[i];                                    \
                        const float bq1 = v_1 + vth_reg[i];                                    \
                        x0_row[i] = bq0;                                                       \
                        x1_row[i] = bq1;                                                       \
                        bool g0 = __float_as_int(bq0) < __float_as_int(o_reg[i]);              \
                        bool g1 = __float_as_int(bq1) < __float_as_int(o_reg[i]);              \
                        if constexpr (RC) {                                                    \
                            g0 = g0 and ((kv_offset + v_0_offset - kstart_reg[i]) < kspan_reg[i]); \
                            g1 = g1 and ((kv_offset + v_1_offset - kstart_reg[i]) < kspan_reg[i]); \
                        }                                                                      \
                        p0_bits |= g0 ? (1u << i) : 0u;                                        \
                        p1_bits |= g1 ? (1u << i) : 0u;
#else
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        x0_row[i] = -v_0;                                                      \
                        x1_row[i] = -v_1;                                                      \
                        bool g0 = v_0 > vth_reg[i];                                            \
                        bool g1 = v_1 > vth_reg[i];                                            \
                        if constexpr (RC) {                                                    \
                            g0 = g0 and ((kv_offset + v_0_offset - kstart_reg[i]) < kspan_reg[i]); \
                            g1 = g1 and ((kv_offset + v_1_offset - kstart_reg[i]) < kspan_reg[i]); \
                        }                                                                      \
                        p0_bits |= g0 ? (1u << i) : 0u;                                        \
                        p1_bits |= g1 ? (1u << i) : 0u;
#endif
                const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
                const bool interior = (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
                #define DSA_SCORE_ROWS(RANGE_CHECK)                                            \
                _Pragma("unroll")                                                              \
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {                                      \
                    auto shifted_accum = accum + i * kNumAccumPerReduce;                       \
                    const auto transform = [&](const uint32_t& j) {                            \
                        return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)]; \
                    };                                                                         \
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};   \
                    _Pragma("unroll")                                                          \
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {                            \
                        _Pragma("unroll")                                                      \
                        for (uint32_t k = 0; k < 4; k ++)                                      \
                            sum[k] += transform(j * 4 + k);                                    \
                    }                                                                          \
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;                                \
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;                                \
                    _Pragma("unroll")                                                          \
                    for (uint32_t j = 0; j < 2; ++ j) {                                        \
                        const auto offset = static_cast<int>(1u << j);                         \
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);                      \
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);                      \
                    }                                                                          \
                    DSA_SCORE_GATE(i, RANGE_CHECK)                                             \
                }
                if (interior) { DSA_SCORE_ROWS(false) } else { DSA_SCORE_ROWS(true) }
                #undef DSA_SCORE_ROWS
                #undef DSA_SCORE_GATE

                // Emit phase. Gate bits are quad-redundant (all 4 lanes of a
                // quad agree); only lane%4==0 carries a distinct candidate,
                // so ballots have at most 8+8 participants per row.
                if (__any_sync(FULL, p0_bits | p1_bits)) {
                    const uint32_t rows_union = __reduce_or_sync(FULL, p0_bits | p1_bits);
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        if (rows_union & (1u << i)) {
                            const bool e0 = lane_emit and ((p0_bits >> i) & 1u);
                            const bool e1 = lane_emit and ((p1_bits >> i) & 1u);
                            const unsigned m0 = __ballot_sync(FULL, e0);
                            const unsigned m1 = __ballot_sync(FULL, e1);
                            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                            const int cnt = __popc(m0) + __popc(m1);
                            const uint32_t queue_base = (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
                            int qn = qn_reg[i];
                            if (qn + cnt > static_cast<int>(DSA_WARP_QUEUE_CAP)) {
                                int base = 0;
                                if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
                                base = __shfl_sync(FULL, base, 0);
                                drain_queue(row_q, queue_base, qn, base);
                                qn = 0;
                                __syncwarp(FULL);  // queue slots reusable
                            }
                            const unsigned below = (1u << lane_idx) - 1u;
                            const auto feed_hist = [&](const float x) {
                                if (refresh_every > 0) {
#ifdef DSA_BUCKET_GATE4
                                    // one F2I off the stored bucket float —
                                    // bucket-identical to the gate.
                                    int braw = static_cast<int>(x);
#else
                                    int braw = static_cast<int>((x - o_reg[i]) * inv_reg[i]);
#endif
                                    int b = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                                    if (hist_in_smem) {
                                        atomicAdd(smem_hist + i * num_buckets + b, 1);
                                    } else {
                                        atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], 1);
                                    }
                                }
                            };
                            if (e0) {
                                const int pos = qn + __popc(m0 & below);
                                warpq_val[queue_base + pos] = x0_row[i];
                                warpq_idx[queue_base + pos] = static_cast<int32_t>(kv_offset + v_0_offset);
                                feed_hist(x0_row[i]);
                            }
                            if (e1) {
                                const int pos = qn + __popc(m0) + __popc(m1 & below);
                                warpq_val[queue_base + pos] = x1_row[i];
                                warpq_idx[queue_base + pos] = static_cast<int32_t>(kv_offset + v_1_offset);
                                feed_hist(x1_row[i]);
                            }
                            qn_reg[i] = qn + cnt;
                        }
                    }
                }

                if (threadIdx.x == 0 && ((kv_block_idx + 1) % DSA_REFRESH_STRIDE) == 0) {
                    __threadfence_block();
                    *kv_progress_ptr = static_cast<int>(kv_block_idx + 1);
                }
            }

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
                    drain_queue(row_q, queue_base, qn, base);
                }
            }

            empty_q_barriers[0]->arrive();
        }

        // Signal the refresh daemon.
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (threadIdx.x == 0) {
            __threadfence_block();
            *scan_done_flag = 1;
        }
    }
}

} // namespace dsa_litetopk

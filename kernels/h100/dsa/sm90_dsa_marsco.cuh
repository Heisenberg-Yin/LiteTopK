// Route-A stage 2: deep_gemm sm90_fp8_mqa_logits with a SAMPLE+WRITE epilogue.
//
// The ReLU-MQA scoring math (WGMMA + per-head fmaxf(.,0)*weight + weighted sum) is
// COPIED VERBATIM from deep_gemm/impls/sm90_fp8_mqa_logits.cuh. The ONLY change is the
// store block at the end: instead of writing every logit to a dense [Q,S] buffer, we
// write only logits >= a per-row threshold row_th[q] into a compact candidate buffer
// (cand_val/cand_idx with per-row atomic counter cand_cnt). The threshold is precomputed
// by running deep_gemm on a small chunk_spread sample of KV positions.
//
// Namespaced dsa_fused:: so it never clashes with the original template.

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

namespace dsa_marsco {

using namespace deep_gemm;

#ifndef DSA_MERGE_BCOUNT
#define DSA_MERGE_BCOUNT 0
#endif

#ifndef DSA_WARP_QUEUE
#define DSA_WARP_QUEUE 1
#endif

#ifdef DENSE_WRITE
#undef DSA_WARP_QUEUE
#define DSA_WARP_QUEUE 0
#endif

#ifndef DSA_WARP_QUEUE_CAP
#define DSA_WARP_QUEUE_CAP 32
#endif

// Candidate stores are write-once (never re-read by this kernel), so route them
// through streaming (evict-first) stores to keep L2 residency for the reused
// KV/scale data instead of polluting it with dead candidates.
#define DSA_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define DSA_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_dsa_marsco(const uint32_t seq_len, const uint32_t seq_len_kv,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         const float* __restrict__ origin,   // [seq_len], bucket origin over x=-score
                         const float* __restrict__ inv_delta, // [seq_len], bucket scale
                         int32_t* __restrict__ th_bucket,
                         int32_t* __restrict__ bcount,
                         const uint32_t num_buckets,
                         const uint32_t topk,
                         const uint32_t refresh_every,
                         const uint32_t num_kv_splits,
                         float* __restrict__ cand_val,       // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_idx,     // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_cnt,     // [seq_len]
                         const uint32_t cand_cap,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    static constexpr uint32_t kNumMathWarpgroups = kNumMathThreads / 128;
    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    static constexpr uint32_t kWarpgroupsPerKV = BLOCK_KV / WGMMA::M;
    static constexpr uint32_t kNumKVGroups = kNumMathWarpgroups / kWarpgroupsPerKV;
    static constexpr uint32_t kThreadsPerKVGroup = kWarpgroupsPerKV * 128;

    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(BLOCK_KV % WGMMA::M == 0, "BLOCK_KV must be an integer number of WGMMA M tiles");
    DG_STATIC_ASSERT(kWarpgroupsPerKV > 0, "Invalid BLOCK_KV/WGMMA::M combination");
    DG_STATIC_ASSERT(kNumMathWarpgroups % kWarpgroupsPerKV == 0, "Math warpgroups must divide evenly into KV groups");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);

    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
#if DSA_WARP_QUEUE
    auto warpq_count = reinterpret_cast<int32_t*>(barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2));
    auto warpq_val = reinterpret_cast<float*>(warpq_count + kNumMathWarps * BLOCK_Q);
    auto warpq_idx = reinterpret_cast<int32_t*>(warpq_val + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);
#endif

    const bool is_tma_load_warp = kNumMathThreads <= threadIdx.x and threadIdx.x < kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            // Each KV block is consumed by one math group.  A group may contain
            // one WGMMA warpgroup (BLOCK_KV=64) or two (BLOCK_KV=128).
            empty_kv_barriers[i]->init(kThreadsPerKVGroup);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // KV-split scheduling: blockIdx.x selects the q-block (one per CTA, no
    // persistence); blockIdx.y selects a contiguous KV sub-range so all SMs
    // work on the same queries in parallel. Each CTA scans only its window
    // [split_lo, split_hi) intersected with the per-row [cu_start, cu_end).
    const uint32_t block_q_idx = blockIdx.x;
    const uint32_t kv_split = blockIdx.y;
    uint32_t q_iter_idx = 0;
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        // Intersect [start,end) with this CTA's KV split window. Split boundaries
        // are BLOCK_KV-aligned so no KV block is processed by two CTAs.
        const uint32_t total_blocks = math::ceil_div(seq_len_kv, BLOCK_KV);
        const uint32_t blocks_per_split = math::ceil_div(total_blocks, num_kv_splits);
        const uint32_t split_lo = kv_split * blocks_per_split * BLOCK_KV;
        const uint32_t split_hi = min((kv_split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv);
        start = start / 4 * 4;
        if (start < split_lo) start = split_lo;
        if (end > split_hi) end = split_hi;
        uint32_t nkv = (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
        return {(q_iter_offset) % kNumQStages,
                ((q_iter_offset) / kNumQStages) & 1,
                start, nkv};
    };

    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1
        };
    };

    cudaGridDependencySynchronize();

    if (threadIdx.x >= kNumMathThreads) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

#ifdef PROD_DENSE
        // DENSE producer (matches deep_gemm exactly): single warp survives, single
        // lane persistently schedules, NO per-KV-block bar.sync, NO refresh.
        if (not is_tma_load_warp)
            return;
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks) {
            issue_tma_q(0, block_q_idx);
            CUTE_TIE_DECL(load_schedule(0), q_stage_idx, q_phase, kv_start, num_kv_blocks);
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
#else

        // Keep the ENTIRE producer warpgroup (128 threads = 4 warps) alive so the
        // periodic threshold refresh can run across all lanes, mirroring
        // simtopk_ready/tidal's warpgroup-0 producer refresh (no lock/qprev).
        const int prod_tid  = static_cast<int>(threadIdx.x - kNumMathThreads);  // 0..127
        const int prod_warp = prod_tid >> 5;   // 0..3  (one warp per query row)
        const int prod_lane = prod_tid & 31;   // 0..31 (parallel bucket scan)
        const bool load_lane = is_tma_load_warp && cute::elect_one_sync();

        // Warp-parallel refresh: warp `prod_warp` owns query row r; its 32 lanes
        // do a chunked inclusive prefix-sum over that row's emitted-bucket
        // histogram (global bcount) and pick the first bucket whose prefix count
        // reaches topk, then tighten th_bucket monotonically. Reads a possibly
        // stale bcount, which only makes the gate more conservative (safe).
        const auto do_refresh = [&]() {
            const int r = prod_warp;
            if (r < static_cast<int>(BLOCK_Q)) {
                const uint32_t row = block_q_idx * BLOCK_Q + r;
                if (row < seq_len) {
                    const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
                    int carry = 0;
                    int found = static_cast<int>(num_buckets) - 1;
                    bool done = false;
                    for (uint32_t base = 0; base < num_buckets && !done; base += 32) {
                        uint32_t b = base + prod_lane;
                        int v = (b < num_buckets) ? brow[b] : 0;
                        int prefix = v;
                        #pragma unroll
                        for (int off = 1; off < 32; off <<= 1) {
                            int n = __shfl_up_sync(0xffffffffu, prefix, off);
                            if (prod_lane >= off) prefix += n;
                        }
                        int incl = carry + prefix;
                        bool hit = (b < num_buckets) && (incl >= static_cast<int>(topk)) &&
                                   (incl - v < static_cast<int>(topk));
                        unsigned hm = __ballot_sync(0xffffffffu, hit);
                        if (hm) {
                            found = static_cast<int>(base) + (__ffs(hm) - 1);
                            done = true;
                        } else {
                            carry += __shfl_sync(0xffffffffu, prefix, 31);
                        }
                    }
                    if (prod_lane == 0 && found < th_bucket[row]) th_bucket[row] = found;
                }
            }
        };

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (block_q_idx < num_q_blocks) {
            if (load_lane)
                issue_tma_q(0, block_q_idx);

            // One q-block, one KV-split pass. Only the elected load lane issues
            // TMA; all 128 producer threads lockstep per KV block so the periodic
            // refresh reads a consistent bcount snapshot.
            CUTE_TIE_DECL(load_schedule(0), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                if (load_lane) {
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                asm volatile("bar.sync 1, %0;\n" :: "r"(kNumTMAThreads) : "memory");
                if (refresh_every > 0 and ((kv_block_idx + 1) % refresh_every) == 0) {
                    do_refresh();
                }
            }
        }
#endif
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = ptx::get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];
        float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
        int gate_reg[BLOCK_Q];

#if DSA_WARP_QUEUE
        DG_STATIC_ASSERT(DSA_WARP_QUEUE_CAP >= 16, "DSA_WARP_QUEUE_CAP must hold one warp/block row");
        if (lane_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i)
                warpq_count[warp_idx * BLOCK_Q + i] = 0;
        }
        const auto flush_warpq = [&](const uint32_t row_i, const uint32_t row_q, const uint64_t out_base) {
            const uint32_t count_slot = warp_idx * BLOCK_Q + row_i;
            const uint32_t queue_base = count_slot * DSA_WARP_QUEUE_CAP;
            int qn = 0;
            if (lane_idx == 0) qn = warpq_count[count_slot];
            qn = __shfl_sync(0xffffffffu, qn, 0);
            if (qn > 0) {
                int base = 0;
                if (lane_idx == 0) {
                    base = atomicAdd(cand_cnt + row_q, qn);
                    warpq_count[count_slot] = 0;
                }
                base = __shfl_sync(0xffffffffu, base, 0);
                for (int t = static_cast<int>(lane_idx); t < qn; t += 32) {
                    const int w = base + t;
                    if (w < static_cast<int>(cand_cap)) {
                        DSA_ST_CAND_VAL(cand_val[out_base + w], warpq_val[queue_base + t]);
                        DSA_ST_CAND_IDX(cand_idx[out_base + w], warpq_idx[queue_base + t]);
                    }
                }
            }
            __syncwarp(0xffffffffu);
        };
#endif

        // Ping-pong: math warpgroups are partitioned into KV groups. Each group
        // owns a full BLOCK_KV-row block and processes alternating blocks, so while
        // one group runs its divergent gate+emit epilogue the other group's
        // WGMMA issues on the tensor cores (SM scheduler overlaps them).
        const uint32_t group_id      = warpgroup_idx / kWarpgroupsPerKV;
        const uint32_t wg_in_group   = warpgroup_idx - group_id * kWarpgroupsPerKV;
        const uint32_t warp_in_group = warp_idx - group_id * kWarpgroupsPerKV * 4;

        const auto warp_offset = warp_in_group * 16;        // 4 warps per WG, 16 rows per warp
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        if (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(0), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t rq = block_q_idx * BLOCK_Q + i;
                o_reg[i]   = origin[rq];
                inv_reg[i] = inv_delta[rq];
            }

            for (uint32_t kv_block_idx = group_id; kv_block_idx < num_kv_blocks; kv_block_idx += kNumKVGroups) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    gate_reg[i] = th_bucket[block_q_idx * BLOCK_Q + i];
                    // emit <=> floor((x-o)*inv) <= gate  <=>  x < o + (gate+1)/inv
                    vth_reg[i] = o_reg[i] + static_cast<float>(gate_reg[i] + 1) / inv_reg[i];
                }

                float scale_kv_0 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                float scale_kv_1 = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                DG_STATIC_ASSERT(BLOCK_KV == kWarpgroupsPerKV * WGMMA::M, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = mma::sm90::make_smem_desc(
                        smem_kv[kv_stage_idx] + (wg_in_group * WGMMA::M) * kHeadDim + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = mma::sm90::make_smem_desc(
                        smem_q[q_stage_idx] + k * WGMMA::K,
                        mma::sm90::to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();

                empty_kv_barriers[kv_stage_idx]->arrive();

                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto transform = [&](const uint32_t& j) {
                        return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                    };

                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k ++)
                            sum[k] += transform(j * 4 + k);
                    }
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    // ---- SAMPLE+WRITE epilogue (only change vs deep_gemm) ----
                    // Emit only logits >= per-row threshold into a compact candidate buffer.
                    // NOTE: after the __shfl_xor reduction, the 4 lanes in each group of 4
                    // (lane & 3) hold the SAME reduced value and the SAME column. The original
                    // dense store writes redundantly (same addr/value = harmless), but our
                    // atomic emit must elect ONE lane per group or we'd create 4 duplicates.
                    // ---- marsco-style WARP-aggregated sparse epilogue ----
                    // The 4 lanes in each group-of-4 hold the SAME reduced score/col, so
                    // only lane&3==0 carries a distinct candidate (8 v_0 + 8 v_1 per warp).
                    // Instead of a per-lane atomicAdd(cand_cnt) + scattered store, the warp
                    // does ONE atomicAdd(cand_cnt[row], popc) reserving a contiguous run,
                    // then each passer writes at base + its ballot prefix (coalesced-ish).
#ifndef DENSE_WRITE
                    const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                    const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                    const float vth = vth_reg[i];
                    const unsigned FULL = 0xffffffffu;

                    const uint32_t col0 = kv_offset + v_0_offset;
                    const float x0 = -v_0;
                    const bool g0 = seq_k_start[i] <= col0 and col0 < seq_k_end[i] and x0 < vth;
                    const uint32_t col1 = kv_offset + v_1_offset;
                    const float x1 = -v_1;
                    const bool g1 = seq_k_start[i] <= col1 and col1 < seq_k_end[i] and x1 < vth;

                    // Gate first across all 32 lanes. Only if some lane group
                    // has a passing score do we enter the sparse emit epilogue;
                    // inside the epilogue, elect one representative per 4-lane
                    // duplicate group to avoid appending repeated columns.
                    const unsigned gate_m0 = __ballot_sync(FULL, g0);
                    const unsigned gate_m1 = __ballot_sync(FULL, g1);
                    const bool lane_emit = ((lane_idx & 3) == 0);
                    const bool p0 = lane_emit and g0;
                    const bool p1 = lane_emit and g1;

                    if ((gate_m0 | gate_m1) != 0) {
                    const unsigned m0 = __ballot_sync(FULL, p0);
                    const unsigned m1 = __ballot_sync(FULL, p1);
                    const int cnt = __popc(m0) + __popc(m1);
                    if (cnt > 0) {
                        const uint32_t count_slot = warp_idx * BLOCK_Q + i;
                        const uint32_t queue_base = count_slot * DSA_WARP_QUEUE_CAP;
                        int qn = 0;
                        if (lane_idx == 0) qn = warpq_count[count_slot];
                        qn = __shfl_sync(FULL, qn, 0);
                        if (qn > 0 && qn + cnt > static_cast<int>(DSA_WARP_QUEUE_CAP)) {
                            int base = 0;
                            if (lane_idx == 0) {
                                base = atomicAdd(cand_cnt + row_q, qn);
                                warpq_count[count_slot] = 0;
                            }
                            base = __shfl_sync(FULL, base, 0);
                            for (int t = static_cast<int>(lane_idx); t < qn; t += 32) {
                                const int w = base + t;
                                if (w < static_cast<int>(cand_cap)) {
                                    DSA_ST_CAND_VAL(cand_val[out_base + w], warpq_val[queue_base + t]);
                                    DSA_ST_CAND_IDX(cand_idx[out_base + w], warpq_idx[queue_base + t]);
                                }
                            }
                            qn = 0;
                            __syncwarp(FULL);
                        }

                        const unsigned below = (lane_idx == 0) ? 0u : ((1u << lane_idx) - 1u);
                        if (p0) {
                            const int pos = qn + __popc(m0 & below);
                            warpq_val[queue_base + pos] = x0;
                            warpq_idx[queue_base + pos] = static_cast<int32_t>(col0);
                            if (refresh_every > 0) {
                                int braw0 = static_cast<int>((x0 - o_reg[i]) * inv_reg[i]);
                                int b0 = braw0 < 0 ? 0 : (braw0 > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw0);
#if DSA_MERGE_BCOUNT
                                unsigned peers0 = __match_any_sync(m0, b0);
                                int leader0 = __ffs(peers0) - 1;
                                if (static_cast<int>(lane_idx) == leader0)
                                    atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b0], __popc(peers0));
#else
                                atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b0], 1);
#endif
                            }
                        }
                        if (p1) {
                            const int pos = qn + __popc(m0) + __popc(m1 & below);
                            warpq_val[queue_base + pos] = x1;
                            warpq_idx[queue_base + pos] = static_cast<int32_t>(col1);
                            if (refresh_every > 0) {
                                int braw1 = static_cast<int>((x1 - o_reg[i]) * inv_reg[i]);
                                int b1 = braw1 < 0 ? 0 : (braw1 > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw1);
#if DSA_MERGE_BCOUNT
                                unsigned peers1 = __match_any_sync(m1, b1);
                                int leader1 = __ffs(peers1) - 1;
                                if (static_cast<int>(lane_idx) == leader1)
                                    atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b1], __popc(peers1));
#else
                                atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b1], 1);
#endif
                            }
                        }
                        if (lane_idx == 0) warpq_count[count_slot] = qn + cnt;
                        __syncwarp(FULL);
                    }
                    }
#else
                    // DENSE-WRITE baseline: same KV-split scoring, write every logit
                    // into a dense [Q, S] buffer (cand_val reused, cand_cap=S). Match
                    // deep_gemm's epilogue EXACTLY: let ALL 32 lanes store. The 4 lanes
                    // in each group hold the SAME value/col, so the redundant writes hit
                    // identical addresses and the hardware coalesces them into full
                    // transactions -- electing lane&3==0 (8 lanes) instead serializes the
                    // store into poorly-coalesced 32B accesses (~3x slower).
                    {
                        const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                        const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                        const uint32_t col0 = kv_offset + v_0_offset;
                        const uint32_t col1 = kv_offset + v_1_offset;
                        if (seq_k_start[i] <= col0 and col0 < seq_k_end[i])
                            cand_val[out_base + col0] = v_0;
                        if (seq_k_start[i] <= col1 and col1 < seq_k_end[i])
                            cand_val[out_base + col1] = v_1;
                    }
#endif
                }
            }
#if DSA_WARP_QUEUE
             #pragma unroll
             for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                 const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                 if (row_q < seq_len) {
                     const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                     flush_warpq(i, row_q, out_base);
                 }
             }
#endif
            empty_q_barriers[q_stage_idx]->arrive();
        }
    }
}

} // namespace dsa_marsco

// LiteTopK DSA scoring kernel V2 (Blackwell / B200 SM100).
//
// The scoring loop is structurally COPIED from DeepGEMM 2.5's
// `sm100_fp8_mqa_logits.cuh` (commit 891d57b, the version vLLM 0.23 pins),
// which is a full generation faster than the old kernel our V1 port was based
// on: PERSISTENT q-block scheduling (grid = #SMs, Q pipeline prefetches the
// next block's Q/weights), per-row weights held in REGISTERS (loaded once per
// q-block, zero smem traffic in the KV loop), and per-row 32-element TMEM
// loads with early UMMA release.
//
// The ONLY change is the epilogue: instead of storing every logit to a dense
// [Q, S] buffer, scores are gated by a per-row bucket threshold and only the
// passers are written to compact candidate buffers (cand_val/cand_idx with a
// per-row atomic counter), exactly like the V1 LiteTopK kernel:
//   * batched-vote emit: per-row pass bits are collected in registers and the
//     divergent warp-queue push only runs after one warp-uniform __any_sync;
//   * strided gate reload (DSA_GATE_STRIDE) with fdiv-on-change;
//   * warp-local candidate queue (DSA_WARP_QUEUE_CAP) flushed per q-block;
//   * the two idle specialized warps run the background threshold-refresh
//     daemon (32-lane prefix sums over the per-row `bcount` histogram,
//     monotonically tightening `th_bucket`); with persistent scheduling the
//     math warps publish the CURRENT q-block so the daemon refreshes the rows
//     actually being scanned. A stale gate is merely conservative.
//
// Ragged Q (seq_len % BLOCK_Q != 0, the vLLM chunk case) is handled by forcing
// an empty KV range on the padded rows of the final q-block.

#pragma once

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

namespace dsa_marsco_v2 {

using namespace deep_gemm;

#ifndef DSA_WARP_QUEUE_CAP
#define DSA_WARP_QUEUE_CAP 32
#endif

#ifndef DSA_REFRESH_STRIDE
#define DSA_REFRESH_STRIDE 16
#endif

#ifndef DSA_GATE_STRIDE
#define DSA_GATE_STRIDE 16
#endif

// Streaming (evict-first) stores for write-once candidate data.
#define DSA_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define DSA_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_dsa_marsco_v2(const uint32_t seq_len, const uint32_t seq_len_kv,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         const float* __restrict__ origin,     // [seq_len]
                         const float* __restrict__ inv_delta,  // [seq_len]
                         int32_t* __restrict__ th_bucket,      // [seq_len]
                         int32_t* __restrict__ bcount,         // [seq_len, num_buckets]
                         const uint32_t num_buckets,
                         const uint32_t topk,
                         const uint32_t refresh_every,
                         float* __restrict__ cand_val,         // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_idx,       // [seq_len, cand_cap]
                         int32_t* __restrict__ cand_cnt,       // [seq_len]
                         const uint32_t cand_cap,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);

    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const auto sm_idx = blockIdx.x;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }

    // Shared memory configs
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    extern __shared__ __align__(512) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumMathWarpGroups;
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

    // TMA / UMMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i); });

    // TMEM allocation slot + LiteTopK daemon mailboxes + warp-local queues
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2);
    auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
    auto kv_progress_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 2);
    auto cur_qblock_ptr = reinterpret_cast<volatile uint32_t*>(tmem_ptr_in_smem + 3);
    auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
    auto warpq_val = reinterpret_cast<float*>(warpq_count + kNumMathWarps * BLOCK_Q);
    auto warpq_idx = reinterpret_cast<int32_t*>(warpq_val + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);

    // Initialize barriers (split across the two active specialized warps,
    // mirroring the upstream kernel)
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
        *cur_qblock_ptr = 0xffffffffu;
        cutlass::arch::fence_barrier_init();
    }
    if (warp_idx == kSpecWarpStart + 1) {
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations (as upstream)
    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;

    // Persistent block scheduler
    uint32_t block_q_idx = sm_idx, q_iter_idx = 0;
    const auto get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + kNumSMs, q_iter_idx + 1};
    };
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            seq_k_start[i] = cu_seq_len_k_start[q_idx];
            seq_k_end[i] = cu_seq_len_k_end[q_idx];
            if (block_q_idx * BLOCK_Q + i >= seq_len) {
                // Padded row of a ragged final q-block: empty range so its
                // (zero-filled TMA q) scores can never pass the gate.
                seq_k_start[i] = seq_len_kv;
                seq_k_end[i] = 0;
            }
            start = min(start, min(seq_k_start[i], seq_len_kv));
            end = max(end, min(seq_k_end[i], seq_len_kv));
        }
        // TMA alignment requirements for SF KV
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1,
                start, math::ceil_div(end - start, BLOCK_KV)};
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1
        };
    };

    // UMMA settings
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
    constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        const auto issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(&tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(&tensor_map_kv, full_kv_barriers[kv_stage_idx],
                                                            smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales, full_kv_barriers[kv_stage_idx],
                                              smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            full_q_barriers[q_stage_idx]->wait(q_phase);

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    empty_umma_barriers[i]->wait(((num_total_kv_blocks + kv_block_idx) & 1) ^ 1);
                    ptx::tcgen05_after_thread_sync();
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[q_stage_idx], 0, k * UMMA_K);
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            empty_q_barriers[q_stage_idx]->arrive();

            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
        // LiteTopK: background threshold-refresh daemon on the otherwise-idle
        // specialized warps. Math warps publish the current q-block and a
        // coarse KV progress counter; the daemon prefix-sums the published
        // rows' bcount histograms and monotonically tightens th_bucket.
        // Staleness in any mailbox only makes a gate MORE conservative.
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        const bool in_scan_refresh = (refresh_every > 0 && refresh_every != 0x7fffffff);
        if (in_scan_refresh) {
            const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
            const auto refresh_row = [&](const uint32_t row) {
                if (row >= seq_len) return;
                const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
                int carry = 0;
                int found = static_cast<int>(num_buckets) - 1;
                bool done = false;
                for (uint32_t base = 0; base < num_buckets && !done; base += 32) {
                    uint32_t b = base + lane_idx;
                    int v = (b < num_buckets) ? brow[b] : 0;
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
            int last_prog = -1;
            uint32_t last_q = 0xffffffffu;
            while (true) {
                const int done = *scan_done_flag;
                const uint32_t q = *cur_qblock_ptr;
                const int prog = *kv_progress_ptr;
                if (q != 0xffffffffu && (prog != last_prog || q != last_q)) {
                    for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                        refresh_row(q * BLOCK_Q + r);
                    last_prog = prog;
                    last_q = q;
                } else if (done) {
                    break;
                } else {
                    __nanosleep(256);
                }
            }
        }
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto tmem_start = warpgroup_idx * UMMA_N;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;

        auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr, float* accum) {
            constexpr int N = decltype(num_elems_c)::value;
            DG_STATIC_ASSERT(N == 32 or N == 64, "Unsupported TMEM load size");
            using Loader = cute::conditional_t<N == 32,
                cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::SM100_TMEM_LOAD_32dp32b64x>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
            }(cute::make_index_sequence<N>{});
            cutlass::arch::fence_view_async_tmem_load();
        };

        // Local register buffers (weights live in registers across the KV loop
        // -- the key upstream optimization)
        float weights[BLOCK_Q][kNumHeads];
        float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
        int gate_reg[BLOCK_Q];

        const unsigned FULL = 0xffffffffu;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights into registers (once per q-block)
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads; ++ j)
                    weights[i][j] = ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j);
            }

            // LiteTopK per-row gate state (once per q-block)
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
                o_reg[i] = origin[rq];
                inv_reg[i] = inv_delta[rq];
                gate_reg[i] = cute::numeric_limits<int32_t>::max();
            }
            // Reset this warp's candidate queues and publish the current block
            if (lane_idx == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i)
                    warpq_count[warp_idx * BLOCK_Q + i] = 0;
            }
            __syncwarp();
            if (threadIdx.x == 0)
                *cur_qblock_ptr = block_q_idx;

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Strided gate reload; the fdiv reruns only on actual change.
                if ((kv_block_idx % DSA_GATE_STRIDE) == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const int g = th_bucket[min(block_q_idx * BLOCK_Q + i, seq_len - 1)];
                        if (g != gate_reg[i]) {
                            gate_reg[i] = g;
                            // emit <=> floor((x-o)*inv) <= gate <=> x < o + (gate+1)/inv
                            vth_reg[i] = o_reg[i] + static_cast<float>(g + 1) / inv_reg[i];
                        }
                    }
                }

                float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

                full_umma_barriers[warpgroup_idx]->wait((num_total_kv_blocks + kv_block_idx) & 1);
                ptx::tcgen05_after_thread_sync();

                empty_kv_barriers[kv_stage_idx]->arrive();

                const auto kv_offset = kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

                uint32_t pass_bits = 0;
                float x_row[BLOCK_Q];

                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    float accum[kNumHeads];
                    tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads, accum);

                    if (i == BLOCK_Q - 1) {
                        ptx::tcgen05_before_thread_sync();
                        empty_umma_barriers[warpgroup_idx]->arrive();
                    }

                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);
                    const auto transform = [&](const uint32_t& j, const float2& sum) {
                        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                        auto b = make_float2(weights[i][j], weights[i][j + 1]);
                        return __ffma2_rn(a, b, sum);
                    };
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        sum_0 = transform(j, sum_0);
                        sum_1 = transform(j + 2, sum_1);
                    }
                    auto sum = __fadd2_rn(sum_0, sum_1);
                    const float v = scale_kv * (sum.x + sum.y);

                    // LiteTopK gate: collect per-row pass bits, no warp ops yet.
                    const float x = -v;
                    x_row[i] = x;
                    const bool g = seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i] and x < vth_reg[i];
                    pass_bits |= g ? (1u << i) : 0u;
                }

                // Batched-vote emit: enter the divergent machinery only when
                // some lane passed some row (warp-uniform vote).
                if (__any_sync(FULL, pass_bits)) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const bool g = (pass_bits >> i) & 1u;
                        const unsigned m = __ballot_sync(FULL, g);
                        if (m != 0) {
                            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                            const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                            const float x = x_row[i];
                            const int cnt = __popc(m);
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
                            if (g) {
                                const int pos = qn + __popc(m & below);
                                warpq_val[queue_base + pos] = x;
                                warpq_idx[queue_base + pos] = static_cast<int32_t>(kv_offset);
                                if (refresh_every > 0) {
                                    int braw = static_cast<int>((x - o_reg[i]) * inv_reg[i]);
                                    int b = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                                    atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], 1);
                                }
                            }
                            if (lane_idx == 0) warpq_count[count_slot] = qn + cnt;
                            __syncwarp(FULL);
                        }
                    }
                }

                // Publish scan progress for the refresh daemon (coarse).
                if (threadIdx.x == 0 && ((kv_block_idx + 1) % DSA_REFRESH_STRIDE) == 0) {
                    __threadfence_block();
                    *kv_progress_ptr = static_cast<int>(kv_block_idx + 1);
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Flush this q-block's warp queues before moving on.
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                if (row_q < seq_len) {
                    const uint32_t count_slot = warp_idx * BLOCK_Q + i;
                    const uint32_t queue_base = count_slot * DSA_WARP_QUEUE_CAP;
                    const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                    int qn = 0;
                    if (lane_idx == 0) qn = warpq_count[count_slot];
                    qn = __shfl_sync(FULL, qn, 0);
                    if (qn > 0) {
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
                        __syncwarp(FULL);
                    }
                }
            }

            empty_q_barriers[q_stage_idx]->arrive();

            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }

        // Signal the refresh daemon and free tensor memory.
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (threadIdx.x == 0) {
            __threadfence_block();
            *scan_done_flag = 1;
        }
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }
}

} // namespace dsa_marsco_v2

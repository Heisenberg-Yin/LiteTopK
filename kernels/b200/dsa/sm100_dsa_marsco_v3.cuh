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

namespace dsa_marsco_v3 {

using namespace deep_gemm;

#ifndef DSA_DIRECT_THRESH
// Adaptive emit: rows with fewer hits than this per warp per GATE_STRIDE
// window switch from the smem queue to direct ballot-group scatter.
#define DSA_DIRECT_THRESH 8
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
void sm100_dsa_marsco_v3(const uint32_t seq_len, const uint32_t seq_len_kv,
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
                         const uint32_t repair,             // certificate-repair pass
                         const int32_t* __restrict__ th_safe,     // exact fallback gate
                         const int32_t* __restrict__ seed_total,  // seeds emitted (<= th_safe)
                         const int32_t* __restrict__ seed_pred,   // seeds under the predicted gate
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

    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");

    if (repair) {
        // Certificate-repair pass. A row is CERTIFIED exact iff at least
        // topk genuine elements were counted STRICTLY below the predicted
        // gate (scan emissions + predicted-qualifying seeds) and nothing
        // overflowed the buffer: then the true top-k is fully inside the
        // candidate set. CTAs whose rows all certify exit before touching
        // any barrier or TMEM. Failed rows fall back to the exact subset
        // bound (th_safe, provably >= the true k-th, and provably yielding
        // cnt >= topk on the re-scan) with their counters rolled back to
        // the seed base. Caller runs this pass with refresh OFF so the
        // re-emitted counts cannot double-tighten anything.
        bool all_ok = true;
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const uint32_t r = blockIdx.x * BLOCK_Q + i;
            if (r >= seq_len) continue;
            const int c = cand_cnt[r];
            const bool ok = (c - seed_total[r] + seed_pred[r] >= static_cast<int>(topk))
                            && (c <= static_cast<int>(cand_cap));
            all_ok &= ok;
        }
        if (all_ok) return;
        if (threadIdx.x == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t r = blockIdx.x * BLOCK_Q + i;
                if (r >= seq_len) continue;
                const int c = cand_cnt[r];
                const bool ok = (c - seed_total[r] + seed_pred[r] >= static_cast<int>(topk))
                                && (c <= static_cast<int>(cand_cap));
                if (ok) {
                    th_bucket[r] = -(1 << 30);   // certified: gate emits nothing
                } else {
                    th_bucket[r] = th_safe[r];
                    cand_cnt[r] = seed_total[r];
                }
            }
            __threadfence();
        }
        __syncthreads();
    }

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

    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    auto full_umma_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups + i); });

    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * 2);
    auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
    auto kv_progress_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 2);
#ifdef DSA_PERSIST
    auto done_qb_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 3);
    auto ack_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 4);  // [2]
    auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 8);
#else
    auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
#endif
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
#ifdef DSA_HIST_DEFER
#if defined(DSA_BULK_DRAIN) || defined(DSA_LANE_EMIT) || defined(DSA_DIRECT_EMIT) \
    || defined(DSA_QUEUE_NULL) || defined(DSA_EMIT_NULL) || defined(DSA_PERSIST)
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
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                full_umma_barriers[i]->init(1);
                empty_umma_barriers[i]->init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    const bool hist_in_smem = (num_kv_splits == 1) && (refresh_every > 0) &&
                              (refresh_every != 0x7fffffff);
    if (hist_in_smem) {
        for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets; idx += blockDim.x)
            smem_hist[idx] = 0;
#ifdef DSA_HIST_DEFER
        if (threadIdx.x < BLOCK_Q) smem_safe[threadIdx.x] = 0;
#endif
    }
    __syncthreads();

#ifdef DSA_REGS_240
    // A/B round 2: 240/32 (= exactly 65536, zero slack) DEADLOCKED at boot --
    // the setmaxnreg redistribution cannot be satisfied at 100% file
    // utilization. 240/24 keeps the current total (64512, 1024 slack);
    // the specialized paths (TMA/governor/daemon) get squeezed 40->24 and
    // may spill -- realshape + bench arbitrate.
    constexpr uint32_t kNumSpecializedRegisters = 24;
    constexpr uint32_t kNumMathRegisters = 240;
#else
    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 232;
#endif

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
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    empty_umma_barriers[i]->wait((kvg & 1) ^ 1);
                    ptx::tcgen05_after_thread_sync();
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[0], 0, k * UMMA_K);
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N, k, runtime_instr_desc);
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i]));
                }
            }
            empty_q_barriers[0]->arrive();
#ifdef DSA_PERSIST
            kvt_base += num_kv_blocks;
#endif
        }
    } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
        // Spare-warp threshold-refresh daemon (V1 semantics: fixed rows).
        // NOTE: moving this refresh into the math warps' gate-reload point
        // (tidal-style C2) measured 12-13% SLOWER at 256K/512K here: unlike
        // tidal, this kernel has no register spill (setmaxnreg 232/40) and
        // sleeping was only 0.28 cyc/issue — the daemon overlaps well, while
        // inline refresh puts bcount global-read latency on the math warps'
        // critical path (a warpgroup hiccup every GATE_STRIDE blocks).
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        const bool in_scan_refresh = (refresh_every > 0 && refresh_every != 0x7fffffff);
#ifdef DSA_PERSIST
        if (in_scan_refresh) {
            const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
            DSA_QB_LOOP {
#else
        if (in_scan_refresh && block_q_idx < num_q_blocks) {
            const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
#endif
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
                        refresh_row(block_q_idx * BLOCK_Q + r);
                    last_prog = prog;
                } else if (done) {
                    for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
                        refresh_row(block_q_idx * BLOCK_Q + r);
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
                        refresh_row(block_q_idx * BLOCK_Q + r);
                    }
                    last_prog = prog;
                } else if (done) {
                    for (uint32_t r = spare_id, li = 0; r < BLOCK_Q; r += 2, ++li) {
#ifdef DSA_HIST_DEFER
                        drain_hist(r, li);
#endif
                        refresh_row(block_q_idx * BLOCK_Q + r);
                    }
                    break;
                } else {
                    __nanosleep(256);
                }
            }
        }
#endif
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

#if defined(DSA_BUCKET_GATE) && (defined(DSA_OLD_GATE) || defined(DSA_LANE_EMIT) \
    || defined(DSA_DIRECT_EMIT) || defined(DSA_QUEUE_NULL))
#error "DSA_BUCKET_GATE repurposes inv_reg/vth_reg; incompatible with these variants"
#endif
#if defined(DSA_BUCKET_GATE2) && (defined(DSA_BUCKET_GATE) || defined(DSA_OLD_GATE) \
    || defined(DSA_LANE_EMIT) || defined(DSA_DIRECT_EMIT) || defined(DSA_QUEUE_NULL))
#error "DSA_BUCKET_GATE2: incompatible variant combination"
#endif
#if defined(DSA_BUCKET_GATE3) && (defined(DSA_BUCKET_GATE) || defined(DSA_BUCKET_GATE2) \
    || defined(DSA_OLD_GATE) || defined(DSA_LANE_EMIT) || defined(DSA_DIRECT_EMIT) \
    || defined(DSA_QUEUE_NULL) || defined(DSA_PERSIST))
#error "DSA_BUCKET_GATE3: incompatible variant combination"
#endif
#if defined(DSA_BUCKET_GATE4) && (defined(DSA_BUCKET_GATE) || defined(DSA_BUCKET_GATE2) \
    || defined(DSA_BUCKET_GATE3) || defined(DSA_OLD_GATE) || defined(DSA_LANE_EMIT) \
    || defined(DSA_DIRECT_EMIT) || defined(DSA_QUEUE_NULL) || defined(DSA_PERSIST))
#error "DSA_BUCKET_GATE4: incompatible variant combination"
#endif
        // NOTE (DSA_BIT_GATE, tried + REJECTED 2026-07-11): ALU bit-pattern
        // buckets (flipped-float compare + shift bucketing) break recall
        // (0.5-96%): the aminmax span crosses octaves/zero, so bit space is
        // wildly nonuniform (radix-judgment redux) and (found+1)<<k overflows
        // u32 at large k. Column-frequency +2 INT ops also pre-decided the
        // speed. Do not revisit without a threshold-anchored, overflow-safe
        // bucket space AND a hit-frequency-only costing.
        float weights[BLOCK_Q][kNumHeads];
        float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
        uint32_t kstart_reg[BLOCK_Q], kspan_reg[BLOCK_Q];  // unsigned range-check trick
        int gate_reg[BLOCK_Q];
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
            // redundantly in registers (qn_reg), so the hot emit path needs no
            // smem bookkeeping and no shfl broadcast.
            // NOTE (adaptive dual-mode emit, rejected 2026-07-09): switching
            // rows to direct ballot-group scatter when the window hit count
            // is low measured 4-13% SLOWER at every real shape — the extra 5
            // registers + branch alone spill at the 168-reg cap. The queue +
            // register-count path below is the measured optimum for emit.
            int qn_reg[BLOCK_Q];
#ifdef DSA_LANE_EMIT
            uint32_t lcnt = 0;  // 4x8-bit per-row private-slot counters
#endif
#if defined(DSA_EMIT_NULL) || defined(DSA_QUEUE_NULL)
            uint32_t emit_sink = 0;  // keeps the gate live without emit code
#endif
#ifdef DSA_GATE_NULL
            float gate_sink = 0.f;   // keeps scoring live without the gate
#endif
#if defined(DSA_BULK_DRAIN) && defined(DSA_BULK_PP)
            int dr_reg[BLOCK_Q] = {};  // ring drained cursor (multiple of 32)
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
#ifdef DSA_BUCKET_GATE3
                // fully-folded bucket space: the FFMA2 chain runs in bucket
                // units (w' = -inv*w, folded below), the tail is
                // fmaf(scale_kv, sum', c0) -- ONE F2I over the sign gate.
                // Slot repurposing (zero new registers): vth_reg = c0 =
                // -o*inv; o_reg = delta = 1/inv (exact hit-side x
                // reconstruction: x = (bq - c0) * delta).
                vth_reg[i] = -o_reg[i] * inv_reg[i];
                o_reg[i] = 1.0f / inv_reg[i];
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
                gate_reg[i] = cute::numeric_limits<int32_t>::max();
                qn_reg[i] = 0;
                kstart_reg[i] = seq_k_start[i];
                kspan_reg[i] = seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
            }
#if defined(DSA_BUCKET_GATE3) || defined(DSA_BUCKET_GATE4)
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
#ifdef DSA_LANE_EMIT
            // GVR-style ballot-free collection: hits stage into per-lane
            // private smem slots (2 per lane-row, reusing the warpq extent);
            // every GATE_STRIDE window the lanes drain STRAIGHT TO GLOBAL
            // via one redux + shfl-prefix + one atomic per row (amortized
            // 1/16 blocks). Hot path per hit: extract counter + 2 STS +
            // counter bump -- no ballot, no popc, no shfl, no syncwarp.
            const auto drain_lanes = [&]() {
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    const uint32_t c = (lcnt >> (i * 8)) & 0xffu;
                    const uint32_t tot = __reduce_add_sync(FULL, c);
                    if (tot == 0) continue;
                    uint32_t off = c;
                    #pragma unroll
                    for (uint32_t d = 1; d < 32; d <<= 1) {
                        const uint32_t nsh = __shfl_up_sync(FULL, off, d);
                        if (lane_idx >= d) off += nsh;
                    }
                    off -= c;  // exclusive prefix
                    const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                    int base = 0;
                    if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, (int)tot);
                    base = __shfl_sync(FULL, base, 0);
                    const uint32_t qb = (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
                    const uint64_t ob = static_cast<uint64_t>(row_q) * cand_cap;
                    for (uint32_t s = 0; s < c; ++ s) {
                        const int w = base + (int)(off + s);
                        if (w < static_cast<int>(cand_cap)) {
                            const uint32_t sl = qb + lane_idx * 2u + s;
                            DSA_ST_CAND_VAL(cand_val[ob + w], warpq_val[sl]);
                            DSA_ST_CAND_IDX(cand_idx[ob + w], warpq_idx[sl]);
                        }
                    }
                }
                lcnt = 0;
            };
#endif

            // Drain a (warp,row) queue segment to the global candidate
            // buffer. The probe index mapping lives HERE, not on the insert
            // path: it used to run warp-wide per ballot group (~4 predicated
            // instructions dragged by a single hit); at drain time 32 lanes
            // retire DSA_WARP_QUEUE_CAP entries in parallel, so it costs
            // ~1/30th in issue slots and is semantically identical.
#if !defined(DSA_EMIT_NULL) && !defined(DSA_QUEUE_NULL)
#ifdef DSA_BULK_DRAIN
            // Bulk drain: qn is a multiple of 4 (caller-guaranteed), base is
            // 4-aligned by induction (seed counts are padded to 4; every
            // drain adds a multiple of 4) -> both TMA addresses are 16B-
            // aligned. Payload rides the TMA engine: ~5 issue slots replace
            // the per-entry LDS/remap/STG chains (and their register/codegen
            // tax on the whole hot loop). Index remap moved to select output.
            const auto drain_queue = [&](const uint32_t i, const uint32_t row_q,
                                         const uint32_t queue_base, const int qn,
                                         const int base) {
                const int avail = static_cast<int>(cand_cap) - base;
                const int len = min(qn, max(avail, 0)) & ~3;
                if (len > 0 && lane_idx == 0) {
                    const uint64_t ob = static_cast<uint64_t>(row_q) * cand_cap
                                        + static_cast<uint32_t>(base);
                    const uint32_t sv = static_cast<uint32_t>(
                        __cvta_generic_to_shared(warpq_val + queue_base));
                    const uint32_t si = static_cast<uint32_t>(
                        __cvta_generic_to_shared(warpq_idx + queue_base));
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                        :: "l"(cand_val + ob), "r"(sv), "r"(len * 4) : "memory");
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                        :: "l"(cand_idx + ob), "r"(si), "r"(len * 4) : "memory");
                    asm volatile("cp.async.bulk.commit_group;\n");
#ifndef DSA_BULK_PP
                    // The queue slots are refilled right after: wait for the
                    // TMA engine's smem READS (not the gmem writes) to finish.
                    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
                    // (ring mode: the wait moved to the NEXT drain of this
                    // warp — by then the copy is ~32 candidates old.)
                }
                __syncwarp(FULL);
            };
#else
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
#endif
#endif  // !EMIT_NULL && !QUEUE_NULL

            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
#ifdef DSA_PERSIST
                const uint32_t kvg = kvt_base + kv_block_idx;
#else
                const uint32_t kvg = kv_block_idx;
#endif
                CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                if ((kv_block_idx % DSA_GATE_STRIDE) == 0) {
#ifdef DSA_LANE_EMIT
                    if (kv_block_idx > 0) drain_lanes();
#endif
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const int g = th_pf[i];
                        if (g != gate_reg[i]) {
                            gate_reg[i] = g;
#if defined(DSA_BUCKET_GATE4)
                            // edge = float(g+1): exact for small ints, no
                            // division. The gate compares BITS against it.
                            o_reg[i] = static_cast<float>(g + 1);
#elif defined(DSA_BUCKET_GATE) || defined(DSA_BUCKET_GATE2) || defined(DSA_BUCKET_GATE3)
                            // bucket gate compares braw <= gate_reg directly:
                            // nothing to precompute (no edge float, no div).
#elif !defined(DSA_OLD_GATE)
                            // SIGN GATE (default): stores +th_x; the hot path computes
                            // t = fmaf(scale, sums, th_x) = v - C and tests the
                            // SIGN BIT (ISETP on the INT pipe); the FSETP and
                            // the final FMUL are algebraically absorbed.
                            vth_reg[i] = o_reg[i] + static_cast<float>(g + 1) / inv_reg[i];
#else
                            // Stored NEGATED: emit <=> x < vth <=> v > -vth. IEEE
                            // negation is exact, so the gate decision is unchanged
                            // while the hot path drops one FNEG per row per block.
                            vth_reg[i] = -(o_reg[i] + static_cast<float>(g + 1) / inv_reg[i]);
#endif
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i)
                        th_pf[i] = __ldcg(th_bucket + min(block_q_idx * BLOCK_Q + i, seq_len - 1));
                }

                float scale_kv = ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

                full_umma_barriers[warpgroup_idx]->wait(kvg & 1);
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
                DG_STATIC_ASSERT(BLOCK_Q % 2 == 0, "row-pair loads need even BLOCK_Q");
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
#elif defined(DSA_BUCKET_GATE3)
                // Fully-folded bucket gate: the chain summed w'*ReLU (bucket
                // units), so the tail FFMA is form-identical to the sign
                // gate's; the ONLY extra column cost is the F2I. Hits reuse
                // v_row (bucket float) for both feed (1 F2I) and exact x
                // reconstruction ((bq - c0) * delta).
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float bq = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);            \
                        v_row[i] = bq;                                                         \
                        bool g = static_cast<int>(bq) <= gate_reg[i];                          \
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
#elif !defined(DSA_OLD_GATE)
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
#else
                #define DSA_SCORE_GATE(i, RC)                                                  \
                        const float v = scale_kv * (sum.x + sum.y);                            \
                        v_row[i] = v;                                                          \
                        bool g = v > vth_reg[i];                                               \
                        if constexpr (RC)                                                      \
                            g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]);            \
                        pass_bits |= g ? (1u << i) : 0u;
#endif
                const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
                const bool interior = (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
                #define DSA_SCORE_ROWS(RANGE_CHECK)                                            \
                _Pragma("unroll")                                                              \
                for (uint32_t pr = 0; pr < BLOCK_Q / 2; ++ pr) {                               \
                    float accum2[kNumHeads * 2];                                               \
                    tmem_load(cute::Int<kNumHeads * 2>{}, tmem_start + pr * 2 * kNumHeads, accum2); \
                    if (pr == BLOCK_Q / 2 - 1) {                                               \
                        ptx::tcgen05_before_thread_sync();                                     \
                        empty_umma_barriers[warpgroup_idx]->arrive();                          \
                    }                                                                          \
                    _Pragma("unroll")                                                          \
                    for (uint32_t k = 0; k < 2; ++ k) {                                        \
                        const uint32_t i = pr * 2 + k;                                         \
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
#ifdef DSA_EMIT_NULL
                emit_sink += pass_bits;      // gate + vote priced, zero emit
#elif defined(DSA_DIRECT_EMIT)
                // Direct per-hit emission: no queue, no ballot, no drain.
                // Wager: at enriched-bound densities (~0.3 hits per
                // warp-block) the returning atomic's frequency x cost drops
                // below the queue machinery's standing cost. RAW indices
                // (select-side remap).
                if (pass_bits) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        if ((pass_bits >> i) & 1u) {
                            const float x = vth_reg[i] - v_row[i];
                            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                            const int w = atomicAdd(cand_cnt + row_q, 1);
                            if (w < static_cast<int>(cand_cap)) {
                                const uint64_t ob = static_cast<uint64_t>(row_q) * cand_cap;
                                DSA_ST_CAND_VAL(cand_val[ob + w], x);
                                DSA_ST_CAND_IDX(cand_idx[ob + w], static_cast<int32_t>(kv_offset));
                            }
                            if (refresh_every > 0) {
                                int braw = static_cast<int>((x - o_reg[i]) * inv_reg[i]);
                                int b = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                                if (hist_in_smem) {
                                    atomicAdd(smem_hist + i * num_buckets + b, 1);
                                } else {
                                    atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], 1);
                                }
                            }
                        }
                    }
                }
#elif defined(DSA_LANE_EMIT)
                // per-lane, zero collectives: extract counter, 2 STS, bump.
                // Overflow past the 2 private slots (rare at any density
                // between drains) falls through to a direct global append.
                if (pass_bits) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        if ((pass_bits >> i) & 1u) {
                            const float x = vth_reg[i] - v_row[i];
                            const uint32_t c = (lcnt >> (i * 8)) & 0xffu;
                            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                            if (c < 2u) {
                                const uint32_t sl = (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP
                                                    + lane_idx * 2u + c;
                                warpq_val[sl] = x;
                                warpq_idx[sl] = static_cast<int32_t>(kv_offset);
                                lcnt += (1u << (i * 8));
                            } else {
                                const int w = atomicAdd(cand_cnt + row_q, 1);
                                if (w < static_cast<int>(cand_cap)) {
                                    const uint64_t ob = static_cast<uint64_t>(row_q) * cand_cap;
                                    DSA_ST_CAND_VAL(cand_val[ob + w], x);
                                    DSA_ST_CAND_IDX(cand_idx[ob + w], static_cast<int32_t>(kv_offset));
                                }
                            }
                            if (refresh_every > 0) {
                                int braw = static_cast<int>((x - o_reg[i]) * inv_reg[i]);
                                int b = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                                if (hist_in_smem) {
                                    atomicAdd(smem_hist + i * num_buckets + b, 1);
                                } else {
                                    atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], 1);
                                }
                            }
                        }
                    }
                }
#elif defined(DSA_QUEUE_NULL)
                // gate + histogram/refresh priced; queue/drain/writeback gone
                if (__any_sync(FULL, pass_bits)) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        if ((pass_bits >> i) & 1u) {
#ifndef DSA_OLD_GATE
                            const float x = vth_reg[i] - v_row[i];
#else
                            const float x = -v_row[i];
#endif
                            int braw = static_cast<int>((x - o_reg[i]) * inv_reg[i]);
                            int b = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                            if (refresh_every > 0) {
                                if (hist_in_smem) atomicAdd(smem_hist + i * num_buckets + b, 1);
                                else atomicAdd(&bcount[static_cast<uint64_t>(block_q_idx * BLOCK_Q + i) * num_buckets + b], 1);
                            } else emit_sink += static_cast<uint32_t>(b);
                        }
                    }
                }
#else
                if (__any_sync(FULL, pass_bits)) {
                    const uint32_t rows_union = __reduce_or_sync(FULL, pass_bits);
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        if (rows_union & (1u << i)) {
                            const bool g = (pass_bits >> i) & 1u;
                            const unsigned m = __ballot_sync(FULL, g);
                            // Emit via the smem warp queue, but with the fill count
                            // in registers (warp-uniform, every lane tracks it):
                            // the hot path has NO smem bookkeeping, NO shfl
                            // broadcast, NO syncwarp. The returning atomicAdd (L2
                            // round-trip the whole warp must wait on through the
                            // shfl) happens only on drain, ~once per 32 candidates.
                            // (Direct per-group scatter measured 10-15% SLOWER
                            // end-to-end: the round-trip per ballot group is the
                            // expensive part, not the stores. NCU @256K: queue
                            // bookkeeping showed up as wait/short_scoreboard/
                            // branch stalls starving the UMMA pipe.)
                            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                            const int cnt = __popc(m);
                            const uint32_t queue_base = (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
                            int qn = qn_reg[i];
#if defined(DSA_BULK_DRAIN) && defined(DSA_BULK_PP)
                            if (false) {  // ring drains POST-insert (see below)
#else
                            if (qn + cnt > static_cast<int>(DSA_WARP_QUEUE_CAP)) {
#endif
#if defined(DSA_BULK_DRAIN) && defined(DSA_BULK_PP)
                                // unreachable: ring drains post-insert
#elif defined(DSA_BULK_DRAIN)
                                // drain in 4-entry granularity (16B TMA
                                // alignment); the 0-3 leftover entries slide
                                // to the queue front and roll onward.
                                const int qn4 = qn & ~3;
                                if (qn4 > 0) {
                                    int base = 0;
                                    if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn4);
                                    base = __shfl_sync(FULL, base, 0);
                                    drain_queue(i, row_q, queue_base, qn4, base);
                                }
                                const int rem = qn - qn4;
                                if (static_cast<int>(lane_idx) < rem) {
                                    warpq_val[queue_base + lane_idx] = warpq_val[queue_base + qn4 + lane_idx];
                                    warpq_idx[queue_base + lane_idx] = warpq_idx[queue_base + qn4 + lane_idx];
                                }
                                qn = rem;
                                __syncwarp(FULL);  // queue slots reusable
#else
                                int base = 0;
                                if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
                                base = __shfl_sync(FULL, base, 0);
                                drain_queue(i, row_q, queue_base, qn, base);
                                qn = 0;
                                __syncwarp(FULL);  // queue slots reusable
#endif
                            }
                            if (g) {
                                // Lean insert: position + two STS; the probe
                                // index mapping is deferred to drain_queue
                                // (semantically free). The histogram feed is
                                // NOT deferred: a queue-depth of count lag
                                // slows the daemon's tightening and the
                                // looser gate costs more than the feed saves
                                // (measured +0.2..0.9ms @q8192 production).
#if defined(DSA_BUCKET_GATE4)
                                const float x = v_row[i];   // bucket-space value IS the payload
#elif defined(DSA_BUCKET_GATE3)
                                // exact x from the bucket float: (bq - c0) * delta
                                const float x = (v_row[i] - vth_reg[i]) * o_reg[i];
#elif defined(DSA_BUCKET_GATE2)
                                const float x = v_row[i];   // v_row holds x directly
#elif defined(DSA_BUCKET_GATE)
                                const float x = -v_row[i];  // v is the raw scaled score here
#elif !defined(DSA_OLD_GATE)
                                const float x = vth_reg[i] - v_row[i];  // x = -v = th_x - t
#else
                                const float x = -v_row[i];
#endif
                                const unsigned below = (1u << lane_idx) - 1u;
                                const int pos = qn + __popc(m & below);
#if defined(DSA_BULK_DRAIN) && defined(DSA_BULK_PP)
                                warpq_val[queue_base + (pos & 63)] = x;
                                warpq_idx[queue_base + (pos & 63)] = static_cast<int32_t>(kv_offset);
#else
                                warpq_val[queue_base + pos] = x;
                                warpq_idx[queue_base + pos] = static_cast<int32_t>(kv_offset);
#endif
                                if (refresh_every > 0) {
#if defined(DSA_BUCKET_GATE3) || defined(DSA_BUCKET_GATE4)
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
#if defined(DSA_BULK_DRAIN) && defined(DSA_BULK_PP)
                            // ring-64 post-insert drain (logical 2x32 ping-
                            // pong): inserts only ever write the half NOT in
                            // flight (lag <= 31 pre-insert, cnt <= 32, so
                            // writes stay within [dr, dr+63]); once a full
                            // 32-half is written, wait for the PREVIOUS
                            // copy's smem reads (issued one half-life ago,
                            // amortized-free) and ship this half. Fixed
                            // 32-entry drains: no remainder, no slide,
                            // 16B-aligned by construction.
                            if (qn_reg[i] - dr_reg[i] >= 32) {
                                if (lane_idx == 0)
                                    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
                                __syncwarp(FULL);
                                int base = 0;
                                if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, 32);
                                base = __shfl_sync(FULL, base, 0);
                                drain_queue(i, row_q, queue_base + (dr_reg[i] & 63), 32, base);
                                dr_reg[i] += 32;
                            }
#endif
                        }
                    }
                }
#endif  // emit alternatives

                if (threadIdx.x == 0 && ((kv_block_idx + 1) % DSA_REFRESH_STRIDE) == 0) {
                    __threadfence_block();
                    *kv_progress_ptr = static_cast<int>(kvg + 1);
                }
#if defined(DSA_EMIT_NULL) || defined(DSA_QUEUE_NULL)
                if (emit_sink == 0x13371337u) *scan_done_flag = 2;  // unprovable sink
#endif
            }

#if !defined(DSA_EMIT_NULL) && !defined(DSA_QUEUE_NULL)
#ifdef DSA_LANE_EMIT
            drain_lanes();
#endif
            // Flush this CTA's warp queues (counts live in qn_reg).
#ifdef DSA_BULK_DRAIN
            // Arbitrary-length flushes break the 4-alignment other drains
            // rely on. Intra-CTA: all math warps finish draining first
            // (barrier). Inter-CTA (num_kv_splits > 1, small-Q only, never
            // certificate mode): pad each flush to a multiple of 4 with
            // +INF sentinels — select skips them via isfinite().
            cutlass::arch::NamedBarrier(kNumMathThreads, 1).sync();
#endif
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                const int qn = qn_reg[i];
                if (row_q < seq_len && qn > 0) {
                    const uint32_t queue_base = (warp_idx * BLOCK_Q + i) * DSA_WARP_QUEUE_CAP;
#ifdef DSA_BULK_DRAIN
#ifdef DSA_BULK_PP
                    const int fl = qn - dr_reg[i];       // undrained ring tail
                    const int dr0 = dr_reg[i];
#else
                    const int fl = qn;
                    const int dr0 = 0;
#endif
                    const int qn_add = (num_kv_splits > 1) ? ((fl + 3) & ~3) : fl;
                    int base = 0;
                    if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn_add);
                    base = __shfl_sync(FULL, base, 0);
                    // scalar stores, RAW indices (select-side remap covers
                    // slots >= seed base)
                    const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                    for (int t = static_cast<int>(lane_idx); t < qn_add; t += 32) {
                        const int w = base + t;
                        if (w < static_cast<int>(cand_cap)) {
                            DSA_ST_CAND_VAL(cand_val[out_base + w],
                                            t < fl ? warpq_val[queue_base + ((dr0 + t) & 63)] : INFINITY);
                            DSA_ST_CAND_IDX(cand_idx[out_base + w],
                                            t < fl ? warpq_idx[queue_base + ((dr0 + t) & 63)] : 0);
                        }
                    }
#else
                    int base = 0;
                    if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
                    base = __shfl_sync(FULL, base, 0);
                    drain_queue(i, row_q, queue_base, qn, base);
#endif
                }
            }

#endif  // !EMIT_NULL && !QUEUE_NULL
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

} // namespace dsa_marsco_v3

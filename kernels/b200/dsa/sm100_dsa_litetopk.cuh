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

namespace dsa_litetopk {

using namespace deep_gemm;

#ifndef DSA_WARP_QUEUE_CAP
#define DSA_WARP_QUEUE_CAP 64
#endif

#ifndef DSA_REFRESH_STRIDE
// KV blocks between threshold-refresh signals to the spare-warp daemon.
// Swept 2026-07-22 (in-process, order-randomized, byte-identical control, two
// independent runs agreeing to 0.1%): 2 is -2.3% at 1M and ~-1.5% at 256K, but
// ONLY in combination with DSA_TMEM_ROWS=4 + 240/24 registers. On a
// ROWS=2 / 232-40 build the same change is +0.8% at 1M.
//
// ncu 2x2 (fi_port/profile/stride_why/) explains it: the knob's effect is
// build-independent -- it kills the daemon's __nanosleep (sleeping 6716 -> 14
// M warp-cyc, -99.8%, in both) and its only cost is the *kv_progress_ptr store
// (STS +3.657M measured vs +3.670M predicted from the block count; the
// threadfence measures as free). What differs is whether anything can use the
// freed issue slots: ROWS=4 turns them into +8.8% issue rate and -5.0% total
// stall, while ROWS=2 -- already math-pipe-bound by its two 32dp32b64x loads
// and two fences per accumulator -- gets math_pipe_throttle +23.7% and +0.8%
// total stall. ROWS=4 is what creates the headroom STRIDE=2 fills.
// Treat the three as one tuned point; re-sweep if any of them changes.
#define DSA_REFRESH_STRIDE 2
#endif

#ifndef DSA_GATE_STRIDE
#define DSA_GATE_STRIDE 16
#endif
#ifndef DSA_MATH_REGS
#define DSA_MATH_REGS 240
#endif
#ifndef DSA_SPEC_REGS
#define DSA_SPEC_REGS 24
#endif
#ifndef DSA_TMEM_ROWS
// Rows retired per tcgen05.ld. 2 is 32dp32b64x; 4 (32dp32b128x) pulls the whole
// BLOCK_Q accumulator in one load, halving the loads and fences. 4 releases
// TMEM one step earlier at the cost of holding 128 accumulator registers
// instead of 64 -- that used to spill 8 bytes, which is why it was rejected
// once; raising the math warps to 240 registers (DSA_MATH_REGS, paid for by
// dropping the specialized warps to the setmaxnreg minimum of 24) absorbs it.
// The budget is zero-sum and exactly saturated: 8*32*240 + 4*32*24 = 64,512 =
// the 168*384 pool ptxas allocates under launch_bounds(384,1).
#define DSA_TMEM_ROWS 4
#endif
#ifndef DSA_UMMA_STAGES
// Accumulator buffers per math warp group; see kNumUmmaStages below.
// 2 (TMEM double buffer) measured -2.7% at 256K.
#define DSA_UMMA_STAGES 2
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

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto warpgroup_idx = warp_idx / 4;
    const auto lane_idx = ptx::get_lane_idx();
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    constexpr uint32_t kNumUmmaStages = DSA_UMMA_STAGES;
    constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;

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
                gate_reg[i] = cute::numeric_limits<int32_t>::max();
                qn_reg[i] = 0;
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

            // Drain a (warp,row) queue segment to the global candidate
            // buffer. The probe index mapping lives HERE, not on the insert
            // path: it used to run warp-wide per ballot group (~4 predicated
            // instructions dragged by a single hit); at drain time 32 lanes
            // retire DSA_WARP_QUEUE_CAP entries in parallel, so it costs
            // ~1/30th in issue slots and is semantically identical.
#ifndef DSA_EMIT_NULL
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
#endif  // !EMIT_NULL

            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
#ifdef DSA_PERSIST
                const uint32_t kvg = kvt_base + kv_block_idx;
#else
                const uint32_t kvg = kv_block_idx;
#endif
                CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

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
#ifdef DSA_EMIT_NULL
                emit_sink += pass_bits;      // gate + vote priced, zero emit
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
                            if (qn + cnt > static_cast<int>(DSA_WARP_QUEUE_CAP)) {
                                int base = 0;
                                if (lane_idx == 0) base = atomicAdd(cand_cnt + row_q, qn);
                                base = __shfl_sync(FULL, base, 0);
                                drain_queue(i, row_q, queue_base, qn, base);
                                qn = 0;
                                __syncwarp(FULL);  // queue slots reusable
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

                if (threadIdx.x == 0 && ((kv_block_idx + 1) % DSA_REFRESH_STRIDE) == 0) {
                    __threadfence_block();
                    *kv_progress_ptr = static_cast<int>(kvg + 1);
                }
#ifdef DSA_EMIT_NULL
                if (emit_sink == 0x13371337u) *scan_done_flag = 2;  // unprovable sink
#endif
            }

#ifndef DSA_EMIT_NULL
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

#endif  // !EMIT_NULL
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

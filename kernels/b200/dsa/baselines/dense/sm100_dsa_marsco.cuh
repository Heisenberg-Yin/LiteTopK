// Route-A stage 2 (Blackwell / B200 SM100 port): deep_gemm sm100_fp8_mqa_logits
// with a SAMPLE+WRITE epilogue.
//
// The ReLU-MQA scoring math (UMMA tcgen05 + TMEM load + per-head fmaxf(.,0)*weight
// + weighted sum) is structurally COPIED from
// deep_gemm/impls/sm100_fp8_mqa_logits.cuh (the Blackwell counterpart of the
// Hopper sm90_fp8_mqa_logits.cuh the H100 version was based on). The ONLY change
// is the store block at the end: instead of writing every logit to a dense [Q,S]
// buffer, we write only logits >= a per-row threshold row_th[q] into a compact
// candidate buffer (cand_val/cand_idx with per-row atomic counter cand_cnt). The
// threshold is precomputed by running deep_gemm on a small sample of KV positions.
//
// Compared to the Hopper (WGMMA) DSA kernel:
//   * MMA is issued by a dedicated UMMA warp into TMEM (tensor memory), not by
//     the math warpgroups into registers. Math warps read the accumulator via
//     SM100_TMEM_LOAD_* and then run the same divergent gate+emit epilogue.
//   * In the SM100 accumulator layout each math lane owns ONE score column
//     (v_offset = lane_idx). This removes the Hopper 4-lane-duplicate group, so
//     every lane carries a distinct candidate (one candidate per lane), which
//     simplifies the warp-aggregated sparse emit versus the WGMMA m64n8 path.
//   * REFRESH: the Hopper version kept the whole producer warpgroup alive to run
//     a mid-scan warp-parallel threshold refresh (tighten th_bucket from bcount).
//     The SM100 producer is only a TMA-load warp + a UMMA warp, so instead the
//     MATH warps run the refresh themselves (see `do_refresh` below): math warp
//     `w` owns query row (block_q_idx*BLOCK_Q + w), so BLOCK_Q rows map 1:1 onto
//     the BLOCK_Q math warps -- the same "one warp per query row" layout as the
//     Hopper producer refresh. Every `refresh_every` KV blocks each warp does a
//     32-lane prefix-sum over its row's bcount and monotonically tightens the
//     gate. Reading a possibly-stale bcount only makes the gate MORE conservative
//     (never fewer candidates), so recall is preserved and the boundary select
//     still finds the exact top-k.
//
// Namespaced dsa_marsco:: so it never clashes with the original template.

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm100.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace dsa_marsco {

using namespace deep_gemm;
using namespace deep_gemm::sm90;
using namespace deep_gemm::sm100;

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

// How many KV blocks the math warps must advance between two spare-warp threshold
// refreshes ("refresh every N CTA KV-blocks"). This is a sweepable hyper-param.
// Measured on real GLM-5 caches the end-to-end latency is nearly flat across
// N in [2..64] (all within ~1%, recall stays 100%) -- the refresh runs on the
// otherwise-idle spare warps, so its frequency is NOT on the critical path; the
// gate-tightening benefit also saturates after the first few refreshes. N=16 is
// a good default (timely tightening, low poll pressure). Override with
// -DDSA_REFRESH_STRIDE=N.
#ifndef DSA_REFRESH_STRIDE
#define DSA_REFRESH_STRIDE 16
#endif

// Reload the per-row gate (th_bucket) every N KV blocks instead of every block.
// The spare-warp refresh daemon only recomputes thresholds every
// DSA_REFRESH_STRIDE completed KV blocks anyway, and th_bucket tightens
// monotonically, so a stale gate is merely conservative (a few extra candidates,
// recall unaffected). Reloading every block cost 4 global loads + 4 fdivs per
// math thread per KV block; with the stride the loads amortize and the fdiv
// only reruns when the loaded gate actually changed.
// Swept {4,8,16,32} on real caches (1m/8192): monotonically faster with larger
// stride but all within ~1% (45.90/45.47/45.19/45.09 ms); 16 matches the
// refresh-daemon publish cadence (DSA_REFRESH_STRIDE), so reading more often
// than that cannot observe a new gate anyway.
#ifndef DSA_GATE_STRIDE
#define DSA_GATE_STRIDE 16
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
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_dsa_marsco(const uint32_t seq_len, const uint32_t seq_len_kv,
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
    const auto num_q_blocks = ceil_div(seq_len, BLOCK_Q);

    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    // Prefetch TMA descriptors (one lane of the first specialized warp)
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs (weight may be unaligned; kv scale aligned to 512B)
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

    // Align to 512 bytes for swizzle-128B
    extern __shared__ __align__(512) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");

    // Tensor memory: DSA_TMEM_BUFS accumulator tiles [UMMA_M, BLOCK_Q*kNumHeads]
    // per math WG (buffered on KV-block parity). With a single tile the UMMA of
    // block n+1 cannot issue until the math warps finished the TMEM load of
    // block n; with two tiles the UMMA warp works on parity (n+1)&1 while the
    // math warps drain parity n&1.
#ifndef DSA_TMEM_BUFS
#define DSA_TMEM_BUFS 2
#endif
    constexpr uint32_t kNumTmemBufs = DSA_TMEM_BUFS;
    constexpr uint32_t kTmemColsPerBuf = BLOCK_Q * kNumHeads * kNumMathWarpGroups;
    constexpr uint32_t kNumTmemCols = kTmemColsPerBuf * kNumTmemBufs;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory columns");

    // Data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
            SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA / UMMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers     = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers    = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });
    // UMMA barriers are per (math WG, tmem parity): index = wg * 2 + parity.
    auto full_umma_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); });
    auto empty_umma_barriers = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * kNumTmemBufs + i); });

    // Tensor memory allocation slot. Slot [0] is used by the TMEM allocator;
    // slot [1] is reused as a `scan_done` flag and slot [2] as a `kv_progress`
    // counter, both published by the math warps to drive the spare-warp
    // background threshold refresh (see below).
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumMathWarpGroups * kNumTmemBufs * 2);
    auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
    auto kv_progress_ptr = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 2);

#if DSA_WARP_QUEUE
    // Warp-local candidate queue after the barrier/tmem region.
    static constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
    auto warpq_val = reinterpret_cast<float*>(warpq_count + kNumMathWarps * BLOCK_Q);
    auto warpq_idx = reinterpret_cast<int32_t*>(warpq_val + kNumMathWarps * BLOCK_Q * DSA_WARP_QUEUE_CAP);
#endif

    // Initialize barriers
    const bool& is_tma_load_warp = (warp_idx == (kNumMathThreads / 32));
    const bool& is_umma_warp = (warp_idx == (kNumMathThreads / 32 + 1));
    if (is_tma_load_warp and cute::elect_one_sync()) {
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
        #pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups * kNumTmemBufs; ++ i) {
            full_umma_barriers[i]->init(1);
            empty_umma_barriers[i]->init(128);
        }
        cutlass::arch::fence_barrier_init();
        *scan_done_flag = 0;
        *kv_progress_ptr = 0;
    } else if (is_umma_warp) {
        // Allocate tensor memory
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumSpecializedRegisters = 32;
    constexpr uint32_t kNumMathRegisters = 208;

    // KV-split scheduling: blockIdx.x selects the q-block (one per CTA, no
    // persistence); blockIdx.y selects a contiguous KV sub-range so all SMs
    // work on the same queries in parallel. Each CTA scans only its window
    // [split_lo, split_hi) intersected with the per-row [cu_start, cu_end).
    const uint32_t block_q_idx = blockIdx.x;
    const uint32_t kv_split = blockIdx.y;
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
    const auto load_schedule = [&]() -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
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
        const uint32_t total_blocks = ceil_div(seq_len_kv, BLOCK_KV);
        const uint32_t blocks_per_split = ceil_div(total_blocks, num_kv_splits);
        const uint32_t split_lo = kv_split * blocks_per_split * BLOCK_KV;
        const uint32_t split_hi = min((kv_split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv);
        start = start / 4 * 4;
        if (start < split_lo) start = split_lo;
        if (end > split_hi) end = split_hi;
        uint32_t nkv = (end > start) ? ceil_div(end - start, BLOCK_KV) : 0;
        return {0u, 0u, start, nkv};
    };

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

    if (is_tma_load_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& blk_idx) {
            tma_copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim, __nv_fp8_e4m3>(
                &tensor_map_q, full_q_barriers[stage_idx], smem_q[stage_idx], 0, blk_idx * BLOCK_Q * kNumHeads);
            tma_copy<kNumHeads, BLOCK_Q, 0, float>(
                &tensor_map_weights, full_q_barriers[stage_idx], smem_weights[stage_idx], 0, blk_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };

        if (cute::elect_one_sync() and block_q_idx < num_q_blocks) {
            issue_tma_q(0, block_q_idx);
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
                tma_copy<kHeadDim, BLOCK_KV, kHeadDim, __nv_fp8_e4m3>(
                    &tensor_map_kv, full_kv_barriers[kv_stage_idx], smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                tma_copy<BLOCK_KV, 1, 0, float>(
                    &tensor_map_kv_scales, full_kv_barriers[kv_stage_idx], smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }
        }
    } else if (is_umma_warp) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        // Require full allocation
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        if (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
            DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);
                // Buffered accumulator: block n uses tmem parity slot
                // n % kNumTmemBufs; the k-th use of a slot flips its barrier
                // phase, hence (n / kNumTmemBufs) & 1.
                const uint32_t abs_idx = num_total_kv_blocks + kv_block_idx;
                const uint32_t parity = abs_idx % kNumTmemBufs;
                const uint32_t buf_phase = (abs_idx / kNumTmemBufs) & 1;
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                    empty_umma_barriers[i * kNumTmemBufs + parity]->wait(buf_phase ^ 1);
                    #pragma unroll
                    for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                        auto a_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                        auto b_desc = make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                            smem_q[q_stage_idx], 0, k * UMMA_K);
                        cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, i * UMMA_N + parity * kTmemColsPerBuf, k, runtime_instr_desc);
                    }
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i * kNumTmemBufs + parity]));
                }
            }
        }
    } else if (warp_idx >= kNumMathThreads / 32) {
        // Spare specialized warp(s): background dynamic-threshold refresh.
        //
        // Instead of stealing math-warp cycles (the Hopper/earlier-SM100 layout
        // ran refresh inside the math warps), the otherwise-idle spare warps act
        // as a "gate-tightening daemon": while the math warps scan KV, each spare
        // warp repeatedly prefix-sums a query row's `bcount` histogram and lowers
        // `th_bucket` monotonically, until the math warps publish `scan_done`.
        // Reading a stale `bcount` / writing a th the math warps read a bit later
        // is safe -- it only ever makes the gate MORE conservative (never fewer
        // candidates), so recall is preserved. This frees all math warps for the
        // epilogue, which is the actual bottleneck on B200.
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        const bool in_scan_refresh = (refresh_every > 0 && refresh_every != 0x7fffffff);
        if (in_scan_refresh && block_q_idx < num_q_blocks) {
            // Two spare warps (warp kNumMathThreads/32+2 .. +3) split the BLOCK_Q
            // rows round-robin. `kNumSpareRefreshWarps` spare warps exist.
            const uint32_t spare_id = warp_idx - (kNumMathThreads / 32 + 2);
            constexpr uint32_t kNumSpareRefreshWarps = (kNumSpecializedThreads / 32) - 2;
            const auto refresh_row = [&](const uint32_t r) {
                if (r >= static_cast<uint32_t>(BLOCK_Q)) return;
                const uint32_t row = block_q_idx * BLOCK_Q + r;
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
            // Progress-driven refresh: the math warps publish how many KV blocks
            // they have finished (`kv_progress`). Each spare warp refreshes its
            // rows once every DSA_REFRESH_STRIDE completed KV blocks, plus one
            // final pass once the scan is done. This makes the refresh frequency
            // a deterministic, scan-progress-based hyper-parameter (independent
            // of wall-clock), instead of a fixed nanosleep. Reading a slightly
            // stale bcount / publishing th a bit late is safe (gate only tightens
            // conservatively -> recall preserved).
            const auto refresh_all_rows = [&]() {
                for (uint32_t r = spare_id; r < static_cast<uint32_t>(BLOCK_Q); r += kNumSpareRefreshWarps)
                    refresh_row(r);
            };
            int last_refreshed = 0;
            while (true) {
                const int done = *scan_done_flag;
                const int prog = *kv_progress_ptr;   // published every DSA_REFRESH_STRIDE blocks
                if (prog > last_refreshed) {
                    refresh_all_rows();
                    last_refreshed = prog;
                } else if (done) {
                    refresh_all_rows();   // final catch-up pass
                    break;
                } else {
                    __nanosleep(256);     // light poll spacing, not the refresh rate
                }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto& tmem_start = __shfl_sync(0xffffffff, warpgroup_idx * UMMA_N, 0);
        const auto& warp_offset = warp_idx * 32;   // 32 KV rows per math warp
        const auto& v_offset = lane_idx;           // one score column per lane

        float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
        int gate_reg[BLOCK_Q];

#if DSA_WARP_QUEUE
        DG_STATIC_ASSERT(DSA_WARP_QUEUE_CAP >= 32, "DSA_WARP_QUEUE_CAP must hold one warp/block row (one col per lane)");
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

        // NOTE: dynamic-threshold refresh no longer runs in the math warps. It
        // is handled entirely by the spare warps as a background daemon (see the
        // spare-warp branch above), so the math warps stay 100% on the epilogue.
        // The math warps still accumulate `bcount` per emitted candidate (below),
        // which is what the spare-warp refresh reads.

#ifdef DSA_NULL_EPILOGUE
        float null_acc = 0.f;
#endif
        if (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            full_q_barriers[q_stage_idx]->wait(q_phase);

            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t rq = block_q_idx * BLOCK_Q + i;
                o_reg[i]   = origin[rq];
                inv_reg[i] = inv_delta[rq];
                gate_reg[i] = cute::numeric_limits<int32_t>::max();  // force vth compute on 1st reload
            }

            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

#ifndef DSA_NULL_EPILOGUE
                // Gate reload, strided (see DSA_GATE_STRIDE note above). The fdiv
                // keeps the exact same expression/rounding as before, it just runs
                // only when the gate value actually changed.
                if ((kv_block_idx % DSA_GATE_STRIDE) == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const int g = th_bucket[block_q_idx * BLOCK_Q + i];
                        if (g != gate_reg[i]) {
                            gate_reg[i] = g;
                            // emit <=> floor((x-o)*inv) <= gate  <=>  x < o + (gate+1)/inv
                            vth_reg[i] = o_reg[i] + static_cast<float>(g + 1) / inv_reg[i];
                        }
                    }
                }
#endif

                float scale_kv = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_offset);

                // Wait UMMA arrival (this block's tmem parity slot), then release KV empty.
                const uint32_t abs_idx = num_total_kv_blocks + kv_block_idx;
                const uint32_t parity = abs_idx % kNumTmemBufs;
                full_umma_barriers[warpgroup_idx * kNumTmemBufs + parity]->wait((abs_idx / kNumTmemBufs) & 1);
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Load the accumulator tile from tensor memory.
                constexpr uint32_t kNumLDTMElems = kNumHeads * BLOCK_Q;
                DG_STATIC_ASSERT(kNumLDTMElems == 32 or kNumLDTMElems == 64 or kNumLDTMElems == 128, "Invalid kNumLDTMElems");
                uint32_t shifted_accum[kNumLDTMElems];
                const uint32_t tmem_addr = tmem_start + parity * kTmemColsPerBuf;
                auto tmem_load = [&](auto... Is) {
                    if constexpr (kNumLDTMElems == 32) {
                        cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_addr, shifted_accum[Is]...);
                    } else if constexpr (kNumLDTMElems == 64) {
                        cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_addr, shifted_accum[Is]...);
                    } else if constexpr (kNumLDTMElems == 128) {
                        cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_addr, shifted_accum[Is]...);
                    }
                };
                [&]<size_t... Is>(cute::index_sequence<Is...>) { tmem_load(Is...); }(cute::make_index_sequence<kNumLDTMElems>{});
                cutlass::arch::fence_view_async_tmem_load();
                empty_umma_barriers[warpgroup_idx * kNumTmemBufs + parity]->arrive();

                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;

#if !defined(DSA_NULL_EPILOGUE) && !defined(DENSE_WRITE)
                // Per-lane pass bits for the BLOCK_Q rows of this KV block; the
                // emit machinery (per-row ballot + divergent queue push) only runs
                // when ANY lane passed ANY row (one warp vote), which skips the
                // 4 ballots + 4 divergent branches on the ~90% of KV blocks where
                // the whole warp has no candidate.
                uint32_t pass_bits = 0;
                float x_row[BLOCK_Q];
#endif
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    float* accum = reinterpret_cast<float*>(shifted_accum + i * kNumHeads);
                    // ReLU-MQA: sum_h fmaxf(accum[h],0) * weight[i][h], then * scale_kv.
                    // Weights are loaded as float4 (LDS.128, the row is 128B-aligned)
                    // instead of scalar loads; the FFMA accumulation order (heads
                    // j%4<2 -> sum_0, else sum_1, j ascending) is unchanged, so the
                    // result stays bit-identical to the scalar version.
                    auto sum_0 = make_float2(0, 0);
                    auto sum_1 = make_float2(0, 0);
                    const float* wrow = smem_weights[q_stage_idx] + i * kNumHeads;
                    DG_STATIC_ASSERT(kNumHeads % 4 == 0, "Invalid head count");
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads; j += 4) {
                        const float4 w4 = ld_shared(reinterpret_cast<const float4*>(wrow + j));
                        sum_0 = __ffma2_rn(make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0)),
                                           make_float2(w4.x, w4.y), sum_0);
                        sum_1 = __ffma2_rn(make_float2(fmaxf(accum[j + 2], 0), fmaxf(accum[j + 3], 0)),
                                           make_float2(w4.z, w4.w), sum_1);
                    }
                    float v = (sum_0.x + sum_0.y + sum_1.x + sum_1.y) * scale_kv;

                    // ---- SAMPLE+WRITE epilogue (only change vs deep_gemm) ----
                    // Each lane owns ONE score column (v_offset = lane_idx), so
                    // there are no duplicate lane groups: every lane carries a
                    // distinct candidate. The warp does ONE atomicAdd(cand_cnt,
                    // popc) to reserve a contiguous run, then each passer writes
                    // at base + its ballot prefix (coalesced-ish).
#ifdef DSA_NULL_EPILOGUE
                    // Measurement-only build: consume the score without gating or
                    // storing, to time the pure TMA/UMMA/TMEM+reduction floor.
                    null_acc += v;
#elif !defined(DENSE_WRITE)
                    {
                        const uint32_t col = kv_offset + v_offset;
                        const float x = -v;
                        x_row[i] = x;
                        const bool g = seq_k_start[i] <= col and col < seq_k_end[i] and x < vth_reg[i];
                        pass_bits |= g ? (1u << i) : 0u;
                    }
#else
                    // DENSE-WRITE baseline: same KV-split scoring, write every logit
                    // into a dense [Q, S] buffer (cand_val reused, cand_cap=S).
                    {
                        const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                        const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                        const uint32_t col = kv_offset + v_offset;
                        if (seq_k_start[i] <= col and col < seq_k_end[i])
                            cand_val[out_base + col] = v;
                    }
#endif
                }

#if !defined(DSA_NULL_EPILOGUE) && !defined(DENSE_WRITE)
                // Warp-aggregated emit, entered only when some lane passed some
                // row (warp-uniform vote, so all lanes branch together).
                if (__any_sync(0xffffffffu, pass_bits)) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                        const unsigned FULL = 0xffffffffu;
                        const bool g = (pass_bits >> i) & 1u;
                        const unsigned m = __ballot_sync(FULL, g);
                        if (m != 0) {
                            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                            const uint64_t out_base = static_cast<uint64_t>(row_q) * cand_cap;
                            const float x = x_row[i];
                            const uint32_t col = kv_offset + v_offset;
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
                                warpq_idx[queue_base + pos] = static_cast<int32_t>(col);
                                if (refresh_every > 0) {
                                    int braw = static_cast<int>((x - o_reg[i]) * inv_reg[i]);
                                    int b = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
#if DSA_MERGE_BCOUNT
                                    unsigned peers = __match_any_sync(m, b);
                                    int leader = __ffs(peers) - 1;
                                    if (static_cast<int>(lane_idx) == leader)
                                        atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], __popc(peers));
#else
                                    atomicAdd(&bcount[static_cast<uint64_t>(row_q) * num_buckets + b], 1);
#endif
                                }
                            }
                            if (lane_idx == 0) warpq_count[count_slot] = qn + cnt;
                            __syncwarp(FULL);
                        }
                    }
                }
#endif
#if !defined(DENSE_WRITE) && !defined(DSA_NULL_EPILOGUE)
                // Publish scan progress every DSA_REFRESH_STRIDE KV blocks so the
                // spare-warp refresh daemon can trigger. Publishing only every
                // stride (not every block) keeps the __threadfence cost to 1/stride.
                if (threadIdx.x == 0 && ((kv_block_idx + 1) % DSA_REFRESH_STRIDE) == 0) {
                    __threadfence_block();
                    *kv_progress_ptr = static_cast<int>(kv_block_idx + 1);
                }
#endif
            }
#ifdef DSA_NULL_EPILOGUE
            // Impossible-in-practice guard: keeps `null_acc` (and thus the whole
            // scoring chain) alive without any real store traffic.
            if (__float_as_uint(null_acc) == 0xdeadbeefu)
                cand_val[0] = null_acc;
#endif
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
        // Publish scan completion so the spare-warp refresh daemon stops. Any one
        // math thread does it; unconditional so a no-work CTA also releases the
        // spare warps. Publishing late is safe (spare refresh is pure optimization).
        if (threadIdx.x == 0) {
            __threadfence_block();
            *scan_done_flag = 1;
        }
    }

    // Free tensor memory
    __syncthreads();
    if (is_tma_load_warp)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
}

} // namespace dsa_marsco

// TidalDecode GQA inner-product top-k — Blackwell / B200 SM100 port.
//
// Ported from the Hopper WGMMA kernel
// hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel to Blackwell tcgen05 UMMA
// + Tensor Memory, reusing the proven SM100 pipeline / sparse epilogue /
// spare-warp refresh from the DSA port (sm100_dsa_marsco.cuh).
//
// fp16 note: for fp16 a 128B swizzle atom holds 64 half elements, so the UMMA
// K-major descriptors use BLOCK_K=64 (D is split into D/64 swizzle atoms, each
// further into 64/UMMA_K UMMA-K steps). KV is made contiguous per KV head by a
// pre-pass gather kernel (paged -> [hkv, M, D]); this CTA loads it with TMA
// (which lays out the 128B-swizzle smem the descriptors expect) — the same
// approach DSA uses. Query rows (QN=8, GQA group) are TMA-loaded once.
//
// Score = plain inner product <q[qrow], kv[kv]> (no ReLU/weight/scale): the UMMA
// accumulator IS the score. Epilogue is the DSA bucket-gated sparse emit.

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm100.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <type_traits>

namespace litetopk_marsco {

using namespace deep_gemm;
using namespace deep_gemm::sm90;
using namespace deep_gemm::sm100;

#ifndef LITETOPK_REFRESH_STRIDE
#define LITETOPK_REFRESH_STRIDE 16
#endif

// Warp-local candidate queue (ported from the DSA B200 kernel's
// DSA_WARP_QUEUE): passing lanes append into a per-(warp, row) shared-memory
// queue and flush with ONE qcount atomicAdd + coalesced stores once full,
// instead of one atomic + scattered streaming stores per tile.
#ifndef LITETOPK_WARP_QUEUE
#define LITETOPK_WARP_QUEUE 1
#endif
#ifndef LITETOPK_WARP_QUEUE_CAP
#define LITETOPK_WARP_QUEUE_CAP 32
#endif

#define LITETOPK_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define LITETOPK_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))

// Score contiguous per-head KV [hkv, M, D] (gathered from the paged cache) against
// QN=8 padded query rows per GQA head group, emitting bucket-gated sparse
// candidates (x = -IP) into buf_val/buf_idx.
//
//   grid.x : GQA head group / flat-batch row group (so co-resident CTAs of
//            different groups march over the SAME KV tiles in lockstep and
//            hit L2 on the shared corpus in flat-batch mode)
//   grid.y : KV tile index (persistent, stride gridDim.y)
//   grid.y : GQA head group (grid_head) -> kv_head = grid_head / q_group_size
template <class cand_t>
__device__ __forceinline__ cand_t litetopk_cand_cast(float x);
template <> __device__ __forceinline__ __half litetopk_cand_cast<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ float litetopk_cand_cast<float>(float x) { return x; }

template <uint32_t QN, uint32_t kHeadDim,
          uint32_t BM, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename cand_t = __half,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
__global__ __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_litetopk_ip(const uint32_t M,                     // scan bound: rows [start_row, M) are scored
                    const uint32_t kv_row_stride,         // rows per KV head in the [hkv, S, D] tensor (S >= M)
                    const uint32_t q_group_size,          // grid_head -> kv_head divisor
                    const uint32_t logical_rows,          // valid query rows (group <= QN)
                    const uint32_t start_row,             // first KV row scanned
                    const float* __restrict__ origin,     // [Rwork]
                    const float* __restrict__ inv_delta,  // [Rwork]
                    int32_t* __restrict__ th_bucket,      // [Rwork]
                    int32_t* __restrict__ qcount,         // [Rwork] candidate counter
                    int32_t* __restrict__ bcount,         // [Rwork, num_buckets]
                    int32_t* __restrict__ cta_done,       // [n_head_groups] zeroed CTA-completion counters
                                                          // (nullptr: skip the in-kernel final refresh)
                    __half* __restrict__ dense_out,       // non-null: ablation-baseline mode — write every
                                                          // score to [n_head_groups*logical_rows, buf_cap]
                                                          // and skip the sparse gate/emit entirely
                    cand_t* __restrict__ buf_val,         // [Rwork, buf_cap] x=-IP (fp16 or fp32)
                    int32_t* __restrict__ buf_idx,        // [Rwork, buf_cap]
                    const uint32_t buf_cap,
                    const uint32_t num_buckets,
                    const uint32_t topk,
                    const uint32_t refresh_every,
                    const bool store_bucket_space,   // see epilogue: false keeps the legacy
                                                      // "store raw score" behavior, required
                                                      // when tail-mode seed prefill (the
                                                      // arch-agnostic launch_hopper_seed_from_
                                                      // sample_fp16) shares buf_val with the scan
                                                      // — that kernel is unaware of bucket-space
                                                      // storage and always writes raw scores.
                    const __grid_constant__ cute::TmaDescriptor tensor_map_q,   // [Hgrp*QN, D]
                    const __grid_constant__ cute::TmaDescriptor tensor_map_kv) { // [hkv*M, D]
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    const uint32_t grid_head = blockIdx.x;
    const uint32_t kv_head = (q_group_size > 0) ? (grid_head / q_group_size) : grid_head;
    const uint32_t query_base = grid_head * QN;

    // spec=128: TMA+UMMA+2 spare（refresh）；spec=64: TMA+UMMA，refresh 内嵌 math warp。
    // 64 时总线程非 128 倍数 → 不能用 setmaxnreg（warpgroup 粒度），寄存器预算全靠
    // launch_bounds（320 线程 → 204/线程，比 384 线程的 168 多 36）。
    DG_STATIC_ASSERT((kNumSpecializedThreads == 128 or kNumSpecializedThreads == 64)
                     and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
    }
    __syncwarp();

    // UMMA: A = KV [BM, D] K-major, B = q [QN, D] K-major, C = [BM, QN] IP.
    // Per-instruction M caps at 128; BM = 256 runs TWO math warpgroups, each
    // owning a 128-row sub-tile with its own TMEM accumulator (DSA layout).
    constexpr uint32_t UMMA_M = 128;
    constexpr uint32_t UMMA_N = QN;
    DG_STATIC_ASSERT(BM == kNumMathThreads, "one kv row per math lane");
    DG_STATIC_ASSERT(BM == UMMA_M * kNumMathWarpGroups, "BM must be 128 per warpgroup");
    constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::half_t);   // 16
    constexpr uint32_t SW_K = 128 / sizeof(cutlass::half_t);    // 64: 128B swizzle atom (fp16)
    DG_STATIC_ASSERT(kHeadDim % SW_K == 0, "D must be multiple of 64 (fp16 128B swizzle atom)");

    // Large-D support (e.g. MS MARCO 768-d embeddings): a KV tile is split
    // into D-chunks of CHUNK_D columns; the TMA/UMMA pipeline is chunk-
    // granular and the UMMA accumulates chunks into the same TMEM buffer, so
    // smem per stage stays bounded (BM x CHUNK_D). D <= 256 keeps one chunk
    // (identical to the original behaviour). Q is small and stays fully
    // resident ([QN, D]).
    // QN > 8 (flat-batch single-pass, e.g. 64 queries) needs Q fully resident
    // ([QN, D] can be 96KB), so KV chunks shrink to 128 columns to fit.
    // Stage byte budget: 64KB normally; 32KB when the (large) Q block is
    // resident (QN > 8) so 3 stages + Q still fit the 227KB smem limit.
    constexpr uint32_t kStageBytes = (QN > 8) ? 32768 : 65536;
    constexpr uint32_t kChunkCapElems = kStageBytes / (BM * sizeof(__half));
    constexpr uint32_t kChunkPref = (QN > 8) ? (kHeadDim > 128 ? 128 : kHeadDim)
                                             : ((kHeadDim > 256) ? 256 : kHeadDim);
    constexpr uint32_t CHUNK_D = kChunkPref < kChunkCapElems ? kChunkPref : kChunkCapElems;
    constexpr uint32_t kNumDChunks = kHeadDim / CHUNK_D;
    DG_STATIC_ASSERT(kHeadDim % CHUNK_D == 0, "D must be a multiple of CHUNK_D");
    DG_STATIC_ASSERT(QN % 8 == 0 && QN <= 64, "QN must be a multiple of 8, <= 64");

    static constexpr uint32_t SMEM_Q_BYTES = QN * kHeadDim * sizeof(__half);
    static constexpr uint32_t SMEM_KV_TILE_BYTES = BM * CHUNK_D * sizeof(__half);

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    __half* smem_q = reinterpret_cast<__half*>(smem_buffer);
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__half*>(smem_buffer + SMEM_Q_BYTES + SMEM_KV_TILE_BYTES * i);
    });
    // Double-buffered TMEM accumulators: the UMMA warp writes buffer
    // (tile & 1) while the math warps drain the other one, so the tiny N=8 MMA
    // is no longer serialized against the math epilogue tile-by-tile.
    constexpr uint32_t kNumAccBufs = 2;
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_Q_BYTES + SMEM_KV_TILE_BYTES * kNumKVStages);
    auto full_kv_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumKVStages + i; });
    // Indexed [wg * kNumAccBufs + buf]
    auto full_umma_barriers = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + 2 * kNumKVStages + i; });
    auto empty_umma_barriers= PatternVisitor([&](const uint32_t& i) { return barrier_ptr + 2 * kNumKVStages + kNumAccBufs * kNumMathWarpGroups + i; });
    auto q_ready_barrier    = barrier_ptr + 2 * kNumKVStages + 2 * kNumAccBufs * kNumMathWarpGroups;
    auto tmem_ptr_in_smem   = reinterpret_cast<uint32_t*>(q_ready_barrier + 1);
    auto scan_done_flag     = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
    auto kv_progress_ptr    = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 2);

    // Warp-local candidate queue: only for the small-QN (GQA) shape — at QN=64
    // the per-(warp, row) queues would need ~50KB of smem; the direct-atomic
    // emit is used instead (the h100 marsco bundle makes the same choice for
    // bs >= 32).
    constexpr bool kUseWarpQueue = LITETOPK_WARP_QUEUE && (QN <= 8);
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
    auto warpq_val = reinterpret_cast<cand_t*>(warpq_count + (kUseWarpQueue ? kNumMathWarps * QN : 0));
    auto warpq_idx = reinterpret_cast<int32_t*>(warpq_val + (kUseWarpQueue ? kNumMathWarps * QN * LITETOPK_WARP_QUEUE_CAP : 0));
    // Per-row coefficients live in smem (register arrays would need 3*QN regs
    // per thread at QN=64). Each math warp fills all QN slots redundantly
    // (identical values), so no cross-warp sync is needed.
    auto coeff_o    = reinterpret_cast<float*>(warpq_idx + (kUseWarpQueue ? kNumMathWarps * QN * LITETOPK_WARP_QUEUE_CAP : 0));
    auto coeff_inv  = coeff_o + QN;
    auto coeff_gate = reinterpret_cast<int32_t*>(coeff_inv + QN);

    // TMEM is allocated in 32-column granules on tcgen05; QN(=8) columns per math
    // warpgroup would be an illegal sub-granule allocation, so each (warpgroup,
    // buffer) accumulator occupies its own 32-col granule.
    constexpr uint32_t kTmemColsPerWG = QN <= 32 ? 32 : QN;   // 32-col granules
    constexpr uint32_t kNumTmemCols = kTmemColsPerWG * kNumAccBufs * kNumMathWarpGroups;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory columns");
    DG_STATIC_ASSERT(UMMA_N <= kTmemColsPerWG, "QN must fit the accumulator granule");

    const uint32_t num_kv_tiles_all = ceil_div(M, BM);
    const uint32_t start_tile = start_row / BM;

    const bool is_tma_warp  = (warp_idx == kNumMathThreads / 32);
    const bool is_umma_warp = (warp_idx == kNumMathThreads / 32 + 1);

    if (is_tma_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++i) {
            full_kv_barriers[i]->init(1);
            // KV smem is consumed by the UMMA warp only; it releases the stage
            // via umma_arrive (tcgen05.commit) once the chunk's MMAs retire.
            empty_kv_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumAccBufs * kNumMathWarpGroups; ++i) {
            full_umma_barriers[i]->init(1);
            empty_umma_barriers[i]->init(128);
        }
        q_ready_barrier->init(1);
        cutlass::arch::fence_barrier_init();
        *scan_done_flag = 0;
        *kv_progress_ptr = 0;
    } else if (is_umma_warp) {
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    constexpr uint32_t kNumSpecializedRegisters = 40;
    constexpr uint32_t kNumMathRegisters = 200;

    // Warp-collective in-scan threshold refresh: re-derive th_bucket[row] from
    // the live bcount histogram (32-lane prefix scan per 32-bucket batch).
    // Hoisted here so both the spare-warp poller (spec=128) and the math-warp
    // inline call sites (spec=64) can use it.
    const auto refresh_row = [&](const uint32_t r) {
        if (r >= logical_rows) return;
        const uint32_t row = query_base + r;
        const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
        int carry = 0, found = static_cast<int>(num_buckets) - 1; bool done = false;
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
            bool hit = (b < num_buckets) && (incl >= static_cast<int>(topk)) && (incl - v < static_cast<int>(topk));
            unsigned hm = __ballot_sync(0xffffffffu, hit);
            if (hm) { found = static_cast<int>(base) + (__ffs(hm) - 1); done = true; }
            else    { carry += __shfl_sync(0xffffffffu, prefix, 31); }
        }
        if (lane_idx == 0 && found < th_bucket[row]) th_bucket[row] = found;
    };

    if (is_tma_warp) {
        if constexpr (kNumSpecializedThreads == 128)
            cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        if (cute::elect_one_sync()) {
            // Load QN query rows once (K-major, 128B swizzle). Issue the TMA copy
            // first, THEN arrive_and_expect_tx (matches the DSA ordering; expecting
            // before the copy can satisfy the barrier early and let UMMA read
            // un-loaded smem).
            tma_copy<kHeadDim, QN, 128, __half>(
                &tensor_map_q, q_ready_barrier, smem_q, 0, query_base);
            q_ready_barrier->arrive_and_expect_tx(SMEM_Q_BYTES);
            // Stream KV tiles, one D-chunk per pipeline stage.
            uint32_t produced = 0;
            for (uint32_t t = start_tile + blockIdx.y; t < num_kv_tiles_all; t += gridDim.y) {
                #pragma unroll
                for (uint32_t c = 0; c < kNumDChunks; ++c, ++produced) {
                    const uint32_t stage = produced % kNumKVStages;
                    const uint32_t phase = (produced / kNumKVStages) & 1;
                    empty_kv_barriers[stage]->wait(phase ^ 1);
                    tma_copy<CHUNK_D, BM, 128, __half>(
                        &tensor_map_kv, full_kv_barriers[stage], smem_kv[stage], c * CHUNK_D,
                        kv_head * kv_row_stride + t * BM);
                    full_kv_barriers[stage]->arrive_and_expect_tx(SMEM_KV_TILE_BYTES);
                }
            }
        }
    } else if (is_umma_warp) {
        if constexpr (kNumSpecializedThreads == 128)
            cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);
        q_ready_barrier->wait(0);

        auto instr_desc = cute::UMMA::make_instr_desc<cutlass::half_t, cutlass::half_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

        uint32_t consumed = 0;
        uint32_t tiles_done = 0;
        for (uint32_t t = start_tile + blockIdx.y; t < num_kv_tiles_all; t += gridDim.y, ++tiles_done) {
            const uint32_t acc_buf = tiles_done % kNumAccBufs;
            const uint32_t acc_phase = (tiles_done / kNumAccBufs) & 1;
            // Accumulator handoff: wait once per tile (before the first chunk).
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++i)
                empty_umma_barriers[i * kNumAccBufs + acc_buf]->wait(acc_phase ^ 1);
            #pragma unroll
            for (uint32_t c = 0; c < kNumDChunks; ++c, ++consumed) {
                const uint32_t stage = consumed % kNumKVStages;
                const uint32_t phase = (consumed / kNumKVStages) & 1;
                full_kv_barriers[stage]->wait(phase);
                #pragma unroll
                for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
                    // Rebuild the K-major descriptor per K step (DSA style). fp16 uses
                    // SW_K=64 (128B swizzle atom); the chunk is split into CHUNK_D/SW_K
                    // atoms, each with SW_K/UMMA_K=4 K-steps. The swizzle atom is
                    // selected by the smem base offset (sw * MN * SW_K); Q stays fully
                    // resident so its atom index carries the chunk offset.
                    //
                    // NOTE: deep_gemm's relaxed SM100_MMA_F16BF16_SS wrapper (unlike
                    // cute::SM100_MMA_*) has NO elect_one_sync guard inside fma, so the
                    // issue loop must be single-lane: without the guard all 32 lanes
                    // issue the MMA and every accumulate k-step lands 32 times.
                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t sw = 0; sw < CHUNK_D / SW_K; ++sw) {
                            #pragma unroll
                            for (uint32_t kk = 0; kk < SW_K / UMMA_K; ++kk) {
                                const uint32_t kstep = (c * (CHUNK_D / SW_K) + sw) * (SW_K / UMMA_K) + kk;
                                auto a_desc = make_umma_desc<cute::UMMA::Major::K, 0, SW_K, 128>(
                                    smem_kv[stage] + sw * BM * SW_K, i * UMMA_M, kk * UMMA_K);
                                auto b_desc = make_umma_desc<cute::UMMA::Major::K, 0, SW_K, 128>(
                                    smem_q + (c * (CHUNK_D / SW_K) + sw) * QN * SW_K, 0, kk * UMMA_K);
                                SM100_MMA_F16BF16_SS::fma(a_desc, b_desc,
                                                          (i * kNumAccBufs + acc_buf) * kTmemColsPerWG,
                                                          kstep, runtime_instr_desc);
                            }
                        }
                    }
                }
                // Release the KV stage as soon as this chunk's MMAs retire
                // (umma_arrive = tcgen05.commit tracks smem-read completion).
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_kv_barriers[stage]));
            }
            // Hand the accumulated tile to the math warps.
            #pragma unroll
            for (uint32_t i = 0; i < kNumMathWarpGroups; ++i)
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(full_umma_barriers[i * kNumAccBufs + acc_buf]));
        }
    } else if (warp_idx >= kNumMathThreads / 32) {
        // Spare (refresh-poller) warps — only exist when spec = 128.
        if constexpr (kNumSpecializedThreads > 64) {
            cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
            const bool in_scan_refresh = (refresh_every > 0 && refresh_every != 0x7fffffff);
            if (in_scan_refresh) {
                const uint32_t spare_id = warp_idx - (kNumMathThreads / 32 + 2);
                constexpr uint32_t kNumSpare = (kNumSpecializedThreads / 32) - 2;
                // NOTE: no refresh on scan_done — a teardown refresh cannot help
                // this kernel anymore and the host always runs
                // refresh_threshold_from_bcount right after; the teardown pass's
                // cold bcount reads (~3-5 µs) also serialize CTA exit across waves.
                int last = 0;
                while (true) {
                    const int d = *scan_done_flag, p = *kv_progress_ptr;
                    if (p > last) { for (uint32_t r = spare_id; r < logical_rows; r += kNumSpare) refresh_row(r); last = p; }
                    else if (d) break;
                    else __nanosleep(2048);
                }
            }
        }
    } else {
        if constexpr (kNumSpecializedThreads == 128)
            cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const auto& tmem_wg_base = __shfl_sync(0xffffffff, warpgroup_idx * kNumAccBufs * kTmemColsPerWG, 0);
        const uint32_t warp_kv_off = warp_idx * 32;

        // Per-row coefficients live in smem tables (QN can be 64: register
        // arrays would blow the budget). Each math warp fills all QN slots
        // redundantly with identical values — no cross-warp sync needed.
        for (uint32_t r = lane_idx; r < QN; r += 32) {
            const uint32_t row = query_base + r;
            const bool valid = r < logical_rows;
            coeff_o[r]    = valid ? origin[row] : 0.f;
            coeff_inv[r]  = valid ? inv_delta[row] : 0.f;
            coeff_gate[r] = valid ? th_bucket[row] : 0;
        }
        __syncwarp();
        // Gate-reload / refresh-progress cadence in tiles. In in-scan-refresh
        // mode the runtime refresh_every IS the stride (a persistent CTA only
        // strides over ~scan_tiles/gridDim.y tiles, so a fixed stride of 16
        // may simply never fire); otherwise fall back to the compile default.
        const uint32_t gate_stride = (refresh_every > 0 && refresh_every != 0x7fffffff)
                                         ? refresh_every : LITETOPK_REFRESH_STRIDE;

        // C1b：队列计数 qn 是 warp-uniform 的——增量 = __popc(ballot)（全 lane 同值）、
        //   倒队列判断 qn+cnt>CAP 也全 warp 一致，所以 32 个 lane 各持一份寄存器副本
        //   永远相等。免去热路径上每轮 lane0 smem 读 + shfl 广播 + lane0 写回；
        //   smem 中的 warpq_count 槽位保留不用（避免动 host 侧 smem 布局）。
        //   索引恒为 r（kUseWarpQueue ⇒ QN≤8 ⇒ rbase==0），unroll 下静态可解析不落 local。
        int qn_reg[kUseWarpQueue ? QN : 1];
        #pragma unroll
        for (uint32_t r = 0; r < (kUseWarpQueue ? QN : 1u); ++r) qn_reg[r] = 0;
        if constexpr (kUseWarpQueue) {
            DG_STATIC_ASSERT(LITETOPK_WARP_QUEUE_CAP >= 32, "queue must hold one warp ballot");
        }
        // Drain a (warp, row) queue: one qcount atomic + coalesced buf stores.
        // (Only invoked when kUseWarpQueue; compiles harmlessly otherwise.)
        const auto flush_warpq = [&](const uint32_t r, const uint32_t row) {
            const uint32_t slot = warp_idx * QN + r;
            const uint32_t qbase = slot * LITETOPK_WARP_QUEUE_CAP;
            const int qn = qn_reg[r];
            if (qn > 0) {
                qn_reg[r] = 0;
                int base = 0;
                if (lane_idx == 0) base = atomicAdd(qcount + row, qn);
                base = __shfl_sync(0xffffffffu, base, 0);
                const uint64_t out_base = static_cast<uint64_t>(row) * buf_cap;
                for (int t2 = static_cast<int>(lane_idx); t2 < qn; t2 += 32) {
                    const int w = base + t2;
                    if (w < static_cast<int>(buf_cap)) {
                        LITETOPK_ST_CAND_VAL(buf_val[out_base + w], warpq_val[qbase + t2]);
                        LITETOPK_ST_CAND_IDX(buf_idx[out_base + w], warpq_idx[qbase + t2]);
                    }
                }
            }
            __syncwarp(0xffffffffu);
        };

        uint32_t consumed = 0;   // tile-granular (KV stages are UMMA-owned now)
        for (uint32_t t = start_tile + blockIdx.y; t < num_kv_tiles_all; t += gridDim.y, ++consumed) {
            const uint32_t acc_buf = consumed % kNumAccBufs;
            const uint32_t acc_phase = (consumed / kNumAccBufs) & 1;
            const uint32_t tmem_start = tmem_wg_base + acc_buf * kTmemColsPerWG;
            full_umma_barriers[warpgroup_idx * kNumAccBufs + acc_buf]->wait(acc_phase);

            const uint32_t kv_row = t * BM + warp_kv_off + lane_idx;
            const bool kv_valid = kv_row < M;

            // Drain the accumulator in 8-column slices: bounds register
            // pressure at QN=64 (per-slice acc/x/b registers only) and is the
            // identity transform at QN=8.
            #pragma unroll
            for (uint32_t sl = 0; sl < QN / 8; ++sl) {
                const uint32_t rbase = sl * 8;
                uint32_t acc[8];
                [&]<size_t... Is>(cute::index_sequence<Is...>) {
                    cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start + rbase, acc[Is]...);
                }(cute::make_index_sequence<8>{});
                cutlass::arch::fence_view_async_tmem_load();
                // Release the accumulator as soon as the LAST slice is read
                // (before its epilogue) — same handoff point as the pre-slice
                // code, so the UMMA warp is never gated on emit work.
                if (sl == QN / 8 - 1)
                    empty_umma_barriers[warpgroup_idx * kNumAccBufs + acc_buf]->arrive();

                if (dense_out != nullptr) {
                    // "Ours-dense" ablation baseline: same TMA/UMMA/TMEM engine,
                    // dense-store epilogue (the downstream top-k is torch.topk).
                    if (kv_valid) {
                        #pragma unroll
                        for (uint32_t r = 0; r < 8; ++r) {
                            if (rbase + r >= logical_rows) continue;
                            dense_out[static_cast<uint64_t>(grid_head * logical_rows + rbase + r) * buf_cap + kv_row] =
                                __float2half(*reinterpret_cast<float*>(&acc[r]));
                        }
                    }
                } else {
                // Batched-vote emit (ported from the DSA kernel): first compute
                // the slice's row predicates with pure register ops (pass_bits),
                // then ONE __any_sync decides whether the warp enters the emit
                // machinery. __any_sync is warp-uniform, so the whole warp
                // branches together and the per-row ballot/shfl/syncwarp logic
                // inside participates with full masks.
                // 候选缓冲存"桶空间坐标" bq = (x-o)*inv（仿射变换，严格保序，不是
                // 取整后的桶号）而不是原始分数 x：select 全程只需要 bq 的相对顺序
                // （radix/边界选择的比较、排序在 bq 空间和 x 空间等价），原始分数
                // 只在最终 K 个输出处一次性转回（host 侧 val = bq*delta+o，见
                // litetopk_sm100_torch.cu 的 debucket 步骤），不必在扫描/select 的每
                // 候选路径上都保留/重算——仅当 store_bucket_space 时启用；tail-mode
                // 种子预填（launch_hopper_seed_from_sample_fp16，arch-agnostic 共享
                // kernel）不知道这个约定、总是写原始分数，与它共享 buf_val 时必须
                // 退回旧行为（存 x），否则同一缓冲区里桶空间值和原始分数会混着算。
                float cand_reg[8];   // 存入 buf_val 的值: bq(桶空间)或 x(原始分数), 见上方注释
                int b_reg[8];
                uint32_t pass_bits = 0;
                #pragma unroll
                for (uint32_t r = 0; r < 8; ++r) {
                    if (rbase + r >= logical_rows) continue;
                    // 门检/桶号就地只算一次，无条件直线码（ILP 交错；把工作挪进
                    // if(g) 反而串行化，实测 +13%）。fp32 路径桶号与 select 从
                    // 存储值重算的是同一表达式，天然一致；fp16 候选仍需按"实际存储
                    // 值"的圆整重推桶号（不变式）——根据 store_bucket_space 圆整的
                    // 对象是 bq 还是 x。
                    const float score = *reinterpret_cast<float*>(&acc[r]);
                    const float x = -score;
                    const float bq = (x - coeff_o[rbase + r]) * coeff_inv[rbase + r];
                    const int braw = static_cast<int>(bq);
                    cand_reg[r] = store_bucket_space ? bq : x;
                    if constexpr (std::is_same_v<cand_t, __half>) {
                        int br;
                        if (store_bucket_space) {
                            br = static_cast<int>(__half2float(__float2half(bq)));
                        } else {
                            const float xr = __half2float(__float2half(x));
                            br = static_cast<int>((xr - coeff_o[rbase + r]) * coeff_inv[rbase + r]);
                        }
                        b_reg[r] = br < 0 ? 0 : (br > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : br);
                    } else {
                        b_reg[r] = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                    }
                    const bool g = kv_valid && (braw <= coeff_gate[rbase + r]);
                    if (g) pass_bits |= (1u << r);
                }
                if (__any_sync(0xffffffffu, pass_bits != 0u)) {
                if constexpr (!kUseWarpQueue) {
                    // Direct-atomic emit with PARALLEL row reservations: lane r
                    // reserves row r's slots with one warp-wide atomicAdd batch
                    // (8 L2 round-trips overlap in a single instruction), then
                    // each passing lane writes its ballot-prefix slot directly.
                    unsigned m8[8];
                    #pragma unroll
                    for (uint32_t r = 0; r < 8; ++r)
                        m8[r] = __ballot_sync(0xffffffffu, (pass_bits >> r) & 1u);
                    int my_base = 0;
                    if (lane_idx < 8 && rbase + lane_idx < logical_rows && m8[lane_idx] != 0)
                        my_base = atomicAdd(qcount + query_base + rbase + lane_idx,
                                            __popc(m8[lane_idx]));
                    #pragma unroll
                    for (uint32_t r = 0; r < 8; ++r) {
                        if (rbase + r >= logical_rows) continue;
                        const unsigned m = m8[r];
                        if (m == 0) continue;
                        const int base = __shfl_sync(0xffffffffu, my_base, r);
                        if ((pass_bits >> r) & 1u) {
                            const uint32_t row = query_base + rbase + r;
                            const unsigned below = (lane_idx == 0) ? 0u : ((1u << lane_idx) - 1u);
                            const int w = base + __popc(m & below);
                            if (w < static_cast<int>(buf_cap)) {
                                LITETOPK_ST_CAND_VAL(buf_val[static_cast<uint64_t>(row) * buf_cap + w], litetopk_cand_cast<cand_t>(cand_reg[r]));
                                LITETOPK_ST_CAND_IDX(buf_idx[static_cast<uint64_t>(row) * buf_cap + w], static_cast<int32_t>(kv_row));
                            }
                            if (refresh_every > 0)
                                atomicAdd(&bcount[static_cast<uint64_t>(row) * num_buckets + b_reg[r]], 1);
                        }
                    }
                } else {
                #pragma unroll
                for (uint32_t r = 0; r < 8; ++r) {
                    if (rbase + r >= logical_rows) continue;
                    const uint32_t row = query_base + rbase + r;
                    const float cand = cand_reg[r];
                    const int b = b_reg[r];
                    const bool g = (pass_bits >> r) & 1u;

                    const unsigned m = __ballot_sync(0xffffffffu, g);
                    if (m != 0) {
                        const int cnt = __popc(m);
                        const unsigned below = (lane_idx == 0) ? 0u : ((1u << lane_idx) - 1u);
                        if constexpr (kUseWarpQueue) {
                            const uint32_t slot = warp_idx * QN + rbase + r;
                            const uint32_t qbase = slot * LITETOPK_WARP_QUEUE_CAP;
                            int qn = qn_reg[r];   // rbase==0（QN≤8），r 静态
                            if (qn + cnt > static_cast<int>(LITETOPK_WARP_QUEUE_CAP)) {
                                int base = 0;
                                if (lane_idx == 0) base = atomicAdd(qcount + row, qn);
                                base = __shfl_sync(0xffffffffu, base, 0);
                                const uint64_t out_base = static_cast<uint64_t>(row) * buf_cap;
                                for (int t2 = static_cast<int>(lane_idx); t2 < qn; t2 += 32) {
                                    const int w = base + t2;
                                    if (w < static_cast<int>(buf_cap)) {
                                        LITETOPK_ST_CAND_VAL(buf_val[out_base + w], warpq_val[qbase + t2]);
                                        LITETOPK_ST_CAND_IDX(buf_idx[out_base + w], warpq_idx[qbase + t2]);
                                    }
                                }
                                qn = 0;
                                __syncwarp(0xffffffffu);
                            }
                            if (g) {
                                const int pos = qn + __popc(m & below);
                                warpq_val[qbase + pos] = litetopk_cand_cast<cand_t>(cand);
                                warpq_idx[qbase + pos] = static_cast<int32_t>(kv_row);
                                if (refresh_every > 0) {
                                    atomicAdd(&bcount[static_cast<uint64_t>(row) * num_buckets + b], 1);
                                }
                            }
                            qn_reg[r] = qn + cnt;
                            __syncwarp(0xffffffffu);
                        } else {
                            // unreachable: the !kUseWarpQueue shape uses the
                            // parallel-reservation emit above
                            (void)cnt; (void)below; (void)b; (void)cand;
                        }
                    }
                }
                }  // kUseWarpQueue
                }  // __any_sync(pass_bits)
                }  // dense_out == nullptr (sparse path)
            }  // slice loop

            if (((consumed + 1) % gate_stride) == 0) {
                // spec=64：refresh 内嵌在 math warp（无 spare 警戒 warp）。放在
                // empty_umma arrive 之后（此处），与 UMMA 的下一 tile MMA 重叠；
                // 8 个 math warp 均分行，然后紧接着的 gate 重读立刻用上收紧的 th。
                if constexpr (kNumSpecializedThreads == 64) {
                    if (refresh_every > 0 && refresh_every != 0x7fffffff) {
                        for (uint32_t r = warp_idx; r < logical_rows; r += kNumMathWarps)
                            refresh_row(r);
                    }
                }
                // Periodically re-read the (possibly refresh-tightened) gate. A
                // stale gate only over-emits, so recall is unaffected. Warp-
                // redundant writes of identical values — benign.
                for (uint32_t r = lane_idx; r < QN; r += 32)
                    coeff_gate[r] = (r < logical_rows) ? th_bucket[query_base + r] : 0;
                __syncwarp();
                if (threadIdx.x == 0) {
                    __threadfence_block();
                    *kv_progress_ptr = static_cast<int>(consumed + 1);
                }
            }
        }
        if constexpr (kUseWarpQueue) {
            #pragma unroll
            for (uint32_t r = 0; r < QN; ++r)
                if (r < logical_rows) flush_warpq(r, query_base + r);
        }
        if (threadIdx.x == 0) { __threadfence_block(); *scan_done_flag = 1; }
    }

    __syncthreads();   // all of this CTA's emissions / bcount atomics issued

    // Last-CTA-per-head-group final threshold refresh. Replaces the separate
    // host refresh kernel between scan and select (one less launch). The
    // threadfence + atomic completion counter is the standard release/acquire
    // handshake: when a CTA's atomicAdd returns gridDim.y-1, every sibling
    // CTA's bcount atomics are L2-visible; bcount is then read with __ldcg to
    // bypass this SM's (possibly stale) L1.
    if (cta_done != nullptr) {
        auto last_cta_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 3);
        if (threadIdx.x == 0) {
            __threadfence();
            *last_cta_flag = (atomicAdd(cta_done + grid_head, 1) == static_cast<int>(gridDim.y) - 1);
        }
        __syncthreads();
        if (*last_cta_flag && warp_idx < kNumMathThreads / 32) {
            constexpr uint32_t kNumMathWarpsF = kNumMathThreads / 32;
            for (uint32_t r = warp_idx; r < logical_rows; r += kNumMathWarpsF) {
                const uint32_t row = query_base + r;
                const int32_t* brow = bcount + static_cast<uint64_t>(row) * num_buckets;
                int carry = 0, found = static_cast<int>(num_buckets) - 1; bool done = false;
                for (uint32_t base = 0; base < num_buckets && !done; base += 32) {
                    uint32_t b = base + lane_idx;
                    int v = (b < num_buckets) ? __ldcg(brow + b) : 0;
                    int prefix = v;
                    #pragma unroll
                    for (int off = 1; off < 32; off <<= 1) {
                        int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
                        if (static_cast<int>(lane_idx) >= off) prefix += nsh;
                    }
                    int incl = carry + prefix;
                    bool hit = (b < num_buckets) && (incl >= static_cast<int>(topk)) && (incl - v < static_cast<int>(topk));
                    unsigned hm = __ballot_sync(0xffffffffu, hit);
                    if (hm) { found = static_cast<int>(base) + (__ffs(hm) - 1); done = true; }
                    else    { carry += __shfl_sync(0xffffffffu, prefix, 31); }
                }
                if (lane_idx == 0 && found < th_bucket[row]) th_bucket[row] = found;
            }
        }
    }

    if (is_tma_warp)
        cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
}

} // namespace litetopk_marsco

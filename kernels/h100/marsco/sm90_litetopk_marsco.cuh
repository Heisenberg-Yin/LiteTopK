// TidalDecode / MS MARCO GQA inner-product top-k — H100 (SM90) port of
// kernels/b200/marsco/sm100_litetopk_marsco.cuh.
//
// The B200 kernel was itself ported FROM the Hopper WGMMA kernel
// (hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel); this file brings the
// B200-side improvements BACK to SM90 on top of the deep_gemm-style pipeline:
//   * chunked-D TMA pipeline (MS MARCO 768-d embeddings) — unchanged;
//   * bucket-space candidate storage (store_bucket_space) + fp16 rounding
//     invariant (bucket re-derived from the ROUNDED stored value) — unchanged;
//   * register-count warp queues (qn_reg) for QN<=8, parallel-reservation
//     direct-atomic emit for QN=64 — unchanged;
//   * persistent tile-strided grid, gate-stride reload, kv-progress publish,
//     last-CTA in-kernel final threshold refresh — unchanged.
//
// What is REPLACED (no SM90 equivalent):
//   * tcgen05 UMMA issued by a dedicated warp + TMEM double-buffered
//     accumulators + 32dp32b8x column-slice loads  ->  WGMMA m64nQNk16
//     (F32F16F16) issued by the math warpgroups, register accumulators.
//     The full/empty_umma barrier pairs and the acc double-buffering vanish
//     (the accumulator lives in the consumer's registers to begin with).
//   * B200's "1 math lane = 1 KV row" TMEM mapping  ->  the WGMMA fragment:
//     each thread holds rows (lane/4, lane/4+8) x column pair
//     ((lane%4)*2, +1) per 8-column group — every accumulator element is a
//     UNIQUE (kv row, q col) candidate, so emit needs NO lane election at
//     all (unlike the DSA port, where quad redundancy comes from the head
//     reduction). Ballots per q column have 8 participating lanes carrying
//     up to 2 candidates each.
//
// Roles: math warps [0, kNumMathThreads/32); then 1 TMA producer warp; the
// remaining specialized warps (spec/32 - 1) are refresh pollers (the B200
// spec=64 build inlined refresh into the math warps because its second spec
// warp was the UMMA issuer — on SM90 that warp is free and polls instead).

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <type_traits>

namespace litetopk_marsco {

using namespace deep_gemm;

#ifndef LITETOPK_REFRESH_STRIDE
#define LITETOPK_REFRESH_STRIDE 16
#endif

#ifndef LITETOPK_WARP_QUEUE
#define LITETOPK_WARP_QUEUE 1
#endif
#ifndef LITETOPK_WARP_QUEUE_CAP
#define LITETOPK_WARP_QUEUE_CAP 32
#endif

#define LITETOPK_ST_CAND_VAL(dst, v) __stcs(&(dst), (v))
#define LITETOPK_ST_CAND_IDX(dst, v) __stcs(&(dst), (v))

// fp16 WGMMA wrapper in the deep_gemm mma::sm90 style (DG only ships FP8/BF16
// selectors; MS MARCO embeddings are fp16).
template <int N_, typename MMA>
struct FP16MMA {
    template <size_t ...Idx>
    CUTLASS_DEVICE static void call_fma_impl(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d, cute::index_sequence<Idx...>) {
        using namespace cute::SM90::GMMA;
        MMA::fma(desc_a, desc_b, d[Idx]..., (scale_d ? ScaleOut::One : ScaleOut::Zero));
    }
    CUTLASS_DEVICE static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d) {
        call_fma_impl(desc_a, desc_b, d, scale_d, cute::make_index_sequence<N_ / 2>{});
    }
    static constexpr int M = 64;
    static constexpr int N = N_;
    static constexpr int K = 16;
    static constexpr int kNumAccum = M * N / 128;
};

template <int N>
struct FP16MMASelector {
    static constexpr auto select_mma() {
        using namespace cute::SM90::GMMA;
        constexpr auto MK = Major::K;
        if constexpr (N == 8)  return MMA_64x8x16_F32F16F16_SS<MK, MK>();
        if constexpr (N == 16) return MMA_64x16x16_F32F16F16_SS<MK, MK>();
        if constexpr (N == 32) return MMA_64x32x16_F32F16F16_SS<MK, MK>();
        if constexpr (N == 64) return MMA_64x64x16_F32F16F16_SS<MK, MK>();
    }
    static constexpr auto select_type() {
        return FP16MMA<N, decltype(select_mma())>();
    }
    using type = decltype(select_type());
};

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
void sm90_litetopk_ip(const uint32_t M,                     // scan bound: rows [start_row, M) are scored
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
                    __half* __restrict__ dense_out,       // non-null: ablation-baseline mode
                    cand_t* __restrict__ buf_val,         // [Rwork, buf_cap] x=-IP (fp16 or fp32)
                    int32_t* __restrict__ buf_idx,        // [Rwork, buf_cap]
                    const uint32_t buf_cap,
                    const uint32_t num_buckets,
                    const uint32_t topk,
                    const uint32_t refresh_every,
                    const bool store_bucket_space,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_q,   // [Hgrp*QN, D]
                    const __grid_constant__ cute::TmaDescriptor tensor_map_kv) { // [hkv*M, D]
    using WGMMA = typename FP16MMASelector<QN>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& lane_idx = ptx::get_lane_idx();

    const uint32_t grid_head = blockIdx.x;
    const uint32_t kv_head = (q_group_size > 0) ? (grid_head / q_group_size) : grid_head;
    const uint32_t query_base = grid_head * QN;

    DG_STATIC_ASSERT((kNumSpecializedThreads == 128 or kNumSpecializedThreads == 64)
                     and kNumMathThreads % 128 == 0, "Invalid threads");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
    }
    __syncwarp();

    // WGMMA: A = KV [BM, D] K-major, B = q [QN, D] K-major, C = [BM, QN] IP.
    // M caps at 64 per instruction; each math warpgroup owns a 64-row sub-tile.
    DG_STATIC_ASSERT(BM == WGMMA::M * kNumMathWarpGroups, "BM must be 64 rows per math warpgroup");
    constexpr uint32_t WGMMA_K = WGMMA::K;                       // 16
    constexpr uint32_t SW_K = 128 / sizeof(cutlass::half_t);     // 64: 128B swizzle atom (fp16)
    DG_STATIC_ASSERT(kHeadDim % SW_K == 0, "D must be multiple of 64 (fp16 128B swizzle atom)");

    // Chunked-D pipeline (identical policy to the B200 kernel).
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
    auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__half*>(smem_buffer + SMEM_Q_BYTES + SMEM_KV_TILE_BYTES * i);
    });
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_Q_BYTES + SMEM_KV_TILE_BYTES * kNumKVStages);
    auto full_kv_barriers   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_kv_barriers  = utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + kNumKVStages + i; });
    auto q_ready_barrier    = barrier_ptr + 2 * kNumKVStages;
    // Slot 0 reserved (held the TMEM pointer on B200 — layout parity with the
    // host smem accounting); 1 = scan_done, 2 = kv progress, 3 = last-CTA flag.
    auto aux_ptr            = reinterpret_cast<uint32_t*>(q_ready_barrier + 1);
    auto scan_done_flag     = reinterpret_cast<volatile int*>(aux_ptr + 1);
    auto kv_progress_ptr    = reinterpret_cast<volatile int*>(aux_ptr + 2);

    constexpr bool kUseWarpQueue = LITETOPK_WARP_QUEUE && (QN <= 8);
    constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
    auto warpq_count = reinterpret_cast<int32_t*>(aux_ptr + 4);
    auto warpq_val = reinterpret_cast<cand_t*>(warpq_count + (kUseWarpQueue ? kNumMathWarps * QN : 0));
    auto warpq_idx = reinterpret_cast<int32_t*>(warpq_val + (kUseWarpQueue ? kNumMathWarps * QN * LITETOPK_WARP_QUEUE_CAP : 0));
    auto coeff_o    = reinterpret_cast<float*>(warpq_idx + (kUseWarpQueue ? kNumMathWarps * QN * LITETOPK_WARP_QUEUE_CAP : 0));
    auto coeff_inv  = coeff_o + QN;
    auto coeff_gate = reinterpret_cast<int32_t*>(coeff_inv + QN);

    const uint32_t num_kv_tiles_all = math::ceil_div(M, BM);
    const uint32_t start_tile = start_row / BM;

    const bool is_tma_warp = (warp_idx == kNumMathThreads / 32);

    if (is_tma_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++i) {
            full_kv_barriers[i]->init(1);
            // Every math thread consumes the stage through its own wgmma and
            // arrives after warpgroup_wait<0> (no UMMA-warp proxy on SM90).
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        q_ready_barrier->init(1);
        cutlass::arch::fence_barrier_init();
        *scan_done_flag = 0;
        *kv_progress_ptr = 0;
    }
    __syncthreads();

    // No setmaxnreg: the shipped spec=64 config launches 320 threads (not a
    // multiple of 128 warpgroups), same as the B200 kernel — the budget comes
    // from launch_bounds (320 threads -> 204 regs). The SM90 math side is
    // register-light anyway (kNumAccum = QN/2 per thread, no TMEM staging).

    // Warp-collective in-scan threshold refresh (verbatim B200).
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
        if (cute::elect_one_sync()) {
            // Load QN query rows once (K-major, 128B swizzle); copy first,
            // THEN arrive_and_expect_tx (the DSA/B200 ordering).
            tma::copy<kHeadDim, QN, 128, __half>(
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
                    tma::copy<CHUNK_D, BM, 128, __half>(
                        &tensor_map_kv, full_kv_barriers[stage], smem_kv[stage], c * CHUNK_D,
                        kv_head * kv_row_stride + t * BM);
                    full_kv_barriers[stage]->arrive_and_expect_tx(SMEM_KV_TILE_BYTES);
                }
            }
        }
    } else if (warp_idx > kNumMathThreads / 32) {
        // Refresh-poller warps (spec/32 - 1 of them; on B200 this slot was the
        // UMMA issuer + optional pollers).
        const bool in_scan_refresh = (refresh_every > 0 && refresh_every != 0x7fffffff);
        if (in_scan_refresh) {
            const uint32_t spare_id = warp_idx - (kNumMathThreads / 32 + 1);
            constexpr uint32_t kNumSpare = (kNumSpecializedThreads / 32) - 1;
            int last = 0;
            while (true) {
                const int d = *scan_done_flag, p = *kv_progress_ptr;
                if (p > last) { for (uint32_t r = spare_id; r < logical_rows; r += kNumSpare) refresh_row(r); last = p; }
                else if (d) break;
                else __nanosleep(2048);
            }
        }
    } else {
        // WGMMA fragment geometry: warp owns 16 KV rows of its warpgroup's
        // m64 sub-tile; each thread holds rows (lane/4, lane/4+8) and, per
        // 8-column group g, the column pair g*8 + (lane%4)*2 + {0,1}:
        //   accum[g*4+0] = (r0, c0), [g*4+1] = (r0, c0+1),
        //   accum[g*4+2] = (r1, c0), [g*4+3] = (r1, c0+1).
        const auto& warpgroup_idx = warp_idx / 4;
        const uint32_t warp_kv_off = warpgroup_idx * WGMMA::M + (warp_idx % 4) * 16;
        const uint32_t r0_off = lane_idx / 4;
        const uint32_t c_pair = (lane_idx % 4) * 2;
        float accum[WGMMA::kNumAccum];

        for (uint32_t r = lane_idx; r < QN; r += 32) {
            const uint32_t row = query_base + r;
            const bool valid = r < logical_rows;
            coeff_o[r]    = valid ? origin[row] : 0.f;
            coeff_inv[r]  = valid ? inv_delta[row] : 0.f;
            coeff_gate[r] = valid ? th_bucket[row] : 0;
        }
        __syncwarp();
        const uint32_t gate_stride = (refresh_every > 0 && refresh_every != 0x7fffffff)
                                         ? refresh_every : LITETOPK_REFRESH_STRIDE;

        // Register-count warp queues (B200 C1b): qn is warp-uniform, every
        // lane tracks a redundant copy — no smem bookkeeping on the hot path;
        // the warpq_count smem slots stay allocated-but-unused (host layout).
        int qn_reg[kUseWarpQueue ? QN : 1];
        #pragma unroll
        for (uint32_t r = 0; r < (kUseWarpQueue ? QN : 1u); ++r) qn_reg[r] = 0;
        if constexpr (kUseWarpQueue) {
            DG_STATIC_ASSERT(LITETOPK_WARP_QUEUE_CAP >= 16, "queue must hold one 8+8 ballot pair");
        }
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

        q_ready_barrier->wait(0);

        uint32_t consumed_chunks = 0;
        uint32_t consumed = 0;   // tiles
        for (uint32_t t = start_tile + blockIdx.y; t < num_kv_tiles_all; t += gridDim.y, ++consumed) {
            // Score the tile: accumulate all D chunks into the register
            // accumulator (scale_d=0 only on the very first k-step).
            #pragma unroll
            for (uint32_t c = 0; c < kNumDChunks; ++c, ++consumed_chunks) {
                const uint32_t stage = consumed_chunks % kNumKVStages;
                const uint32_t phase = (consumed_chunks / kNumKVStages) & 1;
                full_kv_barriers[stage]->wait(phase);
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t sw = 0; sw < CHUNK_D / SW_K; ++sw) {
                    #pragma unroll
                    for (uint32_t kk = 0; kk < SW_K / WGMMA_K; ++kk) {
                        const uint32_t kstep = (c * (CHUNK_D / SW_K) + sw) * (SW_K / WGMMA_K) + kk;
                        // 128B-swizzle atoms: byte geometry identical to the
                        // fp8 D=128 case (64 halfs = 128B rows, SBO = 1024).
                        auto desc_a = mma::sm90::make_smem_desc(
                            smem_kv[stage] + sw * BM * SW_K + (warpgroup_idx * WGMMA::M) * SW_K + kk * WGMMA_K,
                            static_cast<int>(cute::SM90::GMMA::LayoutType::B128), 0, SW_K * 8 * sizeof(__half));
                        auto desc_b = mma::sm90::make_smem_desc(
                            smem_q + (c * (CHUNK_D / SW_K) + sw) * QN * SW_K + kk * WGMMA_K,
                            static_cast<int>(cute::SM90::GMMA::LayoutType::B128), 0, SW_K * 8 * sizeof(__half));
                        WGMMA::wgmma(desc_a, desc_b, accum, kstep > 0);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();
                empty_kv_barriers[stage]->arrive();
            }

            const uint32_t kv_r0 = t * BM + warp_kv_off + r0_off;
            const uint32_t kv_r1 = kv_r0 + 8;
            const bool v0 = kv_r0 < M;
            const bool v1 = kv_r1 < M;

            #pragma unroll
            for (uint32_t g = 0; g < QN / 8; ++g) {
                const float* acc4 = accum + g * 4;

                if (dense_out != nullptr) {
                    // "Ours-dense" ablation: same engine, dense-store epilogue.
                    #pragma unroll
                    for (uint32_t j = 0; j < 4; ++j) {
                        const uint32_t col = g * 8 + c_pair + (j & 1);
                        const uint32_t kvr = (j & 2) ? kv_r1 : kv_r0;
                        if (col < logical_rows && kvr < M)
                            dense_out[static_cast<uint64_t>(grid_head * logical_rows + col) * buf_cap + kvr] =
                                __float2half(acc4[j]);
                    }
                    continue;
                }

                // Gate the 4 held values with straight-line register code
                // (bit j of pass_bits: j&1 = column parity, j&2 = row select),
                // then ONE __any_sync gates the emit machinery (B200 pattern).
                float cand_reg[4];
                int b_reg[4];
                uint32_t pass_bits = 0;
                #pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    const uint32_t col = g * 8 + c_pair + (j & 1);
                    const bool kv_valid = (j & 2) ? v1 : v0;
                    const float score = acc4[j];
                    const float x = -score;
                    const float bq = (x - coeff_o[col]) * coeff_inv[col];
                    const int braw = static_cast<int>(bq);
                    cand_reg[j] = store_bucket_space ? bq : x;
                    if constexpr (std::is_same_v<cand_t, __half>) {
                        // fp16 invariant: bucket re-derived from the ROUNDED
                        // stored value (bq or x, per store_bucket_space).
                        int br;
                        if (store_bucket_space) {
                            br = static_cast<int>(__half2float(__float2half(bq)));
                        } else {
                            const float xr = __half2float(__float2half(x));
                            br = static_cast<int>((xr - coeff_o[col]) * coeff_inv[col]);
                        }
                        b_reg[j] = br < 0 ? 0 : (br > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : br);
                    } else {
                        b_reg[j] = braw < 0 ? 0 : (braw > static_cast<int>(num_buckets) - 1 ? static_cast<int>(num_buckets) - 1 : braw);
                    }
                    const bool gpass = kv_valid && (col < logical_rows) && (braw <= coeff_gate[col]);
                    if (gpass) pass_bits |= (1u << j);
                }

                if (__any_sync(0xffffffffu, pass_bits != 0u)) {
                if constexpr (!kUseWarpQueue) {
                    // Direct-atomic emit with PARALLEL per-column reservations
                    // (B200 m8 pattern): lane qc<8 reserves column qc's slots
                    // in one warp-wide atomicAdd batch, then each holder
                    // writes at base + its ballot prefix. Ballot for column
                    // qc: the 8 lanes with lane%4 == qc/2, row via j-bit.
                    unsigned m0a[8], m1a[8];
                    #pragma unroll
                    for (uint32_t qc = 0; qc < 8; ++qc) {
                        const bool mine = (c_pair == (qc & ~1u));
                        m0a[qc] = __ballot_sync(0xffffffffu, mine && ((pass_bits >> (qc & 1)) & 1u));
                        m1a[qc] = __ballot_sync(0xffffffffu, mine && ((pass_bits >> (2 + (qc & 1))) & 1u));
                    }
                    int my_base = 0;
                    if (lane_idx < 8 && g * 8 + lane_idx < logical_rows) {
                        const int cnt = __popc(m0a[lane_idx]) + __popc(m1a[lane_idx]);
                        if (cnt > 0)
                            my_base = atomicAdd(qcount + query_base + g * 8 + lane_idx, cnt);
                    }
                    #pragma unroll
                    for (uint32_t qc = 0; qc < 8; ++qc) {
                        const uint32_t col = g * 8 + qc;
                        if (col >= logical_rows) continue;
                        const unsigned m0 = m0a[qc], m1 = m1a[qc];
                        if ((m0 | m1) == 0) continue;
                        const int base = __shfl_sync(0xffffffffu, my_base, qc);
                        const uint32_t row = query_base + col;
                        const unsigned below = (1u << lane_idx) - 1u;
                        const bool mine = (c_pair == (qc & ~1u));
                        const uint32_t j0 = (qc & 1);
                        const uint32_t j1 = 2 + (qc & 1);
                        if (mine && ((pass_bits >> j0) & 1u)) {
                            const int w = base + __popc(m0 & below);
                            if (w < static_cast<int>(buf_cap)) {
                                LITETOPK_ST_CAND_VAL(buf_val[static_cast<uint64_t>(row) * buf_cap + w], litetopk_cand_cast<cand_t>(cand_reg[j0]));
                                LITETOPK_ST_CAND_IDX(buf_idx[static_cast<uint64_t>(row) * buf_cap + w], static_cast<int32_t>(kv_r0));
                            }
                            if (refresh_every > 0)
                                atomicAdd(&bcount[static_cast<uint64_t>(row) * num_buckets + b_reg[j0]], 1);
                        }
                        if (mine && ((pass_bits >> j1) & 1u)) {
                            const int w = base + __popc(m0) + __popc(m1 & below);
                            if (w < static_cast<int>(buf_cap)) {
                                LITETOPK_ST_CAND_VAL(buf_val[static_cast<uint64_t>(row) * buf_cap + w], litetopk_cand_cast<cand_t>(cand_reg[j1]));
                                LITETOPK_ST_CAND_IDX(buf_idx[static_cast<uint64_t>(row) * buf_cap + w], static_cast<int32_t>(kv_r1));
                            }
                            if (refresh_every > 0)
                                atomicAdd(&bcount[static_cast<uint64_t>(row) * num_buckets + b_reg[j1]], 1);
                        }
                    }
                } else {
                    // Warp-queue emit (QN <= 8): per q column, two ballots
                    // (row0 / row1 holders) feed one queue insert — the same
                    // 8+8-lane shape as the DSA SM90 emit, minus the election
                    // (every candidate here is already unique).
                    #pragma unroll
                    for (uint32_t qc = 0; qc < 8; ++qc) {
                        const uint32_t col = g * 8 + qc;   // == qc (QN<=8 => g==0)
                        if (col >= logical_rows) continue;
                        const uint32_t row = query_base + col;
                        const bool mine = (c_pair == (qc & ~1u));
                        const uint32_t j0 = (qc & 1);
                        const uint32_t j1 = 2 + (qc & 1);
                        const bool h0 = mine && ((pass_bits >> j0) & 1u);
                        const bool h1 = mine && ((pass_bits >> j1) & 1u);
                        const unsigned m0 = __ballot_sync(0xffffffffu, h0);
                        const unsigned m1 = __ballot_sync(0xffffffffu, h1);
                        if ((m0 | m1) == 0) continue;
                        const int cnt = __popc(m0) + __popc(m1);
                        const uint32_t slot = warp_idx * QN + col;
                        const uint32_t qbase = slot * LITETOPK_WARP_QUEUE_CAP;
                        int qn = qn_reg[col];
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
                        const unsigned below = (1u << lane_idx) - 1u;
                        if (h0) {
                            const int pos = qn + __popc(m0 & below);
                            warpq_val[qbase + pos] = litetopk_cand_cast<cand_t>(cand_reg[j0]);
                            warpq_idx[qbase + pos] = static_cast<int32_t>(kv_r0);
                            if (refresh_every > 0)
                                atomicAdd(&bcount[static_cast<uint64_t>(row) * num_buckets + b_reg[j0]], 1);
                        }
                        if (h1) {
                            const int pos = qn + __popc(m0) + __popc(m1 & below);
                            warpq_val[qbase + pos] = litetopk_cand_cast<cand_t>(cand_reg[j1]);
                            warpq_idx[qbase + pos] = static_cast<int32_t>(kv_r1);
                            if (refresh_every > 0)
                                atomicAdd(&bcount[static_cast<uint64_t>(row) * num_buckets + b_reg[j1]], 1);
                        }
                        qn_reg[col] = qn + cnt;
                        __syncwarp(0xffffffffu);
                    }
                }  // kUseWarpQueue
                }  // __any_sync(pass_bits)
            }  // column-group loop

            if (((consumed + 1) % gate_stride) == 0) {
                // Periodically re-read the (possibly refresh-tightened) gate.
                // Stale gates only over-emit — recall-safe.
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

    // Last-CTA-per-head-group final threshold refresh (verbatim B200: the
    // threadfence + completion-counter release/acquire handshake).
    if (cta_done != nullptr) {
        auto last_cta_flag = reinterpret_cast<volatile int*>(aux_ptr + 3);
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
}

} // namespace litetopk_marsco

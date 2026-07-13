// Hopper D512 gate-count kernel (production extract).
//
// 抽自 _proto_wgmma_cta64_gate.cu 的最快生产路径，固定生产配置、去掉所有诊断开关
// (CLOCK_SPLIT / NOHIST / GATE_COMPUTE_ONLY / GEMM_ONLY_SINK / PINGPONG_LOCK /
//  ANTIPHASE_ENTRY / STARTUP_STAGGER / KSLICE / OPERAND_RELEASE) 与 main/harness。
//
// 固定配置：D=512, BM=BN=64, K_TILE=64, K_STAGES=8, C_STAGE=2, C_TILES_PER_CTA=64,
//   3WG ping-pong(WG0 producer / WG1+WG2 math), A 全驻留, B 单条 3D TMA, SS-WGMMA,
//   coord16 V2 epilogue(无 max guard 的 balanced-tree count), free-mbar 轻量 release。

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <stdint.h>

#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>

#include "hopper_topk.h"

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 512;
constexpr int K_TILE = 64;
constexpr int K_STAGES = D / K_TILE;            // 8
constexpr int C_STAGE = 2;
#ifndef HOPPER_C_TILES_PER_CTA
#define HOPPER_C_TILES_PER_CTA 64
#endif
constexpr int C_TILES_PER_CTA = HOPPER_C_TILES_PER_CTA;
constexpr int SAMPLE_C_TILES_PER_CTA = 16;
static_assert(SAMPLE_C_TILES_PER_CTA >= C_STAGE,
              "sample C tiles per CTA must be at least C_STAGE");
constexpr int WARPGROUPS = 3;
constexpr int THREADS = WARPGROUPS * 128;       // 384
#ifndef HOPPER_THR_REFRESH_GROUP
#define HOPPER_THR_REFRESH_GROUP 0
#endif
constexpr int THR_REFRESH_GROUP = HOPPER_THR_REFRESH_GROUP;
// Diagnostic-only ablation switches for the sparse epilogue (do not change
// correctness of the production path; used to attribute time inside the tail).
//   HOPPER_SPARSE_NO_HIST=1  : skip the per-element atomicAdd(bcount)
//   HOPPER_SPARSE_FAKE_HIST_STORE=1 : replace sparse-tail bcount atomicAdd
//     with a diagnostic plain global store. Correctness is intentionally not
//     preserved; this isolates atomic/L2-serialization cost from gate math.
//   HOPPER_SPARSE_NOSTORE_HIST=1 : keep gate/bucket/th work live but skip
//     bcount writes entirely. Diagnostic-only; correctness is not preserved.
//   HOPPER_SPARSE_NO_WRITE=1 : skip the gate + qcount-atomic + buf write
#ifndef HOPPER_SPARSE_NO_HIST
#define HOPPER_SPARSE_NO_HIST 0
#endif
#ifndef HOPPER_SPARSE_NOSTORE_HIST
#define HOPPER_SPARSE_NOSTORE_HIST 0
#endif
#ifndef HOPPER_SPARSE_FAKE_HIST_STORE
#define HOPPER_SPARSE_FAKE_HIST_STORE 0
#endif
#ifndef HOPPER_SPARSE_MERGE_HIST
#define HOPPER_SPARSE_MERGE_HIST 0
#endif
#ifndef HOPPER_SPARSE_NO_WRITE
#define HOPPER_SPARSE_NO_WRITE 0
#endif
#ifndef HOPPER_SPARSE_WARP_BATCH_WRITE
#define HOPPER_SPARSE_WARP_BATCH_WRITE 0
#endif
#ifndef HOPPER_SPARSE_BUCKET_WRITE
#define HOPPER_SPARSE_BUCKET_WRITE 0
#endif
// Dense-write epilogue variant (batch=64, D=512): instead of reserving a
// contiguous slot via atomicAdd(qcount) and compacting candidates into a small
// [R,BUF] buffer, scatter each passing score directly into a pre-zeroed
// [R,M] dense candidate tensor at its own column.  The epilogue then only needs
// an atomicAdd to the histogram (for threshold) and a plain store; no qcount
// return dependency, no scatter-position bookkeeping.  Select scans the dense
// tensor with buf_idx=nullptr/qcount=nullptr (column == corpus id) and skips 0.
#ifndef HOPPER_SPARSE_DENSE_WRITE
#define HOPPER_SPARSE_DENSE_WRITE 0
#endif
#ifndef HOPPER_SPARSE_BCOUNT_SHARDS
#define HOPPER_SPARSE_BCOUNT_SHARDS 1
#endif
#if HOPPER_SPARSE_BUCKET_WRITE && HOPPER_SPARSE_BCOUNT_SHARDS != 1
#error "HOPPER_SPARSE_BUCKET_WRITE requires HOPPER_SPARSE_BCOUNT_SHARDS=1"
#endif
#if HOPPER_SPARSE_BUCKET_WRITE && HOPPER_SPARSE_REFRESH_FROM_BUF == 1
#error "HOPPER_SPARSE_BUCKET_WRITE requires explicit bucket coordinates"
#endif
#if HOPPER_SPARSE_BUCKET_WRITE && HOPPER_SPARSE_MERGE_HIST
#error "HOPPER_SPARSE_BUCKET_WRITE uses bcount as cursor and cannot merge hist atomics"
#endif
constexpr int BCOUNT_SHARDS = HOPPER_SPARSE_BCOUNT_SHARDS;
#ifndef HOPPER_M8_REG_ROW_QUEUE
// Default candidate writeback is direct atomicAdd.  The register-row queue path
// is intentionally opt-in; measurements show it only helps for extremely large
// k where candidate pressure dominates.
#define HOPPER_M8_REG_ROW_QUEUE 0
#endif
#ifndef HOPPER_SPARSE_REFRESH_FROM_BUF
#define HOPPER_SPARSE_REFRESH_FROM_BUF 0
#endif
constexpr int REFRESH_FROM_BUF = HOPPER_SPARSE_REFRESH_FROM_BUF;
constexpr int REFRESH_NB_MAX = 64;
#ifndef HOPPER_SPARSE_REFRESH_ROWS
#define HOPPER_SPARSE_REFRESH_ROWS 4
#endif
#ifndef HOPPER_RFB_FLUSH_SKIP_ABOVE_GATE
#define HOPPER_RFB_FLUSH_SKIP_ABOVE_GATE 1
#endif
#ifndef HOPPER_RFB_HIST_MATRIX_ADD
#define HOPPER_RFB_HIST_MATRIX_ADD 0
#endif
#ifndef HOPPER_RFB_GLOBAL_BATCH_ADD
#define HOPPER_RFB_GLOBAL_BATCH_ADD 1
#endif
#ifndef HOPPER_RFB_CANDIDATE_PARALLEL
#define HOPPER_RFB_CANDIDATE_PARALLEL 0
#endif
#ifndef HOPPER_RFB_EARLY_UNLOCK
#define HOPPER_RFB_EARLY_UNLOCK 1
#endif
#ifndef HOPPER_RFB_PRODUCER_ONLY
#define HOPPER_RFB_PRODUCER_ONLY 0
#endif
#ifndef HOPPER_RFB_FLATTEN_ONE_SYNC
#define HOPPER_RFB_FLATTEN_ONE_SYNC 0
#endif
#ifndef HOPPER_RFB_FLATTEN_PARALLEL_FLUSH
#define HOPPER_RFB_FLATTEN_PARALLEL_FLUSH 0
#endif
constexpr int REFRESH_ROWS = HOPPER_SPARSE_REFRESH_ROWS;
constexpr int REFRESH_HIST_BYTES =
    (REFRESH_FROM_BUF > 0) ? BM * REFRESH_NB_MAX * int(sizeof(int32_t)) : 0;

constexpr int A_STAGE_ELEMS = BM * K_TILE;
constexpr int B_K_STAGE_ELEMS = BN * K_TILE;
constexpr int B_STAGE_ELEMS = K_STAGES * B_K_STAGE_ELEMS;
constexpr int A_STAGE_BYTES = A_STAGE_ELEMS * int(sizeof(__half));
constexpr int B_STAGE_BYTES = B_STAGE_ELEMS * int(sizeof(__half));
constexpr int A_BYTES = K_STAGES * A_STAGE_BYTES;
constexpr int B_BYTES = C_STAGE * B_STAGE_BYTES;
constexpr int A_BARRIERS = K_STAGES;
constexpr int B_BARRIERS = C_STAGE;
constexpr int FREE_BARRIERS = C_STAGE;
constexpr int SMEM_BYTES =
    A_BYTES + B_BYTES + (A_BARRIERS + B_BARRIERS + FREE_BARRIERS) * int(sizeof(uint64_t));
constexpr int SPARSE_SMEM_BYTES = SMEM_BYTES + REFRESH_HIST_BYTES;

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ int nth_set_bit_u32(unsigned mask, int n) {
  int bit = 0;
#pragma unroll
  for (int k = 0; k < 32; ++k) {
    bit = __ffs(mask) - 1;
    if (k == n) return bit;
    mask &= (mask - 1);
  }
  return bit;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, int arrivals) {
  uint32_t smem = smem_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem), "r"(arrivals));
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* barrier, uint32_t bytes) {
  uint32_t smem = smem_u32(barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* barrier) {
  uint32_t smem = smem_u32(barrier);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, int phase) {
  uint32_t smem = smem_u32(barrier);
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "L_wait_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@p bra L_done_%=;\n"
      "bra L_wait_%=;\n"
      "L_done_%=:\n"
      "}\n" ::"r"(smem),
      "r"(phase)
      : "memory");
}

template <int Id>
__device__ __forceinline__ void named_barrier_sync_128() {
  asm volatile("bar.sync %0, 128;" ::"n"(Id) : "memory");
}

__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* tensor_map,
                                            uint64_t* barrier, int c0, int c1) {
  uint32_t smem_ptr = smem_u32(smem_dst);
  uint32_t smem_mbar = smem_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_ptr),
      "l"(tensor_map), "r"(c0), "r"(c1), "r"(smem_mbar)
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(void* smem_dst, const CUtensorMap* tensor_map,
                                            uint64_t* barrier, int c0, int c1, int c2) {
  uint32_t smem_ptr = smem_u32(smem_dst);
  uint32_t smem_mbar = smem_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(smem_ptr),
      "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(smem_mbar)
      : "memory");
}

__device__ __forceinline__ void cp_async_ca_shared_global_4(void* smem_dst, const void* gmem_src) {
  uint32_t smem_ptr = smem_u32(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_ptr), "l"(gmem_src)
               : "memory");
}

__device__ __forceinline__ void cp_async_ca_shared_global_16(void* smem_dst, const void* gmem_src) {
  uint32_t smem_ptr = smem_u32(smem_dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_ptr), "l"(gmem_src)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

template <int Base, class TensorCoord>
__device__ __forceinline__ int cute_coord_row(TensorCoord const& tCcC) {
  return int(cute::get<0>(tCcC(Base)));
}

__device__ __noinline__ void smalln_refresh_threshold(
    int query_base, int tid, int32_t* __restrict__ th, int32_t* __restrict__ bcount,
    int NB, int K) {
  constexpr int QN = 32;
  for (int row_local = tid; row_local < QN; row_local += 128) {
    int row = query_base + row_local;
    int cum = 0;
    int new_th = NB - 1;
    for (int b = 0; b < NB; ++b) {
      int bc = 0;
      CUTE_UNROLL
      for (int s = 0; s < BCOUNT_SHARDS; ++s) {
        bc += bcount[((row * NB + b) * BCOUNT_SHARDS) + s];
      }
      cum += bc;
      if (cum >= K) {
        new_th = b;
        break;
      }
    }
    atomicMin(&th[row], new_th);
  }
}

__device__ __noinline__ void smalln_refresh_threshold_rows(
    int query_base, int rows, int tid, int32_t* __restrict__ th,
    int32_t* __restrict__ bcount, int NB, int K) {
  for (int row_local = tid; row_local < rows; row_local += 128) {
    int row = query_base + row_local;
    int cum = 0;
    int new_th = NB - 1;
    for (int b = 0; b < NB; ++b) {
      int bc = 0;
      CUTE_UNROLL
      for (int s = 0; s < BCOUNT_SHARDS; ++s) {
        bc += bcount[((row * NB + b) * BCOUNT_SHARDS) + s];
      }
      cum += bc;
      if (cum >= K) {
        new_th = b;
        break;
      }
    }
    atomicMin(&th[row], new_th);
  }
}

template <int KS, class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void cute_emit_k_stage(TiledMma& tiled_mma, TensorA const& tCrA,
                                                  TensorB const& tCrB, TensorC& tCrC) {
  using namespace cute;
  auto tCrA_k = tCrA(_, _, _, Int<KS>{});
  auto tCrB_k = tCrB(_, _, _, Int<KS>{});
  CUTE_UNROLL
  for (int kk = 0; kk < size<2>(tCrA_k); ++kk) {
    cute::gemm(tiled_mma, tCrA_k(_, _, kk), tCrB_k(_, _, kk), tCrC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
}

// Emit active K stages for one C tile, then early-release: wait<1> returns before
// the committed group fully retires so the producer can refill the B stage
// earlier; the caller does the final wait<0> before consuming the accumulator.
template <int ACTIVE_STAGES, class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void cute_emit_gemm_only_active(
    TiledMma& tiled_mma, TensorA const& tCrA, TensorB const& tCrB, TensorC& tCrC) {
  using namespace cute;
  cute::clear(tCrC);
  cute::warpgroup_fence_operand(tCrC);
  cute::warpgroup_arrive();
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  if constexpr (ACTIVE_STAGES > 0) cute_emit_k_stage<0>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 1) cute_emit_k_stage<1>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 2) cute_emit_k_stage<2>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 3) cute_emit_k_stage<3>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 4) cute_emit_k_stage<4>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 5) cute_emit_k_stage<5>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 6) cute_emit_k_stage<6>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 7) cute_emit_k_stage<7>(tiled_mma, tCrA, tCrB, tCrC);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<1>();
}

template <class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void cute_emit_gemm_only_runtime(
    int active_stages, TiledMma& tiled_mma, TensorA const& tCrA,
    TensorB const& tCrB, TensorC& tCrC) {
  switch (active_stages) {
    case 1: cute_emit_gemm_only_active<1>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 2: cute_emit_gemm_only_active<2>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 3: cute_emit_gemm_only_active<3>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 4: cute_emit_gemm_only_active<4>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 5: cute_emit_gemm_only_active<5>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 6: cute_emit_gemm_only_active<6>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 7: cute_emit_gemm_only_active<7>(tiled_mma, tCrA, tCrB, tCrC); break;
    default: cute_emit_gemm_only_active<8>(tiled_mma, tCrA, tCrB, tCrC); break;
  }
}

template <int ACTIVE_STAGES, class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void cute_emit_gemm_only_active_k12(
    TiledMma& tiled_mma, TensorA const& tCrA, TensorB const& tCrB, TensorC& tCrC) {
  using namespace cute;
  cute::clear(tCrC);
  cute::warpgroup_fence_operand(tCrC);
  cute::warpgroup_arrive();
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  if constexpr (ACTIVE_STAGES > 0) cute_emit_k_stage<0>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 1) cute_emit_k_stage<1>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 2) cute_emit_k_stage<2>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 3) cute_emit_k_stage<3>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 4) cute_emit_k_stage<4>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 5) cute_emit_k_stage<5>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 6) cute_emit_k_stage<6>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 7) cute_emit_k_stage<7>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 8) cute_emit_k_stage<8>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 9) cute_emit_k_stage<9>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 10) cute_emit_k_stage<10>(tiled_mma, tCrA, tCrB, tCrC);
  if constexpr (ACTIVE_STAGES > 11) cute_emit_k_stage<11>(tiled_mma, tCrA, tCrB, tCrC);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<1>();
}

template <class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void cute_emit_gemm_only_runtime_k12(
    int active_stages, TiledMma& tiled_mma, TensorA const& tCrA,
    TensorB const& tCrB, TensorC& tCrC) {
  switch (active_stages) {
    case 1: cute_emit_gemm_only_active_k12<1>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 2: cute_emit_gemm_only_active_k12<2>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 3: cute_emit_gemm_only_active_k12<3>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 4: cute_emit_gemm_only_active_k12<4>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 5: cute_emit_gemm_only_active_k12<5>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 6: cute_emit_gemm_only_active_k12<6>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 7: cute_emit_gemm_only_active_k12<7>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 8: cute_emit_gemm_only_active_k12<8>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 9: cute_emit_gemm_only_active_k12<9>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 10: cute_emit_gemm_only_active_k12<10>(tiled_mma, tCrA, tCrB, tCrC); break;
    case 11: cute_emit_gemm_only_active_k12<11>(tiled_mma, tCrA, tCrB, tCrC); break;
    default: cute_emit_gemm_only_active_k12<12>(tiled_mma, tCrA, tCrB, tCrC); break;
  }
}

// K-chunk streaming config for D>512. Base is streamed along K in a ring of
// KCHUNK_RING K-tiles; query stays resident for up to KMAX_STAGES.
//   HOPPER_KCHUNK_RING  : base K-tile ring depth (producer run-ahead window).
//   HOPPER_KCHUNK_PIPE  : number of WGMMAs kept in flight before waiting; the
//     consumer commits each K-tile as its own batch and only waits<PIPE-1>, so
//     consecutive MMAs (and the TMA of later tiles) overlap.  PIPE==1 reproduces
//     the original fully-serialized wait<0>-per-tile behaviour.  Require
//     KCHUNK_RING > KCHUNK_PIPE so released slots are not refilled before the
//     still-in-flight WGMMAs finish reading them.
constexpr int KMAX_STAGES = 16;   // up to D=1024
#ifndef HOPPER_KCHUNK_RING
#define HOPPER_KCHUNK_RING 12
#endif
#ifndef HOPPER_KCHUNK_PIPE
#define HOPPER_KCHUNK_PIPE 1
#endif
// Diagnostic: force D<=512 through the K-chunk streaming path too (default off),
// so the full-D-resident path and the K-chunk path can be A/B compared at D=512.
#ifndef HOPPER_KCHUNK_FORCE
#define HOPPER_KCHUNK_FORCE 0
#endif
// Number of math warpgroups in the K-chunk kernels (producer is always 1 extra
// warpgroup). Default 2 (WG1/WG2 ping-pong). Set 3 to test whether a third
// consumer closes the DRAM-bandwidth gap (i.e. consumers were the bottleneck
// keeping the ring full). KCHUNK_RING must be divisible by this.
#ifndef HOPPER_KCHUNK_MATH_WG
#define HOPPER_KCHUNK_MATH_WG 3
#endif
constexpr int KCHUNK_RING = HOPPER_KCHUNK_RING;
constexpr int KCHUNK_PIPE = HOPPER_KCHUNK_PIPE;
constexpr int KCHUNK_FORCE = HOPPER_KCHUNK_FORCE;
constexpr int KCHUNK_MATH_WG = HOPPER_KCHUNK_MATH_WG;
constexpr int KCHUNK_WG = 1 + KCHUNK_MATH_WG;          // producer + math groups
constexpr int KCHUNK_THREADS = KCHUNK_WG * 128;
constexpr int KCHUNK_SEG = KCHUNK_RING / KCHUNK_MATH_WG;  // ring slots per math WG
static_assert(KCHUNK_RING % KCHUNK_MATH_WG == 0,
              "KCHUNK_RING must be divisible by KCHUNK_MATH_WG");
static_assert(KCHUNK_SEG > KCHUNK_PIPE, "KCHUNK_SEG must exceed KCHUNK_PIPE");
static_assert(KCHUNK_MATH_WG >= 1 && KCHUNK_MATH_WG <= 3, "1..3 math warpgroups");

// Accumulate one K-tile (K_TILE wide) into the C fragment. Caller manages the
// surrounding warpgroup_arrive / commit / wait and the accumulate_ reset.
template <class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void kchunk_mma_one_ktile(
    TiledMma& tiled_mma, TensorA const& tCrA_slot, TensorB const& tCrB_kt, TensorC& tCrC) {
  using namespace cute;
  CUTE_UNROLL
  for (int kk = 0; kk < size<2>(tCrA_slot); ++kk) {
    cute::gemm(tiled_mma, tCrA_slot(_, _, kk), tCrB_kt(_, _, kk), tCrC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
}

// Stream all k_stages K-tiles of one C tile through one HALF of the base ring
// into tCrC. The ring is split so WG1 (even C-tiles) and WG2 (odd C-tiles) can
// both run WGMMA concurrently (3WG ping-pong) while sharing the resident query
// A. `half_base`/`half_size` scope this warpgroup's ring region; `consumed` is
// this warpgroup's monotonic K-tile counter within its half. Each K-tile is its
// own WGMMA batch and the consumer keeps up to KCHUNK_PIPE batches in flight.
template <class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void kchunk_stream_one_ctile(
    TiledMma& mma, TensorA const& tCrA, TensorB const& tCrB, TensorC& tCrC,
    uint64_t* bar_filled, uint64_t* bar_free, int& consumed,
    int half_base, int half_size, int k_stages, bool releaser) {
  using namespace cute;
  int base_consumed = consumed;
  cute::clear(tCrC);
  cute::warpgroup_fence_operand(tCrC);
  mma.accumulate_ = GMMA::ScaleOut::Zero;
  for (int ks = 0; ks < k_stages; ++ks) {
    int slot = half_base + (consumed % half_size);
    int phase = (consumed / half_size) & 1;
    mbarrier_wait(&bar_filled[slot], phase);
    cute::warpgroup_arrive();
    kchunk_mma_one_ktile(mma, tCrA(_, _, _, ks), tCrB(_, _, _, slot), tCrC);
    cute::warpgroup_commit_batch();
    if (ks >= KCHUNK_PIPE - 1) {
      cute::warpgroup_wait<KCHUNK_PIPE - 1>();
      if (releaser) {
        int rel = consumed - (KCHUNK_PIPE - 1);
        mbarrier_arrive(&bar_free[half_base + (rel % half_size)]);
      }
    }
    ++consumed;
  }
  cute::warpgroup_wait<0>();
  if (releaser) {
    int first = base_consumed + k_stages - (KCHUNK_PIPE - 1);
    if (first < base_consumed) first = base_consumed;
    for (int rel = first; rel < base_consumed + k_stages; ++rel) {
      mbarrier_arrive(&bar_free[half_base + (rel % half_size)]);
    }
  }
  cute::warpgroup_fence_operand(tCrC);
}

// Variant of kchunk_stream_one_ctile for the m64n32 layout where the streamed
// ring holds the A operand (base) and the resident operand is B (query). Used by
// the smalln m64n32 kchunk kernels.
template <class TiledMma, class TensorA, class TensorB, class TensorC>
__device__ __forceinline__ void kchunk_stream_one_ctile_aring(
    TiledMma& mma, TensorA const& tCrA, TensorB const& tCrB, TensorC& tCrC,
    uint64_t* bar_filled, uint64_t* bar_free, int& consumed,
    int seg_base, int seg_size, int k_stages, bool releaser) {
  using namespace cute;
  int base_consumed = consumed;
  cute::clear(tCrC);
  cute::warpgroup_fence_operand(tCrC);
  mma.accumulate_ = GMMA::ScaleOut::Zero;
  for (int ks = 0; ks < k_stages; ++ks) {
    int slot = seg_base + (consumed % seg_size);
    int phase = (consumed / seg_size) & 1;
    mbarrier_wait(&bar_filled[slot], phase);
    cute::warpgroup_arrive();
    kchunk_mma_one_ktile(mma, tCrA(_, _, _, slot), tCrB(_, _, _, ks), tCrC);
    cute::warpgroup_commit_batch();
    if (ks >= KCHUNK_PIPE - 1) {
      cute::warpgroup_wait<KCHUNK_PIPE - 1>();
      if (releaser) {
        int rel = consumed - (KCHUNK_PIPE - 1);
        mbarrier_arrive(&bar_free[seg_base + (rel % seg_size)]);
      }
    }
    ++consumed;
  }
  cute::warpgroup_wait<0>();
  if (releaser) {
    int first = base_consumed + k_stages - (KCHUNK_PIPE - 1);
    if (first < base_consumed) first = base_consumed;
    for (int rel = first; rel < base_consumed + k_stages; ++rel) {
      mbarrier_arrive(&bar_free[seg_base + (rel % seg_size)]);
    }
  }
  cute::warpgroup_fence_operand(tCrC);
}

// Producer feed for one SEGMENT of the base ring: streams the K-tiles of every
// c_rel that belongs to this math warpgroup (c_rel % KCHUNK_MATH_WG == parity)
// so the math warpgroups are fed independently from KCHUNK_MATH_WG producer
// threads sharing the resident query A.
__device__ __forceinline__ void kchunk_produce_half(
    __half* smem_b, const CUtensorMap* b_map, uint64_t* bar_filled, uint64_t* bar_free,
    int parity, int seg_base, int seg_size, int bb_slot_elems,
    int cta_c_group_start, int total_c_tiles, int k_stages, int c_tiles_per_cta) {
  int produced = 0;
  for (int c_rel = parity; c_rel < c_tiles_per_cta; c_rel += KCHUNK_MATH_WG) {
    int c_tile = cta_c_group_start + c_rel;
    if (c_tile >= total_c_tiles) break;
    for (int ks = 0; ks < k_stages; ++ks) {
      int slot = seg_base + (produced % seg_size);
      int phase = (produced / seg_size - 1) & 1;
      if (produced >= seg_size) {
        mbarrier_wait(&bar_free[slot], phase);
      }
      mbarrier_expect_tx(&bar_filled[slot], bb_slot_elems * int(sizeof(__half)));
      tma_load_2d(smem_b + slot * bb_slot_elems, b_map, &bar_filled[slot],
                  ks * K_TILE, c_tile * BN);
      ++produced;
    }
  }
}

#define HOPPER_CUDRV(call)                                                                  \
  do {                                                                                      \
    CUresult err__ = (call);                                                                \
    if (err__ != CUDA_SUCCESS) {                                                            \
      const char* name__ = nullptr;                                                         \
      const char* msg__ = nullptr;                                                          \
      cuGetErrorName(err__, &name__);                                                       \
      cuGetErrorString(err__, &msg__);                                                      \
      std::fprintf(stderr, "CUDA driver error %s:%d %s: %s %s\n", __FILE__, __LINE__,       \
                   #call, name__ ? name__ : "?", msg__ ? msg__ : "?");                      \
      std::abort();                                                                         \
    }                                                                                       \
  } while (0)

}  // namespace

extern "C" __global__ __launch_bounds__(THREADS, 1) void hopper_score_to_dense_ip_kernel(
    const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap b_map,
    __half* __restrict__ dense_scores, int total_c_tiles, int M_stride, int k_stages,
    int c_tiles_per_cta) {
  extern __shared__ __align__(128) unsigned char smem[];

  __half* smem_a = reinterpret_cast<__half*>(smem);
  __half* smem_b = reinterpret_cast<__half*>(smem + A_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + A_BYTES + B_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_b = barriers + A_BARRIERS;
  uint64_t* barriers_free = barriers + A_BARRIERS + B_BARRIERS;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int cta_m_base = blockIdx.x * BM;
  int cta_c_group_start = blockIdx.y * c_tiles_per_cta;

  for (int s = tid; s < A_BARRIERS + B_BARRIERS + FREE_BARRIERS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0 && tid == 0) {
    for (int ks = 0; ks < k_stages; ++ks) {
      mbarrier_expect_tx(&barriers_a[ks], A_STAGE_BYTES);
      tma_load_2d(smem_a + ks * A_STAGE_ELEMS, &a_map, &barriers_a[ks], ks * K_TILE, cta_m_base);
    }
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int c_tile = cta_c_group_start + pre;
      if (c_tile < total_c_tiles) {
        mbarrier_expect_tx(&barriers_b[pre], k_stages * B_K_STAGE_ELEMS * int(sizeof(__half)));
        tma_load_3d(smem_b + pre * B_STAGE_ELEMS, &b_map, &barriers_b[pre], 0, c_tile * BN, 0);
      }
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < c_tiles_per_cta; ++refill) {
        int refill_tile = cta_c_group_start + refill;
        if (refill_tile >= total_c_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_b[refill_stage],
                           k_stages * B_K_STAGE_ELEMS * int(sizeof(__half)));
        tma_load_3d(smem_b + refill_stage * B_STAGE_ELEMS, &b_map, &barriers_b[refill_stage], 0,
                    refill_tile * BN, 0);
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<D>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<K_STAGES>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{}, cute::make_shape(cute::Int<BN>{}, cute::Int<K_TILE>{},
                                        cute::Int<K_STAGES>{}, cute::Int<C_STAGE>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_b)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  for (int ks = 0; ks < k_stages; ++ks) {
    mbarrier_wait(&barriers_a[ks], 0);
  }

  for (int c_rel = 0; c_rel < c_tiles_per_cta; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int c_tile = cta_c_group_start + c_rel;
    if (warpgroup == owner_wg && c_tile < total_c_tiles) {
      mbarrier_wait(&barriers_b[stage], phase);
      if (stage == 0) {
        cute_emit_gemm_only_runtime(
            k_stages, cute_tiled_mma, cute_tCrA,
            cute_tCrB(cute::_, cute::_, cute::_, cute::_, cute::Int<0>{}), cute_tCrC);
      } else {
        cute_emit_gemm_only_runtime(
            k_stages, cute_tiled_mma, cute_tCrA,
            cute_tCrB(cute::_, cute::_, cute::_, cute::_, cute::Int<1>{}), cute_tCrC);
      }
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);

      CUTE_UNROLL
      for (int i = 0; i < 32; ++i) {
        int r = int(cute::get<0>(cute_tCcC(i)));
        int c = int(cute::get<1>(cute_tCcC(i)));
        int global_col = c_tile * BN + c;
        dense_scores[(cta_m_base + r) * M_stride + global_col] = __float2half(-float(cute_tCrC(i)));
      }
    }
  }
}

// ===========================================================================
// K-chunk streaming m64n64 dense kernel for D > 512 (batch=64).
//   query A (64 x D) stays resident; base B (64 x D per column tile) is
//   streamed along K through a KCHUNK_RING ring.  Single math warpgroup (WG1).
// ===========================================================================
extern "C" __global__ __launch_bounds__(KCHUNK_THREADS, 1)
void hopper_score_to_dense_ip_kchunk_kernel(
    const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap b_map,
    __half* __restrict__ dense_scores, int total_c_tiles, int M_stride, int k_stages,
    int c_tiles_per_cta) {
  // Query A: resident [BM, K_TILE] x KMAX slots.  Base B: ring of [BN, K_TILE].
  constexpr int QA_SLOT_ELEMS = BM * K_TILE;
  constexpr int QA_ELEMS = KMAX_STAGES * QA_SLOT_ELEMS;
  constexpr int QA_BYTES = QA_ELEMS * int(sizeof(__half));
  constexpr int BB_SLOT_ELEMS = BN * K_TILE;
  constexpr int BB_ELEMS = KCHUNK_RING * BB_SLOT_ELEMS;
  constexpr int BB_BYTES = BB_ELEMS * int(sizeof(__half));
  constexpr int RING_BARS = KCHUNK_RING;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_a = reinterpret_cast<__half*>(smem);            // query resident
  __half* smem_b = reinterpret_cast<__half*>(smem + QA_BYTES); // base ring
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QA_BYTES + BB_BYTES);
  uint64_t* bar_filled = barriers;                  // [RING_BARS]
  uint64_t* bar_free = barriers + RING_BARS;        // [RING_BARS]
  uint64_t* bar_ardy = barriers + 2 * RING_BARS;    // [1] query-ready

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpgroup = tid >> 7;
  int math_warp = (tid >> 5) - warpgroup * 4;
  int cta_m_base = blockIdx.x * BM;
  int cta_c_group_start = blockIdx.y * c_tiles_per_cta;

  for (int s = tid; s < 2 * RING_BARS + 1; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0) {
    // One producer thread per math warpgroup feeds its own ring segment so the
    // math warpgroups are fed independently. Query A is loaded once (shared).
    if (tid == 0) {
      mbarrier_expect_tx(&bar_ardy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
      for (int ks = 0; ks < k_stages; ++ks) {
        tma_load_2d(smem_a + ks * QA_SLOT_ELEMS, &a_map, &bar_ardy[0], ks * K_TILE, cta_m_base);
      }
    }
    int prod = tid >> 5;  // warp index within producer warpgroup (0..3)
    if (prod < KCHUNK_MATH_WG && (tid & 31) == 0) {
      kchunk_produce_half(smem_b, &b_map, bar_filled, bar_free, /*parity=*/prod,
                          /*seg_base=*/prod * KCHUNK_SEG, KCHUNK_SEG, BB_SLOT_ELEMS,
                          cta_c_group_start, total_c_tiles, k_stages, c_tiles_per_cta);
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<KMAX_STAGES>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<BN>{}, cute::Int<K_TILE>{}, cute::Int<KCHUNK_RING>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_b)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  // N-way ping-pong: math WG p owns C-tiles with (c_rel % KCHUNK_MATH_WG == p),
  // each consuming from its own ring segment [p*SEG, (p+1)*SEG).
  int my_parity = warpgroup - 1;            // WG1->0, WG2->1, WG3->2
  int my_seg_base = my_parity * KCHUNK_SEG;
  bool releaser = (math_warp == 0 && lane == 0);

  mbarrier_wait(&bar_ardy[0], 0);

  int consumed = 0;
  for (int c_rel = my_parity; c_rel < c_tiles_per_cta; c_rel += KCHUNK_MATH_WG) {
    int c_tile = cta_c_group_start + c_rel;
    if (c_tile >= total_c_tiles) break;

    kchunk_stream_one_ctile(cute_tiled_mma, cute_tCrA, cute_tCrB, cute_tCrC,
                            bar_filled, bar_free, consumed, my_seg_base, KCHUNK_SEG,
                            k_stages, releaser);
    CUTE_UNROLL
    for (int i = 0; i < 32; ++i) {
      int r = int(cute::get<0>(cute_tCcC(i)));
      int c = int(cute::get<1>(cute_tCcC(i)));
      int global_col = c_tile * BN + c;
      dense_scores[(cta_m_base + r) * M_stride + global_col] = __float2half(-float(cute_tCrC(i)));
    }
  }
}

extern "C" __global__ __launch_bounds__(THREADS, 1) void hopper_smalln_score_m64n32_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    __half* __restrict__ dense_scores, int total_base_tiles, int M_stride, int k_stages) {
  constexpr int QN = 32;
  constexpr int QB_K_STAGE_ELEMS = QN * K_TILE;
  constexpr int QB_STAGE_ELEMS = K_STAGES * QB_K_STAGE_ELEMS;
  constexpr int QB_STAGE_BYTES = QB_STAGE_ELEMS * int(sizeof(__half));
  constexpr int QB_BYTES = C_STAGE * QB_STAGE_BYTES;
  constexpr int BASE_TILE_ELEMS = K_STAGES * A_STAGE_ELEMS;
  constexpr int BASE_TILE_BYTES = BASE_TILE_ELEMS * int(sizeof(__half));
  constexpr int BASE_BYTES = C_STAGE * BASE_TILE_BYTES;
  constexpr int Q_SMEM_BYTES =
      BASE_BYTES + QB_BYTES + (A_BARRIERS + B_BARRIERS + FREE_BARRIERS) * int(sizeof(uint64_t));

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_a = reinterpret_cast<__half*>(smem);
  __half* smem_b = reinterpret_cast<__half*>(smem + BASE_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + BASE_BYTES + QB_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_b = barriers + A_BARRIERS;
  uint64_t* barriers_free = barriers + A_BARRIERS + B_BARRIERS;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int base_tile_group_start = blockIdx.x * C_TILES_PER_CTA;
  int query_tile = blockIdx.y;
  int query_base = query_tile * QN;

  for (int s = tid; s < A_BARRIERS + B_BARRIERS + FREE_BARRIERS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0 && tid == 0) {
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        mbarrier_expect_tx(&barriers_a[pre], k_stages * A_STAGE_BYTES);
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[pre], ks * K_TILE, base_tile * BM);
        }
      }
    }
    for (int pre = 0; pre < C_STAGE; ++pre) {
      mbarrier_expect_tx(&barriers_b[pre],
                         k_stages * QB_K_STAGE_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_b + pre * QB_STAGE_ELEMS, &query_map, &barriers_b[pre], 0, query_base, 0);
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = base_tile_group_start + refill;
        if (refill_tile >= total_base_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_a[refill_stage], k_stages * A_STAGE_BYTES);
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[refill_stage], ks * K_TILE, refill_tile * BM);
        }
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<32>, cute::Int<D>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<32>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<K_STAGES>{},
                       cute::Int<C_STAGE>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{}, cute::make_shape(cute::Int<32>{}, cute::Int<K_TILE>{},
                                        cute::Int<K_STAGES>{}, cute::Int<C_STAGE>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_b)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int base_tile = base_tile_group_start + c_rel;
    if (warpgroup == owner_wg && base_tile < total_base_tiles) {
      mbarrier_wait(&barriers_a[stage], phase);
      if (c_rel < C_STAGE) {
        mbarrier_wait(&barriers_b[stage], 0);
      }
      cute_emit_gemm_only_runtime(
          k_stages, cute_tiled_mma, cute_tCrA(cute::_, cute::_, cute::_, cute::_, stage),
          cute_tCrB(cute::_, cute::_, cute::_, cute::_, stage), cute_tCrC);
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      CUTE_UNROLL
      for (int i = 0; i < 16; ++i) {
        int base_rel = int(cute::get<0>(cute_tCcC(i)));
        int q_rel = int(cute::get<1>(cute_tCcC(i)));
        int global_base = base_tile * BM + base_rel;
        int global_q = query_base + q_rel;
        if (global_base < M_stride) {
          dense_scores[(size_t)global_q * M_stride + global_base] =
              __float2half(-float(cute_tCrC(i)));
        }
      }
    }
  }
  __syncwarp();
}

// ===========================================================================
// K-chunk streaming m64n32 kernels for D > 512 (batch<=32).
//
// Layout differs from the full-D residence path: the base tile (64 x D) is
// streamed along K in a KCHUNK_RING ring of K_TILE-wide slots while the whole
// query tile (32 x D) stays resident.  This keeps smem ~80KB at D=768/1024.
//
// Pipeline (single math warpgroup = WG1; WG0 is the TMA producer):
//   per base C-tile:
//     producer issues KS K-tile base loads through the ring (filled/free bars);
//     consumer waits each slot, does one K-tile WGMMA, signals free.
//   query is loaded once per CTA up front (all KS K-tiles resident).
// ===========================================================================
extern "C" __global__ __launch_bounds__(KCHUNK_THREADS, 1)
void hopper_smalln_score_m64n32_kchunk_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    __half* __restrict__ dense_scores, int total_base_tiles, int M_stride, int k_stages) {
  constexpr int QN = 32;
  // Query: full-D resident, single buffer.  [QN, K_TILE] per K-tile, KMAX slots.
  constexpr int QB_K_TILE_ELEMS = QN * K_TILE;
  constexpr int QB_ELEMS = KMAX_STAGES * QB_K_TILE_ELEMS;
  constexpr int QB_BYTES = QB_ELEMS * int(sizeof(__half));
  // Base ring: KCHUNK_RING slots of [BM, K_TILE].
  constexpr int AB_SLOT_ELEMS = BM * K_TILE;
  constexpr int AB_ELEMS = KCHUNK_RING * AB_SLOT_ELEMS;
  constexpr int AB_BYTES = AB_ELEMS * int(sizeof(__half));
  // Barriers: KCHUNK_RING filled + KCHUNK_RING free + 1 query-ready.
  constexpr int RING_BARS = KCHUNK_RING;
  constexpr int QRDY_BARS = 1;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QB_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QB_BYTES + AB_BYTES);
  uint64_t* bar_filled = barriers;                    // [RING_BARS]
  uint64_t* bar_free = barriers + RING_BARS;          // [RING_BARS]
  uint64_t* bar_qrdy = barriers + 2 * RING_BARS;      // [1]

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpgroup = tid >> 7;
  int base_tile_group_start = blockIdx.x * C_TILES_PER_CTA;
  int query_tile = blockIdx.y;
  int query_base = query_tile * QN;

  for (int s = tid; s < 2 * RING_BARS + QRDY_BARS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  // -------- Producer warpgroup (WG0) --------
  if (warpgroup == 0) {
    if (tid == 0) {
      // Query: one 3D TMA brings all KS K-tiles resident.
      mbarrier_expect_tx(&bar_qrdy[0], k_stages * QB_K_TILE_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    }
    // One producer thread per math warpgroup streams its own ring segment; base
    // tiles are split by (c_rel % KCHUNK_MATH_WG) to match the consumers.
    int prod = tid >> 5;  // warp index within producer warpgroup (0..3)
    if (prod < KCHUNK_MATH_WG && (tid & 31) == 0) {
      kchunk_produce_half(smem_a, &base_map, bar_filled, bar_free, /*parity=*/prod,
                          /*seg_base=*/prod * KCHUNK_SEG, KCHUNK_SEG, AB_SLOT_ELEMS,
                          base_tile_group_start, total_base_tiles, k_stages, C_TILES_PER_CTA);
    }
    return;
  }

  // -------- Consumer warpgroup (WG1) --------
  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<32>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<32>,
                                 cute::Int<K_TILE>>());
  // Base slot: one K-tile [BM, K_TILE].  Query: [32, K_TILE, KMAX].
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<KCHUNK_RING>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<32>{}, cute::Int<K_TILE>{}, cute::Int<KMAX_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  // N-way ping-pong: math WG p owns base tiles with (c_rel % KCHUNK_MATH_WG==p),
  // consuming from ring segment [p*SEG, (p+1)*SEG).
  int my_parity = warpgroup - 1;            // WG1->0, WG2->1, WG3->2
  int my_seg_base = my_parity * KCHUNK_SEG;
  int math_warp = (tid >> 5) - warpgroup * 4;
  bool releaser = (math_warp == 0 && lane == 0);

  // Query resident wait (once).
  mbarrier_wait(&bar_qrdy[0], 0);

  int consumed = 0;
  for (int c_rel = my_parity; c_rel < C_TILES_PER_CTA; c_rel += KCHUNK_MATH_WG) {
    int base_tile = base_tile_group_start + c_rel;
    if (base_tile >= total_base_tiles) break;
    kchunk_stream_one_ctile_aring(cute_tiled_mma, cute_tCrA, cute_tCrB, cute_tCrC,
                                  bar_filled, bar_free, consumed, my_seg_base, KCHUNK_SEG,
                                  k_stages, releaser);
    CUTE_UNROLL
    for (int i = 0; i < 16; ++i) {
      int base_rel = int(cute::get<0>(cute_tCcC(i)));
      int q_rel = int(cute::get<1>(cute_tCcC(i)));
      int global_base = base_tile * BM + base_rel;
      int global_q = query_base + q_rel;
      if (global_base < M_stride) {
        dense_scores[(size_t)global_q * M_stride + global_base] =
            __float2half(-float(cute_tCrC(i)));
      }
    }
  }
  __syncwarp();
}

extern "C" __global__ __launch_bounds__(THREADS, 1) void hopper_smalln_score_to_sparse_m64n32_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    __half* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    int total_base_tiles, int start_base_tile, int M_stride, int BUF, int NB, int K,
    int k_stages) {
  constexpr int QN = 32;
  constexpr int QB_K_STAGE_ELEMS = QN * K_TILE;
  constexpr int QB_STAGE_ELEMS = K_STAGES * QB_K_STAGE_ELEMS;
  constexpr int QB_STAGE_BYTES = QB_STAGE_ELEMS * int(sizeof(__half));
  constexpr int QB_BYTES = C_STAGE * QB_STAGE_BYTES;
  constexpr int BASE_TILE_ELEMS = K_STAGES * A_STAGE_ELEMS;
  constexpr int BASE_TILE_BYTES = BASE_TILE_ELEMS * int(sizeof(__half));
  constexpr int BASE_BYTES = C_STAGE * BASE_TILE_BYTES;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_a = reinterpret_cast<__half*>(smem);
  __half* smem_b = reinterpret_cast<__half*>(smem + BASE_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + BASE_BYTES + QB_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_b = barriers + A_BARRIERS;
  uint64_t* barriers_free = barriers + A_BARRIERS + B_BARRIERS;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int base_tile_group_start = start_base_tile + blockIdx.x * C_TILES_PER_CTA;
  int query_tile = blockIdx.y;
  int query_base = query_tile * QN;

  for (int s = tid; s < A_BARRIERS + B_BARRIERS + FREE_BARRIERS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  // Match the batch=64 sparse path: periodically tighten each query row's
  // threshold from the accumulated histogram.  smalln uses blockIdx.x as the
  // base-tile group axis, so refresh on that axis instead of blockIdx.y.
  if (blockIdx.x > 0 && THR_REFRESH_GROUP > 0 &&
      (blockIdx.x % THR_REFRESH_GROUP) == 0) {
    if (warpgroup == 0) {
      smalln_refresh_threshold(query_base, tid, th, bcount, NB, K);
    }
  }

  if (warpgroup == 0 && tid == 0) {
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        mbarrier_expect_tx(&barriers_a[pre], k_stages * A_STAGE_BYTES);
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[pre], ks * K_TILE, base_tile * BM);
        }
      }
    }
    for (int pre = 0; pre < C_STAGE; ++pre) {
      mbarrier_expect_tx(&barriers_b[pre],
                         k_stages * QB_K_STAGE_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_b + pre * QB_STAGE_ELEMS, &query_map, &barriers_b[pre], 0, query_base, 0);
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = base_tile_group_start + refill;
        if (refill_tile >= total_base_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_a[refill_stage], k_stages * A_STAGE_BYTES);
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[refill_stage], ks * K_TILE, refill_tile * BM);
        }
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<32>, cute::Int<D>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<32>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<K_STAGES>{},
                       cute::Int<C_STAGE>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{}, cute::make_shape(cute::Int<32>{}, cute::Int<K_TILE>{},
                                        cute::Int<K_STAGES>{}, cute::Int<C_STAGE>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_b)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int base_tile = base_tile_group_start + c_rel;
    if (warpgroup == owner_wg && base_tile < total_base_tiles) {
      mbarrier_wait(&barriers_a[stage], phase);
      if (c_rel < C_STAGE) {
        mbarrier_wait(&barriers_b[stage], 0);
      }
      cute_emit_gemm_only_runtime(
          k_stages, cute_tiled_mma, cute_tCrA(cute::_, cute::_, cute::_, cute::_, stage),
          cute_tCrB(cute::_, cute::_, cute::_, cute::_, stage), cute_tCrC);
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      CUTE_UNROLL
      for (int p = 0; p < 16; p += 4) {
        CUTE_UNROLL
        for (int s = 0; s < 2; ++s) {
          int row = query_base + int(cute::get<1>(cute_tCcC(p + s)));
          __half vals[2];
          int idxs[2];
          int n = 0;
          CUTE_UNROLL
          for (int k2 = 0; k2 < 2; ++k2) {
            int i = p + s + k2 * 2;
            int base_rel = int(cute::get<0>(cute_tCcC(i)));
            int global_base = base_tile * BM + base_rel;
            float vf = -float(cute_tCrC(i));
            int braw = int((vf - origin[row]) * inv_delta[row]);
            int valid = (global_base < M_stride) && (braw < NB);
            int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
            int pred = valid && (b <= th[row]);
#if !HOPPER_SPARSE_NO_HIST && HOPPER_SPARSE_REFRESH_FROM_BUF == 0
            if (pred) {
              int shard = (blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5)) % BCOUNT_SHARDS;
              atomicAdd(&bcount[(row * NB + b) * BCOUNT_SHARDS + shard], 1);
            }
#endif
            if (pred) {
              vals[n] = __float2half(vf);
              idxs[n] = global_base;
              ++n;
            }
          }
          if (n > 0) {
            int base = atomicAdd(&qcount[row], n);
            CUTE_UNROLL
            for (int j = 0; j < 2; ++j) {
              if (j >= n) break;
              int pos = base + j;
              if (pos < BUF) {
                buf_val[(size_t)row * BUF + pos] = vals[j];
                buf_idx[(size_t)row * BUF + pos] = idxs[j];
              }
            }
          }
        }
      }
    }
  }
  __syncwarp();
}

template <typename T>
__device__ __forceinline__ T hopper_sparse_value_cast(float v);

template <>
__device__ __forceinline__ __half hopper_sparse_value_cast<__half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ float hopper_sparse_value_cast<float>(float v) {
  return v;
}

// K-chunk streaming sparse m64n32 kernel for D > 512 (batch<=32).  Same ring
// pipeline as the dense kchunk kernel; epilogue is the pair-aggregation sparse
// write used by hopper_smalln_score_to_sparse_m64n32_kernel.
template <typename OutT>
__global__ __launch_bounds__(KCHUNK_THREADS, 1)
void hopper_smalln_score_to_sparse_m64n32_kchunk_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    OutT* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    int total_base_tiles, int start_base_tile, int M_stride, int BUF, int NB, int K,
    int k_stages) {
  constexpr int QN = 32;
  constexpr int QB_K_TILE_ELEMS = QN * K_TILE;
  constexpr int QB_ELEMS = KMAX_STAGES * QB_K_TILE_ELEMS;
  constexpr int QB_BYTES = QB_ELEMS * int(sizeof(__half));
  constexpr int AB_SLOT_ELEMS = BM * K_TILE;
  constexpr int AB_ELEMS = KCHUNK_RING * AB_SLOT_ELEMS;
  constexpr int AB_BYTES = AB_ELEMS * int(sizeof(__half));
  constexpr int RING_BARS = KCHUNK_RING;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QB_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QB_BYTES + AB_BYTES);
  uint64_t* bar_filled = barriers;
  uint64_t* bar_free = barriers + RING_BARS;
  uint64_t* bar_qrdy = barriers + 2 * RING_BARS;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpgroup = tid >> 7;
  int base_tile_group_start = start_base_tile + blockIdx.x * C_TILES_PER_CTA;
  int query_tile = blockIdx.y;
  int query_base = query_tile * QN;

  for (int s = tid; s < 2 * RING_BARS + 1; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  // Periodically tighten thresholds for the 32 query rows in this smalln tile.
  // blockIdx.x is the base-tile group axis for m64n32.
  if (blockIdx.x > 0 && THR_REFRESH_GROUP > 0 &&
      (blockIdx.x % THR_REFRESH_GROUP) == 0) {
    if (warpgroup == 0) {
      smalln_refresh_threshold(query_base, tid, th, bcount, NB, K);
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      mbarrier_expect_tx(&bar_qrdy[0], k_stages * QB_K_TILE_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    }
    int prod = tid >> 5;  // warp index within producer warpgroup (0..3)
    if (prod < KCHUNK_MATH_WG && (tid & 31) == 0) {
      kchunk_produce_half(smem_a, &base_map, bar_filled, bar_free, /*parity=*/prod,
                          /*seg_base=*/prod * KCHUNK_SEG, KCHUNK_SEG, AB_SLOT_ELEMS,
                          base_tile_group_start, total_base_tiles, k_stages, C_TILES_PER_CTA);
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<32>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<32>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<KCHUNK_RING>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<32>{}, cute::Int<K_TILE>{}, cute::Int<KMAX_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<32>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  // N-way ping-pong: math WG p owns base tiles with (c_rel % KCHUNK_MATH_WG==p),
  // consuming from ring segment [p*SEG, (p+1)*SEG).
  int my_parity = warpgroup - 1;            // WG1->0, WG2->1, WG3->2
  int my_seg_base = my_parity * KCHUNK_SEG;
  int math_warp = (tid >> 5) - warpgroup * 4;
  bool releaser = (math_warp == 0 && lane == 0);

  mbarrier_wait(&bar_qrdy[0], 0);

  int consumed = 0;
  for (int c_rel = my_parity; c_rel < C_TILES_PER_CTA; c_rel += KCHUNK_MATH_WG) {
    int base_tile = base_tile_group_start + c_rel;
    if (base_tile >= total_base_tiles) break;

    kchunk_stream_one_ctile_aring(cute_tiled_mma, cute_tCrA, cute_tCrB, cute_tCrC,
                                  bar_filled, bar_free, consumed, my_seg_base, KCHUNK_SEG,
                                  k_stages, releaser);
    CUTE_UNROLL
    for (int p = 0; p < 16; p += 4) {
      CUTE_UNROLL
      for (int s = 0; s < 2; ++s) {
        int row = query_base + int(cute::get<1>(cute_tCcC(p + s)));
        OutT vals[2];
        int idxs[2];
        int n = 0;
        CUTE_UNROLL
        for (int k2 = 0; k2 < 2; ++k2) {
          int i = p + s + k2 * 2;
          int base_rel = int(cute::get<0>(cute_tCcC(i)));
          int global_base = base_tile * BM + base_rel;
          float vf = -float(cute_tCrC(i));
          int braw = int((vf - origin[row]) * inv_delta[row]);
          int valid = (global_base < M_stride) && (braw < NB);
          int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
          int pred = valid && (b <= th[row]);
#if !HOPPER_SPARSE_NO_HIST && HOPPER_SPARSE_REFRESH_FROM_BUF == 0
          if (pred) {
            int shard = (blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5)) % BCOUNT_SHARDS;
            atomicAdd(&bcount[(row * NB + b) * BCOUNT_SHARDS + shard], 1);
          }
#endif
          if (pred) {
            vals[n] = hopper_sparse_value_cast<OutT>(vf);
            idxs[n] = global_base;
            ++n;
          }
        }
        if (n > 0) {
          int base = atomicAdd(&qcount[row], n);
          CUTE_UNROLL
          for (int j = 0; j < 2; ++j) {
            if (j >= n) break;
            int pos = base + j;
            if (pos < BUF) {
              buf_val[(size_t)row * BUF + pos] = vals[j];
              buf_idx[(size_t)row * BUF + pos] = idxs[j];
            }
          }
        }
      }
    }
  }
  __syncwarp();
}

template <int Base, class TensorC, class TensorCoord, typename OutT>
__device__ __forceinline__ void sparse_emit_group16(
    TensorC const& tCrC, TensorCoord const& tCcC, int row, int global_col_base,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    OutT* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    uint8_t* __restrict__ buf_bucket,
    __half* __restrict__ bucket_val, int32_t* __restrict__ bucket_idx, int bucket_cap,
    int BUF, int NB, int K);

extern "C" __global__ __launch_bounds__(THREADS, 1) void hopper_score_to_sparse_ip_kernel(
    const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap b_map,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    int32_t* __restrict__ qprev, int32_t* __restrict__ refresh_lock,
    __half* __restrict__ buf_val, int32_t* __restrict__ buf_idx, uint8_t* __restrict__ buf_bucket,
    __half* __restrict__ bucket_val, int32_t* __restrict__ bucket_idx, int bucket_cap,
    int total_c_tiles, int c_start_tile, int BUF, int NB, int K, int k_stages) {
  extern __shared__ __align__(128) unsigned char smem[];

  __half* smem_a = reinterpret_cast<__half*>(smem);
  __half* smem_b = reinterpret_cast<__half*>(smem + A_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + A_BYTES + B_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_b = barriers + A_BARRIERS;
  uint64_t* barriers_free = barriers + A_BARRIERS + B_BARRIERS;
#if HOPPER_SPARSE_REFRESH_FROM_BUF > 0
  // Only the refresh-from-buf path consumed this scratch histogram; it is unused
  // on the production path (HOPPER_THR_REFRESH_GROUP=0) so keep it guarded.
  int32_t* refresh_hist = reinterpret_cast<int32_t*>(smem + SMEM_BYTES);
  __shared__ int refresh_start_s[BM];
  __shared__ int refresh_end_s[BM];
  __shared__ int refresh_locked_s[BM];
  __shared__ int refresh_prefix_s[BM + 1];
  __shared__ int refresh_done_s;
#endif

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int cta_m_base = blockIdx.x * BM;
  int cta_c_group_start = c_start_tile + blockIdx.y * C_TILES_PER_CTA;

  for (int s = tid; s < A_BARRIERS + B_BARRIERS + FREE_BARRIERS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0 && tid == 0) {
    for (int ks = 0; ks < k_stages; ++ks) {
      mbarrier_expect_tx(&barriers_a[ks], A_STAGE_BYTES);
      tma_load_2d(smem_a + ks * A_STAGE_ELEMS, &a_map, &barriers_a[ks], ks * K_TILE, cta_m_base);
    }
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int c_tile = cta_c_group_start + pre;
      if (c_tile < total_c_tiles) {
        mbarrier_expect_tx(&barriers_b[pre], k_stages * B_K_STAGE_ELEMS * int(sizeof(__half)));
        tma_load_3d(smem_b + pre * B_STAGE_ELEMS, &b_map, &barriers_b[pre], 0, c_tile * BN, 0);
      }
    }
  }

#if HOPPER_THR_REFRESH_GROUP > 0
  // Dead in-kernel threshold-refresh variants removed. The production/optimal
  // build keeps HOPPER_THR_REFRESH_GROUP=0 (the torch loader never overrides it),
  // so this whole region never compiled on the real path. It held the
  // experimental refresh-from-buf implementations (REFRESH_FROM_BUF>0: a locked
  // producer using atomicCAS(refresh_lock)+qprev with FLATTEN_ONE_SYNC /
  // FLATTEN_PARALLEL_FLUSH flush strategies), the RFB_PRODUCER_ONLY lock variant,
  // and a plain bcount-cumsum #else. On the real path the gate is refreshed
  // OUTSIDE this kernel: the scoring epilogue writes bcount inline and
  // launch_hopper_refresh_threshold_from_bcount tightens th between CTA waves.
#endif

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = cta_c_group_start + refill;
        if (refill_tile >= total_c_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_b[refill_stage],
                           k_stages * B_K_STAGE_ELEMS * int(sizeof(__half)));
        tma_load_3d(smem_b + refill_stage * B_STAGE_ELEMS, &b_map, &barriers_b[refill_stage], 0,
                    refill_tile * BN, 0);
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<D>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<K_STAGES>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{}, cute::make_shape(cute::Int<BN>{}, cute::Int<K_TILE>{},
                                        cute::Int<K_STAGES>{}, cute::Int<C_STAGE>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_b)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);
  int cute_row0 = cute_coord_row<0>(cute_tCcC);
  int cute_row1 = cute_coord_row<16>(cute_tCcC);

  for (int ks = 0; ks < k_stages; ++ks) {
    mbarrier_wait(&barriers_a[ks], 0);
  }

  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int c_tile = cta_c_group_start + c_rel;
    if (warpgroup == owner_wg && c_tile < total_c_tiles) {
      mbarrier_wait(&barriers_b[stage], phase);
      if (stage == 0) {
        cute_emit_gemm_only_runtime(
            k_stages, cute_tiled_mma, cute_tCrA,
            cute_tCrB(cute::_, cute::_, cute::_, cute::_, cute::Int<0>{}), cute_tCrC);
      } else {
        cute_emit_gemm_only_runtime(
            k_stages, cute_tiled_mma, cute_tCrA,
            cute_tCrB(cute::_, cute::_, cute::_, cute::_, cute::Int<1>{}), cute_tCrC);
      }
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      int col_base = c_tile * BN;
      sparse_emit_group16<0>(cute_tCrC, cute_tCcC, cta_m_base + cute_row0, col_base,
                             origin, inv_delta, th, qcount, bcount, buf_val, buf_idx, buf_bucket,
                             bucket_val, bucket_idx, bucket_cap, BUF, NB, K);
      sparse_emit_group16<16>(cute_tCrC, cute_tCcC, cta_m_base + cute_row1, col_base,
                              origin, inv_delta, th, qcount, bcount, buf_val, buf_idx, buf_bucket,
                              bucket_val, bucket_idx, bucket_cap, BUF, NB, K);
    }
  }
}

// ===========================================================================
// K-chunk streaming m64n64 sparse kernel for D > 512 (batch=64).
//   query A (64 x D) resident + base B (64 x D) streamed along K (ring).
//   Single math warpgroup; epilogue reuses sparse_emit_group16.
//   Only the default RFB=0 threshold refresh path is supported.
// ===========================================================================
template <typename OutT>
__global__ __launch_bounds__(KCHUNK_THREADS, 1)
void hopper_score_to_sparse_ip_kchunk_kernel(
    const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap b_map,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    OutT* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    int total_c_tiles, int c_start_tile, int BUF, int NB, int K, int k_stages) {
  constexpr int QA_SLOT_ELEMS = BM * K_TILE;
  constexpr int QA_ELEMS = KMAX_STAGES * QA_SLOT_ELEMS;
  constexpr int QA_BYTES = QA_ELEMS * int(sizeof(__half));
  constexpr int BB_SLOT_ELEMS = BN * K_TILE;
  constexpr int BB_ELEMS = KCHUNK_RING * BB_SLOT_ELEMS;
  constexpr int BB_BYTES = BB_ELEMS * int(sizeof(__half));
  constexpr int RING_BARS = KCHUNK_RING;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_a = reinterpret_cast<__half*>(smem);
  __half* smem_b = reinterpret_cast<__half*>(smem + QA_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QA_BYTES + BB_BYTES);
  uint64_t* bar_filled = barriers;
  uint64_t* bar_free = barriers + RING_BARS;
  uint64_t* bar_ardy = barriers + 2 * RING_BARS;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpgroup = tid >> 7;
  int math_warp = (tid >> 5) - warpgroup * 4;
  int cta_m_base = blockIdx.x * BM;
  int cta_c_group_start = c_start_tile + blockIdx.y * C_TILES_PER_CTA;

  for (int s = tid; s < 2 * RING_BARS + 1; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  // Threshold refresh (default RFB=0 path): recompute th from the global
  // histogram once per CTA group before consuming this group's tiles.
  if (warpgroup == 0 && blockIdx.y > 0 && THR_REFRESH_GROUP > 0 &&
      (blockIdx.y % THR_REFRESH_GROUP) == 0) {
    for (int row_local = tid; row_local < BM; row_local += 128) {
      int row = cta_m_base + row_local;
      int cum = 0;
      int new_th = NB - 1;
      for (int b = 0; b < NB; ++b) {
        int bc = 0;
        CUTE_UNROLL
        for (int s = 0; s < BCOUNT_SHARDS; ++s) {
          bc += bcount[((row * NB + b) * BCOUNT_SHARDS) + s];
        }
        cum += bc;
        if (cum >= K) { new_th = b; break; }
      }
      atomicMin(&th[row], new_th);
    }
  }

  if (warpgroup == 0) {
    // One producer thread per math warpgroup feeds its own ring segment.
    if (tid == 0) {
      mbarrier_expect_tx(&bar_ardy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
      for (int ks = 0; ks < k_stages; ++ks) {
        tma_load_2d(smem_a + ks * QA_SLOT_ELEMS, &a_map, &bar_ardy[0], ks * K_TILE, cta_m_base);
      }
    }
    int prod = tid >> 5;  // warp index within producer warpgroup (0..3)
    if (prod < KCHUNK_MATH_WG && (tid & 31) == 0) {
      kchunk_produce_half(smem_b, &b_map, bar_filled, bar_free, /*parity=*/prod,
                          /*seg_base=*/prod * KCHUNK_SEG, KCHUNK_SEG, BB_SLOT_ELEMS,
                          cta_c_group_start, total_c_tiles, k_stages, C_TILES_PER_CTA);
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<BN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<KMAX_STAGES>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<BN>{}, cute::Int<K_TILE>{}, cute::Int<KCHUNK_RING>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  cute::Tensor cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  cute::Tensor cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_b)), CuteSmemLayoutB{});
  cute::Tensor cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  cute::Tensor cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  cute::Tensor cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_cC =
      cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<BN>>{});
  cute::Tensor cute_tCcC = cute_thr_mma.partition_C(cute_cC);
  int cute_row0 = cute_coord_row<0>(cute_tCcC);
  int cute_row1 = cute_coord_row<16>(cute_tCcC);

  // N-way ping-pong: math WG p owns C-tiles with (c_rel % KCHUNK_MATH_WG == p),
  // consuming from its own ring segment [p*SEG, (p+1)*SEG).
  int my_parity = warpgroup - 1;            // WG1->0, WG2->1, WG3->2
  int my_seg_base = my_parity * KCHUNK_SEG;
  bool releaser = (math_warp == 0 && lane == 0);

  mbarrier_wait(&bar_ardy[0], 0);

  int consumed = 0;
  for (int c_rel = my_parity; c_rel < C_TILES_PER_CTA; c_rel += KCHUNK_MATH_WG) {
    int c_tile = cta_c_group_start + c_rel;
    if (c_tile >= total_c_tiles) break;

    kchunk_stream_one_ctile(cute_tiled_mma, cute_tCrA, cute_tCrB, cute_tCrC,
                            bar_filled, bar_free, consumed, my_seg_base, KCHUNK_SEG,
                            k_stages, releaser);
    int col_base = c_tile * BN;
    sparse_emit_group16<0>(cute_tCrC, cute_tCcC, cta_m_base + cute_row0, col_base,
                           origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
                           nullptr, nullptr, nullptr, 0, BUF, NB, K);
    sparse_emit_group16<16>(cute_tCrC, cute_tCcC, cta_m_base + cute_row1, col_base,
                            origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
                            nullptr, nullptr, nullptr, 0, BUF, NB, K);
  }
}

// m64n8 full-kernel SMEM (bytes) for KS resident K-stages: QN=8 query +
// C_STAGE base double-buffer + barriers. Used both to size the launch and to
// derive how many blocks can co-reside per SM.
constexpr int ft_m8_full_smem(int KS) {
  return KS * 8 * K_TILE * int(sizeof(__half))
       + C_STAGE * KS * BM * K_TILE * int(sizeof(__half))
       + (2 * C_STAGE + 1) * int(sizeof(uint64_t));
}
// minBlocksPerMultiprocessor for the m64n8 full kernels. Low D (small KS) frees
// enough SMEM that two CTAs co-reside on one SM (H100 ~228KB/SM), which the
// fixed 12-stage version could never do; high D stays at 1. ~110KB budget per
// CTA keeps room for the static/driver SMEM overhead.
constexpr int ft_m8_min_blocks(int KS) {
  return (2 * ft_m8_full_smem(KS) <= 110 * 1024) ? 2 : 1;
}
// GQA sparse m8 full kernel launch bounds. The kernel launches with
// dim3(THREADS=384) but historically declared __launch_bounds__ for
// KCHUNK_THREADS=512, so ptxas budgeted registers for 512 threads and
// landed at 56 reg -> 3 resident CTAs. Declaring (384, 4) caps registers
// at 40 and lands 4 resident CTAs (occupancy 56% -> 75%), measured
// ~2.5% faster at 768K with neutral 1M and no recall change.
#ifndef HOPPER_GQA_M8_FULL_LB_THREADS
#define HOPPER_GQA_M8_FULL_LB_THREADS THREADS
#endif
#ifndef HOPPER_GQA_M8_FULL_LB_MINBLK
#define HOPPER_GQA_M8_FULL_LB_MINBLK 4
#endif
constexpr int ft_m8_lb_threads() { return HOPPER_GQA_M8_FULL_LB_THREADS; }
constexpr int ft_m8_lb_minblk(int KS) {
  return (HOPPER_GQA_M8_FULL_LB_MINBLK > 0) ? HOPPER_GQA_M8_FULL_LB_MINBLK
                                            : ft_m8_min_blocks(KS);
}

template <int KS>
__global__ __launch_bounds__(KCHUNK_THREADS, ft_m8_min_blocks(KS))
void hopper_smalln_score_m64n8_full_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    __half* __restrict__ dense_scores, int total_base_tiles, int M_stride, int k_stages,
    int logical_rows) {
  constexpr int QN = 8;
  constexpr int M8_FULL_K_STAGES = KS;
  constexpr int QA_SLOT_ELEMS = QN * K_TILE;
  constexpr int QA_ELEMS = M8_FULL_K_STAGES * QA_SLOT_ELEMS;
  constexpr int QA_BYTES = QA_ELEMS * int(sizeof(__half));
  constexpr int BASE_TILE_ELEMS = M8_FULL_K_STAGES * A_STAGE_ELEMS;
  constexpr int BASE_BYTES = C_STAGE * BASE_TILE_ELEMS * int(sizeof(__half));
  constexpr int BARS = C_STAGE + C_STAGE + 1;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QA_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QA_BYTES + BASE_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_free = barriers + C_STAGE;
  uint64_t* bar_qrdy = barriers + 2 * C_STAGE;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int base_tile_group_start = blockIdx.x * C_TILES_PER_CTA;
  int query_base = blockIdx.y * QN;

  for (int s = tid; s < BARS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0 && tid == 0) {
    mbarrier_expect_tx(&bar_qrdy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
    tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        mbarrier_expect_tx(&barriers_a[pre], k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[pre], ks * K_TILE, base_tile * BM);
        }
      }
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = base_tile_group_start + refill;
        if (refill_tile >= total_base_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_a[refill_stage],
                           k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[refill_stage], ks * K_TILE, refill_tile * BM);
        }
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<QN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<QN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{},
                       cute::Int<C_STAGE>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<QN>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  auto cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  auto cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  auto cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  auto cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  auto cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_cC = cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  mbarrier_wait(&bar_qrdy[0], 0);
  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int base_tile = base_tile_group_start + c_rel;
    if (warpgroup == owner_wg && base_tile < total_base_tiles) {
      mbarrier_wait(&barriers_a[stage], phase);
      cute_emit_gemm_only_active_k12<KS>(
          cute_tiled_mma, cute_tCrA(cute::_, cute::_, cute::_, cute::_, stage),
          cute_tCrB, cute_tCrC);
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      CUTE_UNROLL
      for (int i = 0; i < cute::size(cute_tCrC); ++i) {
        int base_rel = int(cute::get<0>(cute_tCcC(i)));
        int q_rel = int(cute::get<1>(cute_tCcC(i)));
        int global_base = base_tile * BM + base_rel;
        int global_q = query_base + q_rel;
        if (q_rel < logical_rows && global_base < M_stride) {
          dense_scores[(size_t)global_q * M_stride + global_base] =
              __float2half(-float(cute_tCrC(i)));
        }
      }
    }
  }
}

// Per-row threshold refresh against a plain [R, NB] histogram: smallest bucket
// whose cumulative count reaches K, lowered via atomicMin. Used by the fused
// full-store + in-GEMM bucket-histogram kernel below.
__device__ __noinline__ void smalln_refresh_threshold_rows_plain(
    int query_base, int rows, int tid, int32_t* __restrict__ th,
    const int32_t* __restrict__ bcount, int NB, int K) {
  for (int row_local = tid; row_local < rows; row_local += 128) {
    int row = query_base + row_local;
    int cum = 0;
    int new_th = NB - 1;
    for (int b = 0; b < NB; ++b) {
      cum += bcount[row * NB + b];
      if (cum >= K) {
        new_th = b;
        break;
      }
    }
    atomicMin(&th[row], new_th);
  }
}

// Fused full-store + in-GEMM bucket histogram. Same coalesced full store as
// hopper_smalln_score_m64n8_full_kernel (every column written), but the
#if 0  // RETIRED: in-GEMM histogram kernel (dense_bucket_fused only)
// epilogue ALSO accumulates a bucket histogram into global bcount — gated by a
// sample-seeded threshold bucket `gate[row]`: only scores landing in bucket
// b <= gate emit an atomicAdd. The seeded gate is conservative (>= the final
// threshold), so all buckets <= final-th are counted exactly; this is what the
// downstream threshold/select need, while atomics drop from O(M) to ~O(k).
template <int KS>
__global__ __launch_bounds__(KCHUNK_THREADS, ft_m8_min_blocks(KS))
void hopper_smalln_score_m64n8_full_hist_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ gate,
    int32_t* __restrict__ bcount,
    __half* __restrict__ dense_scores, int total_base_tiles, int M_stride, int k_stages,
    int logical_rows, int NB, int K, int start_base_tile) {
  constexpr int QN = 8;
  constexpr int M8_FULL_K_STAGES = KS;
  constexpr int QA_SLOT_ELEMS = QN * K_TILE;
  constexpr int QA_ELEMS = M8_FULL_K_STAGES * QA_SLOT_ELEMS;
  constexpr int QA_BYTES = QA_ELEMS * int(sizeof(__half));
  constexpr int BASE_TILE_ELEMS = M8_FULL_K_STAGES * A_STAGE_ELEMS;
  constexpr int BASE_BYTES = C_STAGE * BASE_TILE_ELEMS * int(sizeof(__half));
  constexpr int BARS = C_STAGE + C_STAGE + 1;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QA_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QA_BYTES + BASE_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_free = barriers + C_STAGE;
  uint64_t* bar_qrdy = barriers + 2 * C_STAGE;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int base_tile_group_start = start_base_tile + blockIdx.x * C_TILES_PER_CTA;
  int query_base = blockIdx.y * QN;

  for (int s = tid; s < BARS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  // Dynamic gate tightening: the sample-derived gate is conservative; as tail
  // buckets accumulate in bcount, periodically lower it so later tile groups
  // emit fewer bucket atomics. bcount here is the plain [R, NB] layout.
  if (blockIdx.x > 0 && THR_REFRESH_GROUP > 0 &&
      (blockIdx.x % THR_REFRESH_GROUP) == 0) {
    if (warpgroup == 0) {
      smalln_refresh_threshold_rows_plain(query_base, logical_rows, tid, gate, bcount, NB, K);
    }
  }

  if (warpgroup == 0 && tid == 0) {
    mbarrier_expect_tx(&bar_qrdy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
    tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        mbarrier_expect_tx(&barriers_a[pre], k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[pre], ks * K_TILE, base_tile * BM);
        }
      }
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = base_tile_group_start + refill;
        if (refill_tile >= total_base_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_a[refill_stage],
                           k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[refill_stage], ks * K_TILE, refill_tile * BM);
        }
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<QN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<QN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{},
                       cute::Int<C_STAGE>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<QN>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  auto cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  auto cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  auto cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  auto cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  auto cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_cC = cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  // Hoist per-row bucket coords/cutoff into registers (<= QN rows) so the
  // epilogue reads them from registers. Gate before bucketization: b <= gate
  // is equivalent to vf <= origin + gate / inv_delta.
  float r_origin[QN];
  float r_invd[QN];
  float r_cutoff[QN];
  int r_gate[QN];
  CUTE_UNROLL
  for (int r = 0; r < QN; ++r) {
    int grow = query_base + r;
    bool valid = (r < logical_rows);
    float o = valid ? origin[grow] : 0.0f;
    float inv = valid ? inv_delta[grow] : 1.0f;
    int g = valid ? gate[grow] : -1;
    r_origin[r] = o;
    r_invd[r] = inv;
    r_gate[r] = g;
    r_cutoff[r] = valid ? (o + float(g) / inv) : -INFINITY;
  }

  mbarrier_wait(&bar_qrdy[0], 0);
  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int base_tile = base_tile_group_start + c_rel;
    if (warpgroup == owner_wg && base_tile < total_base_tiles) {
      mbarrier_wait(&barriers_a[stage], phase);
      cute_emit_gemm_only_active_k12<KS>(
          cute_tiled_mma, cute_tCrA(cute::_, cute::_, cute::_, cute::_, stage),
          cute_tCrB, cute_tCrC);
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      CUTE_UNROLL
      for (int i = 0; i < cute::size(cute_tCrC); ++i) {
        int base_rel = int(cute::get<0>(cute_tCcC(i)));
        int q_rel = int(cute::get<1>(cute_tCcC(i)));
        int global_base = base_tile * BM + base_rel;
        int global_q = query_base + q_rel;
        if (q_rel < logical_rows && global_base < M_stride) {
          float vf = -float(cute_tCrC(i));
          dense_scores[(size_t)global_q * M_stride + global_base] = __float2half(vf);
          // Gate first; only bucketize survivors.
          if (vf <= r_cutoff[q_rel]) {
            int braw = int((vf - r_origin[q_rel]) * r_invd[q_rel]);
            int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
            atomicAdd(&bcount[global_q * NB + b], 1);
          }
        }
      }
    }
  }
}

#endif  // hist kernel
extern "C" __global__ __launch_bounds__(KCHUNK_THREADS, 1)
void hopper_smalln_score_m64n8_kchunk_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    __half* __restrict__ dense_scores, int total_base_tiles, int M_stride, int k_stages,
    int logical_rows) {
  constexpr int QN = 8;
  constexpr int QB_K_TILE_ELEMS = QN * K_TILE;
  constexpr int QB_ELEMS = KMAX_STAGES * QB_K_TILE_ELEMS;
  constexpr int QB_BYTES = QB_ELEMS * int(sizeof(__half));
  constexpr int AB_SLOT_ELEMS = BM * K_TILE;
  constexpr int AB_ELEMS = KCHUNK_RING * AB_SLOT_ELEMS;
  constexpr int AB_BYTES = AB_ELEMS * int(sizeof(__half));
  constexpr int RING_BARS = KCHUNK_RING;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QB_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QB_BYTES + AB_BYTES);
  uint64_t* bar_filled = barriers;
  uint64_t* bar_free = barriers + RING_BARS;
  uint64_t* bar_qrdy = barriers + 2 * RING_BARS;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpgroup = tid >> 7;
  int base_tile_group_start = blockIdx.x * C_TILES_PER_CTA;
  int query_base = blockIdx.y * QN;

  for (int s = tid; s < 2 * RING_BARS + 1; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0) {
    if (tid == 0) {
      mbarrier_expect_tx(&bar_qrdy[0], k_stages * QB_K_TILE_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    }
    int prod = tid >> 5;
    if (prod < KCHUNK_MATH_WG && (tid & 31) == 0) {
      kchunk_produce_half(smem_a, &base_map, bar_filled, bar_free, /*parity=*/prod,
                          /*seg_base=*/prod * KCHUNK_SEG, KCHUNK_SEG, AB_SLOT_ELEMS,
                          base_tile_group_start, total_base_tiles, k_stages, C_TILES_PER_CTA);
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<QN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<QN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<KCHUNK_RING>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<QN>{}, cute::Int<K_TILE>{}, cute::Int<KMAX_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  auto cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  auto cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  auto cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  auto cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  auto cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_cC = cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  int my_parity = warpgroup - 1;
  int my_seg_base = my_parity * KCHUNK_SEG;
  int math_warp = (tid >> 5) - warpgroup * 4;
  bool releaser = (math_warp == 0 && lane == 0);

  mbarrier_wait(&bar_qrdy[0], 0);

  int consumed = 0;
  for (int c_rel = my_parity; c_rel < C_TILES_PER_CTA; c_rel += KCHUNK_MATH_WG) {
    int base_tile = base_tile_group_start + c_rel;
    if (base_tile >= total_base_tiles) break;
    kchunk_stream_one_ctile_aring(cute_tiled_mma, cute_tCrA, cute_tCrB, cute_tCrC,
                                  bar_filled, bar_free, consumed, my_seg_base, KCHUNK_SEG,
                                  k_stages, releaser);
    CUTE_UNROLL
    for (int i = 0; i < cute::size(cute_tCrC); ++i) {
      int base_rel = int(cute::get<0>(cute_tCcC(i)));
      int q_rel = int(cute::get<1>(cute_tCcC(i)));
      int global_base = base_tile * BM + base_rel;
      int global_q = query_base + q_rel;
      if (q_rel < logical_rows && global_base < M_stride) {
        dense_scores[(size_t)global_q * M_stride + global_base] =
            __float2half(-float(cute_tCrC(i)));
      }
    }
  }
}

template <int KS>
__global__ __launch_bounds__(KCHUNK_THREADS, ft_m8_min_blocks(KS))
void hopper_smalln_score_to_sparse_m64n8_full_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    __half* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    int total_base_tiles, int start_base_tile, int M_stride, int BUF, int NB, int K,
    int k_stages, int logical_rows) {
  constexpr int QN = 8;
  constexpr int M8_FULL_K_STAGES = KS;
  constexpr int QA_SLOT_ELEMS = QN * K_TILE;
  constexpr int QA_ELEMS = M8_FULL_K_STAGES * QA_SLOT_ELEMS;
  constexpr int QA_BYTES = QA_ELEMS * int(sizeof(__half));
  constexpr int BASE_TILE_ELEMS = M8_FULL_K_STAGES * A_STAGE_ELEMS;
  constexpr int BASE_BYTES = C_STAGE * BASE_TILE_ELEMS * int(sizeof(__half));
  constexpr int BARS = C_STAGE + C_STAGE + 1;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QA_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QA_BYTES + BASE_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_free = barriers + C_STAGE;
  uint64_t* bar_qrdy = barriers + 2 * C_STAGE;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int base_tile_group_start = start_base_tile + blockIdx.x * C_TILES_PER_CTA;
  int query_base = blockIdx.y * QN;

  for (int s = tid; s < BARS; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (blockIdx.x > 0 && THR_REFRESH_GROUP > 0 &&
      (blockIdx.x % THR_REFRESH_GROUP) == 0) {
    if (warpgroup == 0) {
      smalln_refresh_threshold_rows(query_base, logical_rows, tid, th, bcount, NB, K);
    }
  }

  if (warpgroup == 0 && tid == 0) {
    mbarrier_expect_tx(&bar_qrdy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
    tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        mbarrier_expect_tx(&barriers_a[pre], k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[pre], ks * K_TILE, base_tile * BM);
        }
      }
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = base_tile_group_start + refill;
        if (refill_tile >= total_base_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_a[refill_stage],
                           k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          tma_load_2d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                      &base_map, &barriers_a[refill_stage], ks * K_TILE, refill_tile * BM);
        }
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<QN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<QN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{},
                       cute::Int<C_STAGE>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<QN>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  auto cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  auto cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  auto cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  auto cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  auto cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_cC = cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  mbarrier_wait(&bar_qrdy[0], 0);
  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int base_tile = base_tile_group_start + c_rel;
    if (warpgroup == owner_wg && base_tile < total_base_tiles) {
      mbarrier_wait(&barriers_a[stage], phase);
      cute_emit_gemm_only_active_k12<KS>(
          cute_tiled_mma, cute_tCrA(cute::_, cute::_, cute::_, cute::_, stage),
          cute_tCrB, cute_tCrC);
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      CUTE_UNROLL
      for (int i = 0; i < cute::size(cute_tCrC); ++i) {
        int q_rel = int(cute::get<1>(cute_tCcC(i)));
        if (q_rel >= logical_rows) continue;
        int row = query_base + q_rel;
        int base_rel = int(cute::get<0>(cute_tCcC(i)));
        int global_base = base_tile * BM + base_rel;
        float vf = -float(cute_tCrC(i));
        int braw = int((vf - origin[row]) * inv_delta[row]);
        int valid = (global_base < M_stride) && (braw < NB);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        int pred = valid && (b <= th[row]);
#if !HOPPER_SPARSE_NO_HIST && HOPPER_SPARSE_REFRESH_FROM_BUF == 0
        if (pred) {
          int shard = (blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5)) % BCOUNT_SHARDS;
          atomicAdd(&bcount[(row * NB + b) * BCOUNT_SHARDS + shard], 1);
        }
#endif
        if (pred) {
          int pos = atomicAdd(&qcount[row], 1);
          if (pos < BUF) {
            buf_val[(size_t)row * BUF + pos] = __float2half(vf);
            buf_idx[(size_t)row * BUF + pos] = global_base;
          }
        }
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(KCHUNK_THREADS, 1)
void hopper_smalln_score_to_sparse_m64n8_kchunk_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    __half* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    int total_base_tiles, int start_base_tile, int M_stride, int BUF, int NB, int K,
    int k_stages, int logical_rows) {
  constexpr int QN = 8;
  constexpr int QB_K_TILE_ELEMS = QN * K_TILE;
  constexpr int QB_ELEMS = KMAX_STAGES * QB_K_TILE_ELEMS;
  constexpr int QB_BYTES = QB_ELEMS * int(sizeof(__half));
  constexpr int AB_SLOT_ELEMS = BM * K_TILE;
  constexpr int AB_ELEMS = KCHUNK_RING * AB_SLOT_ELEMS;
  constexpr int AB_BYTES = AB_ELEMS * int(sizeof(__half));
  constexpr int RING_BARS = KCHUNK_RING;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QB_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QB_BYTES + AB_BYTES);
  uint64_t* bar_filled = barriers;
  uint64_t* bar_free = barriers + RING_BARS;
  uint64_t* bar_qrdy = barriers + 2 * RING_BARS;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warpgroup = tid >> 7;
  int base_tile_group_start = start_base_tile + blockIdx.x * C_TILES_PER_CTA;
  int query_base = blockIdx.y * QN;

  for (int s = tid; s < 2 * RING_BARS + 1; s += blockDim.x) {
    mbarrier_init(&barriers[s], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (blockIdx.x > 0 && THR_REFRESH_GROUP > 0 &&
      (blockIdx.x % THR_REFRESH_GROUP) == 0) {
    if (warpgroup == 0) {
      smalln_refresh_threshold_rows(query_base, logical_rows, tid, th, bcount, NB, K);
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      mbarrier_expect_tx(&bar_qrdy[0], k_stages * QB_K_TILE_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    }
    int prod = tid >> 5;
    if (prod < KCHUNK_MATH_WG && (tid & 31) == 0) {
      kchunk_produce_half(smem_a, &base_map, bar_filled, bar_free, /*parity=*/prod,
                          /*seg_base=*/prod * KCHUNK_SEG, KCHUNK_SEG, AB_SLOT_ELEMS,
                          base_tile_group_start, total_base_tiles, k_stages, C_TILES_PER_CTA);
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<QN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<QN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<KCHUNK_RING>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<QN>{}, cute::Int<K_TILE>{}, cute::Int<KMAX_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  auto cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  auto cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  auto cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  auto cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  auto cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_cC = cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  int my_parity = warpgroup - 1;
  int my_seg_base = my_parity * KCHUNK_SEG;
  int math_warp = (tid >> 5) - warpgroup * 4;
  bool releaser = (math_warp == 0 && lane == 0);

  mbarrier_wait(&bar_qrdy[0], 0);

  int consumed = 0;
  for (int c_rel = my_parity; c_rel < C_TILES_PER_CTA; c_rel += KCHUNK_MATH_WG) {
    int base_tile = base_tile_group_start + c_rel;
    if (base_tile >= total_base_tiles) break;
    kchunk_stream_one_ctile_aring(cute_tiled_mma, cute_tCrA, cute_tCrB, cute_tCrC,
                                  bar_filled, bar_free, consumed, my_seg_base, KCHUNK_SEG,
                                  k_stages, releaser);
    CUTE_UNROLL
    for (int i = 0; i < cute::size(cute_tCrC); ++i) {
      int q_rel = int(cute::get<1>(cute_tCcC(i)));
      if (q_rel >= logical_rows) continue;
      int row = query_base + q_rel;
      int base_rel = int(cute::get<0>(cute_tCcC(i)));
      int global_base = base_tile * BM + base_rel;
      float vf = -float(cute_tCrC(i));
      int braw = int((vf - origin[row]) * inv_delta[row]);
      int valid = (global_base < M_stride) && (braw < NB);
      int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
      int pred = valid && (b <= th[row]);
#if !HOPPER_SPARSE_NO_HIST && HOPPER_SPARSE_REFRESH_FROM_BUF == 0
      if (pred) {
        int shard = (blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5)) % BCOUNT_SHARDS;
        atomicAdd(&bcount[(row * NB + b) * BCOUNT_SHARDS + shard], 1);
      }
#endif
      if (pred) {
        int pos = atomicAdd(&qcount[row], 1);
        if (pos < BUF) {
          buf_val[(size_t)row * BUF + pos] = __float2half(vf);
          buf_idx[(size_t)row * BUF + pos] = global_base;
        }
      }
    }
  }
}

__device__ __forceinline__ void sparse_bucket_append(
    int row, int bucket, __half value, int idx,
    int32_t* __restrict__ th, int32_t* __restrict__ bcount,
    __half* __restrict__ bucket_val, int32_t* __restrict__ bucket_idx,
    int bucket_cap, int NB);

template <typename OutT>
__global__ void hopper_seed_from_sample_kernel(
    const __half* __restrict__ sample_val, const int32_t* __restrict__ sample_idx,
    int R, int K, int BUF, int NB,
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    int32_t* __restrict__ qprev, int32_t* __restrict__ refresh_lock,
    OutT* __restrict__ buf_val, int32_t* __restrict__ buf_idx, uint8_t* __restrict__ buf_bucket,
    __half* __restrict__ bucket_val, int32_t* __restrict__ bucket_idx, int bucket_cap) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float mn = INFINITY;
  float mx = -INFINITY;
  for (int i = tid; i < K; i += blockDim.x) {
    float v = __half2float(sample_val[(size_t)row * K + i]);
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  __shared__ float smin[256];
  __shared__ float smax[256];
  smin[tid] = mn;
  smax[tid] = mx;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) {
      smin[tid] = fminf(smin[tid], smin[tid + off]);
      smax[tid] = fmaxf(smax[tid], smax[tid + off]);
    }
    __syncthreads();
  }
  for (int i = tid; i < NB * BCOUNT_SHARDS; i += blockDim.x) {
    bcount[row * NB * BCOUNT_SHARDS + i] = 0;
  }
  __syncthreads();
  float o = smin[0];
  float span = fmaxf(smax[0] - o, 1.0e-20f);
  float inv = float(NB - 1) / span;
  if (tid == 0) {
    origin[row] = o;
    inv_delta[row] = inv;
    qcount[row] = K;
    if (qprev) qprev[row] = K < BUF ? K : BUF;
    if (refresh_lock) refresh_lock[row] = 0;
    th[row] = NB - 1;
  }
  for (int i = tid; i < K; i += blockDim.x) {
    __half hv = sample_val[(size_t)row * K + i];
    int id = sample_idx[(size_t)row * K + i];
#if HOPPER_SPARSE_DENSE_WRITE
    // buf_val is the [R, M] dense candidate tensor (BUF == M).  Scatter the
    // sample winner to its own corpus column so the tail (which skips the
    // sample prefix) and select see a consistent dense matrix.
    if (id >= 0 && id < BUF) {
      buf_val[(size_t)row * BUF + id] = hopper_sparse_value_cast<OutT>(__half2float(hv));
    }
#else
    if (i < BUF) {
      buf_val[(size_t)row * BUF + i] = hopper_sparse_value_cast<OutT>(__half2float(hv));
      buf_idx[(size_t)row * BUF + i] = id;
    }
#endif
    float v = __half2float(hv);
    int braw = int((v - o) * inv);
    int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
#if !HOPPER_SPARSE_DENSE_WRITE
    if (i < BUF && buf_bucket) {
      buf_bucket[(size_t)row * BUF + i] = (uint8_t)b;
    }
#endif
#if HOPPER_SPARSE_BUCKET_WRITE
    if (bucket_val) {
      sparse_bucket_append(row, b, hv, id, th, bcount, bucket_val, bucket_idx, bucket_cap, NB);
    }
#else
    int slot = atomicAdd(&bcount[(row * NB + b) * BCOUNT_SHARDS], 1);
    (void)slot;
    (void)bucket_val;
    (void)bucket_idx;
    (void)bucket_cap;
#endif
  }
}

#if 0  // RETIRED: dense path bucket-coords kernel
__global__ void hopper_dense_bucket_coords_kernel(
    const __half* __restrict__ dense_scores, int R, int M,
    float* __restrict__ origin, float* __restrict__ inv_delta, int NB) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float mn = INFINITY;
  float mx = -INFINITY;
  for (int i = tid; i < M; i += blockDim.x) {
    float v = __half2float(dense_scores[(size_t)row * M + i]);
    if (isfinite(v)) {
      mn = fminf(mn, v);
      mx = fmaxf(mx, v);
    }
  }
  __shared__ float smin[256];
  __shared__ float smax[256];
  smin[tid] = mn;
  smax[tid] = mx;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) {
      smin[tid] = fminf(smin[tid], smin[tid + off]);
      smax[tid] = fmaxf(smax[tid], smax[tid + off]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    float lo = smin[0];
    float hi = smax[0];
    if (!isfinite(lo) || !isfinite(hi)) {
      lo = 0.0f;
      hi = 1.0f;
    }
    float span = fmaxf(hi - lo, 1.0e-20f);
    origin[row] = lo;
    inv_delta[row] = float(NB - 1) / span;
  }
}

#endif  // dense_bucket_coords_kernel
__global__ void hopper_negate_kernel(__half* values, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    values[i] = __float2half(-__half2float(values[i]));
  }
}

__device__ __forceinline__ void sparse_bucket_append(
    int row, int bucket, __half value, int idx,
    int32_t* __restrict__ th, int32_t* __restrict__ bcount,
    __half* __restrict__ bucket_val, int32_t* __restrict__ bucket_idx,
    int bucket_cap, int NB) {
  int logical_slot = atomicAdd(&bcount[row * NB + bucket], 1);
  if (logical_slot >= bucket_cap) {
    atomicMin(&th[row], bucket);
  }
  int phys = bucket * bucket_cap + logical_slot;
  if (phys < NB * bucket_cap) {
    size_t off = (size_t)row * NB * bucket_cap + phys;
    bucket_val[off] = value;
    bucket_idx[off] = idx;
  }
}

template <int Base, class TensorC, class TensorCoord, typename OutT>
__device__ __forceinline__ void sparse_emit_group16(
    TensorC const& tCrC, TensorCoord const& tCcC, int row, int global_col_base,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    OutT* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    uint8_t* __restrict__ buf_bucket,
    __half* __restrict__ bucket_val, int32_t* __restrict__ bucket_idx, int bucket_cap,
    int BUF, int NB, int K) {
  (void)row;
  (void)K;
  // WGMMA m64n64 C-fragment layout: within one Base group of 16 accumulators,
  // a thread's elements cover exactly TWO rows, selected by (i & 2):
  //   (i&2)==0  -> row rA = get<0>(tCcC(Base+0))
  //   (i&2)!=0  -> row rB = get<0>(tCcC(Base+2)) == rA + 8
  // Aggregate each row's passing candidates in registers, then do ONE
  // qcount atomic per row + one contiguous write run, instead of a per-column
  // ballot + atomic + scattered write.  Rows are read from the real coords so
  // the value<->id<->row binding is exact.
  int rA = int(cute::get<0>(tCcC(Base + 0)));
  int rB = int(cute::get<0>(tCcC(Base + 2)));
  int gA = blockIdx.x * BM + rA;
  int gB = blockIdx.x * BM + rB;
  float oA = origin[gA], invA = inv_delta[gA];
  float oB = origin[gB], invB = inv_delta[gB];
  int gateA = th[gA], gateB = th[gB];
  OutT vA[8]; int cA[8]; uint8_t bA[8]; int nA = 0;
  OutT vB[8]; int cB[8]; uint8_t bB[8]; int nB = 0;
  CUTE_UNROLL
  for (int i = 0; i < 16; ++i) {
    int elem_col = int(cute::get<1>(tCcC(Base + i)));
    float vf = -float(tCrC(Base + i));
    bool isB = (i & 2) != 0;
    float o = isB ? oB : oA;
    float inv = isB ? invB : invA;
    int gate = isB ? gateB : gateA;
#if HOPPER_SPARSE_REFRESH_FROM_BUF == 1
    // RFB=1 rebuilds buckets during refresh, so the epilogue only needs an
    // online gate. int(x) <= gate for x>=0 is equivalent to x < gate+1, and
    // x<0 should pass bucket 0 as before.
    int pred = ((vf - o) * inv) < float(gate + 1);
#else
    int braw = int((vf - o) * inv);
    int valid = braw < NB;
    int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
    int pred = valid && (b <= gate);
#endif
#if !HOPPER_SPARSE_NO_HIST && HOPPER_SPARSE_REFRESH_FROM_BUF == 0
#if HOPPER_SPARSE_NOSTORE_HIST
    asm volatile("" ::"r"(pred), "r"(b), "r"(gate), "r"(isB ? gB : gA) : "memory");
#elif HOPPER_SPARSE_FAKE_HIST_STORE
    if (pred) {
      volatile int32_t* vbcount = bcount;
      int shard = (blockIdx.y * (THREADS / 32) + (threadIdx.x >> 5)) % BCOUNT_SHARDS;
      vbcount[((isB ? gB : gA) * NB + b) * BCOUNT_SHARDS + shard] = 1;
    }
#else
#if HOPPER_SPARSE_BUCKET_WRITE
    if (pred) {
      int brow = isB ? gB : gA;
      sparse_bucket_append(brow, b, __float2half(vf), global_col_base + elem_col,
                           th, bcount, bucket_val, bucket_idx, bucket_cap, NB);
    }
#else
#if HOPPER_SPARSE_MERGE_HIST
    unsigned pred_mask = __ballot_sync(0xffffffffu, pred != 0);
    if (pred) {
      int hist_row = isB ? rB : rA;
      int key = hist_row * NB + b;
      unsigned peers = __match_any_sync(pred_mask, key);
      int leader = __ffs(peers) - 1;
      if ((threadIdx.x & 31) == leader) {
        int shard = (blockIdx.y * (THREADS / 32) + (threadIdx.x >> 5)) % BCOUNT_SHARDS;
        atomicAdd(&bcount[((isB ? gB : gA) * NB + b) * BCOUNT_SHARDS + shard], __popc(peers));
      }
    }
#else
    if (pred) {
      int shard = (blockIdx.y * (THREADS / 32) + (threadIdx.x >> 5)) % BCOUNT_SHARDS;
      atomicAdd(&bcount[((isB ? gB : gA) * NB + b) * BCOUNT_SHARDS + shard], 1);
    }
#endif
#endif
#endif
#endif
#if !HOPPER_SPARSE_BUCKET_WRITE
    if (pred) {
#if HOPPER_SPARSE_DENSE_WRITE
      // Scatter directly into the pre-zeroed [R, M] dense candidate tensor.
      // buf_val is the dense tensor base; BUF carries M (row stride).  Column is
      // the corpus id, so select reads it back with buf_idx=nullptr.
      int brow = isB ? gB : gA;
      buf_val[(size_t)brow * BUF + (global_col_base + elem_col)] =
          hopper_sparse_value_cast<OutT>(vf);
#else
      if (isB) {
        vB[nB] = hopper_sparse_value_cast<OutT>(vf);
        cB[nB] = global_col_base + elem_col;
#if HOPPER_SPARSE_REFRESH_FROM_BUF == 2
        bB[nB] = (uint8_t)b;
#endif
        ++nB;
      } else {
        vA[nA] = hopper_sparse_value_cast<OutT>(vf);
        cA[nA] = global_col_base + elem_col;
#if HOPPER_SPARSE_REFRESH_FROM_BUF == 2
        bA[nA] = (uint8_t)b;
#endif
        ++nA;
      }
#endif
    }
#endif
  }
#if HOPPER_SPARSE_NO_WRITE
  (void)buf_val; (void)buf_idx; (void)buf_bucket; (void)qcount;
#else
#if HOPPER_SPARSE_DENSE_WRITE
  // All stores already happened in-loop; nothing to flush.
  (void)buf_idx; (void)buf_bucket; (void)qcount;
  (void)vA; (void)cA; (void)nA; (void)vB; (void)cB; (void)nB;
#elif HOPPER_SPARSE_BUCKET_WRITE
  (void)buf_val; (void)buf_idx; (void)buf_bucket; (void)qcount;
#else
  if (nA > 0) {
    int base = atomicAdd(&qcount[gA], nA);
    CUTE_UNROLL
    for (int j = 0; j < 8; ++j) {
      if (j >= nA) break;
      int w = base + j;
      if (w < BUF) {
        buf_val[(size_t)gA * BUF + w] = vA[j];
#if HOPPER_SPARSE_REFRESH_FROM_BUF == 2
        buf_bucket[(size_t)gA * BUF + w] = bA[j];
#endif
        buf_idx[(size_t)gA * BUF + w] = cA[j];
      }
    }
  }
  if (nB > 0) {
    int base = atomicAdd(&qcount[gB], nB);
    CUTE_UNROLL
    for (int j = 0; j < 8; ++j) {
      if (j >= nB) break;
      int w = base + j;
      if (w < BUF) {
        buf_val[(size_t)gB * BUF + w] = vB[j];
#if HOPPER_SPARSE_REFRESH_FROM_BUF == 2
        buf_bucket[(size_t)gB * BUF + w] = bB[j];
#endif
        buf_idx[(size_t)gB * BUF + w] = cB[j];
      }
    }
  }
#endif
#endif
}

namespace {

// A: row-major (N, D) viewed as 2D tile (box_cols=K_TILE wide, box_rows=BM tall).
void encode_a_map(CUtensorMap* map, const __half* ptr, int rows, int dim) {
  constexpr uint32_t rank = 2;
  uint64_t global_dim[rank] = {static_cast<uint64_t>(dim), static_cast<uint64_t>(rows)};
  uint64_t global_stride[rank - 1] = {static_cast<uint64_t>(dim * int(sizeof(__half)))};
  uint32_t box_dim[rank] = {static_cast<uint32_t>(K_TILE), static_cast<uint32_t>(BM)};
  uint32_t elem_stride[rank] = {1, 1};
  HOPPER_CUDRV(cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, const_cast<__half*>(ptr), global_dim,
      global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// B: row-major (M, D) viewed as 3D (K_inner=64, rows=BN, K_outer=8) so one TMA
// fetches a whole BN x D C tile.  The inner 64-wide box matches the 128B swizzle
// period => SMEM layout identical to 8 separate 64x64 loads.
void encode_b_map(CUtensorMap* map, const __half* ptr, int rows, int dim, int k_stages) {
  constexpr uint32_t rank = 3;
  constexpr int k_inner = K_TILE;
  uint64_t global_dim[rank] = {static_cast<uint64_t>(k_inner), static_cast<uint64_t>(rows),
                               static_cast<uint64_t>(k_stages)};
  uint64_t global_stride[rank - 1] = {static_cast<uint64_t>(dim * int(sizeof(__half))),
                                      static_cast<uint64_t>(k_inner * int(sizeof(__half)))};
  uint32_t box_dim[rank] = {static_cast<uint32_t>(k_inner), static_cast<uint32_t>(BN),
                            static_cast<uint32_t>(k_stages)};
  uint32_t elem_stride[rank] = {1, 1, 1};
  HOPPER_CUDRV(cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, const_cast<__half*>(ptr), global_dim,
      global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void encode_b32_map(CUtensorMap* map, const __half* ptr, int rows, int dim, int k_stages) {
  constexpr uint32_t rank = 3;
  constexpr int k_inner = K_TILE;
  constexpr int rows_per_tile = 32;
  uint64_t global_dim[rank] = {static_cast<uint64_t>(k_inner), static_cast<uint64_t>(rows),
                               static_cast<uint64_t>(k_stages)};
  uint64_t global_stride[rank - 1] = {static_cast<uint64_t>(dim * int(sizeof(__half))),
                                      static_cast<uint64_t>(k_inner * int(sizeof(__half)))};
  uint32_t box_dim[rank] = {static_cast<uint32_t>(k_inner), static_cast<uint32_t>(rows_per_tile),
                            static_cast<uint32_t>(k_stages)};
  uint32_t elem_stride[rank] = {1, 1, 1};
  HOPPER_CUDRV(cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, const_cast<__half*>(ptr), global_dim,
      global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void encode_bsmall_map(
    CUtensorMap* map, const __half* ptr, int rows, int dim, int k_stages, int rows_per_tile) {
  constexpr uint32_t rank = 3;
  constexpr int k_inner = K_TILE;
  uint64_t global_dim[rank] = {static_cast<uint64_t>(k_inner), static_cast<uint64_t>(rows),
                               static_cast<uint64_t>(k_stages)};
  uint64_t global_stride[rank - 1] = {static_cast<uint64_t>(dim * int(sizeof(__half))),
                                      static_cast<uint64_t>(k_inner * int(sizeof(__half)))};
  uint32_t box_dim[rank] = {static_cast<uint32_t>(k_inner), static_cast<uint32_t>(rows_per_tile),
                            static_cast<uint32_t>(k_stages)};
  uint32_t elem_stride[rank] = {1, 1, 1};
  HOPPER_CUDRV(cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, const_cast<__half*>(ptr), global_dim,
      global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// Tidal NHD page storage K view:
//   kv_data[page, 0(K), page_entry=0, kv_head, dim]
// For page_size=1, fixed kv_head rows are strided by 2 * Hkv * D elements.
// The 3D TMA box {K_TILE, BM, 1} writes the same BM x K_TILE swizzled tile
// expected by the existing WGMMA shared-memory layout.
void encode_tidal_k_nhd_map(
    CUtensorMap* map, const __half* ptr, int pages, int hkv, int dim,
    int rows_per_tile = BM) {
  constexpr uint32_t rank = 3;
  uint64_t global_dim[rank] = {
      static_cast<uint64_t>(dim), static_cast<uint64_t>(pages), static_cast<uint64_t>(hkv)};
  uint64_t global_stride[rank - 1] = {
      static_cast<uint64_t>(2 * hkv * dim * int(sizeof(__half))),
      static_cast<uint64_t>(dim * int(sizeof(__half)))};
  uint32_t box_dim[rank] = {
      static_cast<uint32_t>(K_TILE), static_cast<uint32_t>(rows_per_tile), 1u};
  uint32_t elem_stride[rank] = {1, 1, 1};
  HOPPER_CUDRV(cuTensorMapEncodeTiled(
      map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, const_cast<__half*>(ptr), global_dim,
      global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

bool g_dense_smem_attr_set = false;
bool g_sparse_smem_attr_set = false;
bool g_smalln_gmma_smem_attr_set = false;
bool g_smalln_sparse_smem_attr_set = false;
bool g_smalln_gmma_kchunk_attr_set = false;
bool g_smalln8_gmma_full_attr_set = false;
bool g_smalln8_sparse_full_attr_set = false;
bool g_smalln8_gmma_full_hist_attr_set = false;
bool g_smalln8_gmma_kchunk_attr_set = false;
bool g_smalln8_sparse_kchunk_attr_set = false;

// Allow the m64n8 full kernel to also serve D<=512 (k_stages<=8), not just the
// D in (512,768] range it was originally gated to. With QN=8 the resident query
// is tiny (~12KB); the full m64n8 SMEM at the 12-stage alloc is ~204KB < 227KB
// (H100 opt-in), and D<=512 needs even less, so residency is never the limit.
// Whether it actually beats the m64n32 fallback for low D is measured; the env
// HOPPER_SMALLN8_LOW_D (default on) lets us A/B it.
static inline bool smalln8_low_d_enabled() {
  static int v = -1;
  if (v < 0) {
    const char* e = std::getenv("HOPPER_SMALLN8_LOW_D");
    v = (e == nullptr || e[0] != '0') ? 1 : 0;
  }
  return v != 0;
}

// K-chunk smem byte budget shared by all m64n* kchunk kernels (query resident
// + base ring + barriers). BM==BN so the query/base K-tile sizes match.
constexpr int KCHUNK_Q_BYTES = KMAX_STAGES * BM * K_TILE * int(sizeof(__half));
constexpr int KCHUNK_B_BYTES = KCHUNK_RING * BN * K_TILE * int(sizeof(__half));
constexpr int KCHUNK_SMEM_BYTES =
    KCHUNK_Q_BYTES + KCHUNK_B_BYTES + (2 * KCHUNK_RING + 1) * int(sizeof(uint64_t));

// SMEM bytes for an m64n8 full kernel instantiated with KS resident K-stages.
// Thin alias of ft_m8_full_smem (defined near the kernels) so the launch sizing
// and the kernel's own layout stay in lock-step.
constexpr int m8_full_smem_bytes(int KS) { return ft_m8_full_smem(KS); }


// ---------------------------------------------------------------------------
// GQA multi-base m64n8 kernels.
// Base is stored as [Hkv, M, D] but encoded as flattened rows [Hkv*M, D].
// blockIdx.y is the KV-head id; query rows are padded to 8 per KV group.
// ---------------------------------------------------------------------------
template <int KS>
__global__ __launch_bounds__(KCHUNK_THREADS, ft_m8_min_blocks(KS))
void hopper_gqa_smalln_score_m64n8_full_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    const int32_t* __restrict__ page_indices,
    const __half* __restrict__ raw_kv_data,
    __half* __restrict__ dense_scores, int total_base_tiles, int M_stride, int base_head_stride,
    int k_stages, int logical_rows, int base_is_paged, int index_stride, int q_group_size) {
  constexpr int QN = 8;
  constexpr int M8_FULL_K_STAGES = KS;
  constexpr int QA_SLOT_ELEMS = QN * K_TILE;
  constexpr int QA_ELEMS = M8_FULL_K_STAGES * QA_SLOT_ELEMS;
  constexpr int QA_BYTES = QA_ELEMS * int(sizeof(__half));
  constexpr int BASE_TILE_ELEMS = M8_FULL_K_STAGES * A_STAGE_ELEMS;
  constexpr int BASE_BYTES = C_STAGE * BASE_TILE_ELEMS * int(sizeof(__half));
  constexpr int BARS = C_STAGE + C_STAGE + 1;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QA_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QA_BYTES + BASE_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_free = barriers + C_STAGE;
  uint64_t* bar_qrdy = barriers + 2 * C_STAGE;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int base_tile_group_start = blockIdx.x * C_TILES_PER_CTA;
  int grid_head = blockIdx.y;
  int kv_head = (base_is_paged == 2) ? (grid_head / q_group_size) : grid_head;
  int query_base = (base_is_paged == 2) ? (grid_head * QN) : (kv_head * QN);
  int base_row_offset = kv_head * base_head_stride;

  for (int ss = tid; ss < BARS; ss += blockDim.x) {
    mbarrier_init(&barriers[ss], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0 && base_is_paged == 2) {
    using StoreElement = cute::half_t;
    using StoreSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                    cute::GMMA::Major::K, StoreElement, cute::Int<BM>,
                                    cute::Int<K_TILE>>());
    using StoreSmemLayoutA = decltype(cute::tile_to_shape(
        StoreSmemAtomA{},
        cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{},
                         cute::Int<C_STAGE>{})));
    auto store_sA = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<StoreElement*>(smem_a)), StoreSmemLayoutA{});
    int raw_hkv = gridDim.y / q_group_size;
    int dim = k_stages * K_TILE;

    if (tid == 0) {
      mbarrier_expect_tx(&bar_qrdy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    }
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        for (int ks = 0; ks < k_stages; ++ks) {
          for (int e = tid; e < BM * (K_TILE / 8); e += 128) {
            int r = e / (K_TILE / 8);
            int kk = (e - r * (K_TILE / 8)) * 8;
            int logical_row = base_tile * BM + r;
            int page = __ldg(page_indices + grid_head * index_stride + logical_row);
            int feat = ks * K_TILE + kk;
            const __half* src = raw_kv_data + (((page * 2) * raw_hkv + kv_head) * dim + feat);
            cp_async_ca_shared_global_16(&store_sA(r, kk, ks, pre), src);
          }
        }
        cp_async_commit_group();
        cp_async_wait_group_0();
        named_barrier_sync_128<7>();
        if (tid == 0) {
          asm volatile("membar.cta;" ::: "memory");
          mbarrier_arrive(&barriers_a[pre]);
        }
      }
    }
    for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
      int refill_tile = base_tile_group_start + refill;
      if (refill_tile >= total_base_tiles) break;
      int refill_stage = refill % C_STAGE;
      int free_phase = (refill / C_STAGE - 1) & 1;
      if (tid == 0) {
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
      }
      named_barrier_sync_128<7>();
      for (int ks = 0; ks < k_stages; ++ks) {
        for (int e = tid; e < BM * (K_TILE / 8); e += 128) {
          int r = e / (K_TILE / 8);
          int kk = (e - r * (K_TILE / 8)) * 8;
          int logical_row = refill_tile * BM + r;
          int page = __ldg(page_indices + grid_head * index_stride + logical_row);
          int feat = ks * K_TILE + kk;
          const __half* src = raw_kv_data + (((page * 2) * raw_hkv + kv_head) * dim + feat);
          cp_async_ca_shared_global_16(&store_sA(r, kk, ks, refill_stage), src);
        }
      }
      cp_async_commit_group();
      cp_async_wait_group_0();
      named_barrier_sync_128<7>();
      if (tid == 0) {
        asm volatile("membar.cta;" ::: "memory");
        mbarrier_arrive(&barriers_a[refill_stage]);
      }
    }
    return;
  }

  if (warpgroup == 0 && tid == 0) {
    mbarrier_expect_tx(&bar_qrdy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
    tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        mbarrier_expect_tx(&barriers_a[pre], k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          if (base_is_paged) {
            tma_load_3d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[pre], ks * K_TILE, base_tile * BM, kv_head);
          } else {
            tma_load_2d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[pre], ks * K_TILE,
                        base_row_offset + base_tile * BM);
          }
        }
      }
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = base_tile_group_start + refill;
        if (refill_tile >= total_base_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_a[refill_stage],
                           k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          if (base_is_paged) {
            tma_load_3d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[refill_stage], ks * K_TILE,
                        refill_tile * BM, kv_head);
          } else {
            tma_load_2d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[refill_stage], ks * K_TILE,
                        base_row_offset + refill_tile * BM);
          }
        }
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<QN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<QN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{},
                       cute::Int<C_STAGE>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<QN>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  auto cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  auto cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  auto cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  auto cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  auto cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_cC = cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_tCcC = cute_thr_mma.partition_C(cute_cC);

  mbarrier_wait(&bar_qrdy[0], 0);
  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int base_tile = base_tile_group_start + c_rel;
    if (warpgroup == owner_wg && base_tile < total_base_tiles) {
      mbarrier_wait(&barriers_a[stage], phase);
      cute_emit_gemm_only_active_k12<KS>(
          cute_tiled_mma, cute_tCrA(cute::_, cute::_, cute::_, cute::_, stage),
          cute_tCrB, cute_tCrC);
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      CUTE_UNROLL
      for (int i = 0; i < cute::size(cute_tCrC); ++i) {
        int base_rel = int(cute::get<0>(cute_tCcC(i)));
        int q_rel = int(cute::get<1>(cute_tCcC(i)));
        int global_base = base_tile * BM + base_rel;
        int global_q = query_base + q_rel;
        if (q_rel < logical_rows && global_base < M_stride) {
          dense_scores[(size_t)global_q * M_stride + global_base] =
              __float2half(-float(cute_tCrC(i)));
        }
      }
    }
  }
}

template <int KS, bool USE_REG_ROW_QUEUE, int GROUP = 8>
__global__ __launch_bounds__(ft_m8_lb_threads(), ft_m8_lb_minblk(KS))
void hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel(
    const __grid_constant__ CUtensorMap base_map, const __grid_constant__ CUtensorMap query_map,
    const int32_t* __restrict__ page_indices,
    const __half* __restrict__ raw_kv_data,
    const float* __restrict__ origin, const float* __restrict__ inv_delta,
    int32_t* __restrict__ th, int32_t* __restrict__ qcount, int32_t* __restrict__ bcount,
    __half* __restrict__ buf_val, int32_t* __restrict__ buf_idx,
    int total_base_tiles, int start_base_tile, int M_stride, int base_head_stride,
    int BUF, int NB, int K, int k_stages, int logical_rows, int base_is_paged,
    int index_stride, int q_group_size) {
  constexpr int QN = 8;
  static_assert(GROUP >= 1 && GROUP <= QN, "GROUP must be in [1, 8]");
  constexpr int RQ_ROWS = GROUP;
  constexpr int M8_FULL_K_STAGES = KS;
  constexpr int QA_SLOT_ELEMS = QN * K_TILE;
  constexpr int QA_ELEMS = M8_FULL_K_STAGES * QA_SLOT_ELEMS;
  constexpr int QA_BYTES = QA_ELEMS * int(sizeof(__half));
  constexpr int BASE_TILE_ELEMS = M8_FULL_K_STAGES * A_STAGE_ELEMS;
  constexpr int BASE_BYTES = C_STAGE * BASE_TILE_ELEMS * int(sizeof(__half));
  constexpr int BARS = C_STAGE + C_STAGE + 1;

  extern __shared__ __align__(128) unsigned char smem[];
  __half* smem_q = reinterpret_cast<__half*>(smem);
  __half* smem_a = reinterpret_cast<__half*>(smem + QA_BYTES);
  uint64_t* barriers = reinterpret_cast<uint64_t*>(smem + QA_BYTES + BASE_BYTES);
  uint64_t* barriers_a = barriers;
  uint64_t* barriers_free = barriers + C_STAGE;
  uint64_t* bar_qrdy = barriers + 2 * C_STAGE;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int warpgroup = tid >> 7;
  int math_warp = warp - warpgroup * 4;
  int base_tile_group_start = start_base_tile + blockIdx.x * C_TILES_PER_CTA;
  int grid_head = blockIdx.y;
  int kv_head = (base_is_paged == 2) ? (grid_head / q_group_size) : grid_head;
  int query_base = (base_is_paged == 2) ? (grid_head * QN) : (kv_head * QN);
  int base_row_offset = kv_head * base_head_stride;
  // Opt-in register-row queue writeback.  It can reduce qcount atomics under
  // extremely large k, but direct atomicAdd is the default/eval path.
  constexpr bool m8_use_reg_row_queue =
      HOPPER_M8_REG_ROW_QUEUE && USE_REG_ROW_QUEUE;
#if HOPPER_M8_REG_ROW_QUEUE
  __half gqa_rq_val0, gqa_rq_val1, gqa_rq_val2, gqa_rq_val3;
  __half gqa_rq_val4, gqa_rq_val5, gqa_rq_val6, gqa_rq_val7;
  int32_t gqa_rq_idx0, gqa_rq_idx1, gqa_rq_idx2, gqa_rq_idx3;
  int32_t gqa_rq_idx4, gqa_rq_idx5, gqa_rq_idx6, gqa_rq_idx7;
  int gqa_rq_count0 = 0, gqa_rq_count1 = 0, gqa_rq_count2 = 0, gqa_rq_count3 = 0;
  int gqa_rq_count4 = 0, gqa_rq_count5 = 0, gqa_rq_count6 = 0, gqa_rq_count7 = 0;
#endif
  for (int ss = tid; ss < BARS; ss += blockDim.x) {
    mbarrier_init(&barriers[ss], 1);
  }
  fence_proxy_async_shared_cta();
  __syncthreads();

  if (warpgroup == 0 && base_is_paged == 2) {
    using StoreElement = cute::half_t;
    using StoreSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                    cute::GMMA::Major::K, StoreElement, cute::Int<BM>,
                                    cute::Int<K_TILE>>());
    using StoreSmemLayoutA = decltype(cute::tile_to_shape(
        StoreSmemAtomA{},
        cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{},
                         cute::Int<C_STAGE>{})));
    auto store_sA = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<StoreElement*>(smem_a)), StoreSmemLayoutA{});
    int raw_hkv = gridDim.y / q_group_size;
    int dim = k_stages * K_TILE;

    if (tid == 0) {
      mbarrier_expect_tx(&bar_qrdy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
      tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    }
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        for (int ks = 0; ks < k_stages; ++ks) {
          for (int e = tid; e < BM * (K_TILE / 8); e += 128) {
            int r = e / (K_TILE / 8);
            int kk = (e - r * (K_TILE / 8)) * 8;
            int logical_row = base_tile * BM + r;
            int page = __ldg(page_indices + grid_head * index_stride + logical_row);
            int feat = ks * K_TILE + kk;
            const __half* src = raw_kv_data + (((page * 2) * raw_hkv + kv_head) * dim + feat);
            cp_async_ca_shared_global_16(&store_sA(r, kk, ks, pre), src);
          }
        }
        cp_async_commit_group();
        cp_async_wait_group_0();
        named_barrier_sync_128<7>();
        if (tid == 0) {
          asm volatile("membar.cta;" ::: "memory");
          mbarrier_arrive(&barriers_a[pre]);
        }
      }
    }
    for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
      int refill_tile = base_tile_group_start + refill;
      if (refill_tile >= total_base_tiles) break;
      int refill_stage = refill % C_STAGE;
      int free_phase = (refill / C_STAGE - 1) & 1;
      if (tid == 0) {
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
      }
      named_barrier_sync_128<7>();
      for (int ks = 0; ks < k_stages; ++ks) {
        for (int e = tid; e < BM * (K_TILE / 8); e += 128) {
          int r = e / (K_TILE / 8);
          int kk = (e - r * (K_TILE / 8)) * 8;
          int logical_row = refill_tile * BM + r;
          int page = __ldg(page_indices + grid_head * index_stride + logical_row);
          int feat = ks * K_TILE + kk;
          const __half* src = raw_kv_data + (((page * 2) * raw_hkv + kv_head) * dim + feat);
          cp_async_ca_shared_global_16(&store_sA(r, kk, ks, refill_stage), src);
        }
      }
      cp_async_commit_group();
      cp_async_wait_group_0();
      named_barrier_sync_128<7>();
      if (tid == 0) {
        asm volatile("membar.cta;" ::: "memory");
        mbarrier_arrive(&barriers_a[refill_stage]);
      }
    }
    return;
  }

  if (blockIdx.x > 0 && THR_REFRESH_GROUP > 0 &&
      (blockIdx.x % THR_REFRESH_GROUP) == 0) {
    if (warpgroup == 0) {
      smalln_refresh_threshold_rows(query_base, logical_rows, tid, th, bcount, NB, K);
    }
  }

  if (warpgroup == 0 && tid == 0) {
    mbarrier_expect_tx(&bar_qrdy[0], k_stages * QA_SLOT_ELEMS * int(sizeof(__half)));
    tma_load_3d(smem_q, &query_map, &bar_qrdy[0], 0, query_base, 0);
    for (int pre = 0; pre < C_STAGE; ++pre) {
      int base_tile = base_tile_group_start + pre;
      if (base_tile < total_base_tiles) {
        mbarrier_expect_tx(&barriers_a[pre], k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          if (base_is_paged) {
            tma_load_3d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[pre], ks * K_TILE, base_tile * BM, kv_head);
          } else {
            tma_load_2d(smem_a + pre * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[pre], ks * K_TILE,
                        base_row_offset + base_tile * BM);
          }
        }
      }
    }
  }

  if (warpgroup == 0) {
    if (tid == 0) {
      for (int refill = C_STAGE; refill < C_TILES_PER_CTA; ++refill) {
        int refill_tile = base_tile_group_start + refill;
        if (refill_tile >= total_base_tiles) break;
        int refill_stage = refill % C_STAGE;
        int free_phase = (refill / C_STAGE - 1) & 1;
        mbarrier_wait(&barriers_free[refill_stage], free_phase);
        mbarrier_expect_tx(&barriers_a[refill_stage],
                           k_stages * A_STAGE_ELEMS * int(sizeof(__half)));
        for (int ks = 0; ks < k_stages; ++ks) {
          if (base_is_paged) {
            tma_load_3d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[refill_stage], ks * K_TILE,
                        refill_tile * BM, kv_head);
          } else {
            tma_load_2d(smem_a + refill_stage * BASE_TILE_ELEMS + ks * A_STAGE_ELEMS,
                        &base_map, &barriers_a[refill_stage], ks * K_TILE,
                        base_row_offset + refill_tile * BM);
          }
        }
      }
    }
    return;
  }

  using CuteElement = cute::half_t;
  using CuteTileShape = cute::Shape<cute::Int<BM>, cute::Int<QN>, cute::Int<K_TILE>>;
  using CuteAtomLayout = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;
  using CuteTiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<CuteElement, CuteElement, float, CuteTileShape>(),
      CuteAtomLayout{}));
  using CuteSmemAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<BM>,
                                 cute::Int<K_TILE>>());
  using CuteSmemAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                 cute::GMMA::Major::K, CuteElement, cute::Int<QN>,
                                 cute::Int<K_TILE>>());
  using CuteSmemLayoutA = decltype(cute::tile_to_shape(
      CuteSmemAtomA{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{},
                       cute::Int<C_STAGE>{})));
  using CuteSmemLayoutB = decltype(cute::tile_to_shape(
      CuteSmemAtomB{},
      cute::make_shape(cute::Int<QN>{}, cute::Int<K_TILE>{}, cute::Int<M8_FULL_K_STAGES>{})));

  CuteTiledMma cute_tiled_mma;
  auto cute_thr_mma = cute_tiled_mma.get_thread_slice(threadIdx.x % 128);
  auto cute_sA = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_a)), CuteSmemLayoutA{});
  auto cute_sB = cute::make_tensor(
      cute::make_smem_ptr(reinterpret_cast<CuteElement*>(smem_q)), CuteSmemLayoutB{});
  auto cute_tCrA = cute_thr_mma.partition_fragment_A(cute_sA);
  auto cute_tCrB = cute_thr_mma.partition_fragment_B(cute_sB);
  auto cute_tCrC =
      cute::partition_fragment_C(cute_tiled_mma, cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_cC = cute::make_identity_tensor(cute::Shape<cute::Int<BM>, cute::Int<QN>>{});
  auto cute_tCcC = cute_thr_mma.partition_C(cute_cC);


#if HOPPER_M8_REG_ROW_QUEUE
#define GQA_RQ_FLUSH(QR, VAL, IDX, CNT)                                      \
  do {                                                                       \
    if constexpr (m8_use_reg_row_queue) {                                    \
      if ((QR) < logical_rows) {                                             \
        unsigned gqa_rq_mask = __activemask();                               \
        int gqa_rq_cnt = (CNT);                                              \
        int gqa_rq_row = query_base + (QR);                                  \
        int gqa_rq_base = 0;                                                 \
        if (lane == 0 && gqa_rq_cnt > 0) {                                   \
          gqa_rq_base = atomicAdd(&qcount[gqa_rq_row], gqa_rq_cnt);          \
        }                                                                    \
        gqa_rq_base = __shfl_sync(gqa_rq_mask, gqa_rq_base, 0);              \
        int gqa_rq_write_cnt = gqa_rq_cnt;                                   \
        if (gqa_rq_base + gqa_rq_write_cnt > BUF) {                          \
          gqa_rq_write_cnt = BUF - gqa_rq_base;                              \
        }                                                                    \
        if (lane < gqa_rq_write_cnt) {                                       \
          int gqa_rq_pos = gqa_rq_base + lane;                               \
          if (gqa_rq_pos < BUF) {                                            \
            buf_val[(size_t)gqa_rq_row * BUF + gqa_rq_pos] = (VAL);          \
            buf_idx[(size_t)gqa_rq_row * BUF + gqa_rq_pos] = (IDX);          \
          }                                                                  \
        }                                                                    \
        (CNT) = 0;                                                           \
      }                                                                      \
    }                                                                        \
  } while (0)

#define GQA_RQ_APPEND_GROUP(QR, VAL, IDX, CNT, PEERS, PEER_N)                 \
  do {                                                                       \
    int gqa_rq_cur = (CNT);                                                  \
    if (gqa_rq_cur + (PEER_N) > 32) {                                        \
      GQA_RQ_FLUSH((QR), (VAL), (IDX), (CNT));                               \
      gqa_rq_cur = 0;                                                        \
    }                                                                        \
    int gqa_rq_target_rank = lane - gqa_rq_cur;                              \
    int gqa_rq_do_store =                                                    \
        (gqa_rq_target_rank >= 0 && gqa_rq_target_rank < (PEER_N));          \
    int gqa_rq_src_lane =                                                    \
        gqa_rq_do_store ? nth_set_bit_u32((PEERS), gqa_rq_target_rank) : 0;  \
    float gqa_rq_src_vf = __shfl_sync(active_mask, vf, gqa_rq_src_lane);     \
    int gqa_rq_src_idx =                                                     \
        __shfl_sync(active_mask, global_base, gqa_rq_src_lane);              \
    if (gqa_rq_do_store) {                                                   \
      (VAL) = __float2half(gqa_rq_src_vf);                                   \
      (IDX) = gqa_rq_src_idx;                                                \
    }                                                                        \
    (CNT) = gqa_rq_cur + (PEER_N);                                           \
  } while (0)

#define GQA_RQ_APPEND_FIXED_ROW_MASKED(QR, VAL, IDX, CNT, MASK)                \
  do {                                                                        \
    if ((QR) < logical_rows) {                                                \
      unsigned gqa_rq_peers = pred_mask & (MASK);                             \
      int gqa_rq_peer_n = __popc(gqa_rq_peers);                               \
      if (gqa_rq_peer_n != 0) {                                               \
        GQA_RQ_APPEND_GROUP((QR), (VAL), (IDX), (CNT), gqa_rq_peers,          \
                            gqa_rq_peer_n);                                   \
      }                                                                       \
    }                                                                         \
  } while (0)
#endif

  mbarrier_wait(&bar_qrdy[0], 0);
  for (int c_rel = 0; c_rel < C_TILES_PER_CTA; ++c_rel) {
    int owner_wg = 1 + (c_rel & 1);
    int stage = c_rel % C_STAGE;
    int phase = (c_rel / C_STAGE) & 1;
    int base_tile = base_tile_group_start + c_rel;
    if (warpgroup == owner_wg && base_tile < total_base_tiles) {
      mbarrier_wait(&barriers_a[stage], phase);
      cute_emit_gemm_only_active_k12<KS>(
          cute_tiled_mma, cute_tCrA(cute::_, cute::_, cute::_, cute::_, stage),
          cute_tCrB, cute_tCrC);
      if (math_warp == 0 && lane == 0) {
        mbarrier_arrive(&barriers_free[stage]);
      }
      cute::warpgroup_wait<0>();
      cute::warpgroup_fence_operand(cute_tCrC);
      CUTE_UNROLL
      for (int i = 0; i < cute::size(cute_tCrC); ++i) {
        int q_rel = int(cute::get<1>(cute_tCcC(i)));
        if constexpr (!m8_use_reg_row_queue) {
          if (q_rel >= logical_rows) continue;
        }
        int row = query_base + q_rel;
        int base_rel = int(cute::get<0>(cute_tCcC(i)));
        int global_base = base_tile * BM + base_rel;
        float vf = -float(cute_tCrC(i));
        int q_valid = q_rel < logical_rows;
        int b = 0;
        int pred = 0;
        if (q_valid) {
          int braw = int((vf - origin[row]) * inv_delta[row]);
          int valid = (global_base < M_stride) && (braw < NB);
          b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
          pred = valid && (b <= th[row]);
        }
        unsigned active_mask = __activemask();
        unsigned pred_mask = __ballot_sync(active_mask, pred != 0);
        if (pred_mask != 0) {
#if !HOPPER_SPARSE_NO_HIST && HOPPER_SPARSE_REFRESH_FROM_BUF == 0
          if (pred) {
            int shard = ((int(blockIdx.y) * 1315423911) ^ (blockIdx.x * (THREADS / 32)) ^ (threadIdx.x >> 5)) % BCOUNT_SHARDS;
            atomicAdd(&bcount[(row * NB + b) * BCOUNT_SHARDS + shard], 1);
          }
#endif
#if HOPPER_M8_REG_ROW_QUEUE
            if constexpr (m8_use_reg_row_queue) {

              if constexpr (GROUP <= 4) {
                if ((i & 1) == 0) {
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(0, gqa_rq_val0, gqa_rq_idx0, gqa_rq_count0, 0x11111111u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(2, gqa_rq_val2, gqa_rq_idx2, gqa_rq_count2, 0x22222222u);
                } else {
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(1, gqa_rq_val1, gqa_rq_idx1, gqa_rq_count1, 0x11111111u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(3, gqa_rq_val3, gqa_rq_idx3, gqa_rq_count3, 0x22222222u);
                }
              } else {
                if ((i & 1) == 0) {
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(0, gqa_rq_val0, gqa_rq_idx0, gqa_rq_count0, 0x11111111u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(2, gqa_rq_val2, gqa_rq_idx2, gqa_rq_count2, 0x22222222u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(4, gqa_rq_val4, gqa_rq_idx4, gqa_rq_count4, 0x44444444u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(6, gqa_rq_val6, gqa_rq_idx6, gqa_rq_count6, 0x88888888u);
                } else {
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(1, gqa_rq_val1, gqa_rq_idx1, gqa_rq_count1, 0x11111111u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(3, gqa_rq_val3, gqa_rq_idx3, gqa_rq_count3, 0x22222222u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(5, gqa_rq_val5, gqa_rq_idx5, gqa_rq_count5, 0x44444444u);
                  GQA_RQ_APPEND_FIXED_ROW_MASKED(7, gqa_rq_val7, gqa_rq_idx7, gqa_rq_count7, 0x88888888u);
                }
              }
            } else
#endif
            {
              if (pred) {
                unsigned peers = __match_any_sync(pred_mask, q_rel);
                int leader = __ffs(peers) - 1;
                int rank = __popc(peers & ((1u << lane) - 1));
                int base_pos = 0;
                if (lane == leader) {
                  base_pos = atomicAdd(&qcount[row], __popc(peers));
                }
                base_pos = __shfl_sync(peers, base_pos, leader);
                int pos = base_pos + rank;
                if (pos < BUF) {
                  buf_val[(size_t)row * BUF + pos] = __float2half(vf);
                  buf_idx[(size_t)row * BUF + pos] = global_base;
                }
              }
            }
        }
      }
    }
  }
#if HOPPER_M8_REG_ROW_QUEUE
  if constexpr (m8_use_reg_row_queue) {
    GQA_RQ_FLUSH(0, gqa_rq_val0, gqa_rq_idx0, gqa_rq_count0);
    GQA_RQ_FLUSH(1, gqa_rq_val1, gqa_rq_idx1, gqa_rq_count1);
    GQA_RQ_FLUSH(2, gqa_rq_val2, gqa_rq_idx2, gqa_rq_count2);
    GQA_RQ_FLUSH(3, gqa_rq_val3, gqa_rq_idx3, gqa_rq_count3);
    if constexpr (GROUP > 4) {
      GQA_RQ_FLUSH(4, gqa_rq_val4, gqa_rq_idx4, gqa_rq_count4);
      GQA_RQ_FLUSH(5, gqa_rq_val5, gqa_rq_idx5, gqa_rq_count5);
      GQA_RQ_FLUSH(6, gqa_rq_val6, gqa_rq_idx6, gqa_rq_count6);
      GQA_RQ_FLUSH(7, gqa_rq_val7, gqa_rq_idx7, gqa_rq_count7);
    }
  }
#undef GQA_RQ_APPEND_FIXED_ROW_MASKED
#undef GQA_RQ_APPEND_GROUP
#undef GQA_RQ_FLUSH
#endif
}

// Per-KS one-time max-dynamic-smem opt-in (one flag slot per KS in [1,12]).
static bool g_m8_full_attr[13] = {false};
// RETIRED: hist path attr (dense_bucket_fused only)
// static bool g_m8_hist_attr[13] = {false};
static bool g_m8_sparse_attr[13] = {false};

// Dispatch an m64n8 full *dense* score kernel by runtime k_stages (1..12).
#define FT_M8_FULL_DENSE_LAUNCH(KS)                                              \
  do {                                                                          \
    constexpr int _smem = m8_full_smem_bytes(KS);                               \
    if (!g_m8_full_attr[KS]) {                                                  \
      cudaFuncSetAttribute(hopper_smalln_score_m64n8_full_kernel<KS>,           \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, _smem); \
      g_m8_full_attr[KS] = true;                                                \
    }                                                                           \
    dim3 grid(base_tile_groups, query_tiles, 1);                                \
    hopper_smalln_score_m64n8_full_kernel<KS><<<grid, dim3(THREADS, 1, 1), _smem, stream>>>( \
        base_map, query_map, dense_scores, total_base_tiles, M, k_stages, logical_rows); \
  } while (0)

#if 0  // RETIRED: hist dispatch macro
// Dispatch the fused full-store + histogram kernel by runtime k_stages.
#define FT_M8_FULL_HIST_LAUNCH(KS)                                              \
  do {                                                                          \
    constexpr int _smem = m8_full_smem_bytes(KS);                               \
    if (!g_m8_hist_attr[KS]) {                                                  \
      cudaFuncSetAttribute(hopper_smalln_score_m64n8_full_hist_kernel<KS>,      \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, _smem); \
      g_m8_hist_attr[KS] = true;                                                \
    }                                                                           \
    dim3 grid(base_tile_groups, query_tiles, 1);                                \
    hopper_smalln_score_m64n8_full_hist_kernel<KS><<<grid, dim3(THREADS, 1, 1), _smem, stream>>>( \
        base_map, query_map, origin, inv_delta, gate, bcount, dense_scores,     \
        total_base_tiles, M, k_stages, logical_rows, NB, K, start_base_tile);   \
  } while (0)

#endif  // FT_M8_FULL_HIST_LAUNCH
// Dispatch the sparse (gather) full kernel by runtime k_stages.
#define FT_M8_FULL_SPARSE_LAUNCH(KS)                                            \
  do {                                                                          \
    constexpr int _smem = m8_full_smem_bytes(KS);                               \
    if (!g_m8_sparse_attr[KS]) {                                               \
      cudaFuncSetAttribute(hopper_smalln_score_to_sparse_m64n8_full_kernel<KS>, \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, _smem); \
      g_m8_sparse_attr[KS] = true;                                             \
    }                                                                           \
    dim3 grid(base_tile_groups, query_tiles, 1);                                \
    hopper_smalln_score_to_sparse_m64n8_full_kernel<KS><<<grid, dim3(THREADS, 1, 1), _smem, stream>>>( \
        base_map, query_map, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx, \
        total_base_tiles, start_base_tile, M, BUF, NB, K, k_stages, logical_rows); \
  } while (0)


static bool g_gqa_m8_full_attr[13] = {false};
static bool g_gqa_m8_sparse_attr[13] = {false};
static bool g_gqa_m8_sparse_regrow_attr[13][2] = {{false}};

#define FT_GQA_M8_FULL_DENSE_LAUNCH(KS)                                          \
  do {                                                                          \
    constexpr int _smem = m8_full_smem_bytes(KS);                               \
    if (!g_gqa_m8_full_attr[KS]) {                                              \
      cudaFuncSetAttribute(hopper_gqa_smalln_score_m64n8_full_kernel<KS>,        \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, _smem); \
      g_gqa_m8_full_attr[KS] = true;                                            \
    }                                                                           \
    dim3 grid(base_tile_groups, hkv, 1);                                         \
    hopper_gqa_smalln_score_m64n8_full_kernel<KS><<<grid, dim3(THREADS, 1, 1), _smem, stream>>>( \
        base_map, query_map, page_indices, raw_kv_data, dense_scores, total_base_tiles, M, base_head_stride, \
        k_stages, logical_rows, base_is_paged, index_stride, q_group_size);      \
  } while (0)

#if HOPPER_M8_REG_ROW_QUEUE
#define FT_GQA_M8_FULL_SPARSE_LAUNCH(KS)                                        \
  do {                                                                          \
    constexpr int _base_smem = m8_full_smem_bytes(KS);                          \
    dim3 grid(base_tile_groups, hkv, 1);                                         \
    if (logical_rows <= 4) {                                                    \
      if (!g_gqa_m8_sparse_regrow_attr[KS][0]) {                                \
        cudaFuncSetAttribute(hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel<KS, true, 4>, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, _base_smem); \
        g_gqa_m8_sparse_regrow_attr[KS][0] = true;                              \
      }                                                                         \
      hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel<KS, true, 4><<<grid, dim3(THREADS, 1, 1), _base_smem, stream>>>( \
          base_map, query_map, page_indices, raw_kv_data, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx, \
          total_base_tiles, start_base_tile, M, base_head_stride, BUF, NB, K, k_stages, logical_rows, \
          base_is_paged, index_stride, q_group_size);                            \
    } else {                                                                    \
      if (!g_gqa_m8_sparse_regrow_attr[KS][1]) {                                \
        cudaFuncSetAttribute(hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel<KS, true, 8>, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, _base_smem); \
        g_gqa_m8_sparse_regrow_attr[KS][1] = true;                              \
      }                                                                         \
      hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel<KS, true, 8><<<grid, dim3(THREADS, 1, 1), _base_smem, stream>>>( \
          base_map, query_map, page_indices, raw_kv_data, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx, \
          total_base_tiles, start_base_tile, M, base_head_stride, BUF, NB, K, k_stages, logical_rows, \
          base_is_paged, index_stride, q_group_size);                            \
    }                                                                           \
  } while (0)
#else
#define FT_GQA_M8_FULL_SPARSE_LAUNCH(KS)                                        \
  do {                                                                          \
    constexpr int _base_smem = m8_full_smem_bytes(KS);                          \
    dim3 grid(base_tile_groups, hkv, 1);                                         \
    if (!g_gqa_m8_sparse_attr[KS]) {                                            \
      cudaFuncSetAttribute(hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel<KS, false>, \
                           cudaFuncAttributeMaxDynamicSharedMemorySize, _base_smem); \
      g_gqa_m8_sparse_attr[KS] = true;                                          \
    }                                                                           \
    hopper_gqa_smalln_score_to_sparse_m64n8_full_kernel<KS, false><<<grid, dim3(THREADS, 1, 1), _base_smem, stream>>>( \
        base_map, query_map, page_indices, raw_kv_data, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx, \
        total_base_tiles, start_base_tile, M, base_head_stride, BUF, NB, K, k_stages, logical_rows, \
        base_is_paged, index_stride, q_group_size);                              \
  } while (0)
#endif

#define FT_M8_DISPATCH_KS(LAUNCH)                                               \
  switch (k_stages) {                                                           \
    case 1: LAUNCH(1); break;   case 2: LAUNCH(2); break;                       \
    case 3: LAUNCH(3); break;   case 4: LAUNCH(4); break;                       \
    case 5: LAUNCH(5); break;   case 6: LAUNCH(6); break;                       \
    case 7: LAUNCH(7); break;   case 8: LAUNCH(8); break;                       \
    case 9: LAUNCH(9); break;   case 10: LAUNCH(10); break;                     \
    case 11: LAUNCH(11); break; default: LAUNCH(12); break;                     \
  }



__global__ void hopper_refresh_threshold_from_bcount_kernel(
    int32_t* __restrict__ th, const int32_t* __restrict__ bcount,
    int R, int NB, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= R) return;
  int old_th = th[row];
  int cum = 0;
  int new_th = old_th;
  bool found = false;
  for (int b = 0; b < NB; ++b) {
    int bc = 0;
    for (int s = 0; s < BCOUNT_SHARDS; ++s) {
      bc += bcount[((row * NB + b) * BCOUNT_SHARDS) + s];
    }
    cum += bc;
    if (cum >= K) {
      new_th = b;
      found = true;
      break;
    }
  }
  if (found && new_th < old_th) th[row] = new_th;
}

}  // namespace

static void launch_hopper_score_to_dense_ip_fp16_impl(
    const __half* x, const __half* base, int R, int M, int dim,
    __half* dense_scores, cudaStream_t stream, int c_tiles_per_cta) {
  int k_stages = dim / K_TILE;
  int total_c_tiles = M / BN;
  int m_tiles = R / BM;
  int c_tile_groups = (total_c_tiles + c_tiles_per_cta - 1) / c_tiles_per_cta;

  CUtensorMap a_map{};
  CUtensorMap b_map{};
  encode_a_map(&a_map, x, R, dim);

  dim3 grid(m_tiles, c_tile_groups, 1);
  dim3 block(THREADS, 1, 1);

  // D > 512 (k_stages > 8) exceeds full-D smem residence; use the K-chunk path.
  // Base is loaded per K-tile (2D), so encode it like the A map (2D, dim wide).
  if (k_stages > K_STAGES || KCHUNK_FORCE) {
    CUtensorMap b2d_map{};
    encode_a_map(&b2d_map, base, M, dim);
    static bool s_dense_kchunk_attr = false;
    if (!s_dense_kchunk_attr) {
      cudaFuncSetAttribute(hopper_score_to_dense_ip_kchunk_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KCHUNK_SMEM_BYTES);
      s_dense_kchunk_attr = true;
    }
    hopper_score_to_dense_ip_kchunk_kernel<<<grid, dim3(KCHUNK_THREADS, 1, 1), KCHUNK_SMEM_BYTES, stream>>>(
        a_map, b2d_map, dense_scores, total_c_tiles, M, k_stages, c_tiles_per_cta);
    return;
  }

  encode_b_map(&b_map, base, M, dim, k_stages);
  if (!g_dense_smem_attr_set) {
    cudaFuncSetAttribute(hopper_score_to_dense_ip_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    g_dense_smem_attr_set = true;
  }

  hopper_score_to_dense_ip_kernel<<<grid, block, SMEM_BYTES, stream>>>(
      a_map, b_map, dense_scores, total_c_tiles, M, k_stages, c_tiles_per_cta);
}

#if 0  // RETIRED: dense launcher (fused_ip_dense only)
void launch_hopper_score_to_dense_ip_fp16(const __half* x, const __half* base, int R, int M,
                                          int dim,
                                          __half* dense_scores, cudaStream_t stream) {
  launch_hopper_score_to_dense_ip_fp16_impl(
      x, base, R, M, dim, dense_scores, stream, C_TILES_PER_CTA);
}

#endif  // launch_hopper_score_to_dense_ip_fp16
void launch_hopper_sample_score_to_dense_ip_fp16(
    const __half* x, const __half* base, int R, int M, int dim,
    __half* dense_scores, cudaStream_t stream) {
  launch_hopper_score_to_dense_ip_fp16_impl(
      x, base, R, M, dim, dense_scores, stream, SAMPLE_C_TILES_PER_CTA);
}


void launch_hopper_gqa_smalln_score_dense_ip_gmma_m64n8_fp16(
    const __half* x, const __half* base3d, int hkv, int npad_per_group, int logical_rows,
    int M, int base_head_stride, int dim, __half* dense_scores, cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int qn8 = 8;
  constexpr int m8_full_k_stages = 12;
  if (!(npad_per_group == qn8 && logical_rows >= 1 && logical_rows <= qn8 &&
        k_stages <= m8_full_k_stages)) return;
  int total_base_tiles = (M + BM - 1) / BM;
  int base_tile_groups = (total_base_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_a_map(&base_map, base3d, hkv * base_head_stride, dim);
  encode_bsmall_map(&query_map, x, hkv * npad_per_group, dim, k_stages, qn8);
  int base_is_paged = 0;
  const int32_t* page_indices = nullptr;
  const __half* raw_kv_data = nullptr;
  int index_stride = 0;
  int q_group_size = 1;
  FT_M8_DISPATCH_KS(FT_GQA_M8_FULL_DENSE_LAUNCH);
}

void launch_hopper_gqa_smalln_score_to_sparse_ip_gmma_m64n8_fp16(
    const __half* x, const __half* base3d, int hkv, int npad_per_group, int logical_rows,
    int M, int base_head_stride, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int qn8 = 8;
  constexpr int m8_full_k_stages = 12;
  if (!(npad_per_group == qn8 && logical_rows >= 1 && logical_rows <= qn8 &&
        k_stages <= m8_full_k_stages)) return;
  int total_base_tiles = (M + BM - 1) / BM;
  int start_base_tile = start_col / BM;
  int tail_tiles = total_base_tiles - start_base_tile;
  if (tail_tiles <= 0) return;
  int base_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_a_map(&base_map, base3d, hkv * base_head_stride, dim);
  encode_bsmall_map(&query_map, x, hkv * npad_per_group, dim, k_stages, qn8);
  int base_is_paged = 0;
  const int32_t* page_indices = nullptr;
  const __half* raw_kv_data = nullptr;
  int index_stride = 0;
  int q_group_size = 1;
  FT_M8_DISPATCH_KS(FT_GQA_M8_FULL_SPARSE_LAUNCH);
}

void launch_hopper_gqa_paged_score_dense_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, int hkv, int npad_per_group, int logical_rows,
    int M, int dim, __half* dense_scores, cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int qn8 = 8;
  constexpr int m8_full_k_stages = 12;
  if (!(npad_per_group == qn8 && logical_rows >= 1 && logical_rows <= qn8 &&
        k_stages <= m8_full_k_stages)) return;
  int total_base_tiles = (M + BM - 1) / BM;
  int base_tile_groups = (total_base_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int base_head_stride = 0;
  int base_is_paged = 1;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_tidal_k_nhd_map(&base_map, kv_data, M, hkv, dim);
  encode_bsmall_map(&query_map, x, hkv * npad_per_group, dim, k_stages, qn8);
  const int32_t* page_indices = nullptr;
  const __half* raw_kv_data = nullptr;
  int index_stride = 0;
  int q_group_size = 1;
  FT_M8_DISPATCH_KS(FT_GQA_M8_FULL_DENSE_LAUNCH);
}

void launch_hopper_gqa_paged_score_to_sparse_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, int hkv, int npad_per_group, int logical_rows,
    int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int qn8 = 8;
  constexpr int m8_full_k_stages = 12;
  if (!(npad_per_group == qn8 && logical_rows >= 1 && logical_rows <= qn8 &&
        k_stages <= m8_full_k_stages)) return;
  int total_base_tiles = (M + BM - 1) / BM;
  int start_base_tile = start_col / BM;
  int tail_tiles = total_base_tiles - start_base_tile;
  if (tail_tiles <= 0) return;
  int base_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int base_head_stride = 0;
  int base_is_paged = 1;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_tidal_k_nhd_map(&base_map, kv_data, M, hkv, dim);
  encode_bsmall_map(&query_map, x, hkv * npad_per_group, dim, k_stages, qn8);
  const int32_t* page_indices = nullptr;
  const __half* raw_kv_data = nullptr;
  int index_stride = 0;
  int q_group_size = 1;
  FT_M8_DISPATCH_KS(FT_GQA_M8_FULL_SPARSE_LAUNCH);
}

void launch_hopper_gqa_group_indexed_score_dense_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, const int32_t* group_indices,
    int hkv, int logical_rows, int index_stride, int M, int physical_pages, int dim,
    __half* dense_scores, cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int qn8 = 8;
  constexpr int m8_full_k_stages = 12;
  if (!(logical_rows >= 1 && logical_rows <= qn8 && k_stages <= m8_full_k_stages)) return;
  int total_base_tiles = (M + BM - 1) / BM;
  int base_tile_groups = (total_base_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int base_head_stride = 0;
  int base_is_paged = 2;
  int q_group_size = 1;
  const int32_t* page_indices = group_indices;
  const __half* raw_kv_data = kv_data;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_tidal_k_nhd_map(&base_map, kv_data, physical_pages, hkv, dim, /*rows_per_tile=*/1);
  encode_bsmall_map(&query_map, x, hkv * qn8, dim, k_stages, qn8);
  FT_M8_DISPATCH_KS(FT_GQA_M8_FULL_DENSE_LAUNCH);
}

void launch_hopper_gqa_group_indexed_score_to_sparse_ip_gmma_m64n8_fp16(
    const __half* x, const __half* kv_data, const int32_t* group_indices,
    int hkv, int logical_rows, int index_stride, int M, int physical_pages, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int qn8 = 8;
  constexpr int m8_full_k_stages = 12;
  if (!(logical_rows >= 1 && logical_rows <= qn8 && k_stages <= m8_full_k_stages)) return;
  int total_base_tiles = (M + BM - 1) / BM;
  int start_base_tile = start_col / BM;
  int tail_tiles = total_base_tiles - start_base_tile;
  if (tail_tiles <= 0) return;
  int base_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int base_head_stride = 0;
  int base_is_paged = 2;
  int q_group_size = 1;
  const int32_t* page_indices = group_indices;
  const __half* raw_kv_data = kv_data;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_tidal_k_nhd_map(&base_map, kv_data, physical_pages, hkv, dim, /*rows_per_tile=*/1);
  encode_bsmall_map(&query_map, x, hkv * qn8, dim, k_stages, qn8);
  FT_M8_DISPATCH_KS(FT_GQA_M8_FULL_SPARSE_LAUNCH);
}

void launch_hopper_smalln_score_dense_ip_gmma_m64n32_fp16(
    const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim,
    __half* dense_scores, cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int m8_full_k_stages = 12;
  // m64n8 serves D in (512,768] always; when HOPPER_SMALLN8_LOW_D is on it also
  // takes D<=512 (k_stages<=8) via the same full kernel (its GEMM loop and SMEM
  // alloc cover k_stages 1..12 uniformly).
  bool use_m8 = Npad <= 8 && k_stages <= m8_full_k_stages &&
                (k_stages > K_STAGES || smalln8_low_d_enabled());
  if (use_m8) {
    constexpr int qn8 = 8;
    int total_base_tiles = (M + BM - 1) / BM;
    int base_tile_groups = (total_base_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
    int query_tiles = (Npad + qn8 - 1) / qn8;
    CUtensorMap base_map{};
    CUtensorMap query_map{};
    encode_a_map(&base_map, base, M, dim);
    encode_bsmall_map(&query_map, x, Npad, dim, k_stages, qn8);
    if (k_stages <= m8_full_k_stages) {
      // Per-KS instantiation: SMEM tracks the actual D, not the 12-stage max.
      FT_M8_DISPATCH_KS(FT_M8_FULL_DENSE_LAUNCH);
      return;
    }
    constexpr int qb_bytes = KMAX_STAGES * qn8 * K_TILE * int(sizeof(__half));
    constexpr int ab_bytes = KCHUNK_RING * BM * K_TILE * int(sizeof(__half));
    constexpr int kc_smem_bytes =
        qb_bytes + ab_bytes + (2 * KCHUNK_RING + 1) * int(sizeof(uint64_t));
    if (!g_smalln8_gmma_kchunk_attr_set) {
      cudaFuncSetAttribute(hopper_smalln_score_m64n8_kchunk_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kc_smem_bytes);
      g_smalln8_gmma_kchunk_attr_set = true;
    }
    dim3 grid(base_tile_groups, query_tiles, 1);
    hopper_smalln_score_m64n8_kchunk_kernel<<<grid, dim3(KCHUNK_THREADS, 1, 1), kc_smem_bytes, stream>>>(
        base_map, query_map, dense_scores, total_base_tiles, M, k_stages, logical_rows);
    return;
  }
  constexpr int qn = 32;

  int total_base_tiles = (M + BM - 1) / BM;
  int base_tile_groups = (total_base_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int query_tiles = (Npad + qn - 1) / qn;

  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_a_map(&base_map, base, M, dim);
  encode_b32_map(&query_map, x, Npad, dim, k_stages);

  dim3 grid(base_tile_groups, query_tiles, 1);
  dim3 block(THREADS, 1, 1);

  // D > 512 (k_stages > 8) exceeds full-D smem residence; use the K-chunk path.
  if (k_stages > K_STAGES) {
    constexpr int qb_bytes = KMAX_STAGES * qn * K_TILE * int(sizeof(__half));
    constexpr int ab_bytes = KCHUNK_RING * BM * K_TILE * int(sizeof(__half));
    constexpr int kc_smem_bytes =
        qb_bytes + ab_bytes + (2 * KCHUNK_RING + 1) * int(sizeof(uint64_t));
    if (!g_smalln_gmma_kchunk_attr_set) {
      cudaFuncSetAttribute(hopper_smalln_score_m64n32_kchunk_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kc_smem_bytes);
      g_smalln_gmma_kchunk_attr_set = true;
    }
    hopper_smalln_score_m64n32_kchunk_kernel<<<grid, dim3(KCHUNK_THREADS, 1, 1), kc_smem_bytes, stream>>>(
        base_map, query_map, dense_scores, total_base_tiles, M, k_stages);
    return;
  }

  constexpr int qb_k_stage_elems = qn * K_TILE;
  constexpr int qb_stage_elems = K_STAGES * qb_k_stage_elems;
  constexpr int qb_stage_bytes = qb_stage_elems * int(sizeof(__half));
  constexpr int qb_bytes = C_STAGE * qb_stage_bytes;
  constexpr int base_tile_elems = K_STAGES * A_STAGE_ELEMS;
  constexpr int base_tile_bytes = base_tile_elems * int(sizeof(__half));
  constexpr int base_bytes = C_STAGE * base_tile_bytes;
  constexpr int smem_bytes =
      base_bytes + qb_bytes + (A_BARRIERS + B_BARRIERS + FREE_BARRIERS) * int(sizeof(uint64_t));

  if (!g_smalln_gmma_smem_attr_set) {
    cudaFuncSetAttribute(hopper_smalln_score_m64n32_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    g_smalln_gmma_smem_attr_set = true;
  }

  hopper_smalln_score_m64n32_kernel<<<grid, block, smem_bytes, stream>>>(
      base_map, query_map, dense_scores, total_base_tiles, M, k_stages);
}

#if 0  // RETIRED: hist launcher
// Fused full-store + in-GEMM bucket-histogram launcher (m64n8, D in (512,768],
// k<=128). Writes the [logical_rows, M] negated-IP dense matrix AND atomic-adds
// the gated bucket histogram into bcount over base tiles [start_col/BM, end).
void launch_hopper_smalln_score_dense_hist_ip_gmma_m64n8_fp16(
    const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim, int NB, int K,
    const float* origin, const float* inv_delta, int32_t* gate, int32_t* bcount,
    __half* dense_scores, cudaStream_t stream, int start_col) {
  int k_stages = dim / K_TILE;
  constexpr int m8_full_k_stages = 12;
  bool ok = Npad <= 8 && k_stages <= m8_full_k_stages &&
            (k_stages > K_STAGES || smalln8_low_d_enabled());
  if (!ok) return;
  constexpr int qn8 = 8;
  int total_base_tiles = (M + BM - 1) / BM;
  int start_base_tile = start_col / BM;
  if (start_base_tile < 0) start_base_tile = 0;
  if (start_base_tile > total_base_tiles) start_base_tile = total_base_tiles;
  int tail_tiles = total_base_tiles - start_base_tile;
  if (tail_tiles <= 0) return;
  int base_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int query_tiles = (Npad + qn8 - 1) / qn8;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_a_map(&base_map, base, M, dim);
  encode_bsmall_map(&query_map, x, Npad, dim, k_stages, qn8);
  // Per-KS instantiation: SMEM tracks the actual D, not the 12-stage max.
  FT_M8_DISPATCH_KS(FT_M8_FULL_HIST_LAUNCH);
}

#endif  // launch_hopper_smalln_score_dense_hist_ip_gmma_m64n8_fp16
void launch_hopper_smalln_score_to_sparse_ip_gmma_m64n32_fp16(
    const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    __half* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  constexpr int m8_full_k_stages = 12;
  bool use_m8 = Npad <= 8 && k_stages <= m8_full_k_stages &&
                (k_stages > K_STAGES || smalln8_low_d_enabled());
  if (use_m8) {
    constexpr int qn8 = 8;
    int total_base_tiles = (M + BM - 1) / BM;
    int start_base_tile = start_col / BM;
    int tail_tiles = total_base_tiles - start_base_tile;
    if (tail_tiles <= 0) return;
    int base_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
    int query_tiles = (Npad + qn8 - 1) / qn8;
    CUtensorMap base_map{};
    CUtensorMap query_map{};
    encode_a_map(&base_map, base, M, dim);
    encode_bsmall_map(&query_map, x, Npad, dim, k_stages, qn8);
    if (k_stages <= m8_full_k_stages) {
      // Per-KS instantiation: SMEM tracks the actual D, not the 12-stage max.
      FT_M8_DISPATCH_KS(FT_M8_FULL_SPARSE_LAUNCH);
      return;
    }
    constexpr int qb_bytes = KMAX_STAGES * qn8 * K_TILE * int(sizeof(__half));
    constexpr int ab_bytes = KCHUNK_RING * BM * K_TILE * int(sizeof(__half));
    constexpr int kc_smem_bytes =
        qb_bytes + ab_bytes + (2 * KCHUNK_RING + 1) * int(sizeof(uint64_t));
    if (!g_smalln8_sparse_kchunk_attr_set) {
      cudaFuncSetAttribute(hopper_smalln_score_to_sparse_m64n8_kchunk_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kc_smem_bytes);
      g_smalln8_sparse_kchunk_attr_set = true;
    }
    dim3 grid(base_tile_groups, query_tiles, 1);
    hopper_smalln_score_to_sparse_m64n8_kchunk_kernel<<<grid, dim3(KCHUNK_THREADS, 1, 1), kc_smem_bytes, stream>>>(
        base_map, query_map, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
        total_base_tiles, start_base_tile, M, BUF, NB, K, k_stages, logical_rows);
    return;
  }
  constexpr int qn = 32;

  int total_base_tiles = (M + BM - 1) / BM;
  int start_base_tile = start_col / BM;
  int tail_tiles = total_base_tiles - start_base_tile;
  if (tail_tiles <= 0) return;
  int base_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int query_tiles = (Npad + qn - 1) / qn;

  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_a_map(&base_map, base, M, dim);
  encode_b32_map(&query_map, x, Npad, dim, k_stages);

  dim3 grid(base_tile_groups, query_tiles, 1);
  dim3 block(THREADS, 1, 1);

  // D > 512 (k_stages > 8) exceeds full-D smem residence; use the K-chunk path.
  if (k_stages > K_STAGES) {
    constexpr int qb_bytes = KMAX_STAGES * qn * K_TILE * int(sizeof(__half));
    constexpr int ab_bytes = KCHUNK_RING * BM * K_TILE * int(sizeof(__half));
    constexpr int kc_smem_bytes =
        qb_bytes + ab_bytes + (2 * KCHUNK_RING + 1) * int(sizeof(uint64_t));
    static bool s_kc_sparse_attr = false;
    if (!s_kc_sparse_attr) {
      cudaFuncSetAttribute(hopper_smalln_score_to_sparse_m64n32_kchunk_kernel<__half>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kc_smem_bytes);
      s_kc_sparse_attr = true;
    }
    hopper_smalln_score_to_sparse_m64n32_kchunk_kernel<__half><<<grid, dim3(KCHUNK_THREADS, 1, 1), kc_smem_bytes, stream>>>(
        base_map, query_map, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
        total_base_tiles, start_base_tile, M, BUF, NB, K, k_stages);
    return;
  }

  constexpr int qb_k_stage_elems = qn * K_TILE;
  constexpr int qb_stage_elems = K_STAGES * qb_k_stage_elems;
  constexpr int qb_stage_bytes = qb_stage_elems * int(sizeof(__half));
  constexpr int qb_bytes = C_STAGE * qb_stage_bytes;
  constexpr int base_tile_elems = K_STAGES * A_STAGE_ELEMS;
  constexpr int base_tile_bytes = base_tile_elems * int(sizeof(__half));
  constexpr int base_bytes = C_STAGE * base_tile_bytes;
  constexpr int smem_bytes =
      base_bytes + qb_bytes + (A_BARRIERS + B_BARRIERS + FREE_BARRIERS) * int(sizeof(uint64_t));

  if (!g_smalln_sparse_smem_attr_set) {
    cudaFuncSetAttribute(hopper_smalln_score_to_sparse_m64n32_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    g_smalln_sparse_smem_attr_set = true;
  }

  hopper_smalln_score_to_sparse_m64n32_kernel<<<grid, block, smem_bytes, stream>>>(
      base_map, query_map, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
      total_base_tiles, start_base_tile, M, BUF, NB, K, k_stages);
}

void launch_hopper_smalln_score_to_sparse_ip_gmma_m64n32_fp32(
    const __half* x, const __half* base, int Npad, int logical_rows, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    float* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream) {
  (void)logical_rows;
  int k_stages = dim / K_TILE;
  if (!(k_stages > K_STAGES && Npad > 8)) return;
  constexpr int qn = 32;
  int total_base_tiles = (M + BM - 1) / BM;
  int start_base_tile = start_col / BM;
  int tail_tiles = total_base_tiles - start_base_tile;
  if (tail_tiles <= 0) return;
  int base_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;
  int query_tiles = (Npad + qn - 1) / qn;
  CUtensorMap base_map{};
  CUtensorMap query_map{};
  encode_a_map(&base_map, base, M, dim);
  encode_b32_map(&query_map, x, Npad, dim, k_stages);
  constexpr int qb_bytes = KMAX_STAGES * qn * K_TILE * int(sizeof(__half));
  constexpr int ab_bytes = KCHUNK_RING * BM * K_TILE * int(sizeof(__half));
  constexpr int kc_smem_bytes =
      qb_bytes + ab_bytes + (2 * KCHUNK_RING + 1) * int(sizeof(uint64_t));
  static bool s_kc_sparse_fp32_attr = false;
  if (!s_kc_sparse_fp32_attr) {
    cudaFuncSetAttribute(hopper_smalln_score_to_sparse_m64n32_kchunk_kernel<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, kc_smem_bytes);
    s_kc_sparse_fp32_attr = true;
  }
  dim3 grid(base_tile_groups, query_tiles, 1);
  hopper_smalln_score_to_sparse_m64n32_kchunk_kernel<float><<<grid, dim3(KCHUNK_THREADS, 1, 1), kc_smem_bytes, stream>>>(
      base_map, query_map, origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
      total_base_tiles, start_base_tile, M, BUF, NB, K, k_stages);
}


void launch_hopper_refresh_threshold_from_bcount(
    int32_t* th, const int32_t* bcount, int R, int NB, int K, cudaStream_t stream) {
  if (R <= 0 || NB <= 0 || K <= 0) return;
  int block = 128;
  int grid = (R + block - 1) / block;
  hopper_refresh_threshold_from_bcount_kernel<<<grid, block, 0, stream>>>(th, bcount, R, NB, K);
}

__global__ void hopper_add_sample_idx_offset_kernel(
    int32_t* __restrict__ sample_idx, int total_items, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total_items) {
    sample_idx[i] += offset;
  }
}

void launch_hopper_add_sample_idx_offset(
    int32_t* sample_idx, int total_items, int offset, cudaStream_t stream) {
  if (total_items <= 0 || offset == 0) return;
  int block = 256;
  int grid = (total_items + block - 1) / block;
  hopper_add_sample_idx_offset_kernel<<<grid, block, 0, stream>>>(
      sample_idx, total_items, offset);
}

void launch_hopper_seed_from_sample_fp16(
    const __half* sample_val, const int32_t* sample_idx,
    int R, int K, int BUF, int NB,
    float* origin, float* inv_delta, int32_t* th, int32_t* qcount, int32_t* bcount,
    int32_t* qprev, int32_t* refresh_lock,
    __half* buf_val, int32_t* buf_idx, uint8_t* buf_bucket,
    __half* bucket_val, int32_t* bucket_idx, int bucket_cap, cudaStream_t stream) {
  hopper_seed_from_sample_kernel<__half><<<R, 256, 0, stream>>>(
      sample_val, sample_idx, R, K, BUF, NB, origin, inv_delta, th, qcount, bcount,
      qprev, refresh_lock, buf_val, buf_idx, buf_bucket, bucket_val, bucket_idx, bucket_cap);
}

void launch_hopper_seed_from_sample_fp32(
    const __half* sample_val, const int32_t* sample_idx,
    int R, int K, int BUF, int NB,
    float* origin, float* inv_delta, int32_t* th, int32_t* qcount, int32_t* bcount,
    int32_t* qprev, int32_t* refresh_lock,
    float* buf_val, int32_t* buf_idx, uint8_t* buf_bucket,
    __half* bucket_val, int32_t* bucket_idx, int bucket_cap, cudaStream_t stream) {
  hopper_seed_from_sample_kernel<float><<<R, 256, 0, stream>>>(
      sample_val, sample_idx, R, K, BUF, NB, origin, inv_delta, th, qcount, bcount,
      qprev, refresh_lock, buf_val, buf_idx, buf_bucket, bucket_val, bucket_idx, bucket_cap);
}

void launch_hopper_score_to_sparse_ip_fp16(
    const __half* x, const __half* base, int R, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    int32_t* qprev, int32_t* refresh_lock,
    __half* buf_val, int32_t* buf_idx, uint8_t* buf_bucket,
    __half* bucket_val, int32_t* bucket_idx, int bucket_cap, int BUF, int NB, int K,
    cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  int total_c_tiles = M / BN;
  int c_start_tile = start_col / BN;
  int tail_tiles = total_c_tiles - c_start_tile;
  if (tail_tiles <= 0) return;
  int m_tiles = R / BM;
  int c_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;

  CUtensorMap a_map{};
  CUtensorMap b_map{};
  encode_a_map(&a_map, x, R, dim);

  dim3 grid(m_tiles, c_tile_groups, 1);
  dim3 block(THREADS, 1, 1);

  // D > 512 (k_stages > 8) exceeds full-D smem residence; use the K-chunk path.
  // The K-chunk sparse kernel only supports the default RFB=0 refresh and the
  // plain compacted-buffer write (no bucket-write / refresh-from-buf / dense
  // write), so the wrapper must not enable those when D>512.
  if (k_stages > K_STAGES || KCHUNK_FORCE) {
    CUtensorMap b2d_map{};
    encode_a_map(&b2d_map, base, M, dim);
    static bool s_sparse_kchunk_attr = false;
    if (!s_sparse_kchunk_attr) {
      cudaFuncSetAttribute(hopper_score_to_sparse_ip_kchunk_kernel<__half>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, KCHUNK_SMEM_BYTES);
      s_sparse_kchunk_attr = true;
    }
    hopper_score_to_sparse_ip_kchunk_kernel<__half><<<grid, dim3(KCHUNK_THREADS, 1, 1), KCHUNK_SMEM_BYTES, stream>>>(
        a_map, b2d_map, origin, inv_delta, th, qcount, bcount,
        buf_val, buf_idx, total_c_tiles, c_start_tile, BUF, NB, K, k_stages);
    return;
  }

  encode_b_map(&b_map, base, M, dim, k_stages);

  if (!g_sparse_smem_attr_set) {
    cudaFuncSetAttribute(hopper_score_to_sparse_ip_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SPARSE_SMEM_BYTES);
    g_sparse_smem_attr_set = true;
  }

  hopper_score_to_sparse_ip_kernel<<<grid, block, SPARSE_SMEM_BYTES, stream>>>(
      a_map, b_map, origin, inv_delta, th, qcount, bcount, qprev, refresh_lock,
      buf_val, buf_idx, buf_bucket, bucket_val, bucket_idx, bucket_cap,
      total_c_tiles, c_start_tile, BUF, NB, K, k_stages);
}

void launch_hopper_score_to_sparse_ip_fp32(
    const __half* x, const __half* base, int R, int M, int dim, int start_col,
    const float* origin, const float* inv_delta,
    int32_t* th, int32_t* qcount, int32_t* bcount,
    float* buf_val, int32_t* buf_idx, int BUF, int NB, int K,
    cudaStream_t stream) {
  int k_stages = dim / K_TILE;
  int total_c_tiles = M / BN;
  int c_start_tile = start_col / BN;
  int tail_tiles = total_c_tiles - c_start_tile;
  if (tail_tiles <= 0) return;
  int m_tiles = R / BM;
  int c_tile_groups = (tail_tiles + C_TILES_PER_CTA - 1) / C_TILES_PER_CTA;

  CUtensorMap a_map{};
  CUtensorMap b2d_map{};
  encode_a_map(&a_map, x, R, dim);
  encode_a_map(&b2d_map, base, M, dim);
  dim3 grid(m_tiles, c_tile_groups, 1);

  static bool s_sparse_kchunk_fp32_attr = false;
  if (!s_sparse_kchunk_fp32_attr) {
    cudaFuncSetAttribute(hopper_score_to_sparse_ip_kchunk_kernel<float>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, KCHUNK_SMEM_BYTES);
    s_sparse_kchunk_fp32_attr = true;
  }
  hopper_score_to_sparse_ip_kchunk_kernel<float><<<grid, dim3(KCHUNK_THREADS, 1, 1), KCHUNK_SMEM_BYTES, stream>>>(
      a_map, b2d_map, origin, inv_delta, th, qcount, bcount,
      buf_val, buf_idx, total_c_tiles, c_start_tile, BUF, NB, K, k_stages);
}

#if 0  // RETIRED: dense bucket coords launcher
void launch_hopper_dense_bucket_coords_fp16(
    const __half* dense_scores, int R, int M,
    float* origin, float* inv_delta, int NB, cudaStream_t stream) {
  hopper_dense_bucket_coords_kernel<<<R, 256, 0, stream>>>(
      dense_scores, R, M, origin, inv_delta, NB);
}

#endif  // launch_hopper_dense_bucket_coords_fp16
void launch_hopper_negate_fp16(__half* values, int n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  hopper_negate_kernel<<<grid, block, 0, stream>>>(values, n);
}

#if 0  // RETIRED: smalln exact-IP + wmma dense paths (slower than sparse)
namespace {

constexpr int SMALLN_THREADS = 128;
constexpr int SMALLN_CHUNK = 4096;
constexpr int SMALLN_LOCAL_K = 8;
constexpr int SMALLN_KMAX = 128;
constexpr float SMALLN_NEG_INF = -3.4028234663852886e38f;
constexpr int SMALLN_QN = 32;
constexpr int SMALLN_A_TILE_ELEMS = K_STAGES * A_STAGE_ELEMS;
constexpr int SMALLN_A_TILE_BYTES = SMALLN_A_TILE_ELEMS * int(sizeof(__half));
constexpr int SMALLN_A_BYTES = C_STAGE * SMALLN_A_TILE_BYTES;
constexpr int SMALLN_B_K_STAGE_ELEMS = SMALLN_QN * K_TILE;
constexpr int SMALLN_B_STAGE_ELEMS = K_STAGES * SMALLN_B_K_STAGE_ELEMS;
constexpr int SMALLN_B_STAGE_BYTES = SMALLN_B_STAGE_ELEMS * int(sizeof(__half));
constexpr int SMALLN_B_BYTES = C_STAGE * SMALLN_B_STAGE_BYTES;
constexpr int SMALLN_GMMA_SMEM_BYTES =
    SMALLN_A_BYTES + SMALLN_B_BYTES +
    (A_BARRIERS + B_BARRIERS + FREE_BARRIERS) * int(sizeof(uint64_t));

__device__ __forceinline__ void smalln_insert(float v, int idx, float* vals, int* ids, int K) {
  int limit = K < SMALLN_KMAX ? K : SMALLN_KMAX;
  if (v <= vals[limit - 1]) return;
  int pos = limit - 1;
  while (pos > 0 && v > vals[pos - 1]) {
    vals[pos] = vals[pos - 1];
    ids[pos] = ids[pos - 1];
    --pos;
  }
  vals[pos] = v;
  ids[pos] = idx;
}

__device__ __forceinline__ void smalln_insert_local(float v, int idx, float* vals, int* ids) {
  if (v <= vals[SMALLN_LOCAL_K - 1]) return;
  int pos = SMALLN_LOCAL_K - 1;
  while (pos > 0 && v > vals[pos - 1]) {
    vals[pos] = vals[pos - 1];
    ids[pos] = ids[pos - 1];
    --pos;
  }
  vals[pos] = v;
  ids[pos] = idx;
}

__device__ __forceinline__ float smalln_dot_dim(const __half* x, const __half* y, int dim) {
  const half2* x2 = reinterpret_cast<const half2*>(x);
  const half2* y2 = reinterpret_cast<const half2*>(y);
  float acc = 0.0f;
  for (int i = 0; i < dim / 2; ++i) {
    float2 p = __half22float2(__hmul2(x2[i], y2[i]));
    acc += p.x + p.y;
  }
  return acc;
}

__global__ void smalln_ip_partial_kernel(
    const __half* __restrict__ x, const __half* __restrict__ base,
    int N, int M, int dim, int K, int num_chunks,
    __half* __restrict__ partial_val, int32_t* __restrict__ partial_idx) {
  int row = blockIdx.x;
  int chunk = blockIdx.y;
  int start = chunk * SMALLN_CHUNK;
  int end = start + SMALLN_CHUNK;
  if (end > M) end = M;

  float local_v[SMALLN_LOCAL_K];
  int local_i[SMALLN_LOCAL_K];
  CUTE_UNROLL
  for (int i = 0; i < SMALLN_LOCAL_K; ++i) {
    local_v[i] = SMALLN_NEG_INF;
    local_i[i] = -1;
  }

  const __half* xrow = x + (size_t)row * dim;
  for (int col = start + threadIdx.x; col < end; col += blockDim.x) {
    float v = smalln_dot_dim(xrow, base + (size_t)col * dim, dim);
    smalln_insert_local(v, col, local_v, local_i);
  }

  __shared__ float sh_v[SMALLN_THREADS * SMALLN_LOCAL_K];
  __shared__ int sh_i[SMALLN_THREADS * SMALLN_LOCAL_K];
  CUTE_UNROLL
  for (int i = 0; i < SMALLN_LOCAL_K; ++i) {
    int off = threadIdx.x * SMALLN_LOCAL_K + i;
    sh_v[off] = local_v[i];
    sh_i[off] = local_i[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float best_v[SMALLN_KMAX];
    int best_i[SMALLN_KMAX];
    CUTE_UNROLL
    for (int i = 0; i < SMALLN_KMAX; ++i) {
      best_v[i] = SMALLN_NEG_INF;
      best_i[i] = -1;
    }
    int total = SMALLN_THREADS * SMALLN_LOCAL_K;
    for (int i = 0; i < total; ++i) {
      if (sh_i[i] >= 0) {
        smalln_insert(sh_v[i], sh_i[i], best_v, best_i, K);
      }
    }
    size_t out = ((size_t)row * num_chunks + chunk) * K;
    for (int i = 0; i < K; ++i) {
      partial_val[out + i] = __float2half(best_v[i]);
      partial_idx[out + i] = best_i[i];
    }
  }
}

__global__ void smalln_ip_merge_kernel(
    const __half* __restrict__ partial_val, const int32_t* __restrict__ partial_idx,
    int N, int K, int num_chunks,
    __half* __restrict__ out_val, int32_t* __restrict__ out_idx) {
  int row = blockIdx.x;
  float best_v[SMALLN_KMAX];
  int best_i[SMALLN_KMAX];
  CUTE_UNROLL
  for (int i = 0; i < SMALLN_KMAX; ++i) {
    best_v[i] = SMALLN_NEG_INF;
    best_i[i] = -1;
  }
  if (threadIdx.x == 0) {
    for (int c = 0; c < num_chunks; ++c) {
      size_t off = ((size_t)row * num_chunks + c) * K;
      for (int j = 0; j < K; ++j) {
        int id = partial_idx[off + j];
        if (id >= 0) {
          smalln_insert(__half2float(partial_val[off + j]), id, best_v, best_i, K);
        }
      }
    }
    size_t out = (size_t)row * K;
    for (int j = 0; j < K; ++j) {
      out_val[out + j] = __float2half(best_v[j]);
      out_idx[out + j] = best_i[j];
    }
  }
}

}  // namespace

void launch_hopper_smalln_ip_topk_fp16(
    const __half* x, const __half* base, int N, int M, int dim, int K,
    __half* partial_val, int32_t* partial_idx, int num_chunks,
    __half* out_val, int32_t* out_idx, cudaStream_t stream) {
  dim3 grid_partial(N, num_chunks, 1);
  smalln_ip_partial_kernel<<<grid_partial, SMALLN_THREADS, 0, stream>>>(
      x, base, N, M, dim, K, num_chunks, partial_val, partial_idx);
  smalln_ip_merge_kernel<<<N, 1, 0, stream>>>(
      partial_val, partial_idx, N, K, num_chunks, out_val, out_idx);
}

namespace {

__global__ void smalln_wmma_score_dense_ip_kernel(
    const __half* __restrict__ x, const __half* __restrict__ base,
    int Npad, int M, __half* __restrict__ dense_scores) {
  using namespace nvcuda;
  int row_tile = blockIdx.x * 32;
  int col_tile = blockIdx.y * 64;
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int warp_row = warp_id >> 2;
  int warp_col = col_tile + (warp_id & 3) * 16;
  int row_base = row_tile + warp_row * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  CUTE_UNROLL
  for (int k0 = 0; k0 < D; k0 += 16) {
    wmma::load_matrix_sync(a_frag, x + (size_t)row_base * D + k0, D);
    wmma::load_matrix_sync(b_frag, base + (size_t)warp_col * D + k0, D);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  __shared__ float tmp[8 * 16 * 16];
  float* warp_tmp = tmp + warp_id * 16 * 16;
  wmma::store_matrix_sync(warp_tmp, c_frag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 16 * 16; i += 32) {
    int r = i / 16;
    int c = i - r * 16;
    int gr = row_base + r;
    int gc = warp_col + c;
    if (gr < Npad && gc < M) {
      dense_scores[(size_t)gr * M + gc] = __float2half(-warp_tmp[i]);
    }
  }
}

}  // namespace

void launch_hopper_smalln_score_dense_ip_fp16(
    const __half* x, const __half* base, int Npad, int M,
    __half* dense_scores, cudaStream_t stream) {
  dim3 grid((Npad + 31) / 32, (M + 63) / 64, 1);
  smalln_wmma_score_dense_ip_kernel<<<grid, 256, 0, stream>>>(x, base, Npad, M, dense_scores);
}
#endif  // smalln_ip / wmma dense tail

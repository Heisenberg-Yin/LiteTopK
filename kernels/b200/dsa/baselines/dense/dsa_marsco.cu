// Route-A stage 2 host wrapper (Blackwell / B200 SM100 port): launch the DSA
// ReLU-MQA scoring kernel with a sample+write epilogue. Scoring math is the
// deep_gemm SM100 UMMA/TMEM path (tcgen05); only the final dense-store epilogue
// is replaced by compact candidate writes. The radix-select post-kernels are
// architecture-agnostic and are copied verbatim from the Hopper version.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <limits>
#include <tuple>

#include "sm100_dsa_marsco.cuh"

namespace {

static void* driver_handle() {
    static void* h = nullptr;
    if (!h) {
        h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        TORCH_CHECK(h, "failed to load libcuda.so.1");
    }
    return h;
}

static CUresult enc_tiled(CUtensorMap* tm, CUtensorMapDataType dt, cuuint32_t rank,
                          void* addr, const cuuint64_t* dims, const cuuint64_t* strides,
                          const cuuint32_t* box, const cuuint32_t* estrides,
                          CUtensorMapInterleave il, CUtensorMapSwizzle sw,
                          CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill oob) {
    using FT = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                            const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                            const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    static FT f = nullptr;
    if (!f) {
        f = reinterpret_cast<FT>(dlsym(driver_handle(), "cuTensorMapEncodeTiled"));
        TORCH_CHECK(f, "failed to load cuTensorMapEncodeTiled");
    }
    return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

static CUtensorMap make_2d(void* ptr, CUtensorMapDataType dt, int elem_size,
                           int gmem_inner, int gmem_outer,
                           int smem_inner, int smem_outer,
                           long gmem_outer_stride, int swizzle_mode) {
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem_size;
    CUtensorMap tm;
    const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
    const cuuint32_t estrides[2] = {1, 1};
    CUtensorMapSwizzle swizzle =
        swizzle_mode == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle_mode == 64  ? CU_TENSOR_MAP_SWIZZLE_64B :
        swizzle_mode == 32  ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult r = enc_tiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                           CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return tm;
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

constexpr int NUM_HEADS = 32;
constexpr int HEAD_DIM = 128;
#ifndef DSA_BLOCK_Q
#define DSA_BLOCK_Q 4
#endif
#ifndef DSA_BLOCK_KV
// B200 default: large tile. BLOCK_KV=256 with 2 math warpgroups uses the bigger
// Blackwell smem/TMEM and roughly halves KV-iteration / barrier overhead vs the
// old 128/1-WG tile. Measured on real GLM-5 caches this lifts the end-to-end
// speedup vs the dense baseline from ~1.1-1.4x to ~1.5-2.15x (recall unchanged
// at 100%). Override with -DDSA_BLOCK_KV=128 to get the small tile.
#define DSA_BLOCK_KV 256
#endif
// On SM100 the math thread count must equal BLOCK_KV (each math lane owns one
// score column; DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads) in the kernel).
#ifndef DSA_MATH_THREADS
#define DSA_MATH_THREADS DSA_BLOCK_KV
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
constexpr int BLOCK_Q = DSA_BLOCK_Q;
constexpr int BLOCK_KV = DSA_BLOCK_KV;
// Each CTA processes exactly ONE q-block (no persistent scheduling), so a single
// Q stage suffices — the old 3 stages were a leftover from deep_gemm's persistent
// scheduler and wasted 2x16.5KB of smem. The freed smem goes into a deeper KV
// pipeline (3 -> 6 stages, ~223KB of the 227KB smem budget) for better TMA
// latency hiding; measured on real GLM-5 caches KV=6 is the best depth (KV=7
// no longer fits with the warp queue).
#ifndef DSA_Q_STAGES
#define DSA_Q_STAGES 1
#endif
#ifndef DSA_KV_STAGES
#define DSA_KV_STAGES 6
#endif
constexpr int NUM_Q_STAGES = DSA_Q_STAGES;
constexpr int NUM_KV_STAGES = DSA_KV_STAGES;
// One specialized warpgroup (TMA-load warp + UMMA warp + spare warps).
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = DSA_MATH_THREADS;
constexpr int NUM_MATH_WGS = MATH_THREADS / 128;
// B200 (GB200/B200) has 148 SMs.
constexpr int NUM_SMS = 148;


__global__ void seed_bcount_kernel(
    const float* __restrict__ seed_val,
    int seed_k,
    const float* __restrict__ origin,
    const float* __restrict__ inv_delta,
    int32_t* __restrict__ bcount,
    int R,
    int NB) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= R) return;
    float o = origin[row];
    float inv = inv_delta[row];
    for (int i = tid; i < seed_k; i += blockDim.x) {
        float x = seed_val[(size_t)row * seed_k + i];
        int braw = static_cast<int>((x - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (braw < NB) atomicAdd(&bcount[(size_t)row * NB + b], 1);
    }
}

__global__ void refresh_threshold_from_bcount_kernel(
    int32_t* __restrict__ th_bucket,
    const int32_t* __restrict__ bcount,
    int R,
    int NB,
    int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= R) return;
    int old_th = th_bucket[row];
    int cum = 0;
    int new_th = old_th;
    bool found = false;
    for (int b = 0; b < NB; ++b) {
        cum += bcount[(size_t)row * NB + b];
        if (cum >= K) {
            new_th = b;
            found = true;
            break;
        }
    }
    if (found && new_th < old_th) th_bucket[row] = new_th;
}

__device__ __forceinline__ uint32_t compact_enc_float(float v) {
    uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

__global__ void compact_topk_min_idx_marsco_kernel(
    const float* __restrict__ val,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ cnt,
    int R,
    int CAP,
    int K,
    float* __restrict__ out_val,
    int32_t* __restrict__ out_idx) {
    constexpr int BT = 256;
    constexpr int RADIX = 256;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= R) return;
    const float* vrow = val + (size_t)row * CAP;
    const int32_t* irow = idx + (size_t)row * CAP;
    float* ov = out_val + (size_t)row * K;
    int32_t* oi = out_idx + (size_t)row * K;
    int n = cnt[row];
    if (n > CAP) n = CAP;
    if (n < 0) n = 0;
    if (n == 0) return;

    __shared__ uint32_t hist[RADIX];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int cnt_lt;
    __shared__ int w_lt;
    __shared__ int w_eq;
    if (tid == 0) { desired = 0u; kfind = (uint32_t)(K < n ? K : n); }
    __syncthreads();

    uint32_t mask = 0u;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - pass * 8;
        for (int b = tid; b < RADIX; b += BT) hist[b] = 0;
        __syncthreads();
        uint32_t d = desired;
        for (int j = tid; j < n; j += BT) {
            uint32_t e = compact_enc_float(vrow[j]);
            if ((e & mask) == (d & mask)) atomicAdd(&hist[(e >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0;
            uint32_t kf = kfind;
            for (int b = 0; b < RADIX; ++b) {
                uint32_t h = hist[b];
                if (acc < kf && kf <= acc + h) {
                    desired = d | (uint32_t(b) << shift);
                    kfind = kf - acc;
                    break;
                }
                acc += h;
            }
        }
        __syncthreads();
        mask |= 0xffu << shift;
    }
    uint32_t pivot = desired;

    if (tid == 0) { cnt_lt = 0; w_lt = 0; w_eq = 0; }
    __syncthreads();
    int local_lt = 0;
    for (int j = tid; j < n; j += BT) {
        if (compact_enc_float(vrow[j]) < pivot) local_lt++;
    }
    atomicAdd(&cnt_lt, local_lt);
    __syncthreads();
    int take_eq = K - cnt_lt;
    if (take_eq < 0) take_eq = 0;

    for (int j = tid; j < n; j += BT) {
        uint32_t e = compact_enc_float(vrow[j]);
        if (e < pivot) {
            int w = atomicAdd(&w_lt, 1);
            if (w < K) { ov[w] = vrow[j]; oi[w] = irow[j]; }
        } else if (e == pivot) {
            int o = atomicAdd(&w_eq, 1);
            if (o < take_eq) {
                int w = cnt_lt + o;
                if (w < K) { ov[w] = vrow[j]; oi[w] = irow[j]; }
            }
        }
    }
    for (int j = tid + n; j < K; j += BT) {
        ov[j] = INFINITY;
        oi[j] = 0;
    }
}

// Threshold-aware select mirroring tidal/marsco's flashtopk_select_thr_kernel.
// bucket <  th : DEFINITELY in top-K -> direct copy (no radix)
// bucket == th : boundary -> radix-select the remaining k_eq = K - cnt_below
// bucket >  th : prune (never emitted by the scan; guarded for safety)
// Falls back to a full radix over bucket<=th when the gate is only a loose upper
// bound (cnt_below >= K) or the buffer is underfilled (cnt_below+cnt_eq < K).
__global__ void compact_topk_min_thr_marsco_kernel(
    const float* __restrict__ val,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ cnt,
    const float* __restrict__ origin,
    const float* __restrict__ inv_delta,
    const int32_t* __restrict__ th_in,
    int R,
    int CAP,
    int K,
    int NB,
    float* __restrict__ out_val,
    int32_t* __restrict__ out_idx) {
    constexpr int BT = 256;
    constexpr int RADIX = 256;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= R) return;
    const float* vrow = val + (size_t)row * CAP;
    const int32_t* irow = idx + (size_t)row * CAP;
    float* ov = out_val + (size_t)row * K;
    int32_t* oi = out_idx + (size_t)row * K;
    int n = cnt[row];
    if (n > CAP) n = CAP;
    if (n < 0) n = 0;

    const float o = origin[row];
    const float inv = inv_delta[row];
    const int th = th_in[row];

    __shared__ uint32_t hist[RADIX];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int s_cnt_bucket_lt;
    __shared__ int s_cnt_bucket_eq;
    __shared__ int s_use_boundary;
    __shared__ int s_select_all;
    __shared__ int s_k_eq;
    __shared__ int s_cnt_lt;
    __shared__ int s_w_pre;
    __shared__ int s_w_lt;
    __shared__ int s_w_eq;

    if (n == 0) {
        for (int j = tid; j < K; j += BT) { ov[j] = INFINITY; oi[j] = 0; }
        return;
    }

    // Pass 1: split the valid candidates into bucket<th and bucket==th.
    if (tid == 0) { s_cnt_bucket_lt = 0; s_cnt_bucket_eq = 0; }
    __syncthreads();
    int my_lt = 0, my_eq = 0;
    for (int j = tid; j < n; j += BT) {
        float v = vrow[j];
        if (!isfinite(v)) continue;
        int braw = static_cast<int>((v - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (b < th) my_lt++;
        else if (b == th) my_eq++;
    }
    atomicAdd(&s_cnt_bucket_lt, my_lt);
    atomicAdd(&s_cnt_bucket_eq, my_eq);
    __syncthreads();

    if (tid == 0) {
        int clt = s_cnt_bucket_lt;
        int ceq = s_cnt_bucket_eq;
        int k_eq = K - clt;
        s_use_boundary = (clt < K) && (k_eq > 0) && (k_eq <= ceq);
        s_select_all = (clt + ceq) < K;
        s_k_eq = s_use_boundary ? k_eq : (K < n ? K : n);
        desired = 0u;
        kfind = static_cast<uint32_t>(s_k_eq);
    }
    __syncthreads();

    #define DSA_IN_SELECT_SET(b) \
        (s_use_boundary ? ((b) == th) : (s_select_all ? true : ((b) <= th)))

    uint32_t mask = 0u;
    #pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - pass * 8;
        for (int b = tid; b < RADIX; b += BT) hist[b] = 0;
        __syncthreads();
        uint32_t d = desired;
        for (int j = tid; j < n; j += BT) {
            float v = vrow[j];
            if (!isfinite(v)) continue;
            int braw = static_cast<int>((v - o) * inv);
            int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
            if (!DSA_IN_SELECT_SET(b)) continue;
            uint32_t e = compact_enc_float(v);
            if ((e & mask) == (d & mask)) atomicAdd(&hist[(e >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0;
            uint32_t kf = kfind;
            for (int b = 0; b < RADIX; ++b) {
                uint32_t h = hist[b];
                if (acc < kf && kf <= acc + h) {
                    desired = d | (uint32_t(b) << shift);
                    kfind = kf - acc;
                    break;
                }
                acc += h;
            }
        }
        __syncthreads();
        mask |= 0xffu << shift;
    }
    uint32_t pivot = desired;

    if (tid == 0) { s_cnt_lt = 0; s_w_pre = 0; s_w_lt = 0; s_w_eq = 0; }
    __syncthreads();
    int lt = 0;
    for (int j = tid; j < n; j += BT) {
        float v = vrow[j];
        if (!isfinite(v)) continue;
        int braw = static_cast<int>((v - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (!DSA_IN_SELECT_SET(b)) continue;
        if (compact_enc_float(v) < pivot) lt++;
    }
    atomicAdd(&s_cnt_lt, lt);
    __syncthreads();
    int cnt_lt = s_cnt_lt;
    int pre_take = s_use_boundary ? s_cnt_bucket_lt : 0;
    int target_k = s_use_boundary ? s_k_eq : (K < n ? K : n);
    int eq_take = target_k - cnt_lt; if (eq_take < 0) eq_take = 0;

    for (int j = tid; j < n; j += BT) {
        float v = vrow[j];
        if (!isfinite(v)) continue;
        int braw = static_cast<int>((v - o) * inv);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (s_use_boundary && b < th) {
            int w = atomicAdd(&s_w_pre, 1);
            if (w < K) { ov[w] = v; oi[w] = irow[j]; }
            continue;
        }
        if (!DSA_IN_SELECT_SET(b)) continue;
        uint32_t e = compact_enc_float(v);
        if (e < pivot) {
            int w = atomicAdd(&s_w_lt, 1);
            int out_pos = pre_take + w;
            if (out_pos < K) { ov[out_pos] = v; oi[out_pos] = irow[j]; }
        } else if (e == pivot) {
            int oo = atomicAdd(&s_w_eq, 1);
            if (oo < eq_take) {
                int w = pre_take + cnt_lt + oo;
                if (w < K) { ov[w] = v; oi[w] = irow[j]; }
            }
        }
    }
    #undef DSA_IN_SELECT_SET
    for (int j = tid + n; j < K; j += BT) {
        ov[j] = INFINITY;
        oi[j] = 0;
    }
}

// Compute the dynamic shared-memory size for the SM100 kernel launch.
static int compute_smem_bytes() {
    const int esz_fp8 = 1, esz_f32 = 4;
    const int smem_q  = BLOCK_Q * NUM_HEADS * HEAD_DIM * esz_fp8;
    const int smem_w  = BLOCK_Q * NUM_HEADS * esz_f32;
    const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
    // kv scale is padded to 512B per stage (matches ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE).
    const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
    // Barriers: Q(full/empty) + KV(full/empty) + UMMA(full/empty per math WG per
    // tmem parity buffer), 8 bytes each (ClusterTransactionBarrier == uint64_t).
    const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 + NUM_MATH_WGS * DSA_TMEM_BUFS * 2;
    const int smem_barriers = num_barriers * 8;
    const int smem_tmem_slot = 4 * (int)sizeof(uint32_t);  // tmem_ptr_in_smem (+padding)
#if DSA_WARP_QUEUE
    const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                           ((int)sizeof(int32_t) + DSA_WARP_QUEUE_CAP * ((int)sizeof(float) + (int)sizeof(int32_t)));
#else
    const int smem_warpq = 0;
#endif
    return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
           NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks +
           smem_barriers + smem_tmem_slot + smem_warpq;
}

static void launch_scoring(
        torch::Tensor& q, torch::Tensor& kv, torch::Tensor& kv_scales, torch::Tensor& weights,
        torch::Tensor& cu_start, torch::Tensor& cu_end,
        torch::Tensor& origin, torch::Tensor& inv_delta, torch::Tensor& th_bucket,
        torch::Tensor& bcount,
        torch::Tensor& cand_val, torch::Tensor& cand_idx, torch::Tensor& cand_cnt,
        int seq_len, int seq_len_kv, int cand_cap, int num_buckets, int topk,
        int refresh_every, int num_kv_splits_override, cudaStream_t stream) {
    const int head_dim = HEAD_DIM, num_heads = NUM_HEADS;
    const int esz_fp8 = 1, esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                        head_dim, seq_len * num_heads, head_dim, BLOCK_Q * num_heads, head_dim, head_dim);
    auto tm_kv = make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                         head_dim, seq_len_kv, head_dim, BLOCK_KV, head_dim, head_dim);
    auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                         ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                        num_heads, seq_len, num_heads, BLOCK_Q, num_heads, 0);

    const int smem = compute_smem_bytes();

    auto kernel = &dsa_marsco::sm100_dsa_marsco<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_SMS, SPEC_THREADS, MATH_THREADS>;
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    // KV-split grid: x = q-blocks, y = KV splits.
    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
    int num_kv_splits;
    if (num_kv_splits_override > 0) {
        num_kv_splits = num_kv_splits_override;
    } else {
        // Over-subscription: target ~4 CTA waves per SM for tail balancing +
        // latency hiding (mirrors the Hopper heuristic, retuned for 148 SMs).
        constexpr int kWaves = 4;
        const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
        num_kv_splits = (kWaves * NUM_SMS + qb - 1) / qb;
        const int max_useful_splits = total_kv_blocks > 0 ? (total_kv_blocks + 1) / 2 : 1;
        if (num_kv_splits > max_useful_splits) num_kv_splits = max_useful_splits;
    }
    if (num_kv_splits < 1) num_kv_splits = 1;
    if (num_kv_splits > total_kv_blocks) num_kv_splits = total_kv_blocks > 0 ? total_kv_blocks : 1;
    dim3 grid((unsigned)num_q_blocks, (unsigned)num_kv_splits, 1);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(), (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
        (uint32_t)num_kv_splits,
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(), (uint32_t)cand_cap,
        tm_q, tm_kv, tm_ks, tm_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mqa_logits_dsa_marsco(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor seed_val,
        torch::Tensor seed_idx,
        int64_t num_buckets64,
        int64_t cand_cap64,
        int64_t topk64,
        int64_t refresh_every64,
        int64_t num_kv_splits_override) {
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() && weights.is_cuda() && origin.is_cuda() && inv_delta.is_cuda() && th_bucket.is_cuda() &&
                seed_val.is_cuda() && seed_idx.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && kv_scales.is_contiguous() && weights.is_contiguous() &&
                cu_start.is_contiguous() && cu_end.is_contiguous() && origin.is_contiguous() && inv_delta.is_contiguous() &&
                th_bucket.is_contiguous() && seed_val.is_contiguous() && seed_idx.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "q must be fp8_e4m3fn");
    TORCH_CHECK(kv.scalar_type() == torch::kFloat8_e4m3fn, "kv must be fp8_e4m3fn");
    TORCH_CHECK(kv_scales.scalar_type() == torch::kFloat, "kv_scales must be fp32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat, "weights must be fp32");
    TORCH_CHECK(origin.scalar_type() == torch::kFloat && inv_delta.scalar_type() == torch::kFloat, "origin/inv_delta must be fp32");
    TORCH_CHECK(th_bucket.scalar_type() == torch::kInt, "th_bucket must be int32");
    TORCH_CHECK(seed_val.scalar_type() == torch::kFloat, "seed_val must be fp32 x=-score");
    TORCH_CHECK(seed_idx.scalar_type() == torch::kInt, "seed_idx must be int32");
    TORCH_CHECK(cu_start.scalar_type() == torch::kInt && cu_end.scalar_type() == torch::kInt,
                "cu_start/cu_end must be int32");

    const int seq_len = (int)q.size(0);
    const int num_heads = (int)q.size(1);
    const int head_dim = (int)q.size(2);
    const int seq_len_kv = (int)kv.size(0);
    (void)cand_cap64;
    const int cand_cap = seq_len_kv;
    TORCH_CHECK(num_heads == NUM_HEADS && head_dim == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(kv.size(1) == HEAD_DIM, "kv D mismatch");
    TORCH_CHECK(origin.numel() == seq_len && inv_delta.numel() == seq_len && th_bucket.numel() == seq_len, "bucket params must have Q elements");
    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    const bool external_refresh = (refresh_every64 < 0);
    const int refresh_every = external_refresh ? 0x7fffffff : static_cast<int>(refresh_every64);
    TORCH_CHECK(num_buckets >= 2 && num_buckets <= 4096, "num_buckets out of range");
    TORCH_CHECK(cand_cap >= 1, "cand_cap must be positive");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1, cand_cap]");
    TORCH_CHECK(refresh_every64 >= -1, "refresh_every must be >= -1 (-1 means post-score external refresh)");
    TORCH_CHECK(seed_val.dim() == 2 && seed_idx.dim() == 2, "seed tensors must be [Q, seed_k]");
    TORCH_CHECK(seed_val.size(0) == seq_len && seed_idx.size(0) == seq_len && seed_val.size(1) == seed_idx.size(1),
                "seed tensor shape mismatch");
    const int seed_k = static_cast<int>(seed_val.size(1));
    TORCH_CHECK(seed_k <= cand_cap, "seed_k must be <= cand_cap");

    auto cand_val = torch::empty({seq_len, cand_cap}, q.options().dtype(torch::kFloat));
    auto cand_idx = torch::empty({seq_len, cand_cap}, q.options().dtype(torch::kInt));
    auto cand_cnt = torch::full({seq_len}, seed_k, q.options().dtype(torch::kInt));
    auto bcount = torch::zeros({seq_len, num_buckets}, q.options().dtype(torch::kInt));
    if (seed_k > 0) {
        cand_val.narrow(1, 0, seed_k).copy_(seed_val);
        cand_idx.narrow(1, 0, seed_k).copy_(seed_idx);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if ((refresh_every > 0 || external_refresh) && seed_k > 0) {
        seed_bcount_kernel<<<seq_len, 256, 0, stream>>>(
            seed_val.data_ptr<float>(), seed_k, origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
            bcount.data_ptr<int32_t>(), seq_len, num_buckets);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    launch_scoring(q, kv, kv_scales, weights, cu_start, cu_end, origin, inv_delta, th_bucket,
                   bcount, cand_val, cand_idx, cand_cnt,
                   seq_len, seq_len_kv, cand_cap, num_buckets, topk, refresh_every,
                   (int)num_kv_splits_override, stream);

    if (external_refresh) {
        int block = 128;
        int grid_r = (seq_len + block - 1) / block;
        refresh_threshold_from_bcount_kernel<<<grid_r, block, 0, stream>>>(
            th_bucket.data_ptr<int32_t>(), bcount.data_ptr<int32_t>(),
            seq_len, num_buckets, topk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return std::make_tuple(cand_val, cand_idx, cand_cnt);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mqa_logits_dsa_marsco_out(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor seed_val,
        torch::Tensor seed_idx,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t cand_cap64,
        int64_t topk64,
        int64_t refresh_every64,
        int64_t num_kv_splits_override) {
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() && weights.is_cuda() && origin.is_cuda() && inv_delta.is_cuda() && th_bucket.is_cuda() &&
                seed_val.is_cuda() && seed_idx.is_cuda() && cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda() && bcount.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && kv_scales.is_contiguous() && weights.is_contiguous() &&
                cu_start.is_contiguous() && cu_end.is_contiguous() && origin.is_contiguous() && inv_delta.is_contiguous() &&
                th_bucket.is_contiguous() && seed_val.is_contiguous() && seed_idx.is_contiguous() &&
                cand_val.is_contiguous() && cand_idx.is_contiguous() && cand_cnt.is_contiguous() && bcount.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn, "q must be fp8_e4m3fn");
    TORCH_CHECK(kv.scalar_type() == torch::kFloat8_e4m3fn, "kv must be fp8_e4m3fn");
    TORCH_CHECK(kv_scales.scalar_type() == torch::kFloat, "kv_scales must be fp32");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat, "weights must be fp32");
    TORCH_CHECK(origin.scalar_type() == torch::kFloat && inv_delta.scalar_type() == torch::kFloat, "origin/inv_delta must be fp32");
    TORCH_CHECK(th_bucket.scalar_type() == torch::kInt, "th_bucket must be int32");
    TORCH_CHECK(seed_val.scalar_type() == torch::kFloat, "seed_val must be fp32 x=-score");
    TORCH_CHECK(seed_idx.scalar_type() == torch::kInt, "seed_idx must be int32");
    TORCH_CHECK(cand_val.scalar_type() == torch::kFloat, "cand_val must be fp32");
    TORCH_CHECK(cand_idx.scalar_type() == torch::kInt && cand_cnt.scalar_type() == torch::kInt && bcount.scalar_type() == torch::kInt,
                "cand_idx/cand_cnt/bcount must be int32");
    TORCH_CHECK(cu_start.scalar_type() == torch::kInt && cu_end.scalar_type() == torch::kInt,
                "cu_start/cu_end must be int32");

    const int seq_len = (int)q.size(0);
    const int num_heads = (int)q.size(1);
    const int head_dim = (int)q.size(2);
    const int seq_len_kv = (int)kv.size(0);
    (void)cand_cap64;
    const int cand_cap = seq_len_kv;
    TORCH_CHECK(num_heads == NUM_HEADS && head_dim == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(kv.size(1) == HEAD_DIM, "kv D mismatch");
    TORCH_CHECK(origin.numel() == seq_len && inv_delta.numel() == seq_len && th_bucket.numel() == seq_len, "bucket params must have Q elements");
    const int num_buckets = static_cast<int>(num_buckets64);
    const int topk = static_cast<int>(topk64);
    const bool external_refresh = (refresh_every64 < 0);
    const int refresh_every = external_refresh ? 0x7fffffff : static_cast<int>(refresh_every64);
    TORCH_CHECK(num_buckets >= 2 && num_buckets <= 4096, "num_buckets out of range");
    TORCH_CHECK(cand_cap >= 1, "cand_cap must be positive");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1, cand_cap]");
    TORCH_CHECK(refresh_every64 >= -1, "refresh_every must be >= -1 (-1 means post-score external refresh)");
    TORCH_CHECK(seed_val.dim() == 2 && seed_idx.dim() == 2, "seed tensors must be [Q, seed_k]");
    TORCH_CHECK(seed_val.size(0) == seq_len && seed_idx.size(0) == seq_len && seed_val.size(1) == seed_idx.size(1),
                "seed tensor shape mismatch");
    const int seed_k = static_cast<int>(seed_val.size(1));
    TORCH_CHECK(seed_k <= cand_cap, "seed_k must be <= cand_cap");

    TORCH_CHECK(cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(),
                "cand_val/cand_idx must be [Q, M]");
    TORCH_CHECK(cand_val.size(0) == seq_len && cand_val.size(1) == cand_cap,
                "cand_val must have shape [Q, seq_len_kv]");
    TORCH_CHECK(cand_idx.size(0) == seq_len && cand_idx.size(1) == cand_cap,
                "cand_idx must have shape [Q, seq_len_kv]");
    TORCH_CHECK(cand_cnt.numel() == seq_len, "cand_cnt must have Q elements");
    TORCH_CHECK(bcount.dim() == 2 && bcount.size(0) == seq_len && bcount.size(1) == num_buckets,
                "bcount must have shape [Q, num_buckets]");
    cand_cnt.fill_(seed_k);
    bcount.zero_();
    if (seed_k > 0) {
        cand_val.narrow(1, 0, seed_k).copy_(seed_val);
        cand_idx.narrow(1, 0, seed_k).copy_(seed_idx);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if ((refresh_every > 0 || external_refresh) && seed_k > 0) {
        seed_bcount_kernel<<<seq_len, 256, 0, stream>>>(
            seed_val.data_ptr<float>(), seed_k, origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
            bcount.data_ptr<int32_t>(), seq_len, num_buckets);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    launch_scoring(q, kv, kv_scales, weights, cu_start, cu_end, origin, inv_delta, th_bucket,
                   bcount, cand_val, cand_idx, cand_cnt,
                   seq_len, seq_len_kv, cand_cap, num_buckets, topk, refresh_every,
                   (int)num_kv_splits_override, stream);

    if (external_refresh) {
        int block = 128;
        int grid_r = (seq_len + block - 1) / block;
        refresh_threshold_from_bcount_kernel<<<grid_r, block, 0, stream>>>(
            th_bucket.data_ptr<int32_t>(), bcount.data_ptr<int32_t>(),
            seq_len, num_buckets, topk);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return std::make_tuple(cand_val, cand_idx, cand_cnt);
}


std::tuple<torch::Tensor, torch::Tensor> compact_topk_min_idx_marsco(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        int64_t k64) {
    TORCH_CHECK(cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(cand_val.is_contiguous() && cand_idx.is_contiguous() && cand_cnt.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(cand_val.scalar_type() == torch::kFloat, "cand_val must be fp32");
    TORCH_CHECK(cand_idx.scalar_type() == torch::kInt && cand_cnt.scalar_type() == torch::kInt, "idx/cnt must be int32");
    TORCH_CHECK(cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(), "candidate tensors must be [R,CAP]");
    int R = static_cast<int>(cand_val.size(0));
    int CAP = static_cast<int>(cand_val.size(1));
    int K = static_cast<int>(k64);
    TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1,CAP]");
    auto out_val = torch::empty({R, K}, cand_val.options());
    auto out_idx = torch::empty({R, K}, cand_idx.options());
    compact_topk_min_idx_marsco_kernel<<<R, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
        R, CAP, K, out_val.data_ptr<float>(), out_idx.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(out_val, out_idx);
}

std::tuple<torch::Tensor, torch::Tensor> compact_topk_min_thr_marsco(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        int64_t num_buckets64,
        int64_t k64) {
    TORCH_CHECK(cand_val.is_cuda() && cand_idx.is_cuda() && cand_cnt.is_cuda() &&
                origin.is_cuda() && inv_delta.is_cuda() && th_bucket.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(cand_val.is_contiguous() && cand_idx.is_contiguous() && cand_cnt.is_contiguous() &&
                origin.is_contiguous() && inv_delta.is_contiguous() && th_bucket.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(cand_val.scalar_type() == torch::kFloat, "cand_val must be fp32");
    TORCH_CHECK(cand_idx.scalar_type() == torch::kInt && cand_cnt.scalar_type() == torch::kInt, "idx/cnt must be int32");
    TORCH_CHECK(origin.scalar_type() == torch::kFloat && inv_delta.scalar_type() == torch::kFloat, "origin/inv_delta must be fp32");
    TORCH_CHECK(th_bucket.scalar_type() == torch::kInt, "th_bucket must be int32");
    TORCH_CHECK(cand_val.dim() == 2 && cand_idx.sizes() == cand_val.sizes(), "candidate tensors must be [R,CAP]");
    int R = static_cast<int>(cand_val.size(0));
    int CAP = static_cast<int>(cand_val.size(1));
    int K = static_cast<int>(k64);
    int NB = static_cast<int>(num_buckets64);
    TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1,CAP]");
    TORCH_CHECK(NB >= 2 && NB <= 4096, "num_buckets out of range");
    TORCH_CHECK(origin.numel() == R && inv_delta.numel() == R && th_bucket.numel() == R,
                "origin/inv_delta/th_bucket must have R elements");
    auto out_val = torch::empty({R, K}, cand_val.options());
    auto out_idx = torch::empty({R, K}, cand_idx.options());
    compact_topk_min_thr_marsco_kernel<<<R, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        R, CAP, K, NB, out_val.data_ptr<float>(), out_idx.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(out_val, out_idx);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mqa_logits_dsa_marsco", &mqa_logits_dsa_marsco,
          "DSA ReLU-MQA scoring (SM100 UMMA/TMEM) with bucket-gated atomic epilogue",
          pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
          pybind11::arg("weights"), pybind11::arg("cu_start"), pybind11::arg("cu_end"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("seed_val"), pybind11::arg("seed_idx"),
          pybind11::arg("num_buckets"), pybind11::arg("cand_cap"), pybind11::arg("topk"),
          pybind11::arg("refresh_every"), pybind11::arg("num_kv_splits")=-1);
    m.def("mqa_logits_dsa_marsco_out", &mqa_logits_dsa_marsco_out,
          "DSA ReLU-MQA scoring (SM100) into externally reusable candidate buffers",
          pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
          pybind11::arg("weights"), pybind11::arg("cu_start"), pybind11::arg("cu_end"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("seed_val"), pybind11::arg("seed_idx"),
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"), pybind11::arg("bcount"),
          pybind11::arg("num_buckets"), pybind11::arg("cand_cap"), pybind11::arg("topk"),
          pybind11::arg("refresh_every"), pybind11::arg("num_kv_splits")=-1);
    m.def("compact_topk_min_idx_marsco", &compact_topk_min_idx_marsco,
          "Qcount-aware top-k-min over compact candidates (atomic variant)");
    m.def("compact_topk_min_thr_marsco", &compact_topk_min_thr_marsco,
          "Threshold-aware top-k-min: bucket<th direct copy + bucket==th boundary radix",
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("num_buckets"), pybind11::arg("topk"));
}

// LiteTopK DSA V2 host wrapper: launches the DeepGEMM-2.5-based persistent
// scoring kernel (sm100_dsa_marsco_v2.cuh) with the sparse candidate epilogue,
// plus the architecture-agnostic radix-select post-kernels (copied verbatim
// from dsa_marsco.cu). Build against the DeepGEMM 2.5 include tree + its
// bundled CUTLASS (NOT the legacy deep_gemm include tree V1 uses).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <limits>
#include <tuple>

#include "sm100_dsa_marsco_v2.cuh"

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
constexpr int BLOCK_Q = 4;         // 128 q*h rows per UMMA tile / 32 heads
constexpr int BLOCK_KV = 256;
constexpr int NUM_Q_STAGES = 3;
constexpr int NUM_KV_STAGES = 3;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;  // 2 math warpgroups on SM100
constexpr int NUM_SMS = 148;       // B200

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

static int compute_smem_bytes() {
    const int esz_fp8 = 1, esz_f32 = 4;
    const int smem_q  = BLOCK_Q * NUM_HEADS * HEAD_DIM * esz_fp8;
    const int smem_w  = BLOCK_Q * NUM_HEADS * esz_f32;
    const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
    const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
    const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 + (MATH_THREADS / 128) * 2;
    const int smem_barriers = num_barriers * 8;
    const int smem_slots = 4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
    const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                           ((int)sizeof(int32_t) + DSA_WARP_QUEUE_CAP * ((int)sizeof(float) + (int)sizeof(int32_t)));
    return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
           NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks +
           smem_barriers + smem_slots + smem_warpq;
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
        int64_t num_kv_splits_override) {  // kept for V1 API compat; unused
    (void)num_kv_splits_override;
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
    const int topk = static_cast<int>(topk64);
    // Sparse-only: honor a caller-provided cap in [topk, S).
    const int cand_cap = (cand_cap64 >= topk && cand_cap64 < seq_len_kv)
                             ? static_cast<int>(cand_cap64) : seq_len_kv;
    TORCH_CHECK(num_heads == NUM_HEADS && head_dim == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(kv.size(1) == HEAD_DIM, "kv D mismatch");
    TORCH_CHECK(origin.numel() == seq_len && inv_delta.numel() == seq_len && th_bucket.numel() == seq_len, "bucket params must have Q elements");
    const int num_buckets = static_cast<int>(num_buckets64);
    const bool external_refresh = (refresh_every64 < 0);
    const int refresh_every = external_refresh ? 0x7fffffff : static_cast<int>(refresh_every64);
    TORCH_CHECK(num_buckets >= 2 && num_buckets <= 4096, "num_buckets out of range");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1, cand_cap]");
    TORCH_CHECK(refresh_every64 >= -1, "refresh_every must be >= -1");
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

    const int esz_fp8 = 1, esz_f32 = 4;
    const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
    auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                        HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM, BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
    auto tm_kv = make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                         HEAD_DIM, seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
    auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                         ks_aligned, 1, BLOCK_KV, 1, 0, 0);
    auto tm_w = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
                        NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

    const int smem = compute_smem_bytes();
    auto kernel = &dsa_marsco_v2::sm100_dsa_marsco_v2<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_SMS, SPEC_THREADS, MATH_THREADS>;
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    dim3 grid((unsigned)NUM_SMS, 1, 1);  // persistent
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(), (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(), (uint32_t)cand_cap,
        tm_q, tm_kv, tm_ks, tm_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

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
          "DSA ReLU-MQA scoring V2 (DeepGEMM-2.5 persistent loop) with sparse candidate epilogue",
          pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
          pybind11::arg("weights"), pybind11::arg("cu_start"), pybind11::arg("cu_end"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("seed_val"), pybind11::arg("seed_idx"),
          pybind11::arg("num_buckets"), pybind11::arg("cand_cap"), pybind11::arg("topk"),
          pybind11::arg("refresh_every"), pybind11::arg("num_kv_splits")=-1);
    m.def("compact_topk_min_idx_marsco", &compact_topk_min_idx_marsco,
          "Qcount-aware top-k-min over compact candidates");
    m.def("compact_topk_min_thr_marsco", &compact_topk_min_thr_marsco,
          "Threshold-aware top-k-min select",
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("num_buckets"), pybind11::arg("topk"));
}

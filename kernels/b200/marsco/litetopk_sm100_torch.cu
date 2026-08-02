// PyTorch host binding for B200 / SM100 inner-product top-k.
//
// Ops:
//   litetopk_sm100::score_sparse(...) -- bucket-gated scan helper.
//   litetopk_sm100::dense_scores(...) -- dense score helper.
//   litetopk_sm100::fused_ip_sparse_b200(...) -- strided sampling, seed
//       parameter construction, SM100 sparse scan, boundary selection, and
//       score reconstruction.
//
// KV layout: contiguous per-head [Hkv, M, D] fp16 (== the benchmark k_cache;
// equals the paged NHD page_size=1 layout with identity group_indices, minus
// the interleaved V pages).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "sm100_litetopk_marsco.cuh"
#include "litetopk_topk.h"

#ifndef LITETOPK_CALLER_DIRECTLT_WORKSPACE
#define LITETOPK_CALLER_DIRECTLT_WORKSPACE 1
#endif
#ifndef LITETOPK_MARSCO_FP8_SCAN
#define LITETOPK_MARSCO_FP8_SCAN 0
#endif

namespace {

template <auto Kernel>
struct ScanKernelTag {
    static constexpr auto value = Kernel;
};

// cudaFuncSetAttribute is device-local state and is illegal during stream
// capture. Configure each concrete kernel specialization once per CUDA device
// during eager warmup. The mutex covers both attribute calls, and a device is
// cached only after both have succeeded.
template <auto Kernel>
static void configure_scan_kernel_once(
    int device, int smem, const char* label) {
    struct Cache {
        std::mutex mutex;
        std::unordered_map<int, int> smem_by_device;
    };
    static Cache cache;

    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto it = cache.smem_by_device.find(device);
    if (it != cache.smem_by_device.end()) {
        TORCH_CHECK(
            it->second == smem, label,
            " dynamic shared-memory size changed for one kernel/device: ",
            it->second, " -> ", smem);
        return;
    }

    const cudaError_t smem_error = cudaFuncSetAttribute(
        Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(
        smem_error == cudaSuccess, label,
        " cudaFuncSetAttribute(max dynamic smem=", smem,
        ") failed on device ", device, ": ",
        cudaGetErrorString(smem_error));
    const cudaError_t carveout_error = cudaFuncSetAttribute(
        Kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    TORCH_CHECK(
        carveout_error == cudaSuccess, label,
        " cudaFuncSetAttribute(max shared-memory carveout) failed on device ",
        device, ": ", cudaGetErrorString(carveout_error));
    cache.smem_by_device.emplace(device, smem);
}

using EncTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                              const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                              const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncTiled get_enc() {
    static EncTiled fn = [] {
        void* p = nullptr;
        cuGetProcAddress("cuTensorMapEncodeTiled", &p, 12000, CU_GET_PROC_ADDRESS_DEFAULT, nullptr);
        return reinterpret_cast<EncTiled>(p);
    }();
    return fn;
}

// 2D TMA descriptor: gmem [gmem_outer, gmem_inner] row-major, smem tile
// [smem_outer, swizzle_atom]. swizzle_mode in bytes.
static CUtensorMap make_2d_typed(
    void* ptr, CUtensorMapDataType data_type, int elem,
    int gmem_inner, long gmem_outer,
    int smem_inner, int smem_outer, int swizzle_mode) {
    if (swizzle_mode != 0) smem_inner = swizzle_mode / elem;
    if (smem_inner > gmem_inner) smem_inner = gmem_inner;
    CUtensorMap tm{};
    const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
    const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
    const cuuint64_t gstrides[1] = {(cuuint64_t)((long)gmem_inner * elem)};
    const cuuint32_t estrides[2] = {1, 1};
    CUtensorMapSwizzle sw =
        swizzle_mode == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle_mode == 64  ? CU_TENSOR_MAP_SWIZZLE_64B :
        swizzle_mode == 32  ? CU_TENSOR_MAP_SWIZZLE_32B : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult r = get_enc()(&tm, data_type, 2, ptr, gdims, gstrides,
                           sdims, estrides, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return tm;
}

static CUtensorMap make_2d_half(
    void* ptr, int gmem_inner, long gmem_outer,
    int smem_inner, int smem_outer, int swizzle_mode) {
    return make_2d_typed(
        ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, /*elem=*/2,
        gmem_inner, gmem_outer, smem_inner, smem_outer, swizzle_mode);
}

// TMA treats native E4M3 payload as opaque bytes; tcgen05 interprets the
// shared-memory bits according to its E4M3 instruction descriptor.
static CUtensorMap make_2d_fp8(
    void* ptr, int gmem_inner, long gmem_outer,
    int smem_inner, int smem_outer, int swizzle_mode) {
    return make_2d_typed(
        ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, /*elem=*/1,
        gmem_inner, gmem_outer, smem_inner, smem_outer, swizzle_mode);
}

// One block per row derives bucket parameters directly from the raw
// fp16 sample scores — pass 1 min/max (o = -max, hi = -min), pass 2 smem
// histogram -> th = bucket of the K-th best — and zeroes bcount/qcount in the
// same launch. There is no candidate prefill because the full scan includes
// sampled rows.
__global__ void litetopk_seed_params_kernel(
    const __half* __restrict__ scores, const int64_t stride,
    const int S, const int NB, const int K,
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket,
    int32_t* __restrict__ bcount, int32_t* __restrict__ qcount,
    int32_t* __restrict__ lt_cnt, int32_t* __restrict__ cand_cnt,
    int32_t* __restrict__ cta_done, const int n_groups
#if LITETOPK_MARSCO_FP8_SCAN
    ,
    const int th_slack) {
#else
    ) {
#endif
    constexpr int BT = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* srow = scores + (size_t)row * stride;

    __shared__ float s_mx[BT];
    __shared__ float s_mn[BT];
    float mx = -INFINITY, mn = INFINITY;
    const int S2 = S / 2 * 2;
    for (int j = tid * 2; j < S2; j += BT * 2) {
        const __half2 h2 = *reinterpret_cast<const __half2*>(srow + j);
        const float2 f2 = __half22float2(h2);
        mx = fmaxf(mx, fmaxf(f2.x, f2.y));
        mn = fminf(mn, fminf(f2.x, f2.y));
    }
    for (int j = S2 + tid; j < S; j += BT) {
        const float v = __half2float(srow[j]);
        mx = fmaxf(mx, v);
        mn = fminf(mn, v);
    }
    s_mx[tid] = mx;
    s_mn[tid] = mn;
    __syncthreads();
    for (int off = BT / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_mx[tid] = fmaxf(s_mx[tid], s_mx[tid + off]);
            s_mn[tid] = fminf(s_mn[tid], s_mn[tid + off]);
        }
        __syncthreads();
    }
    const float o = -s_mx[0];                    // min over x = -score
    const float hi = -s_mn[0];                   // max over x
    const float inv = (NB - 1) / fmaxf(hi - o, 1e-20f);

    extern __shared__ int s_hist[];              // NB ints
    for (int b = tid; b < NB; b += BT) s_hist[b] = 0;
    __syncthreads();
    for (int j = tid; j < S; j += BT) {
        // bucket the fp16-rounded x, same as the scan/select (x = -score is
        // exact in fp16: negation only flips the sign bit)
        const float x = -__half2float(srow[j]);
        int b = static_cast<int>((x - o) * inv);
        b = b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
        atomicAdd(&s_hist[b], 1);
    }
    __syncthreads();
    if (tid == 0) {
        const int kk = K < S ? K : S;
        int cum = 0, th = NB - 1;
        for (int b = 0; b < NB; ++b) {
            cum += s_hist[b];
            if (cum >= kk) { th = b; break; }
        }
#if LITETOPK_MARSCO_FP8_SCAN
        th_bucket[row] = min(th + th_slack, NB - 1);
#else
        th_bucket[row] = th;
#endif
        origin[row] = o;
        // Keep `inv` above for the sample histogram/threshold.  Only the
        // value published to the scan/select path is widened to the same
        // four-fp16-ULP floor as the former ATen chain:
        //   abs -> clamp_min(1) -> *2^-10 -> *4 -> reciprocal -> minimum.
        // In particular, a zero sample span still produces a finite stored
        // inverse even though the raw inverse used for histogramming is huge.
        const float ulp = fmaxf(fabsf(o), 1.0f) * (1.0f / 1024.0f);
        const float inv_max = 1.0f / (ulp * 4.0f);
        inv_delta[row] = fminf(inv, inv_max);
        qcount[row] = 0;
        lt_cnt[row] = 0;
        cand_cnt[row] = 0;
        if (row < n_groups) cta_done[row] = 0;
    }
    for (int b = tid; b < NB; b += BT)
        bcount[(size_t)row * NB + b] = 0;
}

constexpr int QN = 8;
constexpr int BM = 128;
// Fixed KV pipeline depth used by the JIT loader and direct builds.
#ifndef LITETOPK_KV_STAGES
#define LITETOPK_KV_STAGES 6
#endif
constexpr int NUM_KV_STAGES = LITETOPK_KV_STAGES;
constexpr int NUM_SMS = 148;       // B200
// One TMA warp and one UMMA warp; threshold refresh runs on math warps.
constexpr int SPEC_THREADS = 64;
constexpr int MATH_THREADS = 128;  // 1 math warpgroup, BM = 128 (one kv row per lane)

static int stages_for(int head_dim, int bm) {
    if (bm > 128) return 3;                 // 64KB chunk stages
    return head_dim <= 128 ? NUM_KV_STAGES : 3;
}

static int compute_smem_bytes(
    int head_dim, int qn, int bm, bool out_fp32 = false,
    bool fp8_input = false) {
    const int cand_b = out_fp32 ? 4 : 2;   // queue value entries follow cand_t
    const int input_b = fp8_input ? 1 : 2;
    // Mirrors the kernel's constexpr layout (CHUNK_D, warp queue, coeff tables).
    int chunk_d = qn > 8 ? (head_dim > 128 ? 128 : head_dim)
                         : (head_dim > 256 ? 256 : head_dim);
    const int cap = (qn > 8 ? 32768 : 65536) / (bm * input_b);
    if (chunk_d > cap) chunk_d = cap;
    const int smem_q = qn * head_dim * input_b;
    const int smem_kv = bm * chunk_d * input_b;
    // KV full/empty per stage + UMMA full/empty per (warpgroup, acc buffer) + q_ready
    const int num_barriers = stages_for(head_dim, bm) * 2 + (bm / 128) * 2 * 2 + 1;
    const int smem_barriers = num_barriers * 8;
    const int smem_misc = 4 * (int)sizeof(uint32_t);  // tmem slot + flags
    // Warp-local candidate queue: QN <= 8 only (kUseWarpQueue in the kernel).
    const int smem_warpq = qn <= 8
        ? (bm / 32) * qn * ((int)sizeof(int32_t) + LITETOPK_WARP_QUEUE_CAP * (cand_b + (int)sizeof(int32_t)))
        : 0;
    const int smem_coeff =
        qn * (2 * (int)sizeof(float) + (int)sizeof(int32_t)) +
        (fp8_input ? qn * (int)sizeof(float) : 0);
    return smem_q + stages_for(head_dim, bm) * smem_kv + smem_barriers + smem_misc + smem_warpq + smem_coeff;
}

#define LITETOPK_DISPATCH_D(head_dim, ...)                                           \
    do {                                                                          \
        if (head_dim == 64)  {                                                    \
            if constexpr (!kFp8Input) { constexpr int DD = 64; __VA_ARGS__; }      \
            else TORCH_CHECK(false, "fp8 scan requires head_dim multiple of 128"); \
        }                                                                         \
        else if (head_dim == 128) { constexpr int DD = 128; __VA_ARGS__; }        \
        else if (head_dim == 256) { constexpr int DD = 256; __VA_ARGS__; }        \
        else if (head_dim == 512) { constexpr int DD = 512; __VA_ARGS__; }        \
        else if (head_dim == 768) { constexpr int DD = 768; __VA_ARGS__; }        \
        else TORCH_CHECK(false, "unsupported head_dim ", head_dim);               \
    } while (0)

#if LITETOPK_MARSCO_FP8_SCAN
#define LITETOPK_SCAN_INPUT_ARG(input_type) , input_type
#else
#define LITETOPK_SCAN_INPUT_ARG(input_type)
#endif

// Launch the SM100 sparse scan: score kv rows [start_row, scan_end) of the
// contiguous [hkv, kv_row_stride, D] cache against xpad [n_head_groups*QN, D].
template <typename input_t>
static void launch_sm100_scan_impl(
    const at::Tensor& xpad, const at::Tensor& kv_cont,
    int n_head_groups, int q_group_size, int logical_rows,
    int scan_end, int start_row,
    const at::Tensor& origin, const at::Tensor& inv_delta, const at::Tensor& th,
    const at::Tensor& qcount, const at::Tensor& bcount,
    const at::Tensor& buf_val, const at::Tensor& buf_idx,
    int buf_cap, int num_buckets, int topk, int refresh_every, int num_ctas_x,
    cudaStream_t stream, int32_t* cta_done = nullptr, __half* dense_out = nullptr,
    int qn = 8, int bm = 128, bool out_fp32 = false,
    bool store_bucket_space = true,
    const at::Tensor* q_dequant_scale = nullptr,
    const at::Tensor* kv_dequant_scale = nullptr) {
    constexpr bool kFp8Input =
        std::is_same_v<input_t, __nv_fp8_e4m3>;
    const int head_dim = xpad.size(1);
    const int hkv = kv_cont.size(0);
    const long kv_row_stride = kv_cont.size(1);

    auto tm_q = kFp8Input
        ? make_2d_fp8(
              xpad.data_ptr(), head_dim, xpad.size(0), head_dim, qn, 128)
        : make_2d_half(
              xpad.data_ptr(), head_dim, xpad.size(0), head_dim, qn, 128);
    auto tm_kv = kFp8Input
        ? make_2d_fp8(
              kv_cont.data_ptr(), head_dim,
              (long)hkv * kv_row_stride, head_dim, bm, 128)
        : make_2d_half(
              kv_cont.data_ptr(), head_dim,
              (long)hkv * kv_row_stride, head_dim, bm, 128);
    const float* q_scale_ptr =
        q_dequant_scale ? q_dequant_scale->data_ptr<float>() : nullptr;
    const float* kv_scale_ptr =
        kv_dequant_scale ? kv_dequant_scale->data_ptr<float>() : nullptr;

    if (num_ctas_x <= 0) {
        // Persistent tile-strided CTA grid.
        num_ctas_x = std::max(1, 8 * NUM_SMS / std::max(1, n_head_groups));
    }
    const int scan_tiles = (scan_end - start_row + bm - 1) / bm;
    if (scan_tiles > 0 && num_ctas_x > scan_tiles) num_ctas_x = scan_tiles;
    if (num_ctas_x < 1) num_ctas_x = 1;

    LITETOPK_DISPATCH_D(head_dim, {
        constexpr int STAGES = DD <= 128 ? NUM_KV_STAGES : 3;
        auto dispatch = [&](auto kernel_tag, auto* buf_ptr) {
            constexpr auto kernel = decltype(kernel_tag)::value;
            int smem =
                compute_smem_bytes(DD, qn, bm, out_fp32, kFp8Input);
            configure_scan_kernel_once<kernel>(
                xpad.get_device(), smem, "sm100 scan");
            // x = head groups (co-resident groups share L2 on the same tiles), y = tile stride
            dim3 grid(n_head_groups, num_ctas_x, 1);
            kernel<<<grid, SPEC_THREADS + (bm == 256 ? 256 : MATH_THREADS), smem, stream>>>(
                (uint32_t)scan_end, (uint32_t)kv_row_stride, (uint32_t)q_group_size,
                (uint32_t)logical_rows, (uint32_t)start_row,
                origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
                qcount.data_ptr<int>(), bcount.data_ptr<int>(), cta_done, dense_out,
                buf_ptr, buf_idx.data_ptr<int>(),
                (uint32_t)buf_cap, (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
                store_bucket_space,
#if LITETOPK_MARSCO_FP8_SCAN
                q_scale_ptr, kv_scale_ptr,
#endif
                tm_q, tm_kv);
            cudaError_t le = cudaGetLastError();
            TORCH_CHECK(le == cudaSuccess, "launch failed (smem=", smem, "): ", cudaGetErrorString(le));
        };
        if (out_fp32) {
            auto* bp = buf_val.data_ptr<float>();
            if (bm == 256) {
                if (qn == 64) dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<64, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, float LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
                else          dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<QN, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, float LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
            } else {
                if (qn == 64) dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<64, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, float LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
                else          dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<QN, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, float LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
            }
        } else {
            auto* bp = reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>());
            if (bm == 256) {
                if (qn == 64) dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<64, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, __half LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
                else          dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<QN, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, __half LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
            } else {
                if (qn == 64) dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<64, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, __half LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
                else          dispatch(ScanKernelTag<&litetopk_marsco::sm100_litetopk_ip<QN, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, __half LITETOPK_SCAN_INPUT_ARG(input_t)>>{}, bp);
            }
        }
    });
}

static void launch_sm100_scan(
    const at::Tensor& xpad, const at::Tensor& kv_cont,
    int n_head_groups, int q_group_size, int logical_rows,
    int scan_end, int start_row,
    const at::Tensor& origin, const at::Tensor& inv_delta,
    const at::Tensor& th, const at::Tensor& qcount,
    const at::Tensor& bcount, const at::Tensor& buf_val,
    const at::Tensor& buf_idx, int buf_cap, int num_buckets, int topk,
    int refresh_every, int num_ctas_x, cudaStream_t stream,
    int32_t* cta_done = nullptr, __half* dense_out = nullptr,
    int qn = 8, int bm = 128, bool out_fp32 = false,
    bool store_bucket_space = true) {
    launch_sm100_scan_impl<__half>(
        xpad, kv_cont, n_head_groups, q_group_size, logical_rows,
        scan_end, start_row, origin, inv_delta, th, qcount, bcount,
        buf_val, buf_idx, buf_cap, num_buckets, topk, refresh_every,
        num_ctas_x, stream, cta_done, dense_out, qn, bm, out_fp32,
        store_bucket_space);
}

#if LITETOPK_MARSCO_FP8_SCAN
static void launch_sm100_scan_fp8(
    const at::Tensor& xpad, const at::Tensor& kv_cont,
    const at::Tensor& q_dequant_scale,
    const at::Tensor& kv_dequant_scale,
    int n_head_groups, int q_group_size, int logical_rows,
    int scan_end, int start_row,
    const at::Tensor& origin, const at::Tensor& inv_delta,
    const at::Tensor& th, const at::Tensor& qcount,
    const at::Tensor& bcount, const at::Tensor& buf_val,
    const at::Tensor& buf_idx, int buf_cap, int num_buckets, int topk,
    int refresh_every, int num_ctas_x, cudaStream_t stream,
    int32_t* cta_done = nullptr, int qn = 8, int bm = 128,
    bool out_fp32 = false, bool store_bucket_space = true) {
    // Keep the experimental binary surface deliberately narrow. MS MARCO is
    // D=768 with 64 concurrent queries; a single 64x256 specialization avoids
    // adding dozens of unmeasured fp8 kernels to the production extension.
    TORCH_CHECK(
        xpad.size(1) == 768 && qn == 64 && bm == 256 && !out_fp32,
        "experimental fp8 scan requires D=768, qn=64, bm=256, "
        "out_fp32=False");
    TORCH_CHECK(logical_rows == 64, "fp8 scan requires 64 valid rows");
    const long kv_row_stride = kv_cont.size(1);
    auto tm_q = make_2d_fp8(
        xpad.data_ptr(), /*gmem_inner=*/768, xpad.size(0),
        /*smem_inner=*/768, /*smem_outer=*/64, /*swizzle=*/128);
    auto tm_kv = make_2d_fp8(
        kv_cont.data_ptr(), /*gmem_inner=*/768,
        (long)kv_cont.size(0) * kv_row_stride,
        /*smem_inner=*/768, /*smem_outer=*/256, /*swizzle=*/128);
    if (num_ctas_x <= 0) {
        num_ctas_x =
            std::max(1, 8 * NUM_SMS / std::max(1, n_head_groups));
    }
    const int scan_tiles = (scan_end - start_row + 255) / 256;
    if (scan_tiles > 0 && num_ctas_x > scan_tiles)
        num_ctas_x = scan_tiles;
    if (num_ctas_x < 1) num_ctas_x = 1;

    using kernel_t = decltype(
        &litetopk_marsco::sm100_litetopk_ip<
            64, 768, 256, 3, NUM_SMS, SPEC_THREADS, 256, __half,
            __nv_fp8_e4m3>);
    kernel_t kernel =
        &litetopk_marsco::sm100_litetopk_ip<
            64, 768, 256, 3, NUM_SMS, SPEC_THREADS, 256, __half,
            __nv_fp8_e4m3>;
    const int smem = compute_smem_bytes(
        /*head_dim=*/768, /*qn=*/64, /*bm=*/256,
        /*out_fp32=*/false, /*fp8_input=*/true);
    configure_scan_kernel_once<
        &litetopk_marsco::sm100_litetopk_ip<
            64, 768, 256, 3, NUM_SMS, SPEC_THREADS, 256, __half,
            __nv_fp8_e4m3>>(
        xpad.get_device(), smem, "sm100 fp8 scan");
    dim3 grid(n_head_groups, num_ctas_x, 1);
    kernel<<<grid, SPEC_THREADS + 256, smem, stream>>>(
        (uint32_t)scan_end, (uint32_t)kv_row_stride,
        (uint32_t)q_group_size, (uint32_t)logical_rows,
        (uint32_t)start_row, origin.data_ptr<float>(),
        inv_delta.data_ptr<float>(), th.data_ptr<int>(),
        qcount.data_ptr<int>(), bcount.data_ptr<int>(), cta_done,
        /*dense_out=*/nullptr,
        reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
        buf_idx.data_ptr<int>(), (uint32_t)buf_cap,
        (uint32_t)num_buckets, (uint32_t)topk,
        (uint32_t)refresh_every, store_bucket_space,
        q_dequant_scale.data_ptr<float>(),
        kv_dequant_scale.data_ptr<float>(), tm_q, tm_kv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

// ---------------- correctness probe op (scan only, open gate) ----------------
std::tuple<at::Tensor, at::Tensor, at::Tensor> litetopk_sm100_score_sparse(
    const at::Tensor& q_cont, const at::Tensor& kv_cont,
    const at::Tensor& origin, const at::Tensor& inv_delta, at::Tensor& th,
    int64_t q_group_size, int64_t logical_rows, int64_t start_row,
    int64_t num_buckets, int64_t topk, int64_t refresh_every, int64_t buf_cap,
    int64_t num_ctas_x) {
    TORCH_CHECK(q_cont.is_cuda() && kv_cont.is_cuda(), "cuda tensors required");
    TORCH_CHECK(
        origin.is_cuda() && inv_delta.is_cuda() && th.is_cuda(),
        "origin/inv_delta/th must be CUDA tensors");
    TORCH_CHECK(
        q_cont.device() == kv_cont.device() &&
            q_cont.device() == origin.device() &&
            q_cont.device() == inv_delta.device() &&
            q_cont.device() == th.device(),
        "all score_sparse tensors must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(q_cont.device());
    TORCH_CHECK(q_cont.dim() == 2 && kv_cont.dim() == 3, "q_cont [Rwork,D], kv_cont [hkv,M,D]");
    TORCH_CHECK(
        q_cont.is_contiguous() && kv_cont.is_contiguous(),
        "q_cont/kv_cont must be contiguous");
    TORCH_CHECK(q_cont.scalar_type() == at::kHalf && kv_cont.scalar_type() == at::kHalf, "fp16");
    TORCH_CHECK(
        origin.scalar_type() == at::kFloat &&
            inv_delta.scalar_type() == at::kFloat &&
            th.scalar_type() == at::kInt,
        "origin/inv_delta must be float32 and th must be int32");
    TORCH_CHECK(
        origin.is_contiguous() && inv_delta.is_contiguous() &&
            th.is_contiguous(),
        "origin/inv_delta/th must be contiguous");
    const int64_t Rwork64 = q_cont.size(0);
    const int64_t M64 = kv_cont.size(1);
    TORCH_CHECK(
        Rwork64 >= QN && Rwork64 % QN == 0,
        "q_cont.shape[0] must be a positive multiple of ", QN);
    TORCH_CHECK(
        kv_cont.size(0) >= 1 && M64 >= 1,
        "kv_cont must have at least one head and one row");
    TORCH_CHECK(
        origin.dim() == 1 && origin.size(0) == Rwork64 &&
            inv_delta.dim() == 1 && inv_delta.size(0) == Rwork64 &&
            th.dim() == 1 && th.size(0) == Rwork64,
        "origin/inv_delta/th must each have shape [Rwork]");
    TORCH_CHECK(
        q_group_size >= 1 && logical_rows >= 1 && logical_rows <= QN,
        "require q_group_size >= 1 and 1 <= logical_rows <= ", QN);
    TORCH_CHECK(
        start_row >= 0 && start_row <= M64,
        "start_row must be in [0, M]");
    TORCH_CHECK(
        num_buckets >= 2 && num_buckets <= 4096,
        "num_buckets out of range");
    TORCH_CHECK(topk >= 1 && topk <= M64, "require 1 <= topk <= M");
    // This low-level scan/overflow probe intentionally permits buf_cap < topk:
    // qcount reports attempted candidates while stores truncate safely at cap.
    // The public fused top-k op separately enforces buf_cap >= k.
    TORCH_CHECK(buf_cap >= 1 && buf_cap <= M64, "require 1 <= buf_cap <= M");
    const int Rwork = static_cast<int>(Rwork64);
    const int M = static_cast<int>(M64);
    TORCH_CHECK(kv_cont.size(2) == q_cont.size(1), "D mismatch");
    const int n_head_groups = Rwork / QN;
    TORCH_CHECK(
        (n_head_groups - 1) / q_group_size < kv_cont.size(0),
        "q_group_size does not map every query group to a KV head");

    auto opt_h = q_cont.options();
    auto opt_i = q_cont.options().dtype(at::kInt);
    auto buf_val = at::empty({Rwork, buf_cap}, opt_h);
    auto buf_idx = at::empty({Rwork, buf_cap}, opt_i);
    auto qcount = at::zeros({Rwork}, opt_i);
    auto bcount = at::zeros({Rwork, num_buckets}, opt_i);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_sm100_scan(q_cont, kv_cont, n_head_groups, (int)q_group_size, (int)logical_rows,
                      M, (int)start_row, origin, inv_delta, th, qcount, bcount,
                      buf_val, buf_idx, (int)buf_cap, (int)num_buckets, (int)topk,
                      (int)refresh_every, (int)num_ctas_x, stream);
    return std::make_tuple(buf_val, buf_idx, qcount);
}

// ---------------- dense score helper ----------------
// Writes dense [Hq, M] scores; the caller performs top-k selection.
at::Tensor litetopk_sm100_dense_scores(const at::Tensor& x, const at::Tensor& kv_cont,
                                    int64_t num_ctas_x) {
    TORCH_CHECK(x.is_cuda() && kv_cont.is_cuda(), "x/kv must be CUDA");
    TORCH_CHECK(
        x.device() == kv_cont.device(),
        "x/kv must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.dim() == 2 && kv_cont.dim() == 3, "x [Hq,D], kv [Hkv,M,D]");
    TORCH_CHECK(kv_cont.is_contiguous(), "kv_cont must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf && kv_cont.scalar_type() == at::kHalf, "fp16");
    const int64_t Hq = x.size(0);
    const int64_t D = x.size(1);
    const int64_t Hkv = kv_cont.size(0);
    const int64_t M = kv_cont.size(1);
    TORCH_CHECK(Hkv >= 1, "kv_cont must have at least one KV head");
    TORCH_CHECK(Hq >= 1 && M >= 1, "x and kv_cont must have non-empty row dimensions");
    TORCH_CHECK(kv_cont.size(2) == D && Hq % Hkv == 0, "shape mismatch");
    const int64_t group = Hq / Hkv;
    const bool flat_batch = (Hkv == 1 && group > QN);
    static const int kFlatQnD = [] {
        const char* e = getenv("LITETOPK_FLAT_QN");
        return (e && atoi(e) == 64) ? 64 : 8;
    }();
    static const int kBmD = [] {
        const char* e = getenv("LITETOPK_BM");
        return (e && atoi(e) == 256) ? 256 : 128;
    }();
    const int64_t qn = (flat_batch && kFlatQnD == 64 && Hq % 64 == 0) ? 64 : QN;
    TORCH_CHECK(group <= QN || flat_batch, "GQA group size must be in [1, ", QN, "] (or Hkv==1 flat batch)");
    TORCH_CHECK(!flat_batch || Hq % qn == 0, "flat batch requires N to be a multiple of ", qn);
    const int64_t n_groups = flat_batch ? (Hq / qn) : Hkv;
    const int64_t q_group_size = flat_batch ? n_groups : 1;
    const int64_t rows_valid = flat_batch ? qn : group;
    const int64_t Rwork = n_groups * qn;

    at::Tensor xpad;
    if (flat_batch) {
        xpad = x.contiguous();
    } else {
        xpad = at::zeros({Rwork, D}, x.options());
        xpad.view({Hkv, QN, D}).slice(1, 0, group).copy_(x.view({Hkv, group, D}));
    }

    auto opt_i = x.options().dtype(at::kInt);
    auto opt_f = x.options().dtype(at::kFloat);
    auto dense = at::empty({Hq, M}, x.options());
    // Dummies for the unused sparse-path parameters (read but never useful).
    auto origin = at::zeros({Rwork}, opt_f);
    auto inv_delta = at::zeros({Rwork}, opt_f);
    auto th = at::zeros({Rwork}, opt_i);
    auto qcount = at::zeros({Rwork}, opt_i);
    auto bcount = at::zeros({Rwork, 2}, opt_i);
    auto buf_idx = at::zeros({1}, opt_i);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_sm100_scan(xpad, kv_cont, (int)n_groups, (int)q_group_size, (int)rows_valid,
                      (int)M, /*start_row=*/0,
                      origin, inv_delta, th, qcount, bcount,
                      /*buf_val=*/dense, buf_idx,
                      /*buf_cap=*/(int)M, /*num_buckets=*/2, /*topk=*/1,
                      /*refresh_every=*/0, (int)num_ctas_x, stream,
                      /*cta_done=*/nullptr,
                      reinterpret_cast<__half*>(dense.data_ptr<at::Half>()), (int)qn, kBmD);
    return dense;
}

// ---------------- fused sparse pipeline ----------------
static std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_ip_sparse_b200_impl(
    const at::Tensor& x, const at::Tensor& kv_cont,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t sample_size,
    int64_t refresh_every, int64_t num_ctas_x, int64_t sample_mode,
    int64_t qn_arg, int64_t bm_arg, bool out_fp32) {
    TORCH_CHECK(x.is_cuda() && kv_cont.is_cuda(), "x/kv must be CUDA");
    TORCH_CHECK(
        x.device() == kv_cont.device(),
        "x/kv must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.dim() == 2 && kv_cont.dim() == 3, "x [Hq,D], kv [Hkv,M,D]");
    TORCH_CHECK(x.is_contiguous() && kv_cont.is_contiguous(), "x/kv must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf && kv_cont.scalar_type() == at::kHalf, "fp16 required");
    TORCH_CHECK(qn_arg == 0 || qn_arg == 8 || qn_arg == 64,
                "qn must be 0, 8, or 64");
    TORCH_CHECK(bm_arg == 0 || bm_arg == 128 || bm_arg == 256,
                "bm must be 0, 128, or 256");
    const int64_t Hq = x.size(0);
    const int64_t D = x.size(1);
    const int64_t Hkv = kv_cont.size(0);
    const int64_t M = kv_cont.size(1);
    TORCH_CHECK(kv_cont.size(2) == D, "head_dim mismatch");
    TORCH_CHECK(Hkv >= 1 && Hq >= Hkv && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
    const int64_t group = Hq / Hkv;
    // MS MARCO uses flat batch: Hkv == 1 and Hq concurrent queries share one
    // corpus. The kernel maps every query group to KV head 0.
    const bool flat_batch = (Hkv == 1 && group > QN);
    // Flat batch supports QN=8 and QN=64 execution shapes.
    static const int kFlatQn = [] {
        const char* e = getenv("LITETOPK_FLAT_QN");
        return (e && atoi(e) == 64) ? 64 : 8;
    }();
    static const int kBm = [] {
        const char* e = getenv("LITETOPK_BM");
        return (e && atoi(e) == 256) ? 256 : 128;
    }();
    // Explicit args override the env defaults (0 = use env/auto).
    const int eff_flat_qn = qn_arg == 64 ? 64 : (qn_arg == 8 ? 8 : kFlatQn);
    const int eff_bm = bm_arg == 256 ? 256 : (bm_arg == 128 ? 128 : kBm);
    const int64_t qn = (flat_batch && eff_flat_qn == 64 && Hq % 64 == 0) ? 64 : QN;
    TORCH_CHECK(flat_batch, "flat-batch mode requires Hkv==1 and N (=Hq=", Hq,
                ") > QN=", QN);
    TORCH_CHECK(Hq % qn == 0, "flat batch requires N to be a multiple of ", qn);
    TORCH_CHECK(M % 64 == 0, "M must be a multiple of 64");
    TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
    TORCH_CHECK(num_buckets >= 2 && num_buckets <= 4096, "num_buckets out of range");

    TORCH_CHECK(sample_size >= 1, "sample_size must be positive");
    if (sample_size > M) sample_size = M;
    sample_size =
        (sample_size / 64 + (sample_size % 64 != 0)) * 64;
    if (sample_size < k)
        sample_size = (k / 64 + (k % 64 != 0)) * 64;
    if (sample_size > M) sample_size = M;
    TORCH_CHECK(sample_mode == 1,
                "sample_mode must be 1 (strided corpus sampling)");
    const int64_t scan_end = M;
    if (buf_cap < k) buf_cap = k;
    if (buf_cap > M) buf_cap = M;
    const int64_t candidate_cap = buf_cap;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t n_groups = flat_batch ? (Hq / qn) : Hkv;
    const int64_t q_group_size = flat_batch ? n_groups : 1;
    const int64_t rows_valid = flat_batch ? qn : group;
    const int64_t Rwork = n_groups * qn;

    // Every flat-batch query row is real, so no padding is required.
    at::Tensor xpad = x.contiguous();

    // 1) Score strided sample rows.
    const int64_t stride = M / sample_size;
    // This slice selects exactly row i*stride, matching the former
    // arange*stride + index_select path.  Its transposed [D,sample] view has
    // strides [1,stride*D], which cuBLAS consumes directly as a column-major
    // matrix with a padded leading dimension; no sample gather is required.
    auto kv_sample =
        kv_cont.slice(/*dim=*/1, /*start=*/0, sample_size * stride, stride);
    // raw scores (no neg): the fused prep kernel handles x = -score internally.
    at::Tensor sample_scores =
        at::matmul(xpad, kv_sample.select(0, 0).transpose(0, 1));
    auto sample_flat = sample_scores.view({Rwork, sample_size});         // GEMM output is contiguous

    auto opt_h = x.options();
    auto opt_i = x.options().dtype(at::kInt);
    auto opt_f = x.options().dtype(at::kFloat);
    // fp32 output path: candidates, select workspaces and outputs in fp32 —
    // also removes the fp16 bucket-rounding coupling entirely (scan histogram
    // and select both operate on the exact fp32 value).
    auto opt_c = out_fp32 ? opt_f : opt_h;
    auto origin = at::empty({Rwork}, opt_f);
    auto inv_delta = at::empty({Rwork}, opt_f);
    auto th = at::empty({Rwork}, opt_i);
    auto qcount = at::empty({Rwork}, opt_i);
    auto bcount = at::empty({Rwork, num_buckets}, opt_i);
    auto buf_val = at::empty({Rwork, buf_cap}, opt_c);
    auto buf_idx = at::empty({Rwork, buf_cap}, opt_i);
    auto val_pad = at::empty({Rwork, k}, opt_c);
    auto idx_pad = at::empty({Rwork, k}, opt_i);
    auto cand_val = at::empty({Rwork, candidate_cap}, opt_c);
    auto cand_idx = at::empty({Rwork, candidate_cap}, opt_i);
    auto cand_cnt = at::empty({Rwork}, opt_i);
    auto lt_val = at::empty({Rwork, candidate_cap}, opt_c);
    auto lt_idx = at::empty({Rwork, candidate_cap}, opt_i);
    auto lt_cnt = at::empty({Rwork}, opt_i);
    auto cta_done = at::empty({n_groups}, opt_i);
    // Direct-lt finalize needs three temporary arrays. Own them at the
    // fused-op level so PyTorch's caching allocator can reuse the storage and
    // the selector emits no cudaMallocAsync/cudaFreeAsync nodes.
    const size_t direct_part_slots =
#if LITETOPK_CALLER_DIRECTLT_WORKSPACE
        flash_topk_select_thr_mb_direct_part_slots(
            (int)Rwork, (int)buf_cap, (int)k, (int)candidate_cap);
#else
        0;
#endif
    at::Tensor part_val;
    at::Tensor part_idx;
    at::Tensor part_cnt;
    if (direct_part_slots != 0) {
        const int64_t slots = static_cast<int64_t>(direct_part_slots);
        part_val = at::empty({slots, k}, opt_c);
        part_idx = at::empty({slots, k}, opt_i);
        part_cnt = at::empty({slots}, opt_i);
    }

    // 2) Build per-row bucket parameters and reset every counter consumed by
    // the scan/select launches.  The seed grid has one CTA per Rwork row, so
    // it covers lt/cand exactly once and its first n_groups CTAs cover
    // cta_done.  Same-stream ordering makes all stores visible before scan.
    litetopk_seed_params_kernel<<<(int)Rwork, 256, num_buckets * sizeof(int), stream>>>(
        reinterpret_cast<const __half*>(sample_flat.data_ptr<at::Half>()),
        sample_size, (int)sample_size, (int)num_buckets, (int)k,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
        bcount.data_ptr<int>(), qcount.data_ptr<int>(),
        lt_cnt.data_ptr<int>(), cand_cnt.data_ptr<int>(),
        cta_done.data_ptr<int>(), (int)n_groups
#if LITETOPK_MARSCO_FP8_SCAN
        , /*th_slack=*/0
#endif
        );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Bucket-width floor: the select stage re-derives buckets from fp16
    // candidate values, so the bucket width (seed span / NB) must stay at
    // least a few fp16 ULPs of the score magnitude or scan(fp32)/select(fp16)
    // classifications can diverge. litetopk_seed_params_kernel applies this
    // clamp only when publishing inv_delta, after using the raw inverse for
    // its sample histogram and threshold. This removes seven tiny ATen
    // launches from every call while preserving the seed classification.

    // 3) SM100 UMMA/TMEM sparse scan over the full corpus.
    //    refresh_every <= 0 selects sentinel mode: bcount is histogrammed
    //    (0x7fffffff sentinel) and the threshold refreshed once post-scan.
    //    (The histogram is NOT optional: the thr select direct-copies the
    //    bucket<th set and is only exact when the post-scan refresh has
    //    tightened th so that cnt(bucket<th) < k.)
    const int scan_refresh = refresh_every > 0 ? (int)refresh_every : 0x7fffffff;
    // qcount/bcount/cta_done/lt_cnt/cand_cnt zeroing is fused into
    // litetopk_seed_params_kernel.
    if (scan_end > 0) {
        // The last CTA of each head group performs the final threshold
        // update after all candidate histograms are visible.
        launch_sm100_scan(xpad, kv_cont, (int)n_groups, (int)q_group_size, (int)rows_valid,
                          (int)scan_end, /*start_row=*/0,
                          origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
                          (int)buf_cap, (int)num_buckets, (int)k, scan_refresh,
                          (int)num_ctas_x, stream, cta_done.data_ptr<int>(),
                          /*dense_out=*/nullptr, (int)qn, eff_bm, out_fp32,
                          /*store_bucket_space=*/true);
    }

    // 4) Boundary radix select over compact candidates. buf_val / cand_val /
    // lt_val / val_pad all hold BUCKET-SPACE coordinates bq = (x-o)*inv end to
    // end (scan stores bq directly, select never reconstructs x — see the
    // scan epilogue and flashtopk_compact_thr_kernel). bq is an affine
    // (order-preserving) transform of x, so every comparison/sort inside the
    // select machinery is unaffected. Final selector stores alone convert the
    // K winners back to IP, avoiding a host-side affine over [Rwork,K].
    if (out_fp32) {
        launch_flash_topk_select_thr_mb_idx_fp32(
            buf_val.data_ptr<float>(),
            buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)Rwork, (int)buf_cap, (int)k, (int)candidate_cap,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            lt_val.data_ptr<float>(), lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            val_pad.data_ptr<float>(), idx_pad.data_ptr<int>(), stream,
            /*debucket_output=*/true, /*counters_preinitialized=*/true,
            direct_part_slots ? part_val.data_ptr<float>() : nullptr,
            direct_part_slots ? part_idx.data_ptr<int>() : nullptr,
            direct_part_slots ? part_cnt.data_ptr<int>() : nullptr);
    } else {
        launch_flash_topk_select_thr_mb_idx_fp16(
            reinterpret_cast<const __half*>(buf_val.data_ptr<at::Half>()),
            buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)Rwork, (int)buf_cap, (int)k, (int)candidate_cap,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
            cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
            lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
            idx_pad.data_ptr<int>(), /*coords_fp16=*/false, stream, /*skip_zero=*/0,
            /*bucket_space=*/true, /*debucket_output=*/true,
            /*counters_preinitialized=*/true,
            direct_part_slots
                ? reinterpret_cast<__half*>(part_val.data_ptr<at::Half>())
                : nullptr,
            direct_part_slots ? part_idx.data_ptr<int>() : nullptr,
            direct_part_slots ? part_cnt.data_ptr<int>() : nullptr);
    }
    auto val = val_pad.view({n_groups, qn, k}).slice(1, 0, rows_valid).reshape({Hq, k}).contiguous();
    auto idx = idx_pad.view({n_groups, qn, k}).slice(1, 0, rows_valid).reshape({Hq, k}).contiguous();
    return std::make_tuple(val, idx, qcount);
}

std::tuple<at::Tensor, at::Tensor> fused_ip_sparse_b200(
    const at::Tensor& x, const at::Tensor& kv_cont,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t sample_size,
    int64_t refresh_every, int64_t num_ctas_x, int64_t sample_mode,
    int64_t qn_arg, int64_t bm_arg, bool out_fp32) {
    auto result = fused_ip_sparse_b200_impl(
        x, kv_cont, k, num_buckets, buf_cap, sample_size, refresh_every,
        num_ctas_x, sample_mode, qn_arg, bm_arg, out_fp32);
    return std::make_tuple(std::get<0>(result), std::get<1>(result));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_ip_sparse_b200_with_counts(
    const at::Tensor& x, const at::Tensor& kv_cont,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t sample_size,
    int64_t refresh_every, int64_t num_ctas_x, int64_t sample_mode,
    int64_t qn_arg, int64_t bm_arg, bool out_fp32) {
    return fused_ip_sparse_b200_impl(
        x, kv_cont, k, num_buckets, buf_cap, sample_size, refresh_every,
        num_ctas_x, sample_mode, qn_arg, bm_arg, out_fp32);
}

#if LITETOPK_MARSCO_FP8_SCAN
// Experimental, default-off coarse E4M3 scan. The full fp16 corpus remains
// authoritative: the fp8 kernel only proposes `coarse_k` corpus ids, then a
// gather+BMM over the original fp16 tensors re-scores those ids and returns
// the final K. Rowwise scales use dequantized_value = fp8_value * scale.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
fused_ip_sparse_fp8_b200(
    const at::Tensor& x_ref, const at::Tensor& x_fp8,
    const at::Tensor& x_scale, const at::Tensor& kv_ref,
    const at::Tensor& kv_fp8, const at::Tensor& kv_scale,
    int64_t k, int64_t coarse_k, int64_t num_buckets,
    int64_t buf_cap, int64_t sample_size, int64_t refresh_every,
    int64_t num_ctas_x, int64_t sample_mode, bool rerank) {
    TORCH_CHECK(
        x_ref.is_cuda() && x_fp8.is_cuda() && x_scale.is_cuda() &&
            kv_ref.is_cuda() && kv_fp8.is_cuda() && kv_scale.is_cuda(),
        "all fp8 scan tensors must be CUDA tensors");
    TORCH_CHECK(
        x_ref.device() == x_fp8.device() &&
            x_ref.device() == x_scale.device() &&
            x_ref.device() == kv_ref.device() &&
            x_ref.device() == kv_fp8.device() &&
            x_ref.device() == kv_scale.device(),
        "all fp8 scan tensors must be on the same CUDA device");
    const c10::cuda::CUDAGuard device_guard(x_ref.device());
    TORCH_CHECK(
        x_ref.dim() == 2 && x_fp8.dim() == 2 &&
            kv_ref.dim() == 3 && kv_fp8.dim() == 3,
        "x_ref/x_fp8 [B,D], kv_ref/kv_fp8 [1,M,D]");
    TORCH_CHECK(
        x_ref.is_contiguous() && x_fp8.is_contiguous() &&
            x_scale.is_contiguous() && kv_ref.is_contiguous() &&
            kv_fp8.is_contiguous() && kv_scale.is_contiguous(),
        "all fp8 scan tensors must be contiguous");
    TORCH_CHECK(
        x_ref.scalar_type() == at::kHalf &&
            kv_ref.scalar_type() == at::kHalf,
        "reference query/corpus must be fp16");
    TORCH_CHECK(
        x_fp8.scalar_type() == at::kFloat8_e4m3fn &&
            kv_fp8.scalar_type() == at::kFloat8_e4m3fn,
        "coarse query/corpus must be float8_e4m3fn");
    TORCH_CHECK(
        x_scale.scalar_type() == at::kFloat &&
            kv_scale.scalar_type() == at::kFloat,
        "rowwise dequant scales must be float32");
    TORCH_CHECK(
        x_ref.sizes() == x_fp8.sizes() &&
            kv_ref.sizes() == kv_fp8.sizes(),
        "fp16 and fp8 tensor shapes must match");
    const int64_t Hq = x_ref.size(0);
    const int64_t D = x_ref.size(1);
    const int64_t Hkv = kv_ref.size(0);
    const int64_t M = kv_ref.size(1);
    TORCH_CHECK(
        Hq == 64 && Hkv == 1 && D == 768 && kv_ref.size(2) == D,
        "experimental fp8 path requires B=64, Hkv=1, D=768");
    TORCH_CHECK(
        x_scale.dim() == 1 && x_scale.size(0) == Hq,
        "x_scale must have shape [B]");
    TORCH_CHECK(
        kv_scale.dim() == 2 && kv_scale.size(0) == Hkv &&
            kv_scale.size(1) == M,
        "kv_scale must have shape [Hkv,M]");
    TORCH_CHECK(M % 64 == 0, "M must be a multiple of 64");
    TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
    TORCH_CHECK(
        coarse_k > k && coarse_k <= M,
        "require k < coarse_k <= M");
    TORCH_CHECK(
        num_buckets >= 2 && num_buckets <= 4096,
        "num_buckets out of range");
    TORCH_CHECK(
        sample_mode == 1,
        "sample_mode must be 1 (strided corpus sampling)");
    TORCH_CHECK(sample_size >= 1, "sample_size must be positive");
    if (sample_size > M) sample_size = M;
    sample_size =
        (sample_size / 64 + (sample_size % 64 != 0)) * 64;
    if (sample_size < coarse_k)
        sample_size =
            (coarse_k / 64 + (coarse_k % 64 != 0)) * 64;
    if (sample_size > M) sample_size = M;
    if (buf_cap < coarse_k) buf_cap = coarse_k;
    if (buf_cap > M) buf_cap = M;

    constexpr int qn = 64;
    constexpr int bm = 256;
    constexpr int n_groups = 1;
    constexpr int rows_valid = 64;
    const int64_t Rwork = Hq;
    const int64_t candidate_cap = buf_cap;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Build calibration scores from the same quantized values/scales used by
    // the coarse scan. Conversion is sample-only; the M-row scan remains fp8.
    const int64_t stride = M / sample_size;
    auto x_deq = x_fp8.to(at::kHalf).mul(
        x_scale.to(at::kHalf).unsqueeze(1));
    auto kv_sample = kv_fp8.slice(
        /*dim=*/1, /*start=*/0, sample_size * stride, stride);
    auto kv_sample_scale = kv_scale
        .slice(/*dim=*/1, /*start=*/0, sample_size * stride, stride)
        .select(0, 0)
        .to(at::kHalf)
        .unsqueeze(1);
    auto kv_sample_deq =
        kv_sample.select(0, 0).to(at::kHalf).mul(kv_sample_scale);
    auto sample_flat = at::matmul(
        x_deq, kv_sample_deq.transpose(0, 1));

    auto opt_h = x_ref.options();
    auto opt_i = x_ref.options().dtype(at::kInt);
    auto opt_f = x_ref.options().dtype(at::kFloat);
    auto origin = at::empty({Rwork}, opt_f);
    auto inv_delta = at::empty({Rwork}, opt_f);
    auto th = at::empty({Rwork}, opt_i);
    auto qcount = at::empty({Rwork}, opt_i);
    auto bcount = at::empty({Rwork, num_buckets}, opt_i);
    auto buf_val = at::empty({Rwork, buf_cap}, opt_h);
    auto buf_idx = at::empty({Rwork, buf_cap}, opt_i);
    auto val_pad = at::empty({Rwork, coarse_k}, opt_h);
    auto idx_pad = at::empty({Rwork, coarse_k}, opt_i);
    auto cand_val = at::empty({Rwork, candidate_cap}, opt_h);
    auto cand_idx = at::empty({Rwork, candidate_cap}, opt_i);
    auto cand_cnt = at::empty({Rwork}, opt_i);
    auto lt_val = at::empty({Rwork, candidate_cap}, opt_h);
    auto lt_idx = at::empty({Rwork, candidate_cap}, opt_i);
    auto lt_cnt = at::empty({Rwork}, opt_i);
    auto cta_done = at::empty({n_groups}, opt_i);

    const size_t direct_part_slots =
#if LITETOPK_CALLER_DIRECTLT_WORKSPACE
        flash_topk_select_thr_mb_direct_part_slots(
            (int)Rwork, (int)buf_cap, (int)coarse_k,
            (int)candidate_cap);
#else
        0;
#endif
    at::Tensor part_val;
    at::Tensor part_idx;
    at::Tensor part_cnt;
    if (direct_part_slots != 0) {
        const int64_t slots = static_cast<int64_t>(direct_part_slots);
        part_val = at::empty({slots, coarse_k}, opt_h);
        part_idx = at::empty({slots, coarse_k}, opt_i);
        part_cnt = at::empty({slots}, opt_i);
    }

    // Two-bucket initial slack absorbs the tiny fp16 dequantized sample GEMM
    // versus fp32 tcgen05 accumulation mismatch. It only over-emits; the
    // completion refresh still tightens against the actual fp8 scan scores.
    litetopk_seed_params_kernel<<<
        (int)Rwork, 256, num_buckets * sizeof(int), stream>>>(
        reinterpret_cast<const __half*>(
            sample_flat.data_ptr<at::Half>()),
        sample_size, (int)sample_size, (int)num_buckets,
        (int)coarse_k, origin.data_ptr<float>(),
        inv_delta.data_ptr<float>(), th.data_ptr<int>(),
        bcount.data_ptr<int>(), qcount.data_ptr<int>(),
        lt_cnt.data_ptr<int>(), cand_cnt.data_ptr<int>(),
        cta_done.data_ptr<int>(), n_groups, /*th_slack=*/2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int scan_refresh =
        refresh_every > 0 ? (int)refresh_every : 0x7fffffff;
    launch_sm100_scan_fp8(
        x_fp8, kv_fp8, x_scale, kv_scale, n_groups,
        /*q_group_size=*/n_groups, rows_valid, (int)M,
        /*start_row=*/0, origin, inv_delta, th, qcount, bcount,
        buf_val, buf_idx, (int)buf_cap, (int)num_buckets,
        (int)coarse_k, scan_refresh, (int)num_ctas_x, stream,
        cta_done.data_ptr<int>(), qn, bm, /*out_fp32=*/false,
        /*store_bucket_space=*/true);

    launch_flash_topk_select_thr_mb_idx_fp16(
        reinterpret_cast<const __half*>(buf_val.data_ptr<at::Half>()),
        buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
        (int)Rwork, (int)buf_cap, (int)coarse_k,
        (int)candidate_cap, origin.data_ptr<float>(),
        inv_delta.data_ptr<float>(), th.data_ptr<int>(),
        qcount.data_ptr<int>(), (int)num_buckets,
        reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
        cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
        lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
        idx_pad.data_ptr<int>(), /*coords_fp16=*/false, stream,
        /*skip_zero=*/0, /*bucket_space=*/true,
        /*debucket_output=*/true, /*counters_preinitialized=*/true,
        direct_part_slots
            ? reinterpret_cast<__half*>(
                  part_val.data_ptr<at::Half>())
            : nullptr,
        direct_part_slots ? part_idx.data_ptr<int>() : nullptr,
        direct_part_slots ? part_cnt.data_ptr<int>() : nullptr);

    auto coarse_idx = idx_pad.reshape({Hq, coarse_k}).contiguous();
    if (!rerank) {
        auto coarse_val =
            val_pad.reshape({Hq, coarse_k}).contiguous();
        return std::make_tuple(
            coarse_val, coarse_idx, coarse_idx);
    }
    // index_select accepts int32 on CUDA, avoiding a 64-bit index slab.
    auto gathered = at::index_select(
        kv_ref.select(0, 0), /*dim=*/0, coarse_idx.reshape({-1}))
        .view({Hq, coarse_k, D});
    auto ref_scores =
        at::bmm(gathered, x_ref.unsqueeze(2)).squeeze(2);
    auto final_topk =
        at::topk(ref_scores, k, /*dim=*/1, /*largest=*/true,
                 /*sorted=*/false);
    auto final_val = std::get<0>(final_topk);
    auto final_pos = std::get<1>(final_topk);
    auto final_idx = coarse_idx.gather(/*dim=*/1, final_pos);
    return std::make_tuple(final_val, final_idx, coarse_idx);
}
#endif

}  // namespace

TORCH_LIBRARY(litetopk_sm100, m) {
    m.def("score_sparse(Tensor q_cont, Tensor kv_cont, Tensor origin, Tensor inv_delta, Tensor th, "
          "int q_group_size, int logical_rows, int start_row, int num_buckets, int topk, "
          "int refresh_every, int buf_cap, int num_ctas_x=0) -> (Tensor, Tensor, Tensor)");
    m.def("fused_ip_sparse_b200(Tensor x, Tensor kv_cont, int k, int num_buckets, "
          "int buf_cap, int sample_size, int refresh_every, int num_ctas_x, int sample_mode=1, "
          "int qn=0, int bm=0, bool out_fp32=False) -> (Tensor, Tensor)");
    m.def(
        "fused_ip_sparse_b200_with_counts(Tensor x, Tensor kv_cont, int k, "
        "int num_buckets, int buf_cap, int sample_size, int refresh_every, "
        "int num_ctas_x, int sample_mode=1, int qn=0, int bm=0, "
        "bool out_fp32=False) -> (Tensor, Tensor, Tensor)");
#if LITETOPK_MARSCO_FP8_SCAN
    m.def(
        "fused_ip_sparse_fp8_b200(Tensor x_ref, Tensor x_fp8, "
        "Tensor x_scale, Tensor kv_ref, Tensor kv_fp8, Tensor kv_scale, "
        "int k, int coarse_k, int num_buckets, int buf_cap, "
        "int sample_size, int refresh_every, int num_ctas_x, "
        "int sample_mode=1, bool rerank=True) "
        "-> (Tensor, Tensor, Tensor)");
#endif
    m.def("dense_scores(Tensor x, Tensor kv_cont, int num_ctas_x=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(litetopk_sm100, CUDA, m) {
    m.impl("score_sparse", TORCH_FN(litetopk_sm100_score_sparse));
    m.impl("fused_ip_sparse_b200", TORCH_FN(fused_ip_sparse_b200));
    m.impl(
        "fused_ip_sparse_b200_with_counts",
        TORCH_FN(fused_ip_sparse_b200_with_counts));
#if LITETOPK_MARSCO_FP8_SCAN
    m.impl(
        "fused_ip_sparse_fp8_b200",
        TORCH_FN(fused_ip_sparse_fp8_b200));
#endif
    m.impl("dense_scores", TORCH_FN(litetopk_sm100_dense_scores));
}

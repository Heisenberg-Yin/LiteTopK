// PyTorch extension for the H100 LiteTopK DSA indexer. It provides sample
// calibration, the SM90 WGMMA sparse scan, threshold refresh, and radix-select
// post-processing. Build against the DeepGEMM 2.5 include tree and its bundled
// CUTLASS headers for sm_90a.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <atomic>
#include <limits>
#include <mutex>
#include <optional>
#include <tuple>

#include "sm90_dsa_litetopk.cuh"

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
constexpr int BLOCK_KV = 128;      // = 64 (WGMMA M) x 2 math warpgroups
constexpr int NUM_Q_STAGES = 1;   // one q-block per CTA
#ifndef DSA_V3_KV_STAGES
#define DSA_V3_KV_STAGES 4
#ifndef DSA_SCALE_HEADROOM
#define DSA_SCALE_HEADROOM 0.5f  // fine-scale drift headroom above sample max
#endif
#endif
constexpr int NUM_KV_STAGES = DSA_V3_KV_STAGES;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256;  // 2 math warpgroups (each owns one m64 tile)
constexpr int NUM_SMS = 132;       // H100 SXM

// Cache device-dependent launch properties. The SM count sizes the KV-split
// grid, and the scan kernel opts into dynamic shared memory once per device.
constexpr int MAX_TRACKED_CUDA_DEVICES = 64;

static int current_device_sm_count(int device) {
    TORCH_CHECK(device >= 0 && device < MAX_TRACKED_CUDA_DEVICES,
                "CUDA device ordinal out of cached range: ", device);
    static std::atomic<int> counts[MAX_TRACKED_CUDA_DEVICES]{};
    int count = counts[device].load(std::memory_order_acquire);
    if (count == 0) {
        C10_CUDA_CHECK(cudaDeviceGetAttribute(
            &count, cudaDevAttrMultiProcessorCount, device));
        counts[device].store(count, std::memory_order_release);
    }
    return count;
}

static void set_scan_smem_attribute_once(
        const void* kernel, int smem, int device) {
    TORCH_CHECK(device >= 0 && device < MAX_TRACKED_CUDA_DEVICES,
                "CUDA device ordinal out of cached range: ", device);
    static std::once_flag once[MAX_TRACKED_CUDA_DEVICES];
    std::call_once(once[device], [=] {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    });
}

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

// Fused sample preparation, one block per row. It derives the bucket affine
// and initial K-th bucket, publishes the sample histogram as a conservative
// refresh base, and emits every sample position at or above the threshold.
// Those seeds are a superset of the sample top-K; the final select trims them.
__global__ void seed_prep_kernel(
    const float* __restrict__ slog, const int64_t slog_stride,
    const int head, const int NB, const int K, const int cap,
    const int emit_limit,  // only columns j < emit_limit may emit seeds;
                           // probe columns beyond it are histogram/scale-only
    const int probe_stride_tok,  // >0: sample columns are strided probe pages;
                                 // emitted seed index j maps to original
                                 // position (j/64)*probe_stride_tok + (j%64)
    const int hist_stride,       // subsample factor for the minmax+histogram
                                 // passes (threshold estimation only; the
                                 // emit pass still reads everything). Caller
                                 // scales K to the subsample quantile.
    const float headroom,  // extend the bucket scale ABOVE the sample max by
                           // headroom*span (absolute, resolution-preserving
                           // when NB is scaled up with it): drifted scores
                           // land in real buckets instead of clamping to
                           // bucket 0 where refresh can never resolve them
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket,
    int32_t* __restrict__ bcount,
    float* __restrict__ cand_val, int32_t* __restrict__ cand_idx,
    int32_t* __restrict__ cand_cnt) {
    constexpr int BT = 1024;
    constexpr int NSUB = 4;  // sub-histograms to spread smem atomic conflicts
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const float* srow = slog + (size_t)row * slog_stride;

    // pass 1: min/max of the row's FINITE scores (vectorized). -inf appears
    // when the caller passes clean_logits=True full-row logits (dense-select
    // mode) for the out-of-range causal tail; it must not poison the range.
    __shared__ float s_mx[BT / 32];
    __shared__ float s_mn[BT / 32];
    float mx = -INFINITY, mn = INFINITY;
    const int head4 = head / 4 * 4;
    const auto acc = [&](const float s) {
        if (isfinite(s)) {
            mx = fmaxf(mx, s);
            mn = fminf(mn, s);
        }
    };
    for (int j = tid * 4; j < head4; j += BT * 4 * hist_stride) {
        const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
        acc(s4.x); acc(s4.y); acc(s4.z); acc(s4.w);
    }
    if (hist_stride == 1)
        for (int j = head4 + tid; j < head; j += BT)
            acc(srow[j]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
    }
    if (lane == 0) { s_mx[tid >> 5] = mx; s_mn[tid >> 5] = mn; }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (int wgi = 1; wgi < BT / 32; ++wgi) {
            s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
            s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
        }
    }
    __syncthreads();
    float o = -s_mx[0];         // min over x = -score
    const float hi = -s_mn[0];  // max over x
    const float mag = fmaxf(fabsf(hi), fabsf(o));
    const float span = fmaxf(fmaxf(hi - o, mag * 0x1p-8f), 1e-6f);
    o -= headroom * span;       // forward (above-max) drift headroom
    float inv = (NB - 1) / (span * (1.0f + headroom));

    // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
    // smem atomic conflicts, vectorized loads.
    extern __shared__ int s_hist[];  // NSUB * NB ints
    for (int b = tid; b < NSUB * NB; b += BT) s_hist[b] = 0;
    __syncthreads();
    int* my_hist = s_hist + (tid / (BT / NSUB)) * NB;
    const auto bucket_of = [&](const float s) -> int {
        const float x = -s;
        int b = static_cast<int>((x - o) * inv);
        return b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
    };
    for (int j = tid * 4; j < head4; j += BT * 4 * hist_stride) {
        const float4 s4 = *reinterpret_cast<const float4*>(srow + j);
        if (isfinite(s4.x)) atomicAdd(&my_hist[bucket_of(s4.x)], 1);
        if (isfinite(s4.y)) atomicAdd(&my_hist[bucket_of(s4.y)], 1);
        if (isfinite(s4.z)) atomicAdd(&my_hist[bucket_of(s4.z)], 1);
        if (isfinite(s4.w)) atomicAdd(&my_hist[bucket_of(s4.w)], 1);
    }
    if (hist_stride == 1)
        for (int j = head4 + tid; j < head; j += BT) {
            const float s = srow[j];
            if (isfinite(s)) atomicAdd(&my_hist[bucket_of(s)], 1);
        }
    __syncthreads();
    // merge sub-histograms into s_hist[0..NB)
    for (int b = tid; b < NB; b += BT) {
        int c = s_hist[b];
        #pragma unroll
        for (int g = 1; g < NSUB; ++g) c += s_hist[g * NB + b];
        s_hist[b] = c;
    }
    __syncthreads();
    // Publish the sample histogram for scan-time threshold refresh. A
    // calibration-only probe emits no seeds, so its scan-side counts start at
    // zero and are populated by the full scan.
    for (int b = tid; b < NB; b += BT)
        // Probe mode (emit_limit==0): scan-side refresh must start from zero
        // counts — write zeros here, saving the caller a separate memset.
        bcount[(size_t)row * NB + b] = (emit_limit == 0) ? 0 : s_hist[b];
    __shared__ int s_th;
    if (tid == 0) {
        const int kk = K < head ? K : head;
        int cum = 0, th = NB - 1;
        for (int b = 0; b < NB; ++b) {
            cum += s_hist[b];
            if (cum >= kk) { th = b; break; }
        }
        s_th = th;
        th_bucket[row] = th;
        origin[row] = o;
        inv_delta[row] = inv;
    }
    __syncthreads();
    const int th_emit = s_th;

    if (emit_limit == 0) {  // threshold-probe mode: no seeds wanted; skip the
        if (tid == 0)
            cand_cnt[row] = 0;  // whole pass-3 read of the sample
        return;
    }

    // pass 3: emit every position with bucket <= th (compact, unordered).
    // Warp-aggregated: one shared-counter atomic per warp per pass-group.
    __shared__ int s_cnt;
    if (tid == 0) s_cnt = 0;
    __syncthreads();
    float* vrow = cand_val + (size_t)row * cap;
    int32_t* irow = cand_idx + (size_t)row * cap;
    const auto emit_group = [&](const float s, const int j) {
        const int b_of = (isfinite(s) && j < emit_limit) ? bucket_of(s) : NB;
        const bool g = b_of <= th_emit;
        const unsigned m = __ballot_sync(0xffffffffu, g);
        if (m != 0) {
            int base = 0;
            if (lane == 0) base = atomicAdd(&s_cnt, __popc(m));
            base = __shfl_sync(0xffffffffu, base, 0);
            if (g) {
                const int pos = base + __popc(m & ((lane == 0) ? 0u : ((1u << lane) - 1u)));
                if (pos < cap) {
                    // Seeds use the same bucket coordinates as scan candidates:
                    // bq = (x - o) * inv. Select consumes this representation
                    // with the identity affine transform.
                    vrow[pos] = (-s - o) * inv;
                    irow[pos] = probe_stride_tok > 0
                        ? (j >> 6) * probe_stride_tok + (j & 63) : j;
                }
            }
        }
    };
    // Uniform trip counts so every lane always reaches the warp ballots.
    for (int j0 = 0; j0 < head4; j0 += BT * 4) {
        const int j = j0 + tid * 4;
        float4 s4 = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        if (j < head4) s4 = *reinterpret_cast<const float4*>(srow + j);
        emit_group(s4.x, j);
        emit_group(s4.y, j + 1);
        emit_group(s4.z, j + 2);
        emit_group(s4.w, j + 3);
    }
    for (int j0 = head4; j0 < head; j0 += BT) {
        const int j = j0 + tid;
        const float s = (j < head) ? srow[j] : -INFINITY;
        emit_group(s, j);
    }
    __syncthreads();
    if (tid == 0) {
        int c = s_cnt < cap ? s_cnt : cap;
        cand_cnt[row] = c;
    }
}

__device__ __forceinline__ uint32_t compact_enc_float(float v) {
    uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

__global__ void compact_topk_min_idx_litetopk_kernel(
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

__global__ void compact_topk_min_thr_litetopk_kernel(
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
    int32_t* __restrict__ out_idx,
    // bulk-drain mode: scan-emitted indices (slot >= seed_base[row]) are in
    // COMPACTED space; map them back to original positions at output time
    // (only winners pay, K per row). Seeds (< seed_base) are already mapped.
    const uint32_t probe_group,
    const uint64_t probe_magic,
    const uint32_t probe_add_max,
    const int32_t* __restrict__ seed_base) {
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
    const int sbase = (probe_group != 0 && seed_base != nullptr)
                          ? seed_base[row] : 0x7fffffff;
    const auto out_map = [&](const int32_t raw, const int j) -> int32_t {
        if (j < sbase) return raw;
        const uint32_t kvo = static_cast<uint32_t>(raw);
        const uint32_t sup = (uint32_t)(((uint64_t)kvo * probe_magic) >> 42);
        return static_cast<int32_t>(kvo + min((sup + 1) * 64u, probe_add_max));
    };

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
            if (w < K) { ov[w] = v; oi[w] = out_map(irow[j], j); }
            continue;
        }
        if (!DSA_IN_SELECT_SET(b)) continue;
        uint32_t e = compact_enc_float(v);
        if (e < pivot) {
            int w = atomicAdd(&s_w_lt, 1);
            int out_pos = pre_take + w;
            if (out_pos < K) { ov[out_pos] = v; oi[out_pos] = out_map(irow[j], j); }
        } else if (e == pivot) {
            int oo = atomicAdd(&s_w_eq, 1);
            if (oo < eq_take) {
                int w = pre_take + cnt_lt + oo;
                if (w < K) { ov[w] = v; oi[w] = out_map(irow[j], j); }
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
    // SM90: no UMMA full/empty barrier pair (WGMMA is issued by the math
    // warpgroups themselves).
    const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2;
    const int smem_barriers = num_barriers * 8;
    const int smem_slots = 4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
    const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                           ((int)sizeof(int32_t) + DSA_WARP_QUEUE_CAP * ((int)sizeof(float) + (int)sizeof(int32_t)));
    const int smem_hist = BLOCK_Q * 256 * (int)sizeof(int32_t);  // per-CTA refresh
                                                                  // histogram (NB<=256)
    return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
           NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks +
           smem_barriers + smem_slots + smem_warpq + smem_hist;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mqa_logits_dsa_litetopk(
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
    TORCH_CHECK(num_buckets >= 2 && num_buckets <= 256, "num_buckets out of range (smem hist sized for 256)");
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
    auto kernel = &dsa_litetopk::sm90_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_SMS, SPEC_THREADS, MATH_THREADS>;
    const int device = q.get_device();
    set_scan_smem_attribute_once((const void*)kernel, smem, device);

    // KV-split grid: x indexes query blocks and y indexes contiguous KV
    // windows. The split count is sized from the active device.
    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
    int num_kv_splits;
    if (num_kv_splits_override > 0) {
        num_kv_splits = (int)num_kv_splits_override;
    } else {
        constexpr int kWaves = 4;
        const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
        num_kv_splits =
            (kWaves * current_device_sm_count(device) + qb - 1) / qb;
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
        (uint32_t)num_kv_splits, 0u, 0ULL, 0u,
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

// Fused seed/prep: sample scores -> (origin, inv_delta, th_bucket, cand_val,
// cand_idx, cand_cnt, bcount), everything the scan needs, in one launch.
void seed_prep_litetopk_(torch::Tensor slog, int64_t num_buckets64, int64_t topk64,
                       int64_t cand_cap64, int64_t emit_limit64, double headroom,
                       int64_t probe_stride_tok64, int64_t hist_stride64,
                       torch::Tensor origin, torch::Tensor inv_delta,
                       torch::Tensor th_bucket, torch::Tensor bcount,
                       torch::Tensor cand_val, torch::Tensor cand_idx,
                       torch::Tensor cand_cnt) {
    TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
    TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
    TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
    const int Q = (int)slog.size(0);
    const int head = (int)slog.size(1);
    const int NB = (int)num_buckets64;
    const int K = (int)topk64;
    const int cap = (int)cand_cap64;
    TORCH_CHECK(cand_val.size(0) >= Q && cand_val.size(1) == cap, "cand_val shape");
    TORCH_CHECK(bcount.size(0) >= Q && bcount.size(1) == NB, "bcount shape");
    TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
                "slog rows must be 16B aligned");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int seed_smem = 4 * NB * (int)sizeof(int);
    if (seed_smem > 48 * 1024) {
        static bool attr_set2 = false;
        if (!attr_set2) {
            C10_CUDA_CHECK(cudaFuncSetAttribute((void*)seed_prep_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4096 * (int)sizeof(int)));
            attr_set2 = true;
        }
    }
    const int emit_limit = emit_limit64 == 0 ? 0 : (emit_limit64 > 0 ? (int)emit_limit64 : head);
    const int probe_stride_tok = (int)probe_stride_tok64;
    const int hist_stride = hist_stride64 > 1 ? (int)hist_stride64 : 1;
    seed_prep_kernel<<<Q, 1024, seed_smem, stream>>>(
        slog.data_ptr<float>(), slog.stride(0), head, NB, K, cap, emit_limit,
        probe_stride_tok, hist_stride,
        (float)headroom,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(),
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
seed_prep_litetopk(torch::Tensor slog, int64_t num_buckets64, int64_t topk64,
                 int64_t cand_cap64, int64_t emit_limit64, double headroom,
                 int64_t probe_stride_tok64, int64_t hist_stride64) {
    TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
    TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
    TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
    const int Q = (int)slog.size(0);
    const int head = (int)slog.size(1);
    const int NB = (int)num_buckets64;
    const int K = (int)topk64;
    const int cap = (int)cand_cap64;
    TORCH_CHECK(NB >= 2 && NB <= 4096, "num_buckets out of range");
    TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
    // float4 pass requires 16B-aligned rows; fall back is not implemented.
    TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
                "slog rows must be 16B aligned");

    auto opts_f = slog.options();
    auto opts_i = slog.options().dtype(torch::kInt);
    auto origin = torch::empty({Q}, opts_f);
    auto inv_delta = torch::empty({Q}, opts_f);
    auto th_bucket = torch::empty({Q}, opts_i);
    auto bcount = torch::empty({Q, NB}, opts_i);      // fully overwritten
    auto cand_val = torch::empty({Q, cap}, opts_f);
    auto cand_idx = torch::empty({Q, cap}, opts_i);
    auto cand_cnt = torch::empty({Q}, opts_i);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int seed_smem = 4 * NB * (int)sizeof(int);
    if (seed_smem > 48 * 1024) {  // NB > 3072 exceeds the default dynamic limit
        static bool attr_set = false;
        if (!attr_set) {
            C10_CUDA_CHECK(cudaFuncSetAttribute((void*)seed_prep_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4096 * (int)sizeof(int)));
            attr_set = true;
        }
    }
    const int emit_limit = emit_limit64 == 0 ? 0 : (emit_limit64 > 0 ? (int)emit_limit64 : head);
    const int probe_stride_tok = (int)probe_stride_tok64;
    const int hist_stride = hist_stride64 > 1 ? (int)hist_stride64 : 1;
    seed_prep_kernel<<<Q, 1024, seed_smem, stream>>>(
        slog.data_ptr<float>(), slog.stride(0), head, NB, K, cap, emit_limit,
        probe_stride_tok, hist_stride,
        (float)headroom,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(),
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(origin, inv_delta, th_bucket, cand_val, cand_idx,
                           cand_cnt, bcount);
}

// Scan into buffers prepared by seed_prep_litetopk (no seeding of any kind).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mqa_logits_dsa_litetopk_ext(
        torch::Tensor q,
        torch::Tensor kv,
        torch::Tensor kv_scales,
        torch::Tensor weights,
        torch::Tensor cu_start,
        torch::Tensor cu_end,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor bcount,
        int64_t num_buckets64,
        int64_t topk64,
        int64_t refresh_every64,
        int64_t num_kv_splits_override,
        int64_t probe_group64,
        int64_t probe_add_max64) {
    TORCH_CHECK(q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() && weights.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && kv_scales.is_contiguous() && weights.is_contiguous() &&
                cu_start.is_contiguous() && cu_end.is_contiguous() && origin.is_contiguous() && inv_delta.is_contiguous() &&
                th_bucket.is_contiguous() && cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                cand_cnt.is_contiguous() && bcount.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn && kv.scalar_type() == torch::kFloat8_e4m3fn,
                "q/kv must be fp8_e4m3fn");
    const int seq_len = (int)q.size(0);
    const int seq_len_kv = (int)kv.size(0);
    const int cand_cap = (int)cand_val.size(1);
    const int num_buckets = (int)num_buckets64;
    const int topk = (int)topk64;
    TORCH_CHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(cand_val.size(0) == seq_len && cand_idx.sizes() == cand_val.sizes() &&
                cand_cnt.numel() == seq_len && bcount.size(0) == seq_len && bcount.size(1) == num_buckets,
                "prepared buffer shape mismatch");
    const bool external_refresh = (refresh_every64 < 0);
    const int refresh_every = external_refresh ? 0x7fffffff : (int)refresh_every64;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
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
    auto kernel = &dsa_litetopk::sm90_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_SMS, SPEC_THREADS, MATH_THREADS>;
    const int device = q.get_device();
    set_scan_smem_attribute_once((const void*)kernel, smem, device);

    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
    int num_kv_splits;
    if (num_kv_splits_override > 0) {
        num_kv_splits = (int)num_kv_splits_override;
    } else {
        constexpr int kWaves = 4;
        const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
        num_kv_splits =
            (kWaves * current_device_sm_count(device) + qb - 1) / qb;
        const int max_useful_splits = total_kv_blocks > 0 ? (total_kv_blocks + 1) / 2 : 1;
        if (num_kv_splits > max_useful_splits) num_kv_splits = max_useful_splits;
    }
    if (num_kv_splits < 1) num_kv_splits = 1;
    if (num_kv_splits > total_kv_blocks) num_kv_splits = total_kv_blocks > 0 ? total_kv_blocks : 1;
    int grid_q = num_q_blocks;
    dim3 grid((unsigned)grid_q, (unsigned)num_kv_splits, 1);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(), (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
        (uint32_t)num_kv_splits, (uint32_t)probe_group64,
        probe_group64 > 0 ? (((1ULL << 42) + (uint64_t)probe_group64 - 1) / (uint64_t)probe_group64) : 0ULL,
        (uint32_t)probe_add_max64,
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

std::tuple<torch::Tensor, torch::Tensor> compact_topk_min_idx_litetopk(
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
    compact_topk_min_idx_litetopk_kernel<<<R, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
        R, CAP, K, out_val.data_ptr<float>(), out_idx.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(out_val, out_idx);
}

std::tuple<torch::Tensor, torch::Tensor> compact_topk_min_thr_litetopk(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor origin,
        torch::Tensor inv_delta,
        torch::Tensor th_bucket,
        int64_t num_buckets64,
        int64_t k64,
        int64_t probe_group64,
        int64_t probe_add_max64,
        std::optional<torch::Tensor> seed_base,
        std::optional<torch::Tensor> out_val_buf,
        std::optional<torch::Tensor> out_idx_buf) {
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
    auto out_val = out_val_buf.has_value()
                       ? out_val_buf.value()
                       : torch::empty({R, K}, cand_val.options());
    auto out_idx = out_idx_buf.has_value()
                       ? out_idx_buf.value()
                       : torch::empty({R, K}, cand_idx.options());
    TORCH_CHECK(out_val.is_cuda() && out_idx.is_cuda(),
                "select outputs must be CUDA");
    TORCH_CHECK(out_val.is_contiguous() && out_idx.is_contiguous(),
                "select outputs must be contiguous");
    TORCH_CHECK(out_val.scalar_type() == torch::kFloat &&
                out_idx.scalar_type() == torch::kInt,
                "select outputs must be fp32/int32");
    TORCH_CHECK(out_val.dim() == 2 && out_idx.dim() == 2 &&
                out_val.size(0) == R && out_val.size(1) == K &&
                out_idx.size(0) == R && out_idx.size(1) == K,
                "select outputs must be [R, topk]");
    TORCH_CHECK(out_val.get_device() == cand_val.get_device() &&
                out_idx.get_device() == cand_val.get_device(),
                "select outputs must be on the candidate device");
    if (probe_group64 > 0) {
        TORCH_CHECK(seed_base.has_value() && seed_base->is_cuda() &&
                    seed_base->is_contiguous() &&
                    seed_base->scalar_type() == torch::kInt &&
                    seed_base->numel() >= R,
                    "seed_base [R] int32 required with probe_group");
    }
    compact_topk_min_thr_litetopk_kernel<<<R, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        cand_val.data_ptr<float>(), cand_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        R, CAP, K, NB, out_val.data_ptr<float>(), out_idx.data_ptr<int32_t>(),
        (uint32_t)probe_group64,
        probe_group64 > 0 ? (((1ULL << 42) + (uint64_t)probe_group64 - 1) / (uint64_t)probe_group64) : 0ULL,
        (uint32_t)probe_add_max64,
        probe_group64 > 0 ? seed_base->data_ptr<int32_t>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(out_val, out_idx);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("seed_prep_litetopk_", &seed_prep_litetopk_,
          "In-place fused sample prep (caller-owned buffers)",
          pybind11::arg("slog"), pybind11::arg("num_buckets"), pybind11::arg("topk"),
          pybind11::arg("cand_cap"), pybind11::arg("emit_limit"), pybind11::arg("headroom"),
          pybind11::arg("probe_stride_tok"), pybind11::arg("hist_stride"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"),
          pybind11::arg("th_bucket"), pybind11::arg("bcount"),
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"));
    m.def("seed_prep_litetopk", &seed_prep_litetopk,
          "Fused sample prep: scores -> (origin, inv, th, cand bufs, cnt, bcount)",
          pybind11::arg("slog"), pybind11::arg("num_buckets"), pybind11::arg("topk"),
          pybind11::arg("cand_cap"), pybind11::arg("emit_limit") = -1,
          pybind11::arg("headroom") = 0.0,
          pybind11::arg("probe_stride_tok") = 0,
          pybind11::arg("hist_stride") = 1);
    m.def("mqa_logits_dsa_litetopk_ext", &mqa_logits_dsa_litetopk_ext,
          "Sparse scan into buffers prepared by seed_prep_litetopk",
          pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
          pybind11::arg("weights"), pybind11::arg("cu_start"), pybind11::arg("cu_end"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"),
          pybind11::arg("bcount"), pybind11::arg("num_buckets"), pybind11::arg("topk"),
          pybind11::arg("refresh_every"), pybind11::arg("num_kv_splits")=-1,
          pybind11::arg("probe_group")=0, pybind11::arg("probe_add_max")=0);
    m.def("mqa_logits_dsa_litetopk", &mqa_logits_dsa_litetopk,
          "Fused DSA ReLU-MQA scan with sparse candidate emission",
          pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
          pybind11::arg("weights"), pybind11::arg("cu_start"), pybind11::arg("cu_end"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("seed_val"), pybind11::arg("seed_idx"),
          pybind11::arg("num_buckets"), pybind11::arg("cand_cap"), pybind11::arg("topk"),
          pybind11::arg("refresh_every"), pybind11::arg("num_kv_splits")=-1);
    m.def("compact_topk_min_idx_litetopk", &compact_topk_min_idx_litetopk,
          "Qcount-aware top-k-min over compact candidates");
    m.def("compact_topk_min_thr_litetopk", &compact_topk_min_thr_litetopk,
          "Threshold-aware top-k-min select",
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("num_buckets"), pybind11::arg("topk"),
          pybind11::arg("probe_group") = 0, pybind11::arg("probe_add_max") = 0,
          pybind11::arg("seed_base") = pybind11::none(),
          pybind11::arg("out_val") = pybind11::none(),
          pybind11::arg("out_idx") = pybind11::none());
}

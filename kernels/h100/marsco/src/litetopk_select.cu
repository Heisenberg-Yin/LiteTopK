// FlashTopk-v1 CUDA 实现（采样定阈 + 单遍直方图 + 边界桶精确 topk）。
//
//   路线（区别于 radixselect 的 4-pass radix）：借鉴 sample-range + bucket 思路，
//   把「5 遍全量扫描」压成「采样(~1.6%) + 2 遍全量」：
//     1. sample pass：每行单 block 在前缀采样子集(S=max(k,16384))上 in-block radix
//        求采样第 k 小 enc 作分桶上界 hi（采样第 k 小 >= 全集第 k 小，安全），min 作下界 lo。
//        255 个有效桶覆盖 [lo,hi]，桶 255 专放 enc>hi 的 bulk。
//     2. hist pass（全量 1 遍）：每元素分桶，per-row 直方图（shared 累加 + 全局 atomic 归约）。
//     3. threshold：cumsum 求精确阈值桶 T 与 count_below（bucket<T 的精确个数 <k）。
//     4. gather pass（全量 1 遍）：bucket<T 直接写出 out[0:count_below]（确定入选）；
//        bucket==T 收集进 per-row 候选 buffer（CAP 槽）。
//     5. boundary：对候选 buffer 做 in-block 精确选择，取 need=k-count_below 个最小补齐
//        out[count_below:k]，含 tie 处理；overflow 时重扫整行 bucket==T 保证精确。
//
//   单调编码（保序：v1<v2 <=> enc1<enc2）：v>=0: enc=bits^0x80000000 ; v<0: enc=~bits
//   fp16 原生 16-bit 编码（不提升 float），pass 数减半。行首 16B 对齐(N%VEC==0)时 hist/gather
//   走 float4 向量化读。
//   workspace 由内部以 stream-ordered cudaMallocAsync 管理；topk-max 经取负 -> topk-min 实现。

#include "litetopk_select.h"

#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

namespace {

// 256B 对齐（与 TF/CUDA 设备分配的对齐一致），保证各子缓冲首址对齐。
__host__ __device__ inline size_t ft_align256(size_t x) { return (x + 255) & ~((size_t)255); }

// 在调用方预分配的单块 blob 上做 bump 切分；base 传 nullptr 时仅用于累计字节数（size 查询）。
struct WsBump {
    char* base;
    size_t off;
    explicit WsBump(void* p) : base(reinterpret_cast<char*>(p)), off(0) {}
    template <typename U>
    U* take(size_t count) {
        size_t a = ft_align256(off);
        U* ptr = reinterpret_cast<U*>(base + a);
        off = a + count * sizeof(U);
        return ptr;
    }
    size_t used() const { return ft_align256(off); }
};

__device__ __forceinline__ uint32_t ft_enc_of_float(float v) {
    uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? (~bits) : (bits ^ 0x80000000u);
}

// fp16 原生 16-bit 单调编码（不提升到 float），与 fp32 同序。仅低 16 bit，故 radix 2 pass。
__device__ __forceinline__ uint32_t ft_enc_of_half(__half v) {
    uint16_t bits = __half_as_ushort(v);
    uint16_t mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
    uint16_t e = __hisnan(v) ? 0xffffu : (uint16_t)(bits ^ mask);
    return (uint32_t)e;
}

// bf16 原生 16-bit 单调编码（符号位同在 MSB，bit 布局与 fp16 同构），与 fp32 同序。低 16 bit -> 2 pass。
__device__ __forceinline__ uint32_t ft_enc_of_bf16(__nv_bfloat16 v) {
    uint16_t bits = __bfloat16_as_ushort(v);
    uint16_t mask = (bits & 0x8000u) ? 0xffffu : 0x8000u;
    uint16_t e = __hisnan(v) ? 0xffffu : (uint16_t)(bits ^ mask);
    return (uint32_t)e;
}

// 从指针读取并单调编码（用于 radix/分桶比较）。float -> 32-bit；half -> 16-bit。
template <typename T>
__device__ __forceinline__ uint32_t ft_enc_ldg(const T* p);
template <>
__device__ __forceinline__ uint32_t ft_enc_ldg<float>(const float* p) { return ft_enc_of_float(__ldg(p)); }
template <>
__device__ __forceinline__ uint32_t ft_enc_ldg<__half>(const __half* p) { return ft_enc_of_half(__ldg(p)); }
template <>
__device__ __forceinline__ uint32_t ft_enc_ldg<__nv_bfloat16>(const __nv_bfloat16* p) { return ft_enc_of_bf16(__ldg(p)); }

// 从寄存器/局部值直接编码（不能对 local 地址用 __ldg）
template <typename T>
__device__ __forceinline__ uint32_t ft_enc_val(T v);
template <>
__device__ __forceinline__ uint32_t ft_enc_val<float>(float v) { return ft_enc_of_float(v); }
template <>
__device__ __forceinline__ uint32_t ft_enc_val<__half>(__half v) { return ft_enc_of_half(v); }
template <>
__device__ __forceinline__ uint32_t ft_enc_val<__nv_bfloat16>(__nv_bfloat16 v) { return ft_enc_of_bf16(v); }

// 值 -> float（torch 扩展禁用了 __half 隐式转换，须显式 __half2float）。
template <typename T>
__device__ __forceinline__ float ft_to_float(T v);
template <>
__device__ __forceinline__ float ft_to_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ float ft_to_float<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float ft_to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

// 单调编码的逆变换：enc -> 原始值（非 NaN 处 bit 精确还原）。用于 sort 后直接由 key 还原 value，
//   省去回原数组 scatter-read 的 gather 步骤。
__device__ __forceinline__ float ft_dec_to_float(uint32_t enc) {
    uint32_t bits = (enc & 0x80000000u) ? (enc ^ 0x80000000u) : (~enc);
    return __uint_as_float(bits);
}
__device__ __forceinline__ __half ft_dec_to_half(uint32_t enc) {
    uint16_t e = (uint16_t)enc;
    uint16_t bits = (e & 0x8000u) ? (uint16_t)(e ^ 0x8000u) : (uint16_t)(e ^ 0xffffu);
    return __ushort_as_half(bits);
}
__device__ __forceinline__ __nv_bfloat16 ft_dec_to_bf16(uint32_t enc) {
    uint16_t e = (uint16_t)enc;
    uint16_t bits = (e & 0x8000u) ? (uint16_t)(e ^ 0x8000u) : (uint16_t)(e ^ 0xffffu);
    return __ushort_as_bfloat16(bits);
}
template <typename T>
__device__ __forceinline__ T ft_dec_val(uint32_t enc);
template <>
__device__ __forceinline__ float ft_dec_val<float>(uint32_t enc) { return ft_dec_to_float(enc); }
template <>
__device__ __forceinline__ __half ft_dec_val<__half>(uint32_t enc) { return ft_dec_to_half(enc); }
template <>
__device__ __forceinline__ __nv_bfloat16 ft_dec_val<__nv_bfloat16>(uint32_t enc) { return ft_dec_to_bf16(enc); }

// 编码位宽与 radix pass 数：fp32 用 32-bit / 4 pass；fp16 用 16-bit / 2 pass。
// kRevMask：编码域内的「逆序掩码」。对 enc 异或此掩码可整体翻转单调序（升<->降），
//   用于 topk-max：max 的前 K 大 == (enc^kRevMask) 的前 K 小，无需对数据物理取负。
template <typename T> struct EncTraits;
template <> struct EncTraits<float>  { static constexpr int kBits = 32; static constexpr int kPasses = 4; static constexpr uint32_t kRevMask = 0xFFFFFFFFu; };
template <> struct EncTraits<__half> { static constexpr int kBits = 16; static constexpr int kPasses = 2; static constexpr uint32_t kRevMask = 0x0000FFFFu; };
template <> struct EncTraits<__nv_bfloat16> { static constexpr int kBits = 16; static constexpr int kPasses = 2; static constexpr uint32_t kRevMask = 0x0000FFFFu; };

// 向量化读：每线程一条 128-bit load（float4，16B）取代 VEC 条标量 load。
//   fp32: VEC=4（float4）；fp16: VEC=8（float4 重解释为 8 个 __half）。
// 目的是把 hist/gather 那遍删不掉的全量顺序读的 load 指令数降为 1/VEC，
// 减小指令发射 / LSU / MSHR 压力，使内存流水线更饱满（带宽利用率 ~86%->~95%）。
// 不改变搬运总字节数，不引入额外内存往返，精确性不变。
template <typename T> struct VecTraits;
template <> struct VecTraits<float>  { static constexpr int VEC = 4; };
template <> struct VecTraits<__half> { static constexpr int VEC = 8; };
template <> struct VecTraits<__nv_bfloat16> { static constexpr int VEC = 8; };

// 读取 data[vec_idx*VEC .. +VEC-1]（需 16B 对齐）到 out[VEC]。
template <typename T>
__device__ __forceinline__ void ft_load_vec(const T* data, int64_t vec_idx, T* out);
template <>
__device__ __forceinline__ void ft_load_vec<float>(const float* data, int64_t vec_idx, float* out) {
    float4 v = __ldg(reinterpret_cast<const float4*>(data) + vec_idx);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}
template <>
__device__ __forceinline__ void ft_load_vec<__half>(const __half* data, int64_t vec_idx, __half* out) {
    float4 v = __ldg(reinterpret_cast<const float4*>(data) + vec_idx);
    const __half* h = reinterpret_cast<const __half*>(&v);
    #pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = h[e];
}
template <>
__device__ __forceinline__ void ft_load_vec<__nv_bfloat16>(const __nv_bfloat16* data, int64_t vec_idx, __nv_bfloat16* out) {
    float4 v = __ldg(reinterpret_cast<const float4*>(data) + vec_idx);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for (int e = 0; e < 8; ++e) out[e] = h[e];
}

namespace ftns {
constexpr int FT_BLOCK_THREADS = 256;
constexpr int FT_RADIX_BITS = 8;
constexpr int FT_RADIX_DIGITS = 1 << FT_RADIX_BITS; // 256
constexpr int FT_RADIX_MASK = FT_RADIX_DIGITS - 1;
constexpr int FT_NUM_BUCKETS = 256;       // 0..254 = 范围内桶；255 = bulk(enc>hi)
constexpr int FT_RANGE_BUCKETS = 255;     // [lo,hi] 线性映射到 [0,254]
constexpr int FT_MIN_ITEMS_PER_THREAD = 4;
constexpr int FT_MAX_ITEMS_PER_THREAD = 64;

__device__ __forceinline__ uint32_t getDigit(uint32_t val, int bit) {
    return (val >> bit) & FT_RADIX_MASK;
}

// enc -> bucket：enc<=lo->0；lo<enc<=hi->((enc-lo)*254)/span∈[0,254]；enc>hi->255
// 用每 block 预计算的浮点倒数 inv_span = FT_RANGE_BUCKETS/span 做映射，
// 避免每元素 64-bit 整数除法（GPU 上软件模拟，上百周期）。映射对 enc 单调非降，
// 与整数版同序，故 bucket<T 全选 + 边界桶精确选 的正确性不变。
__device__ __forceinline__ int enc_to_bucket(uint32_t enc, uint32_t lo, uint32_t hi, float inv_span) {
    if (enc <= lo) return 0;
    if (enc > hi) return FT_NUM_BUCKETS - 1;           // 255 = bulk
    int b = (int)((float)(enc - lo) * inv_span);
    if (b > FT_RANGE_BUCKETS - 1) b = FT_RANGE_BUCKETS - 1;  // 钳到 254
    if (b < 0) b = 0;
    return b;
}
__device__ __forceinline__ float ft_inv_span(uint32_t span) {
    return (float)FT_RANGE_BUCKETS / (float)span;
}

} // namespace ftns

// ---------------- N<=k 退化：直接返回全部输入 ----------------
template <typename T>
__global__ void ft_copy_all_kernel(const T* __restrict__ in, int B, int N,
                                   T* __restrict__ outv, int32_t* __restrict__ outi) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * N;
    if (idx >= total) return;
    outv[idx] = in[idx];
    outi[idx] = (int32_t)(idx % N);
}
template <typename T>
void ft_copy_all(const T* in, int B, int N, T* outv, int32_t* outi, cudaStream_t stream) {
    int64_t total = (int64_t)B * N;
    int blk = 256; int64_t gr = (total + blk - 1) / blk; if (gr < 1) gr = 1;
    ft_copy_all_kernel<T><<<(int)gr, blk, 0, stream>>>(in, B, N, outv, outi);
}

// ---------------- 1. sample pass：每行单 block 求采样 min 与第 k 小 enc ----------------
// 采样以「前缀连续」方式覆盖整行最前面 sample_cnt*256 个连续元素（sample_stride=256），
// 每段由 256 线程连续读 256 个元素 —— warp 内 lane 连续 => 合并访问。
// in-block 4-pass radix 求采样第 k 小作分桶上界 hi。
template <typename T, int BLOCK_THREADS_>
__global__ void flashtopk_sample_kernel(
    const T* __restrict__ in, int N, int k,
    uint32_t sample_stride, int sample_cnt,
    uint32_t* __restrict__ lo_out,   // [B]
    uint32_t* __restrict__ hi_out,   // [B]
    uint32_t* __restrict__ span_out, // [B]
    uint32_t enc_xor                 // 0=min；kRevMask=max（翻转编码序，免取负）
) {
    using namespace ftns;
    int row = blockIdx.x;
    int tidx = threadIdx.x;
    const T* data = in + (size_t)row * N;

    typedef cub::BlockScan<uint32_t, BLOCK_THREADS_> BlockScan;
    union __align__(16) TempStorage {
        uint32_t digit_counters[FT_RADIX_DIGITS];
        typename BlockScan::TempStorage scan_storage;
        uint32_t reduce_tmp[BLOCK_THREADS_];
    };
    __shared__ TempStorage ts;
    __shared__ uint32_t s_min;
    __shared__ uint32_t s_desired;
    __shared__ uint32_t s_kfind;

    // ---- 局部 min（pass 前先求 lo）：分段连续合并读 ----
    uint32_t local_min = 0xFFFFFFFFu;
    for (int c = 0; c < sample_cnt; ++c) {
        int64_t pos = (int64_t)c * sample_stride + tidx;
        if (pos < N) {
            uint32_t e = ft_enc_ldg<T>(&data[pos]) ^ enc_xor;
            if (e < local_min) local_min = e;
        }
    }
    ts.reduce_tmp[tidx] = local_min;
    __syncthreads();
    for (int s = BLOCK_THREADS_ / 2; s > 0; s >>= 1) {
        if (tidx < s && ts.reduce_tmp[tidx + s] < ts.reduce_tmp[tidx])
            ts.reduce_tmp[tidx] = ts.reduce_tmp[tidx + s];
        __syncthreads();
    }
    if (tidx == 0) { s_min = ts.reduce_tmp[0]; s_desired = 0u; s_kfind = (uint32_t)k; }
    __syncthreads();

    // ---- in-block radix：求采样第 k 小 enc（fp32 4-pass / fp16 2-pass）----
    const int num_passes = EncTraits<T>::kPasses;
    const int start_bit  = EncTraits<T>::kBits - FT_RADIX_BITS;
    uint32_t desiredMask = 0u;
    for (int pass = 0; pass < num_passes; ++pass) {
        int current_bit = start_bit - FT_RADIX_BITS * pass;
        if (tidx < FT_RADIX_DIGITS) ts.digit_counters[tidx] = 0;
        __syncthreads();
        uint32_t desired = s_desired;
        for (int c = 0; c < sample_cnt; ++c) {
            int64_t pos = (int64_t)c * sample_stride + tidx;
            if (pos < N) {
                uint32_t e = ft_enc_ldg<T>(&data[pos]) ^ enc_xor;
                if ((e & desiredMask) == (desired & desiredMask))
                    atomicAdd(&ts.digit_counters[getDigit(e, current_bit)], 1u);
            }
        }
        __syncthreads();
        uint32_t dc = (tidx < FT_RADIX_DIGITS) ? ts.digit_counters[tidx] : 0u;
        __syncthreads();
        uint32_t cumsum;
        BlockScan(ts.scan_storage).InclusiveSum(dc, cumsum);
        __syncthreads();
        uint32_t kf = s_kfind;
        if (tidx < FT_RADIX_DIGITS) {
            uint32_t cum_left = cumsum - dc;
            if (cum_left < kf && kf <= cumsum) {
                s_desired = desired | (tidx << current_bit);
                s_kfind = kf - cum_left;
            }
        }
        __syncthreads();
        desiredMask |= (uint32_t)FT_RADIX_MASK << current_bit;
    }

    if (tidx == 0) {
        uint32_t lo = s_min;
        uint32_t hi = s_desired;          // 采样第 k 小 enc（>= 全集第 k 小）
        if (hi < lo) hi = lo;
        uint32_t span = hi - lo; if (span < 1u) span = 1u;
        lo_out[row] = lo; hi_out[row] = hi; span_out[row] = span;
    }
}

// ---------------- 2. hist pass：全量 1 遍，per-row 直方图 ----------------
template <typename T, int BLOCK_THREADS_>
__global__ void flashtopk_hist_kernel(
    const T* __restrict__ in, uint32_t slice_size,
    int items_per_thread, uint32_t blocks_per_slice, uint32_t num_slices,
    const uint32_t* __restrict__ lo_in, const uint32_t* __restrict__ hi_in,
    const uint32_t* __restrict__ span_in,
    uint32_t* __restrict__ g_hist,   // [num_slices * 256]，需预清零
    uint32_t enc_xor
) {
    using namespace ftns;
    int items_per_block = items_per_thread * BLOCK_THREADS_;
    int tidx = threadIdx.x;
    uint32_t block_idx = blockIdx.x;
    uint32_t slice = block_idx / blocks_per_slice;
    uint32_t blk_in_slice = block_idx % blocks_per_slice;
    if (slice >= num_slices) return;

    __shared__ uint32_t s_hist[FT_NUM_BUCKETS];
    for (int i = tidx; i < FT_NUM_BUCKETS; i += BLOCK_THREADS_) s_hist[i] = 0;
    __syncthreads();

    const T* data = in + (size_t)slice * slice_size;
    uint32_t lo = lo_in[slice], hi = hi_in[slice], span = span_in[slice];
    float inv_span = ft_inv_span(span);
    for (int i = 0; i < items_per_thread; ++i) {
        int64_t idx = (int64_t)blk_in_slice * items_per_block + (int64_t)i * BLOCK_THREADS_ + tidx;
        if (idx < (int64_t)slice_size) {
            uint32_t e = ft_enc_ldg<T>(&data[idx]) ^ enc_xor;
            int b = enc_to_bucket(e, lo, hi, inv_span);
            atomicAdd(&s_hist[b], 1u);
        }
    }
    __syncthreads();
    for (int i = tidx; i < FT_NUM_BUCKETS; i += BLOCK_THREADS_) {
        uint32_t v = s_hist[i];
        if (v) atomicAdd(&g_hist[slice * FT_NUM_BUCKETS + i], v);
    }
}

// ---------------- 2v. hist pass（向量化）：每线程一条 float4 读 VEC 个元素 ----------------
// 仅当 slice_size % VEC == 0 时启用（行首 16B 对齐）。块的元素区间 [vbase*VEC, +)，
// 以 float4 为单位 grid-stride 覆盖；warp 内 lane 连续 float4 => 512B 合并，load 指令 /VEC。
template <typename T, int BLOCK_THREADS_>
__global__ void flashtopk_hist_vec_kernel(
    const T* __restrict__ in, uint32_t slice_size,
    int items_per_thread, uint32_t blocks_per_slice, uint32_t num_slices,
    const uint32_t* __restrict__ lo_in, const uint32_t* __restrict__ hi_in,
    const uint32_t* __restrict__ span_in,
    uint32_t* __restrict__ g_hist,
    uint32_t enc_xor
) {
    using namespace ftns;
    constexpr int VEC = VecTraits<T>::VEC;
    int items_per_block = items_per_thread * BLOCK_THREADS_;
    int vec_per_block = items_per_block / VEC;
    int niter = (vec_per_block + BLOCK_THREADS_ - 1) / BLOCK_THREADS_;
    int tidx = threadIdx.x;
    uint32_t block_idx = blockIdx.x;
    uint32_t slice = block_idx / blocks_per_slice;
    uint32_t blk_in_slice = block_idx % blocks_per_slice;
    if (slice >= num_slices) return;
    int64_t num_vec = (int64_t)slice_size / VEC;
    int64_t vbase = (int64_t)blk_in_slice * vec_per_block;

    __shared__ uint32_t s_hist[FT_NUM_BUCKETS];
    for (int i = tidx; i < FT_NUM_BUCKETS; i += BLOCK_THREADS_) s_hist[i] = 0;
    __syncthreads();

    const T* data = in + (size_t)slice * slice_size;
    uint32_t lo = lo_in[slice], hi = hi_in[slice], span = span_in[slice];
    float inv_span = ft_inv_span(span);
    for (int it = 0; it < niter; ++it) {
        int64_t v = vbase + (int64_t)it * BLOCK_THREADS_ + tidx;
        if (v < num_vec && v < vbase + vec_per_block) {
            T tmp[VEC];
            ft_load_vec<T>(data, v, tmp);
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                uint32_t enc = ft_enc_val<T>(tmp[e]) ^ enc_xor;
                int b = enc_to_bucket(enc, lo, hi, inv_span);
                atomicAdd(&s_hist[b], 1u);
            }
        }
    }
    __syncthreads();
    for (int i = tidx; i < FT_NUM_BUCKETS; i += BLOCK_THREADS_) {
        uint32_t v = s_hist[i];
        if (v) atomicAdd(&g_hist[slice * FT_NUM_BUCKETS + i], v);
    }
}

// ---------------- 3. threshold：cumsum 求精确 T 与 count_below ----------------
template <int BLOCK_THREADS_>
__global__ void flashtopk_threshold_kernel(
    const uint32_t* __restrict__ g_hist, uint32_t num_slices, uint32_t k,
    uint32_t* __restrict__ T_out, uint32_t* __restrict__ cb_out
) {
    using namespace ftns;
    int row = blockIdx.x;
    int tidx = threadIdx.x;
    if ((uint32_t)row >= num_slices) return;

    typedef cub::BlockScan<uint32_t, BLOCK_THREADS_> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ uint32_t s_T;
    __shared__ uint32_t s_cb;
    if (tidx == 0) { s_T = FT_NUM_BUCKETS - 1; s_cb = 0; }
    __syncthreads();

    uint32_t cnt = (tidx < FT_NUM_BUCKETS) ? g_hist[row * FT_NUM_BUCKETS + tidx] : 0u;
    uint32_t cumsum;
    BlockScan(scan_storage).InclusiveSum(cnt, cumsum);
    __syncthreads();
    if (tidx < FT_NUM_BUCKETS) {
        uint32_t cum_left = cumsum - cnt;
        // T = 最小 bucket 满足 cumsum>=k；count_below = 该 bucket 之前的累计(<k)
        if (cum_left < k && k <= cumsum) {
            s_T = (uint32_t)tidx;
            s_cb = cum_left;
        }
    }
    __syncthreads();
    if (tidx == 0) { T_out[row] = s_T; cb_out[row] = s_cb; }
}

// ---------------- 4. gather pass：全量 1 遍 ----------------
//   bucket<T -> 直接写 out[0:count_below]；bucket==T -> 收集进候选 buffer。
template <typename T, int BLOCK_THREADS_>
__global__ void flashtopk_gather_kernel(
    const T* __restrict__ in, uint32_t slice_size, uint32_t K,
    int items_per_thread, uint32_t blocks_per_slice, uint32_t num_slices,
    const uint32_t* __restrict__ lo_in, const uint32_t* __restrict__ hi_in,
    const uint32_t* __restrict__ span_in,
    const uint32_t* __restrict__ T_in, const uint32_t* __restrict__ cb_in,
    uint32_t cap,
    uint32_t* __restrict__ lt_counter,   // [num_slices] zeros
    uint32_t* __restrict__ cand_counter, // [num_slices] zeros
    uint32_t* __restrict__ overflow,     // [1] zeros
    T* __restrict__ cand_val,            // [num_slices * cap]
    int32_t* __restrict__ cand_idx,      // [num_slices * cap]
    T* __restrict__ out_val,             // [num_slices, K]
    int32_t* __restrict__ out_idx,
    uint32_t enc_xor
) {
    using namespace ftns;
    int items_per_block = items_per_thread * BLOCK_THREADS_;
    int tidx = threadIdx.x;
    uint32_t block_idx = blockIdx.x;
    uint32_t slice = block_idx / blocks_per_slice;
    uint32_t blk_in_slice = block_idx % blocks_per_slice;
    if (slice >= num_slices) return;

    const T* data = in + (size_t)slice * slice_size;
    T*       outv = out_val + (size_t)slice * K;
    int32_t* outi = out_idx + (size_t)slice * K;
    uint32_t lo = lo_in[slice], hi = hi_in[slice], span = span_in[slice];
    float inv_span = ft_inv_span(span);
    uint32_t T_b = T_in[slice];
    T* cv = cand_val + (size_t)slice * cap;
    int32_t* ci = cand_idx + (size_t)slice * cap;

    typedef cub::BlockScan<int, BLOCK_THREADS_> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int s_base;

    for (int i = 0; i < items_per_thread; ++i) {
        int64_t idx = (int64_t)blk_in_slice * items_per_block + (int64_t)i * BLOCK_THREADS_ + tidx;
        bool valid = idx < (int64_t)slice_size;
        uint32_t e = valid ? (ft_enc_ldg<T>(&data[idx]) ^ enc_xor) : 0u;
        int b = valid ? enc_to_bucket(e, lo, hi, inv_span) : (FT_NUM_BUCKETS); // 无效给越界桶

        // bucket < T：确定入选，block 内 scan 抢全局连续槽位
        bool is_lt = valid && (b < (int)T_b);
        int local_idx, total;
        BlockScan(scan_storage).ExclusiveSum(is_lt ? 1 : 0, local_idx, total);
        __syncthreads();
        if (tidx == 0 && total > 0) s_base = (int)atomicAdd(&lt_counter[slice], (uint32_t)total);
        __syncthreads();
        if (is_lt) {
            int w = s_base + local_idx;
            if (w < (int)K) { outv[w] = data[idx]; outi[w] = (int)idx; }
        }
        __syncthreads();

        // bucket == T：收集进候选 buffer
        bool is_eq = valid && (b == (int)T_b);
        if (is_eq) {
            uint32_t slot = atomicAdd(&cand_counter[slice], 1u);
            if (slot < cap) { cv[slot] = data[idx]; ci[slot] = (int)idx; }
            else atomicMax(overflow, 1u);
        }
    }
}

// ---------------- 4v. gather pass（向量化）：每线程一条 float4 读 VEC 个元素 ----------------
// 仅当 slice_size % VEC == 0 时启用。每线程本批 VEC 个元素中统计 lt 个数做一次 BlockScan，
// 抢到连续输出槽位后顺序写出；eq 元素逐个 atomic 收集进候选 buffer（语义同标量版）。
template <typename T, int BLOCK_THREADS_>
__global__ void flashtopk_gather_vec_kernel(
    const T* __restrict__ in, uint32_t slice_size, uint32_t K,
    int items_per_thread, uint32_t blocks_per_slice, uint32_t num_slices,
    const uint32_t* __restrict__ lo_in, const uint32_t* __restrict__ hi_in,
    const uint32_t* __restrict__ span_in,
    const uint32_t* __restrict__ T_in, const uint32_t* __restrict__ cb_in,
    uint32_t cap,
    uint32_t* __restrict__ lt_counter,
    uint32_t* __restrict__ cand_counter,
    uint32_t* __restrict__ overflow,
    T* __restrict__ cand_val,
    int32_t* __restrict__ cand_idx,
    T* __restrict__ out_val,
    int32_t* __restrict__ out_idx,
    uint32_t enc_xor
) {
    using namespace ftns;
    constexpr int VEC = VecTraits<T>::VEC;
    int items_per_block = items_per_thread * BLOCK_THREADS_;
    int vec_per_block = items_per_block / VEC;
    int niter = (vec_per_block + BLOCK_THREADS_ - 1) / BLOCK_THREADS_;
    int tidx = threadIdx.x;
    uint32_t block_idx = blockIdx.x;
    uint32_t slice = block_idx / blocks_per_slice;
    uint32_t blk_in_slice = block_idx % blocks_per_slice;
    if (slice >= num_slices) return;
    int64_t num_vec = (int64_t)slice_size / VEC;
    int64_t vbase = (int64_t)blk_in_slice * vec_per_block;

    const T* data = in + (size_t)slice * slice_size;
    T*       outv = out_val + (size_t)slice * K;
    int32_t* outi = out_idx + (size_t)slice * K;
    uint32_t lo = lo_in[slice], hi = hi_in[slice], span = span_in[slice];
    float inv_span = ft_inv_span(span);
    uint32_t T_b = T_in[slice];
    T* cv = cand_val + (size_t)slice * cap;
    int32_t* ci = cand_idx + (size_t)slice * cap;

    typedef cub::BlockScan<int, BLOCK_THREADS_> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;
    __shared__ int s_base;

    for (int it = 0; it < niter; ++it) {
        int64_t v = vbase + (int64_t)it * BLOCK_THREADS_ + tidx;
        bool active = (v < num_vec) && (v < vbase + vec_per_block);
        T tmp[VEC];
        int b[VEC];
        int my_lt = 0;
        if (active) {
            ft_load_vec<T>(data, v, tmp);
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                uint32_t enc = ft_enc_val<T>(tmp[e]) ^ enc_xor;
                b[e] = enc_to_bucket(enc, lo, hi, inv_span);
                if (b[e] < (int)T_b) my_lt++;
            }
        }
        int local_off, total;
        BlockScan(scan_storage).ExclusiveSum(my_lt, local_off, total);
        __syncthreads();
        if (tidx == 0 && total > 0) s_base = (int)atomicAdd(&lt_counter[slice], (uint32_t)total);
        __syncthreads();
        if (active) {
            int w = s_base + local_off;
            int64_t base_elem = v * VEC;
            #pragma unroll
            for (int e = 0; e < VEC; ++e) {
                if (b[e] < (int)T_b) {
                    if (w < (int)K) { outv[w] = tmp[e]; outi[w] = (int)(base_elem + e); }
                    ++w;
                } else if (b[e] == (int)T_b) {
                    uint32_t slot = atomicAdd(&cand_counter[slice], 1u);
                    if (slot < cap) { cv[slot] = tmp[e]; ci[slot] = (int)(base_elem + e); }
                    else atomicMax(overflow, 1u);
                }
            }
        }
        __syncthreads();
    }
}

// ---------------- 5. boundary：候选 buffer 中精确取 need=k-count_below 个最小 ----------------
//   常规路径用 gather 收集的候选 buffer（pop<=cap）。若边界桶 pop>cap（宽分布会发生），
//   候选 buffer 被截断，改为直接在整行 input 上重扫 bucket==T 的元素，保证精确。
template <typename T, int BLOCK_THREADS_>
__global__ void flashtopk_boundary_kernel(
    const T* __restrict__ in, uint32_t slice_size,
    const uint32_t* __restrict__ lo_in, const uint32_t* __restrict__ hi_in,
    const uint32_t* __restrict__ span_in,
    uint32_t K, uint32_t num_slices, uint32_t cap,
    const uint32_t* __restrict__ T_in, const uint32_t* __restrict__ cb_in,
    const uint32_t* __restrict__ cand_counter,
    const T* __restrict__ cand_val,   // [num_slices * cap]
    const int32_t* __restrict__ cand_idx,
    T* __restrict__ out_val,          // [num_slices, K]
    int32_t* __restrict__ out_idx,
    uint32_t enc_xor
) {
    using namespace ftns;
    int row = blockIdx.x;
    int tidx = threadIdx.x;
    if ((uint32_t)row >= num_slices) return;

    uint32_t count_below = cb_in[row];
    if (count_below >= K) return;        // 无需边界补齐
    uint32_t need = K - count_below;
    uint32_t pop = cand_counter[row];
    bool use_full = (pop > cap);         // overflow：候选 buffer 不可信，重扫整行
    uint32_t niter = use_full ? slice_size : pop;
    if (pop == 0) return;

    const T* data = in + (size_t)row * slice_size;
    uint32_t lo = lo_in[row], hi = hi_in[row], span = span_in[row];
    float inv_span = ft_inv_span(span);
    uint32_t T_b = T_in[row];
    const T* cv = cand_val + (size_t)row * cap;
    const int32_t* ci = cand_idx + (size_t)row * cap;
    T*       outv = out_val + (size_t)row * K;
    int32_t* outi = out_idx + (size_t)row * K;

    typedef cub::BlockScan<uint32_t, BLOCK_THREADS_> BlockScanU;
    typedef cub::BlockScan<int, BLOCK_THREADS_> BlockScanI;
    union __align__(16) TempStorage {
        uint32_t digit_counters[FT_RADIX_DIGITS];
        typename BlockScanU::TempStorage scanu;
        typename BlockScanI::TempStorage scani;
    };
    __shared__ TempStorage ts;
    __shared__ uint32_t s_desired;
    __shared__ uint32_t s_kfind;
    __shared__ int s_cnt_lt;
    __shared__ int s_w_lt;
    __shared__ int s_w_eq;

    if (tidx == 0) { s_desired = 0u; s_kfind = need; }
    __syncthreads();

    // in-block radix：求边界桶元素中 need-th 小 enc (= pivot)（fp32 4-pass / fp16 2-pass）
    const int num_passes = EncTraits<T>::kPasses;
    const int start_bit  = EncTraits<T>::kBits - FT_RADIX_BITS;
    uint32_t desiredMask = 0u;
    for (int pass = 0; pass < num_passes; ++pass) {
        int current_bit = start_bit - FT_RADIX_BITS * pass;
        if (tidx < FT_RADIX_DIGITS) ts.digit_counters[tidx] = 0;
        __syncthreads();
        uint32_t desired = s_desired;
        for (uint32_t i = tidx; i < niter; i += BLOCK_THREADS_) {
            uint32_t e; bool member;
            if (use_full) {
                e = ft_enc_ldg<T>(&data[i]) ^ enc_xor;
                member = (enc_to_bucket(e, lo, hi, inv_span) == (int)T_b);
            } else {
                e = ft_enc_ldg<T>(&cv[i]) ^ enc_xor;
                member = true;
            }
            if (member && (e & desiredMask) == (desired & desiredMask))
                atomicAdd(&ts.digit_counters[getDigit(e, current_bit)], 1u);
        }
        __syncthreads();
        uint32_t dc = (tidx < FT_RADIX_DIGITS) ? ts.digit_counters[tidx] : 0u;
        __syncthreads();
        uint32_t cumsum;
        BlockScanU(ts.scanu).InclusiveSum(dc, cumsum);
        __syncthreads();
        uint32_t kf = s_kfind;
        if (tidx < FT_RADIX_DIGITS) {
            uint32_t cum_left = cumsum - dc;
            if (cum_left < kf && kf <= cumsum) {
                s_desired = desired | (tidx << current_bit);
                s_kfind = kf - cum_left;
            }
        }
        __syncthreads();
        desiredMask |= (uint32_t)FT_RADIX_MASK << current_bit;
    }
    uint32_t pivot = s_desired;

    // 统计 enc<pivot 的个数 cnt_lt（这些全选），eq 取 need-cnt_lt 个
    if (tidx == 0) { s_cnt_lt = 0; s_w_lt = 0; s_w_eq = 0; }
    __syncthreads();
    int my_lt = 0;
    for (uint32_t i = tidx; i < niter; i += BLOCK_THREADS_) {
        uint32_t e; bool member;
        if (use_full) {
            e = ft_enc_ldg<T>(&data[i]) ^ enc_xor;
            member = (enc_to_bucket(e, lo, hi, inv_span) == (int)T_b);
        } else {
            e = ft_enc_ldg<T>(&cv[i]) ^ enc_xor;
            member = true;
        }
        if (member && e < pivot) my_lt++;
    }
    int agg_lt, total_lt;
    BlockScanI(ts.scani).ExclusiveSum(my_lt, agg_lt, total_lt);
    __syncthreads();
    if (tidx == 0) s_cnt_lt = total_lt;
    __syncthreads();
    int cnt_lt = s_cnt_lt;
    int eq_take = (int)need - cnt_lt; if (eq_take < 0) eq_take = 0;

    // 写出：enc<pivot -> [count_below, count_below+cnt_lt)；enc==pivot 取 eq_take 个补到 need
    for (uint32_t i = tidx; i < niter; i += BLOCK_THREADS_) {
        T val; uint32_t e; int oidx; bool member;
        if (use_full) {
            val = data[i];
            e = ft_enc_val<T>(val) ^ enc_xor;
            member = (enc_to_bucket(e, lo, hi, inv_span) == (int)T_b);
            oidx = (int)i;
        } else {
            val = cv[i];
            e = ft_enc_val<T>(val) ^ enc_xor;
            member = true;
            oidx = ci[i];
        }
        if (!member) continue;
        if (e < pivot) {
            int w = (int)count_below + atomicAdd(&s_w_lt, 1);
            if (w < (int)K) { outv[w] = val; outi[w] = oidx; }
        } else if (e == pivot) {
            int o = atomicAdd(&s_w_eq, 1);
            if (o < eq_take) {
                int w = (int)count_below + cnt_lt + o;
                if (w < (int)K) { outv[w] = val; outi[w] = oidx; }
            }
        }
    }
}

// ---------------- 阈值复用选择：跳过 sample/hist/threshold ----------------
//   配合融合 _flat_kernel 使用：调用方已持有每行收敛阈值桶 th 与分桶坐标
//   (origin, inv_delta)。流式 gate 只保证 th >= 真实阈值桶（为收紧 buffer 而“宁可多收”），
//   故 th 是真实阈值桶的【安全上界】：bucket>th 的元素分数必高于真实阈值，绝不入选，可安全剪枝。
//   若 th 正好夹住 top-k 边界，则 bucket<th 的候选必然全入选，只需在 bucket==th
//   的边界桶内 select 补齐剩余名额。若 th 只是较松的安全上界（bucket<th 已经 >=K，
//   或 bucket<=th 不足 K），则回退到对 bucket<=th 的全候选 radix-select，保证正确性。
//   buffer 尾部未初始化区域由 qcount 限界跳过。out_idx 默认是 buffer 内列位置；
//   若传入 buf_idx/sample_idx，则直接写最终 corpus id。
template <typename T, typename OT, int BLOCK_THREADS_>
__global__ void flashtopk_select_thr_kernel(
    const T* __restrict__ buf, int BUF, int K,
    const OT* __restrict__ origin_in,    // [R]，dtype 可与 buf 不同（桶坐标自身精度）
    const OT* __restrict__ inv_delta_in, // [R]
    const int32_t* __restrict__ th_in,      // [R]
    const int32_t* __restrict__ qcount_in,  // [R] 每行有效候选数；可超过 BUF，kernel 内 clamp
    const int32_t* __restrict__ buf_idx_in,     // optional [R, BUF]，若非空则 out_idx 写 global idx
    const int32_t* __restrict__ sample_idx_in,  // optional [R, K]，buffer 前 K 个 sample 位置的 global idx
    int NB,
    T* __restrict__ out_val,                // [R, K]
    int32_t* __restrict__ out_idx,
    uint32_t enc_xor
) {
    using namespace ftns;
    int row = blockIdx.x;
    int tidx = threadIdx.x;
    const T* data = buf + (size_t)row * BUF;
    const int32_t* idx_data = buf_idx_in ? (buf_idx_in + (size_t)row * BUF) : nullptr;
    const int32_t* sample_idx = sample_idx_in ? (sample_idx_in + (size_t)row * K) : nullptr;
    T*       outv = out_val + (size_t)row * K;
    int32_t* outi = out_idx + (size_t)row * K;
    // 桶号算术统一在 fp32 进行，与 _flat_kernel 逐位一致（其 origin/inv_delta 亦 .to(fp32) 后再算）。
    float origin = ft_to_float<OT>(origin_in[row]);
    float inv_delta = ft_to_float<OT>(inv_delta_in[row]);
    int th = th_in[row];
    int limit = qcount_in ? qcount_in[row] : BUF;
    if (limit > BUF) limit = BUF;
    if (limit < 0) limit = 0;

    typedef cub::BlockScan<uint32_t, BLOCK_THREADS_> BlockScanU;
    union __align__(16) TempStorage {
        uint32_t digit_counters[FT_RADIX_DIGITS];
        typename BlockScanU::TempStorage scanu;
    };
    __shared__ TempStorage ts;
    __shared__ uint32_t s_desired;
    __shared__ uint32_t s_kfind;
    __shared__ int s_cnt_bucket_lt;
    __shared__ int s_cnt_bucket_eq;
    __shared__ int s_use_boundary;
    __shared__ int s_select_all;
    __shared__ int s_k_eq;
    __shared__ int s_cnt_lt;
    __shared__ int s_w_pre;
    __shared__ int s_w_lt;
    __shared__ int s_w_eq;

    // ---- 先判断能否走边界桶优化：bucket<th 直接写，bucket==th 内 select ----
    if (tidx == 0) { s_cnt_bucket_lt = 0; s_cnt_bucket_eq = 0; }
    __syncthreads();
    int my_bucket_lt = 0;
    int my_bucket_eq = 0;
    for (int i = tidx; i < limit; i += BLOCK_THREADS_) {
        T v = data[i];
        float vf = ft_to_float<T>(v);
        if (!isfinite(vf)) continue;
        int braw = (int)((vf - origin) * inv_delta);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (b < th) my_bucket_lt++;
        else if (b == th) my_bucket_eq++;
    }
    atomicAdd(&s_cnt_bucket_lt, my_bucket_lt);
    atomicAdd(&s_cnt_bucket_eq, my_bucket_eq);
    __syncthreads();

    if (tidx == 0) {
        int cnt_bucket_lt = s_cnt_bucket_lt;
        int cnt_bucket_eq = s_cnt_bucket_eq;
        int k_eq = K - cnt_bucket_lt;
        int use_boundary = (cnt_bucket_lt < K) && (k_eq > 0) && (k_eq <= cnt_bucket_eq);
        int select_all = (cnt_bucket_lt + cnt_bucket_eq) < K;
        s_use_boundary = use_boundary;
        s_select_all = select_all;
        s_k_eq = use_boundary ? k_eq : K;
        s_desired = 0u;
        s_kfind = (uint32_t)s_k_eq;
    }
    __syncthreads();

    // ---- in-block radix：求边界桶或全部候选中的 K-th 小 enc (= pivot) ----
    const int num_passes = EncTraits<T>::kPasses;
    const int start_bit  = EncTraits<T>::kBits - FT_RADIX_BITS;
    uint32_t desiredMask = 0u;
    for (int pass = 0; pass < num_passes; ++pass) {
        int current_bit = start_bit - FT_RADIX_BITS * pass;
        if (tidx < FT_RADIX_DIGITS) ts.digit_counters[tidx] = 0;
        __syncthreads();
        uint32_t desired = s_desired;
        for (int i = tidx; i < limit; i += BLOCK_THREADS_) {
            T v = data[i];
            float vf = ft_to_float<T>(v);
            if (!isfinite(vf)) continue;
            int braw = (int)((vf - origin) * inv_delta);
            int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
            if (s_use_boundary) {
                if (b != th) continue;
            } else if (s_select_all) {
                // The bounded buffer may contain fewer than K candidates in bucket<=th
                // even when the global histogram says th is sufficient. Fall back to
                // selecting over the whole valid buffer so every output slot gets a real id.
            } else {
                if (b > th) continue;
            }
            uint32_t e = ft_enc_val<T>(v) ^ enc_xor;
            if ((e & desiredMask) == (desired & desiredMask))
                atomicAdd(&ts.digit_counters[getDigit(e, current_bit)], 1u);
        }
        __syncthreads();
        uint32_t dc = (tidx < FT_RADIX_DIGITS) ? ts.digit_counters[tidx] : 0u;
        __syncthreads();
        uint32_t cumsum;
        BlockScanU(ts.scanu).InclusiveSum(dc, cumsum);
        __syncthreads();
        uint32_t kf = s_kfind;
        if (tidx < FT_RADIX_DIGITS) {
            uint32_t cum_left = cumsum - dc;
            if (cum_left < kf && kf <= cumsum) {
                s_desired = desired | (tidx << current_bit);
                s_kfind = kf - cum_left;
            }
        }
        __syncthreads();
        desiredMask |= (uint32_t)FT_RADIX_MASK << current_bit;
    }
    uint32_t pivot = s_desired;

    // ---- 统计 select 集合内 enc<pivot 个数；enc==pivot 取剩余名额补齐 ----
    if (tidx == 0) { s_cnt_lt = 0; s_w_pre = 0; s_w_lt = 0; s_w_eq = 0; }
    __syncthreads();
    int my_lt = 0;
    for (int i = tidx; i < limit; i += BLOCK_THREADS_) {
        T v = data[i];
        float vf = ft_to_float<T>(v);
        if (!isfinite(vf)) continue;
        int braw = (int)((vf - origin) * inv_delta);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (s_use_boundary) {
            if (b != th) continue;
        } else if (s_select_all) {
            // Underfilled threshold bucket: count against all buffered candidates.
        } else {
            if (b > th) continue;
        }
        uint32_t e = ft_enc_val<T>(v) ^ enc_xor;
        if (e < pivot) my_lt++;
    }
    atomicAdd(&s_cnt_lt, my_lt);
    __syncthreads();
    int cnt_lt = s_cnt_lt;
    int pre_take = s_use_boundary ? s_cnt_bucket_lt : 0;
    int target_k = s_use_boundary ? s_k_eq : K;
    int eq_take = target_k - cnt_lt; if (eq_take < 0) eq_take = 0;

    // ---- 写出：边界优化下 bucket<th 直接写；select 集合再按 pivot 补齐 ----
    for (int i = tidx; i < limit; i += BLOCK_THREADS_) {
        T v = data[i];
        float vf = ft_to_float<T>(v);
        if (!isfinite(vf)) continue;
        int braw = (int)((vf - origin) * inv_delta);
        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
        if (s_use_boundary && b < th) {
            int w = atomicAdd(&s_w_pre, 1);
            if (w < K) {
                outv[w] = v;
                outi[w] = idx_data ? ((i < K && sample_idx) ? sample_idx[i] : idx_data[i]) : i;
            }
            continue;
        }
        if (s_use_boundary) {
            if (b != th) continue;
        } else if (s_select_all) {
            // Underfilled threshold bucket: write top-K from the full valid buffer.
        } else {
            if (b > th) continue;
        }
        uint32_t e = ft_enc_val<T>(v) ^ enc_xor;
        if (e < pivot) {
            int w = atomicAdd(&s_w_lt, 1);
            int out_pos = pre_take + w;
            if (out_pos < K) {
                outv[out_pos] = v;
                outi[out_pos] = idx_data ? ((i < K && sample_idx) ? sample_idx[i] : idx_data[i]) : i;
            }
        } else if (e == pivot) {
            int o = atomicAdd(&s_w_eq, 1);
            if (o < eq_take) {
                int w = pre_take + cnt_lt + o;
                if (w < K) {
                    outv[w] = v;
                    outi[w] = idx_data ? ((i < K && sample_idx) ? sample_idx[i] : idx_data[i]) : i;
                }
            }
        }
    }
}

template <typename T, typename OT>
void flash_topk_select_thr(const T* buf, int R, int BUF, int K,
                           const OT* origin, const OT* inv_delta,
                           const int32_t* th, const int32_t* qcount, int NB,
                           T* out_val, int* out_idx, cudaStream_t stream,
                           const int32_t* buf_idx = nullptr,
                           const int32_t* sample_idx = nullptr,
                           uint32_t enc_xor = 0u) {
    using namespace ftns;
    if (R <= 0 || K <= 0) return;
    // out_idx 兜底清零：BUF 溢出导致欠填时，未写槽位仍是合法索引(0)，避免下游 gather 越界。
    cudaMemsetAsync(out_idx, 0, sizeof(int) * (size_t)R * K, stream);
    flashtopk_select_thr_kernel<T, OT, FT_BLOCK_THREADS><<<R, FT_BLOCK_THREADS, 0, stream>>>(
        buf, BUF, K, origin, inv_delta, th, qcount, buf_idx, sample_idx, NB, out_val, out_idx, enc_xor);
}

// ==================================================================
//   packed (4B: fp16_score<<16 | off16) 分段 K-min select（fp16 专用）
// ==================================================================
//   buf_pack[R, NSEG*CAP] int32：段 seg 占用 [seg*CAP, seg*CAP+CAP)，每槽 packed。
//   segcnt[R, NSEG]：每段实际候选数（可超 CAP，kernel 内 clamp）。空槽不读、无需 fill。
//   写回 _seg_pack_kernel 已只存过 gate 的候选 → 真 top-K 全在其中，这里只需对有效槽
//   做 in-block radix K-min，并在寄存器内解码 score=packed>>16、id=seg*BLK+off16。
//   消除 Python 侧 (>>/&/view/gather/topk_min) 那串 elementwise 解码开销。
template <int BLOCK_THREADS_, int NSEG_MAX>
__global__ void flashtopk_select_packed_kernel(
    const int32_t* __restrict__ buf_pack, int CAP, int NSEG, int BLK, int M, int K,
    const int32_t* __restrict__ segcnt_in,   // [R, NSEG]
    __half* __restrict__ out_val,            // [R, K]
    int32_t* __restrict__ out_idx            // [R, K]
) {
    using namespace ftns;
    int row = blockIdx.x;
    int tidx = threadIdx.x;
    const int32_t* data = buf_pack + (size_t)row * NSEG * CAP;
    const int32_t* segc = segcnt_in + (size_t)row * NSEG;
    __half* outv = out_val + (size_t)row * K;
    int32_t* outi = out_idx + (size_t)row * K;

    // 段内紧凑遍历：s_segoff[seg] 为各段有效候选的 exclusive prefix sum，
    // s_totalvalid 为本行有效候选总数。把 compact 下标 j∈[0,totalvalid) 映射到
    // buffer 绝对位置 (seg*CAP + (j-s_segoff[seg]))，从而**完全跳过空槽**——
    // 迭代量 = 实际候选数（≈base qcount），不随 CAP 超配膨胀。
    __shared__ int s_segcnt[NSEG_MAX];
    __shared__ int s_segoff[NSEG_MAX + 1];
    __shared__ int s_totalvalid;
    for (int s = tidx; s < NSEG; s += BLOCK_THREADS_) {
        int c = segc[s];
        s_segcnt[s] = c > CAP ? CAP : (c < 0 ? 0 : c);
    }
    __syncthreads();
    if (tidx == 0) {
        int acc = 0;
        for (int s = 0; s < NSEG; ++s) { s_segoff[s] = acc; acc += s_segcnt[s]; }
        s_segoff[NSEG] = acc;
        s_totalvalid = acc;
    }
    __syncthreads();
    const int totalvalid = s_totalvalid;

    typedef cub::BlockScan<uint32_t, BLOCK_THREADS_> BlockScanU;
    union __align__(16) TempStorage {
        uint32_t digit_counters[FT_RADIX_DIGITS];
        typename BlockScanU::TempStorage scanu;
    };
    __shared__ TempStorage ts;
    __shared__ uint32_t s_desired;
    __shared__ uint32_t s_kfind;
    __shared__ int s_w_lt;
    __shared__ int s_w_eq;
    __shared__ int s_cnt_lt;

    if (tidx == 0) { s_desired = 0u; s_kfind = (uint32_t)K; }
    __syncthreads();

    // compact 下标 j -> buffer 绝对位置：二分 s_segoff 找所在段。
    #define FT_COMPACT_POS(j, pos)                                            \
        do {                                                                  \
            int _lo = 0, _hi = NSEG;                                          \
            while (_lo < _hi - 1) {                                           \
                int _mid = (_lo + _hi) >> 1;                                  \
                if (s_segoff[_mid] <= (j)) _lo = _mid; else _hi = _mid;       \
            }                                                                 \
            (pos) = _lo * CAP + ((j) - s_segoff[_lo]);                        \
        } while (0)

    // ---- 2-pass 8-bit radix：求有效候选内第 K 小 enc（pivot），fp16 -> 16 bit ----
    const int num_passes = 2;
    const int start_bit  = 16 - FT_RADIX_BITS;
    uint32_t desiredMask = 0u;
    for (int pass = 0; pass < num_passes; ++pass) {
        int current_bit = start_bit - FT_RADIX_BITS * pass;
        if (tidx < FT_RADIX_DIGITS) ts.digit_counters[tidx] = 0;
        __syncthreads();
        uint32_t desired = s_desired;
        for (int j = tidx; j < totalvalid; j += BLOCK_THREADS_) {
            int i; FT_COMPACT_POS(j, i);
            uint32_t pk = (uint32_t)data[i];
            uint16_t sb = (uint16_t)(pk >> 16);
            uint32_t e = (uint32_t)(sb ^ ((sb & 0x8000u) ? 0xffffu : 0x8000u));
            if ((e & desiredMask) == (desired & desiredMask))
                atomicAdd(&ts.digit_counters[getDigit(e, current_bit)], 1u);
        }
        __syncthreads();
        uint32_t dc = (tidx < FT_RADIX_DIGITS) ? ts.digit_counters[tidx] : 0u;
        __syncthreads();
        uint32_t cumsum;
        BlockScanU(ts.scanu).InclusiveSum(dc, cumsum);
        __syncthreads();
        uint32_t kf = s_kfind;
        if (tidx < FT_RADIX_DIGITS) {
            uint32_t cum_left = cumsum - dc;
            if (cum_left < kf && kf <= cumsum) {
                s_desired = desired | (tidx << current_bit);
                s_kfind = kf - cum_left;
            }
        }
        __syncthreads();
        desiredMask |= (uint32_t)FT_RADIX_MASK << current_bit;
    }
    uint32_t pivot = s_desired;

    // ---- 统计 enc<pivot 个数；enc==pivot 取剩余名额补齐 ----
    if (tidx == 0) { s_cnt_lt = 0; s_w_lt = 0; s_w_eq = 0; }
    __syncthreads();
    int my_lt = 0;
    for (int j = tidx; j < totalvalid; j += BLOCK_THREADS_) {
        int i; FT_COMPACT_POS(j, i);
        uint32_t pk = (uint32_t)data[i];
        uint16_t sb = (uint16_t)(pk >> 16);
        uint32_t e = (uint32_t)(sb ^ ((sb & 0x8000u) ? 0xffffu : 0x8000u));
        if (e < pivot) my_lt++;
    }
    atomicAdd(&s_cnt_lt, my_lt);
    __syncthreads();
    int cnt_lt = s_cnt_lt;
    int eq_take = K - cnt_lt; if (eq_take < 0) eq_take = 0;

    // ---- 写出：enc<pivot 直接写，enc==pivot 补齐 eq_take 个；寄存器内解码 score/id ----
    for (int j = tidx; j < totalvalid; j += BLOCK_THREADS_) {
        int i; FT_COMPACT_POS(j, i);
        int seg = i / CAP;
        uint32_t pk = (uint32_t)data[i];
        uint16_t sb = (uint16_t)(pk >> 16);
        uint32_t e = (uint32_t)(sb ^ ((sb & 0x8000u) ? 0xffffu : 0x8000u));
        int off16 = (int)(pk & 0xFFFFu);
        int id = seg * BLK + off16;
        if (id > M - 1) id = M - 1;
        if (e < pivot) {
            int w = atomicAdd(&s_w_lt, 1);
            if (w < K) { outv[w] = __ushort_as_half(sb); outi[w] = id; }
        } else if (e == pivot) {
            int o = atomicAdd(&s_w_eq, 1);
            if (o < eq_take) {
                int w = cnt_lt + o;
                if (w < K) { outv[w] = __ushort_as_half(sb); outi[w] = id; }
            }
        }
    }
    #undef FT_COMPACT_POS
}

// ==================================================================
//                     启动参数计算与核心启动逻辑
// ==================================================================
struct FTLaunchParams {
    int items_per_thread;
    uint32_t blocks_per_slice;
    uint32_t num_blocks;
    int sample_cnt;
    uint32_t sample_stride;
    uint32_t cap;
};

inline FTLaunchParams ft_compute_params(int B, int N, int k, int sm_count) {
    using namespace ftns;
    int items_per_thread = FT_MIN_ITEMS_PER_THREAD;
    int items_per_block = items_per_thread * FT_BLOCK_THREADS;
    uint32_t blocks_per_slice = (N + items_per_block - 1) / items_per_block;
    while (blocks_per_slice > 1 &&
           (uint32_t)B * blocks_per_slice > (uint32_t)(sm_count * 4)) {
        items_per_thread *= 2;
        if (items_per_thread > FT_MAX_ITEMS_PER_THREAD) { items_per_thread = FT_MAX_ITEMS_PER_THREAD; }
        items_per_block = items_per_thread * FT_BLOCK_THREADS;
        uint32_t nbps = (N + items_per_block - 1) / items_per_block;
        if (nbps == blocks_per_slice) break;
        blocks_per_slice = nbps;
        if (items_per_thread == FT_MAX_ITEMS_PER_THREAD) break;
    }
    if (blocks_per_slice < 1) blocks_per_slice = 1;

    FTLaunchParams p;
    p.items_per_thread = items_per_thread;
    p.blocks_per_slice = blocks_per_slice;
    p.num_blocks = (uint32_t)B * blocks_per_slice;

    // 采样子集：S = max(k, 16384)，capped N。前缀连续采样：
    //   只取整行最前面的 num_seg*256 ≈ S 个连续元素（seg_stride 固定 256）。
    //   kernel 内 pos = c*256 + tidx，warp 内 lane 连续 => 合并访问，且只读前缀。
    int S_target = k > 16384 ? k : 16384;
    if (S_target > N) S_target = N;
    int num_seg = (S_target + FT_BLOCK_THREADS - 1) / FT_BLOCK_THREADS;
    if (num_seg < 1) num_seg = 1;
    uint32_t seg_stride = (uint32_t)FT_BLOCK_THREADS;
    p.sample_stride = seg_stride;
    p.sample_cnt = num_seg;

    // 候选 buffer 容量：边界桶预期 ~k/255，给足余量并设上限
    uint32_t cap = (uint32_t)k;            // 充裕：单桶最多 k 个真候选即够
    uint32_t cap_floor = 4096u;
    if (cap < cap_floor) cap = cap_floor;
    if (cap > (uint32_t)N) cap = (uint32_t)N;
    p.cap = cap;
    return p;
}

// 候选 buffer 容量（与 ft_compute_params 内一致，独立于 sm_count）。
inline uint32_t ft_cap_for(int N, int K) {
    uint32_t cap = (uint32_t)K;
    uint32_t cap_floor = 4096u;
    if (cap < cap_floor) cap = cap_floor;
    if (cap > (uint32_t)N) cap = (uint32_t)N;
    return cap;
}

// flash_topk_min 的 workspace 在调用方 blob 上的切分结果。
template <typename T>
struct FtMinWs {
    uint32_t *lo, *hi, *span, *hist, *Tb, *cb, *lt_cnt, *cand_cnt, *overflow;
    T* cand_val;
    int32_t* cand_idx;
};

// 在 base 上切分 min workspace；返回所需总字节数（base=nullptr 时仅做 size 查询）。
template <typename T>
size_t ft_min_carve(void* base, int B, int N, int K, FtMinWs<T>& w) {
    uint32_t cap = ft_cap_for(N, K);
    size_t capn = (size_t)B * cap;
    WsBump b(base);
    w.lo       = b.take<uint32_t>(B);
    w.hi       = b.take<uint32_t>(B);
    w.span     = b.take<uint32_t>(B);
    w.hist     = b.take<uint32_t>((size_t)B * ftns::FT_NUM_BUCKETS);
    w.Tb       = b.take<uint32_t>(B);
    w.cb       = b.take<uint32_t>(B);
    w.lt_cnt   = b.take<uint32_t>(B);
    w.cand_cnt = b.take<uint32_t>(B);
    w.overflow = b.take<uint32_t>(1);
    w.cand_val = b.take<T>(capn);
    w.cand_idx = b.take<int32_t>(capn);
    return b.used();
}

template <typename T>
size_t flash_topk_min_ws_bytes(int B, int N, int K) {
    if ((int64_t)N <= K) return 0;     // copy_all 路径无需 workspace
    FtMinWs<T> w;
    return ft_min_carve<T>(nullptr, B, N, K, w);
}

// 核心实现：workspace 由调用方以单块 blob（>= flash_topk_min_ws_bytes）传入。
template <typename T>
void flash_topk_min(const T* in, int B, int N, int K,
                    T* out_val, int* out_idx, void* ws, cudaStream_t stream,
                    uint32_t enc_xor = 0u) {
    using namespace ftns;
    // N <= K 退化：直接返回全部 N 个元素（输出第二维应为 N，调用方需保证容量）。
    if ((int64_t)N <= K) {
        ft_copy_all<T>(in, B, N, out_val, out_idx, stream);
        return;
    }

    int sm_count = 0;
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    }
    FTLaunchParams p = ft_compute_params(B, N, K, sm_count);

    FtMinWs<T> w;
    ft_min_carve<T>(ws, B, N, K, w);
    size_t bsz = sizeof(uint32_t) * (size_t)B;
    size_t hsz = sizeof(uint32_t) * (size_t)B * FT_NUM_BUCKETS;

    cudaMemsetAsync(w.hist, 0, hsz, stream);
    cudaMemsetAsync(w.lt_cnt, 0, bsz, stream);
    cudaMemsetAsync(w.cand_cnt, 0, bsz, stream);
    cudaMemsetAsync(w.overflow, 0, sizeof(uint32_t), stream);

    flashtopk_sample_kernel<T, FT_BLOCK_THREADS><<<B, FT_BLOCK_THREADS, 0, stream>>>(
        in, N, K, p.sample_stride, p.sample_cnt, w.lo, w.hi, w.span, enc_xor);

    // hist/gather：行首 16B 对齐（slice_size % VEC == 0）时走 float4 向量化读，
    // 否则退化到标量。items_per_block 恒为 256 的倍数，必整除 VEC。
    constexpr int VEC = VecTraits<T>::VEC;
    bool use_vec = ((N % VEC) == 0);

    if (use_vec) {
        flashtopk_hist_vec_kernel<T, FT_BLOCK_THREADS><<<p.num_blocks, FT_BLOCK_THREADS, 0, stream>>>(
            in, (uint32_t)N, p.items_per_thread, p.blocks_per_slice, (uint32_t)B,
            w.lo, w.hi, w.span, w.hist, enc_xor);
    } else {
        flashtopk_hist_kernel<T, FT_BLOCK_THREADS><<<p.num_blocks, FT_BLOCK_THREADS, 0, stream>>>(
            in, (uint32_t)N, p.items_per_thread, p.blocks_per_slice, (uint32_t)B,
            w.lo, w.hi, w.span, w.hist, enc_xor);
    }

    flashtopk_threshold_kernel<FT_BLOCK_THREADS><<<B, FT_BLOCK_THREADS, 0, stream>>>(
        w.hist, (uint32_t)B, (uint32_t)K, w.Tb, w.cb);

    if (use_vec) {
        flashtopk_gather_vec_kernel<T, FT_BLOCK_THREADS><<<p.num_blocks, FT_BLOCK_THREADS, 0, stream>>>(
            in, (uint32_t)N, (uint32_t)K, p.items_per_thread, p.blocks_per_slice, (uint32_t)B,
            w.lo, w.hi, w.span, w.Tb, w.cb, p.cap,
            w.lt_cnt, w.cand_cnt, w.overflow, w.cand_val, w.cand_idx, out_val, out_idx, enc_xor);
    } else {
        flashtopk_gather_kernel<T, FT_BLOCK_THREADS><<<p.num_blocks, FT_BLOCK_THREADS, 0, stream>>>(
            in, (uint32_t)N, (uint32_t)K, p.items_per_thread, p.blocks_per_slice, (uint32_t)B,
            w.lo, w.hi, w.span, w.Tb, w.cb, p.cap,
            w.lt_cnt, w.cand_cnt, w.overflow, w.cand_val, w.cand_idx, out_val, out_idx, enc_xor);
    }

    flashtopk_boundary_kernel<T, FT_BLOCK_THREADS><<<B, FT_BLOCK_THREADS, 0, stream>>>(
        in, (uint32_t)N, w.lo, w.hi, w.span,
        (uint32_t)K, (uint32_t)B, p.cap, w.Tb, w.cb, w.cand_cnt,
        w.cand_val, w.cand_idx, out_val, out_idx, enc_xor);
}
// ==================================================================
//   multi-block 阈值复用选择（mb）：边界桶拆分 + 小 radix
// ==================================================================
//   背景：THR_MODE=1 放大 buffer（BUF 可达 100k+）后，单 block 串行扫满 BUF 成为
//   select 阶段瓶颈。mb 把"扫 BUF + 分桶"拆到 bpr(<=256) 个 block/行并行。
//
//   核心优化（边界桶拆分，镜像单 block 版三分支）：粗筛时已知阈值桶 th，分两桶收集：
//     b <  th：必定优于边界 —— 写入 lt 缓冲；
//     b == th：边界桶 —— 写入 eq 缓冲；
//     b >  th：必不入选 —— 丢弃。
//   第二段 finalize 按 cnt_lt/cnt_eq 与 K 的关系决策（与单 block 版完全一致以保正确性，
//   因为 th 是采样估计、cnt_lt 不保证 < K）：
//     cnt_lt >= K          ：阈值偏松，对 lt 缓冲 radix 选 K 个；
//     cnt_lt+cnt_eq <= K   ：候选不足（buffer 溢出退化），lt+eq 全部 copy；
//     否则（标准）         ：lt 全部 copy 到 out[0..cnt_lt)，eq radix 选 K-cnt_lt 个补齐。
//   标准路径下 radix 只在小边界桶 eq 上跑（远小于全部 b<=th），且 lt 是纯 copy —— 这是相对
//   "全收集 b<=th 再整体 radix"的提速点。bucket 单调于 enc，故 lt 全优于 eq，分桶 copy 正确。
//
//   原子优化：两类写入都用 cub::BlockScan 在 block 内求前缀和，每 block 每轮各做
//   1 次 atomicAdd 拿基址，幸存线程在 base+本地前缀处直接写 —— 原子次数降到 bpr*iters 量级。
// ─────────────── DENSE 直方图 + 阈值（SMEM 原子，取代 Triton 的 global 原子）───────────────
// 背景：DENSE 模式 _flat 写全量 score 到 [R,M] 后，需要每行阈值桶 th 给 compact 分 b<th/b==th。
//   Triton 只能做 global atomicAdd(bcount)，逐元素冲突严重（原型实测 11ms）。这里改 CUDA：
//   block 内先在 SMEM 累加直方图（SMEM 原子快得多），末尾一次性 flush 到 global bcount —— 原型
//   实测 1.67ms，6.7× 于逐元素 global 原子。向量化 __ldcs(int4) 读 + streaming hint 不污染 L2。
template <typename T, typename OT, int BLOCK_THREADS_>
__global__ void flashtopk_dense_hist_kernel(
    const T* __restrict__ buf, int M, int NB,
    const OT* __restrict__ origin_in, const OT* __restrict__ inv_delta_in,
    int32_t* __restrict__ bcount    // [R, NB]，调用方已清零
) {
    using namespace ftns;
    extern __shared__ int s_hist[];   // [NB]
    int row = blockIdx.y;
    const T* data = buf + (size_t)row * M;
    float origin = ft_to_float<OT>(origin_in[row]);
    float inv_delta = ft_to_float<OT>(inv_delta_in[row]);
    for (int t = threadIdx.x; t < NB; t += BLOCK_THREADS_) s_hist[t] = 0;
    __syncthreads();

    int64_t stride = (int64_t)gridDim.x * BLOCK_THREADS_;
    constexpr int VEC = (int)(sizeof(int4) / sizeof(T));
    bool can_vec = ((size_t)M % VEC == 0);
    if (can_vec) {
        const int4* data4 = reinterpret_cast<const int4*>(data);
        int64_t M4 = (int64_t)M / VEC;
        for (int64_t j4 = (int64_t)blockIdx.x*BLOCK_THREADS_ + threadIdx.x; j4 < M4; j4 += stride) {
            int4 pk = __ldcs(data4 + j4);
            const T* ve = reinterpret_cast<const T*>(&pk);
            #pragma unroll
            for (int t = 0; t < VEC; ++t) {
                float v = ft_to_float<T>(ve[t]);
                if (isfinite(v)) {
                    int b = (int)((v - origin) * inv_delta);
                    b = b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
                    atomicAdd(&s_hist[b], 1);
                }
            }
        }
    } else {
        for (int64_t i = (int64_t)blockIdx.x*BLOCK_THREADS_ + threadIdx.x; i < M; i += stride) {
            float v = ft_to_float<T>(data[i]);
            if (isfinite(v)) {
                int b = (int)((v - origin) * inv_delta);
                b = b < 0 ? 0 : (b > NB - 1 ? NB - 1 : b);
                atomicAdd(&s_hist[b], 1);
            }
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < NB; t += BLOCK_THREADS_)
        if (s_hist[t]) atomicAdd(&bcount[(size_t)row*NB + t], s_hist[t]);
}

// 阈值 kernel：每行一个 block，对 bcount[row,0..NB) 做 cumsum，求最小桶 th 使 prefix>=K。
//   与 Triton 的 th 语义一致（cum>=K 的最小桶号，clamp 到 NB-1）。NB<=256，单 warp 足矣。
__global__ void flashtopk_dense_threshold_kernel(
    const int32_t* __restrict__ bcount, int NB, int K, int32_t* __restrict__ th_out
) {
    int row = blockIdx.x;
    const int32_t* hrow = bcount + (size_t)row * NB;
    // 单线程串行 cumsum（NB<=256，开销可忽略，避免 scan 复杂度）。
    if (threadIdx.x == 0) {
        int cum = 0, th = NB - 1;
        for (int b = 0; b < NB; ++b) {
            cum += hrow[b];
            if (cum >= K) { th = b; break; }
        }
        th_out[row] = th;
    }
}

// warp 聚合写出：本 warp 对 pred 做一次 __ballot，空 warp 零成本跳过；命中 warp 仅 leader
//   做 1 次 atomicAdd 拿基址，lane 用 popc 算偏移直接写。供 compact 主循环按元素调用。
template <typename T>
__device__ __forceinline__ void ft_warp_emit(
        int pred, T v, int32_t id, int lane, unsigned lanemask,
        int32_t* cnt, T* outv, int32_t* outi, int CAP) {
    unsigned m = __ballot_sync(0xffffffffu, pred);
    if (!m) return;
    int leader = __ffs(m) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(cnt, __popc(m));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (pred) { int w = base + __popc(m & lanemask); if (w < CAP) { outv[w] = v; outi[w] = id; } }
}

template <typename T, typename OT, int BLOCK_THREADS_>
__global__ void flashtopk_compact_thr_kernel(
    const T* __restrict__ buf, int M, int CAP,
    const OT* __restrict__ origin_in,    // [R]
    const OT* __restrict__ inv_delta_in, // [R]
    const int32_t* __restrict__ th_in,   // [R]
    int NB, int K,
    const int32_t* __restrict__ buf_idx_in,    // optional [R, M]：scatter 的 corpus id
    const int32_t* __restrict__ sample_idx_in, // optional [R, K]：前 K 个 sample 槽的 corpus id
    const int32_t* __restrict__ qcount_in,     // optional [R]：每行有效候选数（limit），null 则扫满 M
    T* __restrict__ lt_val,              // [R, CAP] b<th 候选
    int32_t* __restrict__ lt_idx,        // [R, CAP] corpus id
    int32_t* __restrict__ lt_cnt,        // [R] b<th 候选数（可超 CAP，下游 clamp）
    T* __restrict__ eq_val,              // [R, CAP] b==th 边界候选
    int32_t* __restrict__ eq_idx,        // [R, CAP] corpus id
    int32_t* __restrict__ eq_cnt         // [R] b==th 候选数（可超 CAP，下游 clamp）
) {
    using namespace ftns;
    int row = blockIdx.y;
    const T* data = buf + (size_t)row * M;
    const int32_t* idx_data = buf_idx_in ? (buf_idx_in + (size_t)row * M) : nullptr;
    const int32_t* sample_idx = sample_idx_in ? (sample_idx_in + (size_t)row * K) : nullptr;
    T*       ltv = lt_val + (size_t)row * CAP;
    int32_t* lti = lt_idx + (size_t)row * CAP;
    T*       eqv = eq_val + (size_t)row * CAP;
    int32_t* eqi = eq_idx + (size_t)row * CAP;
    float origin = ft_to_float<OT>(origin_in[row]);
    float inv_delta = ft_to_float<OT>(inv_delta_in[row]);
    int th = th_in[row];
    int limit = qcount_in ? qcount_in[row] : M;
    if (limit > M) limit = M;
    if (limit < 0) limit = 0;

    // warp 聚合 compact：每轮用 __ballot 拿本 warp 命中 mask，空 warp 直接跳过（零同步），
    //   有命中的 warp 仅 leader 做 1 次 atomicAdd 拿基址，lane 用 popc 算偏移。彻底去掉
    //   block 级 BlockScan + __syncthreads —— DENSE 满 M 扫描下绝大多数轮次无命中，旧版
    //   每轮强制全员同步的开销淹没了访存（实测仅 13% 带宽），warp 版把空轮成本降到一次 ballot。
    const int lane = threadIdx.x & 31;
    const unsigned lanemask = (1u << lane) - 1u;   // 本 lane 之前的位

    int64_t stride = (int64_t)gridDim.x * BLOCK_THREADS_;

    // 向量化读：每线程一次 __ldcs(int4)=16B 拉满访存事务粒度，并用 streaming hint 不污染 L2
    //   （这块 [R,M] 只读一遍无复用）。条件：dense（id=col=i，无 sample_idx 旁路）+ 16B 对齐
    //   （M*sizeof(T)%16==0 → 每行起始对齐）。否则退化为标量逐元素读。
    constexpr int VEC = (int)(sizeof(int4) / sizeof(T));   // half/bf16→8, float→4
    bool can_vec = (idx_data == nullptr) && ((size_t)M % VEC == 0);
    if (can_vec) {
        const int4* data4 = reinterpret_cast<const int4*>(data);
        int64_t M4 = (int64_t)M / VEC;
        int64_t lim4 = (limit + VEC - 1) / VEC; if (lim4 > M4) lim4 = M4;
        int64_t base4 = (int64_t)blockIdx.x * BLOCK_THREADS_;
        for (; base4 < lim4; base4 += stride) {
            int64_t j4 = base4 + threadIdx.x;
            int4 pk; bool have = j4 < lim4;
            if (have) pk = __ldcs(data4 + j4);     // streaming load，绕开 L2 缓存
            const T* ve = reinterpret_cast<const T*>(&pk);
            #pragma unroll
            for (int t = 0; t < VEC; ++t) {
                int is_lt = 0, is_eq = 0; T v = T(); int32_t id = 0;
                int64_t i = j4 * VEC + t;
                if (have && i < limit) {
                    v = ve[t];
                    float vf = ft_to_float<T>(v);
                    if (isfinite(vf)) {
                        int braw = (int)((vf - origin) * inv_delta);
                        int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
                        if (b < th) is_lt = 1; else if (b == th) is_eq = 1;
                        if (is_lt || is_eq) id = (int32_t)i;
                    }
                }
                ft_warp_emit<T>(is_lt, v, id, lane, lanemask, lt_cnt + row, ltv, lti, CAP);
                ft_warp_emit<T>(is_eq, v, id, lane, lanemask, eq_cnt + row, eqv, eqi, CAP);
            }
        }
        return;
    }

    // 标量回退：以 block 为单位推进（base 全员一致），保证尾部轮次 warp 内 32 lane 同进同退。
    int64_t base_i = (int64_t)blockIdx.x * BLOCK_THREADS_;
    for (; base_i < limit; base_i += stride) {
        int64_t i = base_i + threadIdx.x;
        int is_lt = 0, is_eq = 0;
        T v = T(); int32_t id = 0;
        if (i < limit) {
            v = data[i];
            float vf = ft_to_float<T>(v);
            if (isfinite(vf)) {
                int braw = (int)((vf - origin) * inv_delta);
                int b = braw < 0 ? 0 : (braw > NB - 1 ? NB - 1 : braw);
                if (b < th) is_lt = 1;
                else if (b == th) is_eq = 1;     // b>th 丢弃
                if (is_lt || is_eq)
                    id = idx_data ? (((int)i < K && sample_idx) ? sample_idx[i] : idx_data[i])
                                  : (int32_t)i;
            }
        }
        ft_warp_emit<T>(is_lt, v, id, lane, lanemask, lt_cnt + row, ltv, lti, CAP);
        ft_warp_emit<T>(is_eq, v, id, lane, lanemask, eq_cnt + row, eqv, eqi, CAP);
    }
}

// 第二段 finalize：按 cnt_lt / cnt_eq 与 K 的关系决策（镜像单 block 版三分支）。
//   lt 桶（b<th）已全优于 eq 桶（b==th），bucket 单调于 enc 保证。
//   标准路径：lt 全 copy 到 out[0..cnt_lt)，eq 上 radix 选 K-cnt_lt 个补齐。
template <typename T, int BLOCK_THREADS_>
__global__ void flashtopk_boundary_radix_kernel(
    const T* __restrict__ lt_val, const int32_t* __restrict__ lt_idx,
    const int32_t* __restrict__ lt_cnt,
    const T* __restrict__ eq_val, const int32_t* __restrict__ eq_idx,
    const int32_t* __restrict__ eq_cnt,
    int CAP, int K,
    T* __restrict__ out_val, int32_t* __restrict__ out_idx,
    uint32_t enc_xor
) {
    using namespace ftns;
    int row = blockIdx.x;
    int tidx = threadIdx.x;
    const T* ltd = lt_val + (size_t)row * CAP;
    const int32_t* lti = lt_idx + (size_t)row * CAP;
    const T* eqd = eq_val + (size_t)row * CAP;
    const int32_t* eqi = eq_idx + (size_t)row * CAP;
    T*       ov = out_val + (size_t)row * K;
    int32_t* oi = out_idx + (size_t)row * K;

    int cnt_lt = lt_cnt[row]; if (cnt_lt < 0) cnt_lt = 0; if (cnt_lt > CAP) cnt_lt = CAP;
    int cnt_eq = eq_cnt[row]; if (cnt_eq < 0) cnt_eq = 0; if (cnt_eq > CAP) cnt_eq = CAP;

    // radix select 的输入（data/idx）、名额（k_find）、写出起点（base）由分支确定。
    const T* data; const int32_t* idx_data;
    int n;          // radix 输入规模
    int k_find;     // 在 data 上选最小 k_find 个
    int base;       // 写到 out 的起点（前面已 copy 的数量）

    if (cnt_lt >= K) {
        // 阈值偏松：top-K 全在 lt 内，对 lt radix 选 K 个。
        data = ltd; idx_data = lti; n = cnt_lt; k_find = K; base = 0;
    } else if (cnt_lt + cnt_eq <= K) {
        // 候选不足（buffer 溢出截断）：lt+eq 全部 copy，剩余槽位补 0（取代 host 端 memset）。
        for (int i = tidx; i < cnt_lt; i += BLOCK_THREADS_) { ov[i] = ltd[i]; oi[i] = lti[i]; }
        for (int i = tidx; i < cnt_eq; i += BLOCK_THREADS_) {
            int w = cnt_lt + i; if (w < K) { ov[w] = eqd[i]; oi[w] = eqi[i]; }
        }
        int filled = cnt_lt + cnt_eq;
        for (int i = filled + tidx; i < K; i += BLOCK_THREADS_) { oi[i] = 0; }
        return;
    } else {
        // 标准：lt 全 copy 到 out[0..cnt_lt)，eq 上 radix 选 K-cnt_lt 个补齐。
        for (int i = tidx; i < cnt_lt; i += BLOCK_THREADS_) { ov[i] = ltd[i]; oi[i] = lti[i]; }
        data = eqd; idx_data = eqi; n = cnt_eq; k_find = K - cnt_lt; base = cnt_lt;
    }
    __syncthreads();   // 保证 lt copy 完成（与下方共享内存使用分隔）

    typedef cub::BlockScan<uint32_t, BLOCK_THREADS_> BlockScanU;
    union __align__(16) TempStorage {
        uint32_t digit_counters[FT_RADIX_DIGITS];
        typename BlockScanU::TempStorage scanu;
    };
    __shared__ TempStorage ts;
    __shared__ uint32_t s_desired;
    __shared__ uint32_t s_kfind;
    __shared__ int s_cnt_lt;
    __shared__ int s_w_lt;
    __shared__ int s_w_eq;

    if (tidx == 0) { s_desired = 0u; s_kfind = (uint32_t)k_find; }
    __syncthreads();

    // ---- in-block radix：求 data 中第 k_find 小 enc (= pivot) ----
    const int num_passes = EncTraits<T>::kPasses;
    const int start_bit  = EncTraits<T>::kBits - FT_RADIX_BITS;
    uint32_t desiredMask = 0u;
    for (int pass = 0; pass < num_passes; ++pass) {
        int current_bit = start_bit - FT_RADIX_BITS * pass;
        if (tidx < FT_RADIX_DIGITS) ts.digit_counters[tidx] = 0;
        __syncthreads();
        uint32_t desired = s_desired;
        for (int i = tidx; i < n; i += BLOCK_THREADS_) {
            uint32_t e = ft_enc_val<T>(data[i]) ^ enc_xor;
            if ((e & desiredMask) == (desired & desiredMask))
                atomicAdd(&ts.digit_counters[getDigit(e, current_bit)], 1u);
        }
        __syncthreads();
        uint32_t dc = (tidx < FT_RADIX_DIGITS) ? ts.digit_counters[tidx] : 0u;
        __syncthreads();
        uint32_t cumsum;
        BlockScanU(ts.scanu).InclusiveSum(dc, cumsum);
        __syncthreads();
        uint32_t kf = s_kfind;
        if (tidx < FT_RADIX_DIGITS) {
            uint32_t cum_left = cumsum - dc;
            if (cum_left < kf && kf <= cumsum) {
                s_desired = desired | (tidx << current_bit);
                s_kfind = kf - cum_left;
            }
        }
        __syncthreads();
        desiredMask |= (uint32_t)FT_RADIX_MASK << current_bit;
    }
    uint32_t pivot = s_desired;

    // ---- enc<pivot 全取，enc==pivot 取剩余名额补齐 ----
    if (tidx == 0) { s_cnt_lt = 0; s_w_lt = 0; s_w_eq = 0; }
    __syncthreads();
    int my_lt = 0;
    for (int i = tidx; i < n; i += BLOCK_THREADS_) {
        uint32_t e = ft_enc_val<T>(data[i]) ^ enc_xor;
        if (e < pivot) my_lt++;
    }
    atomicAdd(&s_cnt_lt, my_lt);
    __syncthreads();
    int cnt_lt2 = s_cnt_lt;
    int eq_take = k_find - cnt_lt2; if (eq_take < 0) eq_take = 0;

    for (int i = tidx; i < n; i += BLOCK_THREADS_) {
        T vv = data[i];
        uint32_t e = ft_enc_val<T>(vv) ^ enc_xor;
        if (e < pivot) {
            int w = atomicAdd(&s_w_lt, 1);
            int pos = base + w;
            if (pos < K) { ov[pos] = vv; oi[pos] = idx_data[i]; }
        } else if (e == pivot) {
            int o = atomicAdd(&s_w_eq, 1);
            if (o < eq_take) {
                int pos = base + cnt_lt2 + o;
                if (pos < K) { ov[pos] = vv; oi[pos] = idx_data[i]; }
            }
        }
    }
}

template <typename T, typename OT>
void flash_topk_select_thr_mb(const T* buf, const int32_t* buf_idx, const int32_t* sample_idx,
                              int R, int BUF, int K,
                              const OT* origin, const OT* inv_delta,
                              const int32_t* th, const int32_t* qcount, int NB,
                              int CAP, T* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
                              T* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
                              T* out_val, int* out_idx, cudaStream_t stream,
                              uint32_t enc_xor = 0u) {
    using namespace ftns;
    if (R <= 0 || K <= 0) return;
    // cand_*（eq 缓冲）与 lt_*（lt 缓冲）均由调用方（torch op 层）用 at::empty 分配并传入，
    //   走 PyTorch caching allocator 复用，无需初始化（计数器在此清零，val/idx 由有效区间界定）。
    // 仅清两个小计数器（各 R 个 int）；out_idx 的欠填兜底零值改由 finalize kernel 内补，
    //   省掉每次调用 R*K*4B（如 256*4096*4≈4MB）的 memset。
    cudaMemsetAsync(lt_cnt, 0, sizeof(int32_t) * (size_t)R, stream);
    cudaMemsetAsync(cand_cnt, 0, sizeof(int32_t) * (size_t)R, stream);  // eq_cnt
    // 动态 bpr：grid 为 dim3(bpr, R)，行维度已提供 R 路并行；bpr 只需把总 block 数补到
    //   填满 GPU 的量级即可。固定 bpr=256 在 R 较大时会产生 bpr*R 个 block 严重过订阅
    //   （如 R=256→65536 个 block，远超可驻留量），空 block 的调度成本反成瓶颈。
    static int s_num_sm = 0;
    if (s_num_sm == 0) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_num_sm, cudaDevAttrMultiProcessorCount, dev);
        if (s_num_sm <= 0) s_num_sm = 108;
    }
    int bpr_cap = (int)((BUF + FT_BLOCK_THREADS - 1) / FT_BLOCK_THREADS);
    int target_total = s_num_sm * 16;             // 约填满 GPU 的常驻 block 量级
    int bpr = target_total / (R > 0 ? R : 1);
    if (bpr > bpr_cap) bpr = bpr_cap;
    if (bpr < 1) bpr = 1;
    dim3 grid_c(bpr, R);
    // 第一阶段：多 block 并行分桶。b<th→lt 缓冲；b==th→eq(cand) 缓冲。
    flashtopk_compact_thr_kernel<T, OT, FT_BLOCK_THREADS><<<grid_c, FT_BLOCK_THREADS, 0, stream>>>(
        buf, BUF, CAP, origin, inv_delta, th, NB, K,
        buf_idx, sample_idx, qcount,
        lt_val, lt_idx, lt_cnt, cand_val, cand_idx, cand_cnt);
    // 第二阶段：三分支 finalize。标准路径 lt 全 copy + eq 上小 radix 补齐。
    flashtopk_boundary_radix_kernel<T, FT_BLOCK_THREADS><<<R, FT_BLOCK_THREADS, 0, stream>>>(
        lt_val, lt_idx, lt_cnt, cand_val, cand_idx, cand_cnt, CAP, K,
        out_val, out_idx, enc_xor);
}

// DENSE 直方图 + 阈值 host：SMEM 直方图 kernel（归约到 global bcount）+ 阈值 cumsum kernel。
//   调用方分配 bcount[R,NB]（此处清零）与 th[R]。bpr 同 mb 逻辑按 SM 数动态定。
template <typename T, typename OT>
void flash_topk_dense_threshold(const T* buf, int R, int M, int K,
                                const OT* origin, const OT* inv_delta, int NB,
                                int32_t* bcount, int32_t* th, cudaStream_t stream) {
    using namespace ftns;
    if (R <= 0 || K <= 0) return;
    cudaMemsetAsync(bcount, 0, sizeof(int32_t) * (size_t)R * NB, stream);
    static int s_num_sm = 0;
    if (s_num_sm == 0) {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_num_sm, cudaDevAttrMultiProcessorCount, dev);
        if (s_num_sm <= 0) s_num_sm = 108;
    }
    int bpr_cap = (int)(((size_t)M + FT_BLOCK_THREADS - 1) / FT_BLOCK_THREADS);
    int bpr = (s_num_sm * 16) / (R > 0 ? R : 1);
    if (bpr > bpr_cap) bpr = bpr_cap;
    if (bpr < 1) bpr = 1;
    dim3 grid_h(bpr, R);
    flashtopk_dense_hist_kernel<T, OT, FT_BLOCK_THREADS>
        <<<grid_h, FT_BLOCK_THREADS, sizeof(int) * (size_t)NB, stream>>>(
            buf, M, NB, origin, inv_delta, bcount);
    flashtopk_dense_threshold_kernel<<<R, 32, 0, stream>>>(bcount, NB, K, th);
}
} // namespace

// ---- multi-block 阈值复用选择（mb）launch 包装 ------------------------------
//   cand_*（eq 缓冲）与 lt_*（lt 缓冲）均 [R,CAP] val/idx + [R] cnt，由调用方分配。
void launch_flash_topk_select_thr_mb_idx_fp32(
        const float* buf, const int32_t* buf_idx, const int32_t* sample_idx,
        int R, int BUF, int K, int CAP,
        const float* origin, const float* inv_delta,
        const int32_t* th, const int32_t* qcount, int NB,
        float* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
        float* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
        float* out_val, int* out_idx, cudaStream_t stream) {
    flash_topk_select_thr_mb<float, float>(
        buf, buf_idx, sample_idx, R, BUF, K, origin, inv_delta, th, qcount, NB,
        CAP, cand_val, cand_idx, cand_cnt, lt_val, lt_idx, lt_cnt, out_val, out_idx, stream);
}
void launch_flash_topk_select_thr_mb_idx_fp16(
        const __half* buf, const int32_t* buf_idx, const int32_t* sample_idx,
        int R, int BUF, int K, int CAP,
        const void* origin, const void* inv_delta,
        const int32_t* th, const int32_t* qcount, int NB,
        __half* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
        __half* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
        __half* out_val, int* out_idx, bool coords_fp16, cudaStream_t stream) {
    if (coords_fp16) {
        flash_topk_select_thr_mb<__half, __half>(
            buf, buf_idx, sample_idx, R, BUF, K,
            reinterpret_cast<const __half*>(origin), reinterpret_cast<const __half*>(inv_delta),
            th, qcount, NB, CAP, cand_val, cand_idx, cand_cnt,
            lt_val, lt_idx, lt_cnt, out_val, out_idx, stream);
    } else {
        flash_topk_select_thr_mb<__half, float>(
            buf, buf_idx, sample_idx, R, BUF, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            th, qcount, NB, CAP, cand_val, cand_idx, cand_cnt,
            lt_val, lt_idx, lt_cnt, out_val, out_idx, stream);
    }
}
void launch_flash_topk_select_thr_mb_idx_bf16(
        const __nv_bfloat16* buf, const int32_t* buf_idx, const int32_t* sample_idx,
        int R, int BUF, int K, int CAP,
        const void* origin, const void* inv_delta,
        const int32_t* th, const int32_t* qcount, int NB,
        __nv_bfloat16* cand_val, int32_t* cand_idx, int32_t* cand_cnt,
        __nv_bfloat16* lt_val, int32_t* lt_idx, int32_t* lt_cnt,
        __nv_bfloat16* out_val, int* out_idx, bool coords_bf16, cudaStream_t stream) {
    if (coords_bf16) {
        flash_topk_select_thr_mb<__nv_bfloat16, __nv_bfloat16>(
            buf, buf_idx, sample_idx, R, BUF, K,
            reinterpret_cast<const __nv_bfloat16*>(origin), reinterpret_cast<const __nv_bfloat16*>(inv_delta),
            th, qcount, NB, CAP, cand_val, cand_idx, cand_cnt,
            lt_val, lt_idx, lt_cnt, out_val, out_idx, stream);
    } else {
        flash_topk_select_thr_mb<__nv_bfloat16, float>(
            buf, buf_idx, sample_idx, R, BUF, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            th, qcount, NB, CAP, cand_val, cand_idx, cand_cnt,
            lt_val, lt_idx, lt_cnt, out_val, out_idx, stream);
    }
}

// ---- DENSE 直方图+阈值 launch 包装 ------------------------------------------
void launch_flash_topk_dense_threshold_fp32(
        const float* buf, int R, int M, int K,
        const float* origin, const float* inv_delta, int NB,
        int32_t* bcount, int32_t* th, cudaStream_t stream) {
    flash_topk_dense_threshold<float, float>(buf, R, M, K, origin, inv_delta, NB, bcount, th, stream);
}
void launch_flash_topk_dense_threshold_fp16(
        const __half* buf, int R, int M, int K,
        const void* origin, const void* inv_delta, int NB,
        int32_t* bcount, int32_t* th, bool coords_fp16, cudaStream_t stream) {
    if (coords_fp16)
        flash_topk_dense_threshold<__half, __half>(buf, R, M, K,
            reinterpret_cast<const __half*>(origin), reinterpret_cast<const __half*>(inv_delta),
            NB, bcount, th, stream);
    else
        flash_topk_dense_threshold<__half, float>(buf, R, M, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            NB, bcount, th, stream);
}
void launch_flash_topk_dense_threshold_bf16(
        const __nv_bfloat16* buf, int R, int M, int K,
        const void* origin, const void* inv_delta, int NB,
        int32_t* bcount, int32_t* th, bool coords_bf16, cudaStream_t stream) {
    if (coords_bf16)
        flash_topk_dense_threshold<__nv_bfloat16, __nv_bfloat16>(buf, R, M, K,
            reinterpret_cast<const __nv_bfloat16*>(origin), reinterpret_cast<const __nv_bfloat16*>(inv_delta),
            NB, bcount, th, stream);
    else
        flash_topk_dense_threshold<__nv_bfloat16, float>(buf, R, M, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            NB, bcount, th, stream);
}

void launch_flash_topk_select_packed_fp16(
    const int32_t* buf_pack, int R, int CAP, int NSEG, int BLK, int M, int K,
    const int32_t* segcnt, __half* out_val, int* out_idx, cudaStream_t stream) {
    using namespace ftns;
    if (R <= 0 || K <= 0) return;
    // out_idx 兜底清零：欠填槽位仍为合法索引 0，避免下游越界。
    cudaMemsetAsync(out_idx, 0, sizeof(int) * (size_t)R * K, stream);
    flashtopk_select_packed_kernel<FT_BLOCK_THREADS, 1024><<<R, FT_BLOCK_THREADS, 0, stream>>>(
        buf_pack, CAP, NSEG, BLK, M, K, segcnt, out_val, out_idx);
}

size_t flash_topk_min_workspace_bytes_fp32(int B, int N, int K) {
    return flash_topk_min_ws_bytes<float>(B, N, K);
}
size_t flash_topk_min_workspace_bytes_fp16(int B, int N, int K) {
    return flash_topk_min_ws_bytes<__half>(B, N, K);
}
size_t flash_topk_min_workspace_bytes_bf16(int B, int N, int K) {
    return flash_topk_min_ws_bytes<__nv_bfloat16>(B, N, K);
}

void launch_flash_topk_min_fp32_ws(const float* in,
                                   int B, int N, int K,
                                   float* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream) {
    flash_topk_min<float>(in, B, N, K, out_val, out_idx, ws, stream);
}

void launch_flash_topk_min_fp16_ws(const __half* in,
                                   int B, int N, int K,
                                   __half* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream) {
    flash_topk_min<__half>(in, B, N, K, out_val, out_idx, ws, stream);
}

void launch_flash_topk_min_bf16_ws(const __nv_bfloat16* in,
                                   int B, int N, int K,
                                   __nv_bfloat16* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream) {
    flash_topk_min<__nv_bfloat16>(in, B, N, K, out_val, out_idx, ws, stream);
}

// 阈值复用选择（配合融合 _flat_kernel）：跳过 sample/hist/threshold，只跑 gather+boundary。
void launch_flash_topk_select_thr_fp32(const float* buf, int R, int BUF, int K,
                                       const float* origin, const float* inv_delta,
                                       const int32_t* th, const int32_t* qcount, int NB,
                                       float* out_val, int* out_idx, cudaStream_t stream) {
    // fp32 buffer 的桶坐标恒为 fp32。
    flash_topk_select_thr<float, float>(buf, R, BUF, K, origin, inv_delta, th, qcount, NB,
                                        out_val, out_idx, stream);
}
void launch_flash_topk_select_thr_idx_fp32(const float* buf, const int32_t* buf_idx,
                                           const int32_t* sample_idx,
                                           int R, int BUF, int K,
                                           const float* origin, const float* inv_delta,
                                           const int32_t* th, const int32_t* qcount, int NB,
                                           float* out_val, int* out_idx, cudaStream_t stream) {
    flash_topk_select_thr<float, float>(buf, R, BUF, K, origin, inv_delta, th, qcount, NB,
                                        out_val, out_idx, stream, buf_idx, sample_idx);
}

// fp16 buffer：桶坐标可为 fp16（FP16_BUCKET 开）或 fp32（关，默认）。coords_fp16 选择 OT。
void launch_flash_topk_select_thr_fp16(const __half* buf, int R, int BUF, int K,
                                       const void* origin, const void* inv_delta,
                                       const int32_t* th, const int32_t* qcount, int NB,
                                       __half* out_val, int* out_idx,
                                       bool coords_fp16, cudaStream_t stream) {
    if (coords_fp16) {
        flash_topk_select_thr<__half, __half>(
            buf, R, BUF, K,
            reinterpret_cast<const __half*>(origin), reinterpret_cast<const __half*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream);
    } else {
        flash_topk_select_thr<__half, float>(
            buf, R, BUF, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream);
    }
}
void launch_flash_topk_select_thr_idx_fp16(const __half* buf, const int32_t* buf_idx,
                                           const int32_t* sample_idx,
                                           int R, int BUF, int K,
                                           const void* origin, const void* inv_delta,
                                           const int32_t* th, const int32_t* qcount, int NB,
                                           __half* out_val, int* out_idx,
                                           bool coords_fp16, cudaStream_t stream) {
    if (coords_fp16) {
        flash_topk_select_thr<__half, __half>(
            buf, R, BUF, K,
            reinterpret_cast<const __half*>(origin), reinterpret_cast<const __half*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream, buf_idx, sample_idx);
    } else {
        flash_topk_select_thr<__half, float>(
            buf, R, BUF, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream, buf_idx, sample_idx);
    }
}

// bf16 buffer：桶坐标可为 bf16（coords_bf16 开）或 fp32（关，默认）。
void launch_flash_topk_select_thr_bf16(const __nv_bfloat16* buf, int R, int BUF, int K,
                                       const void* origin, const void* inv_delta,
                                       const int32_t* th, const int32_t* qcount, int NB,
                                       __nv_bfloat16* out_val, int* out_idx,
                                       bool coords_bf16, cudaStream_t stream) {
    if (coords_bf16) {
        flash_topk_select_thr<__nv_bfloat16, __nv_bfloat16>(
            buf, R, BUF, K,
            reinterpret_cast<const __nv_bfloat16*>(origin), reinterpret_cast<const __nv_bfloat16*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream);
    } else {
        flash_topk_select_thr<__nv_bfloat16, float>(
            buf, R, BUF, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream);
    }
}
void launch_flash_topk_select_thr_idx_bf16(const __nv_bfloat16* buf, const int32_t* buf_idx,
                                           const int32_t* sample_idx,
                                           int R, int BUF, int K,
                                           const void* origin, const void* inv_delta,
                                           const int32_t* th, const int32_t* qcount, int NB,
                                           __nv_bfloat16* out_val, int* out_idx,
                                           bool coords_bf16, cudaStream_t stream) {
    if (coords_bf16) {
        flash_topk_select_thr<__nv_bfloat16, __nv_bfloat16>(
            buf, R, BUF, K,
            reinterpret_cast<const __nv_bfloat16*>(origin), reinterpret_cast<const __nv_bfloat16*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream, buf_idx, sample_idx);
    } else {
        flash_topk_select_thr<__nv_bfloat16, float>(
            buf, R, BUF, K,
            reinterpret_cast<const float*>(origin), reinterpret_cast<const float*>(inv_delta),
            th, qcount, NB, out_val, out_idx, stream, buf_idx, sample_idx);
    }
}

// 兼容旧接口：内部 cudaMallocAsync 自管 workspace（供 torch 绑定 / 测试使用）。
void launch_flash_topk_min_fp32(const float* in,
                                int B, int N, int K,
                                float* out_val, int* out_idx,
                                cudaStream_t stream) {
    size_t bytes = flash_topk_min_ws_bytes<float>(B, N, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    flash_topk_min<float>(in, B, N, K, out_val, out_idx, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

void launch_flash_topk_min_fp16(const __half* in,
                                int B, int N, int K,
                                __half* out_val, int* out_idx,
                                cudaStream_t stream) {
    size_t bytes = flash_topk_min_ws_bytes<__half>(B, N, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    flash_topk_min<__half>(in, B, N, K, out_val, out_idx, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

void launch_flash_topk_min_bf16(const __nv_bfloat16* in,
                                int B, int N, int K,
                                __nv_bfloat16* out_val, int* out_idx,
                                cudaStream_t stream) {
    size_t bytes = flash_topk_min_ws_bytes<__nv_bfloat16>(B, N, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    flash_topk_min<__nv_bfloat16>(in, B, N, K, out_val, out_idx, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

// topk-max workspace = min workspace（max 通过 enc^kRevMask 直接复用 min kernel，无需 neg 缓冲）。
template <typename T>
static size_t flash_topk_max_ws_bytes(int B, int N, int K) {
    if ((int64_t)B * N <= 0 || K <= 0) return 0;
    return flash_topk_min_ws_bytes<T>(B, N, K);
}

size_t flash_topk_max_workspace_bytes_fp32(int B, int N, int K) {
    return flash_topk_max_ws_bytes<float>(B, N, K);
}
size_t flash_topk_max_workspace_bytes_fp16(int B, int N, int K) {
    return flash_topk_max_ws_bytes<__half>(B, N, K);
}
size_t flash_topk_max_workspace_bytes_bf16(int B, int N, int K) {
    return flash_topk_max_ws_bytes<__nv_bfloat16>(B, N, K);
}

void launch_flash_topk_max_fp32_ws(const float* in,
                                   int B, int N, int K,
                                   float* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream) {
    if ((int64_t)B * N <= 0 || K <= 0) return;
    flash_topk_min<float>(in, B, N, K, out_val, out_idx, ws, stream,
                          EncTraits<float>::kRevMask);
}

void launch_flash_topk_max_fp16_ws(const __half* in,
                                   int B, int N, int K,
                                   __half* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream) {
    if ((int64_t)B * N <= 0 || K <= 0) return;
    flash_topk_min<__half>(in, B, N, K, out_val, out_idx, ws, stream,
                           EncTraits<__half>::kRevMask);
}

void launch_flash_topk_max_bf16_ws(const __nv_bfloat16* in,
                                   int B, int N, int K,
                                   __nv_bfloat16* out_val, int* out_idx,
                                   void* ws, cudaStream_t stream) {
    if ((int64_t)B * N <= 0 || K <= 0) return;
    flash_topk_min<__nv_bfloat16>(in, B, N, K, out_val, out_idx, ws, stream,
                                  EncTraits<__nv_bfloat16>::kRevMask);
}

// 兼容旧接口：内部 cudaMallocAsync 自管 workspace。
void launch_flash_topk_max_fp32(const float* in,
                                int B, int N, int K,
                                float* out_val, int* out_idx,
                                cudaStream_t stream) {
    if ((int64_t)B * N <= 0 || K <= 0) return;
    size_t bytes = flash_topk_max_ws_bytes<float>(B, N, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    launch_flash_topk_max_fp32_ws(in, B, N, K, out_val, out_idx, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

void launch_flash_topk_max_fp16(const __half* in,
                                int B, int N, int K,
                                __half* out_val, int* out_idx,
                                cudaStream_t stream) {
    if ((int64_t)B * N <= 0 || K <= 0) return;
    size_t bytes = flash_topk_max_ws_bytes<__half>(B, N, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    launch_flash_topk_max_fp16_ws(in, B, N, K, out_val, out_idx, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

void launch_flash_topk_max_bf16(const __nv_bfloat16* in,
                                int B, int N, int K,
                                __nv_bfloat16* out_val, int* out_idx,
                                cudaStream_t stream) {
    if ((int64_t)B * N <= 0 || K <= 0) return;
    size_t bytes = flash_topk_max_ws_bytes<__nv_bfloat16>(B, N, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    launch_flash_topk_max_bf16_ws(in, B, N, K, out_val, out_idx, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

void launch_topk_identity_fp32(const float* in, int B, int N,
                               float* out_val, int* out_idx, cudaStream_t stream) {
    if ((int64_t)B * N <= 0) return;
    ft_copy_all<float>(in, B, N, out_val, out_idx, stream);
}

void launch_topk_identity_fp16(const __half* in, int B, int N,
                               __half* out_val, int* out_idx, cudaStream_t stream) {
    if ((int64_t)B * N <= 0) return;
    ft_copy_all<__half>(in, B, N, out_val, out_idx, stream);
}

void launch_topk_identity_bf16(const __nv_bfloat16* in, int B, int N,
                               __nv_bfloat16* out_val, int* out_idx, cudaStream_t stream) {
    if ((int64_t)B * N <= 0) return;
    ft_copy_all<__nv_bfloat16>(in, B, N, out_val, out_idx, stream);
}

namespace {

// 段偏移填充 kernel（offsets[i] = i*K, i in [0,B]）。
__global__ void ft_fill_offsets_kernel(int32_t* off, int B, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= B) off[i] = (int32_t)((int64_t)i * K);
}

// 把 [B,K] 的 value 编码成单调 uint key（降序排序时大 value -> 大 key）。
//   payload 直接用已有的 out_idx（原始列索引），无需另写。
template <typename T>
__global__ void ft_sort_make_keys_kernel(const T* __restrict__ val,
                                         uint32_t* __restrict__ keys,
                                         int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    keys[idx] = ft_enc_val<T>(val[idx]);
}

// sort 后直接由 sorted key 解码出 value（省去回原数组的 scatter-read gather）；
//   sorted idx 已是排好序的原始索引，直接拷到输出。
template <typename T>
__global__ void ft_sort_finalize_kernel(const uint32_t* __restrict__ keys_sorted,
                                        const int32_t* __restrict__ idx_sorted,
                                        T* __restrict__ val_out,
                                        int32_t* __restrict__ idx_out,
                                        int64_t total) {
    int64_t o = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= total) return;
    val_out[o] = ft_dec_val<T>(keys_sorted[o]);
    idx_out[o] = idx_sorted[o];
}

// 行内降序排序核心：value -> 单调 uint key，payload = 原始索引；cub 分段 radix 降序排序后，
//   key 解码回 value、payload 即排好的索引。相比"sort 位置 + gather 回原数组"省掉：
//   (1) gather 的 K 次 scatter-read；(2) val/idx 两次 D2D memcpy。fp16 仅用低 16 bit -> 2 pass。
//   workspace 由调用方传入：keys_in/keys_out [total] + idx_out [total] + offsets [B+1] + cub tmp。

// CUB 分段 sort 的 tmp_bytes（与数据无关，可用 null 指针查询）。
template <typename T>
static size_t topk_sort_cub_tmp_bytes(int B, int K) {
    int64_t total = (int64_t)B * K;
    const int end_bit = EncTraits<T>::kBits;
    size_t tmp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, tmp_bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr,
        (const int32_t*)nullptr, (int32_t*)nullptr,
        (int)total, B, (const int32_t*)nullptr, (const int32_t*)nullptr, 0, end_bit, (cudaStream_t)0);
    return tmp_bytes;
}

template <typename T>
size_t topk_sort_ws_bytes(int B, int K) {
    if (B <= 0 || K <= 1) return 0;
    int64_t total = (int64_t)B * K;
    WsBump b(nullptr);
    b.take<uint32_t>(total);            // keys_in
    b.take<uint32_t>(total);            // keys_out
    b.take<int32_t>(total);             // idx_out
    b.take<int32_t>(B + 1);             // offsets
    size_t fixed = b.used();
    size_t tmp = topk_sort_cub_tmp_bytes<T>(B, K);
    return fixed + ft_align256(tmp);
}

template <typename T>
void topk_sort_desc(T* out_val, int* out_idx, int B, int K, void* ws, cudaStream_t stream) {
    if (B <= 0 || K <= 1) return;          // K<=1 无需排序
    int64_t total = (int64_t)B * K;

    WsBump b(ws);
    uint32_t* keys_in  = b.take<uint32_t>(total);
    uint32_t* keys_out = b.take<uint32_t>(total);
    int32_t*  idx_out  = b.take<int32_t>(total);
    int32_t*  offsets  = b.take<int32_t>(B + 1);
    void*     tmp      = reinterpret_cast<void*>(b.take<char>(0));  // 余下区域作 cub tmp
    size_t    tmp_bytes = topk_sort_cub_tmp_bytes<T>(B, K);

    // 段偏移 0, K, 2K, ...（kernel 填，避免 host->device 同步拷贝）
    {
        int blk = 256;
        int gr = (B + 1 + blk - 1) / blk;
        ft_fill_offsets_kernel<<<gr, blk, 0, stream>>>(offsets, B, K);
    }

    // 生成 keys（payload 直接用 out_idx）
    {
        int blk = 256;
        int64_t gr = (total + blk - 1) / blk; if (gr < 1) gr = 1;
        ft_sort_make_keys_kernel<T><<<(int)gr, blk, 0, stream>>>(out_val, keys_in, total);
    }

    // cub 分段降序 radix sort：key=enc，value=原始索引；fp16 只需排低 16 bit -> 2 pass
    const int end_bit = EncTraits<T>::kBits;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        tmp, tmp_bytes,
        keys_in, keys_out, out_idx, idx_out,
        (int)total, B, offsets, offsets + 1, 0, end_bit, stream);

    // 解码 sorted key -> value，sorted idx -> out_idx（in-place 写回 out_val/out_idx）
    {
        int blk = 256;
        int64_t gr = (total + blk - 1) / blk; if (gr < 1) gr = 1;
        ft_sort_finalize_kernel<T><<<(int)gr, blk, 0, stream>>>(
            keys_out, idx_out, out_val, out_idx, total);
    }
}

} // namespace

size_t topk_sort_desc_workspace_bytes_fp32(int B, int K) {
    return topk_sort_ws_bytes<float>(B, K);
}
size_t topk_sort_desc_workspace_bytes_fp16(int B, int K) {
    return topk_sort_ws_bytes<__half>(B, K);
}
size_t topk_sort_desc_workspace_bytes_bf16(int B, int K) {
    return topk_sort_ws_bytes<__nv_bfloat16>(B, K);
}

void launch_topk_sort_desc_fp32_ws(float* out_val, int* out_idx,
                                   int B, int K, void* ws, cudaStream_t stream) {
    topk_sort_desc<float>(out_val, out_idx, B, K, ws, stream);
}
void launch_topk_sort_desc_fp16_ws(__half* out_val, int* out_idx,
                                   int B, int K, void* ws, cudaStream_t stream) {
    topk_sort_desc<__half>(out_val, out_idx, B, K, ws, stream);
}
void launch_topk_sort_desc_bf16_ws(__nv_bfloat16* out_val, int* out_idx,
                                   int B, int K, void* ws, cudaStream_t stream) {
    topk_sort_desc<__nv_bfloat16>(out_val, out_idx, B, K, ws, stream);
}

// 兼容旧接口：内部 cudaMallocAsync 自管 workspace。
void launch_topk_sort_desc_fp32(float* out_val, int* out_idx,
                                int B, int K, cudaStream_t stream) {
    size_t bytes = topk_sort_ws_bytes<float>(B, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    topk_sort_desc<float>(out_val, out_idx, B, K, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

void launch_topk_sort_desc_fp16(__half* out_val, int* out_idx,
                                int B, int K, cudaStream_t stream) {
    size_t bytes = topk_sort_ws_bytes<__half>(B, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    topk_sort_desc<__half>(out_val, out_idx, B, K, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

void launch_topk_sort_desc_bf16(__nv_bfloat16* out_val, int* out_idx,
                                int B, int K, cudaStream_t stream) {
    size_t bytes = topk_sort_ws_bytes<__nv_bfloat16>(B, K);
    void* ws = nullptr;
    if (bytes) cudaMallocAsync(&ws, bytes, stream);
    topk_sort_desc<__nv_bfloat16>(out_val, out_idx, B, K, ws, stream);
    if (ws) cudaFreeAsync(ws, stream);
}

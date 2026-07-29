// LiteTopK DSA V3 hybrid host wrapper: DeepGEMM-2.5 scoring loop + V1 KV-split;
// scoring kernel (sm100_dsa_litetopk.cuh) with the sparse candidate epilogue,
// plus the architecture-agnostic radix-select post-kernels (copied verbatim
// from dsa_litetopk.cu). Build against the DeepGEMM 2.5 include tree + its
// bundled CUTLASS (NOT the legacy deep_gemm include tree V1 uses).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <limits>
#include <optional>
#include <tuple>

#include "sm100_dsa_litetopk.cuh"

namespace {

using CandidateValue = dsa_litetopk::CandidateValue;

static torch::TensorOptions candidate_options(
        const torch::TensorOptions& options) {
#ifdef DSA_CANDIDATE_U16
    // torch.float16 is only the owning 16-bit storage type here.  CUDA treats
    // its payload as an opaque uint16 score code; no half arithmetic occurs.
    return options.dtype(torch::kHalf);
#else
    return options.dtype(torch::kFloat);
#endif
}

static CandidateValue* candidate_data_ptr(torch::Tensor& tensor) {
#ifdef DSA_CANDIDATE_U16
    return reinterpret_cast<CandidateValue*>(
        tensor.data_ptr<at::Half>());
#else
    return tensor.data_ptr<float>();
#endif
}

static void check_candidate_dtype(const torch::Tensor& tensor) {
#ifdef DSA_CANDIDATE_U16
    TORCH_CHECK(
        tensor.scalar_type() == torch::kHalf,
        "DSA_CANDIDATE_U16 cand_val storage must be float16");
#else
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat,
        "cand_val must be fp32");
#endif
}

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
constexpr int NUM_Q_STAGES = 1;   // one q-block per CTA
#ifndef DSA_V3_KV_STAGES
#define DSA_V3_KV_STAGES 4
#ifndef DSA_SCALE_HEADROOM
#define DSA_SCALE_HEADROOM 0.5f  // fine-scale drift headroom above sample max
#endif
#endif
constexpr int NUM_KV_STAGES = DSA_V3_KV_STAGES;
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

// Fused seed/prep kernel (one block per row, all state in smem — borrows the
// vLLM top_k_per_row engineering): from the sample scores [Q, head] derive the
// per-row bucket params (origin, inv_delta), the initial gate threshold
// (bucket of the K-th best sample score), write the FULL sample histogram into
// bcount (a valid, conservative refresh base: counting genuine row elements
// can only tighten th safely), and emit every sample position with
// bucket <= th as initial candidates — a SUPERSET of the sample top-K, which
// the exact final select trims. Replaces: aminmax + torch.topk/radix seed +
// neg/contiguous copies + host seed copies + seed_bcount_kernel (~6 passes,
// ~10 launches) with 3 passes in 1 launch.
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
    CandidateValue* __restrict__ cand_val,
    int32_t* __restrict__ cand_idx,
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
    const float span = fmaxf(hi - o, 1e-20f);
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
    // Coarse K-th estimate, then REBUILD the scale over just the useful
    // range: [~K-th value (+1 coarse bucket slack) .. sample max + drift
    // headroom]. The bottom of [min,max] cannot contain the final threshold,
    // while headroom prevents scores above the sample maximum from collapsing
    // into bucket 0. The resulting scale concentrates bins around the useful
    // threshold range.
#ifdef DSA_SPARSE_REFRESH_ZERO_BASE
    // The production U16 contract uses emit_limit==0 and a single KV split.
    // Its scan covers the complete KV range and initializes the CTA-local
    // histogram itself, so writing Q*NB zeros to global memory is dead work.
    if (emit_limit != 0)
#endif
        for (int b = tid; b < NB; b += BT)
            // Probe mode (emit_limit==0): scan-side refresh starts from zero.
            bcount[(size_t)row * NB + b] =
                (emit_limit == 0) ? 0 : s_hist[b];
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
    CandidateValue* vrow = cand_val + (size_t)row * cap;
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
#ifdef DSA_BUCKET_GATE4
                    // GATE4 build: candidate values live in BUCKET SPACE
                    // build-wide (the scan writes bq; select is rebased).
                    // Seeds must match: (x - o)*inv, same affine as the scan.
                    vrow[pos] = static_cast<CandidateValue>(
                        (-s - o) * inv);
#else
                    vrow[pos] = static_cast<CandidateValue>(-s);
#endif
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

#ifdef DSA_CHUNKED_EMIT
__global__ void compact_chunked_emit_kernel(
    const uint64_t* __restrict__ workspace,
    CandidateValue* __restrict__ cand_val,
    int32_t* __restrict__ cand_idx,
    int32_t* __restrict__ cand_cnt,
    int rows,
    int cap,
    int num_kv_splits,
    int chunks_per_split,
    uint32_t probe_group,
    uint64_t probe_magic,
    uint32_t probe_add_max) {
    constexpr int kMathWarps = MATH_THREADS / 32;
    const int qblock = blockIdx.x;
    const int tid = threadIdx.x;
    const int source_warp = tid >> 5;
    const int lane = tid & 31;
    const int num_q_blocks = (rows + BLOCK_Q - 1) / BLOCK_Q;
    if (qblock >= num_q_blocks) return;

    const uint64_t num_chunks =
        static_cast<uint64_t>(num_q_blocks) * num_kv_splits *
        chunks_per_split * kMathWarps;
    const uint64_t record_count =
        num_chunks * BLOCK_Q * DSA_EMIT_LANE_SLOTS * 32u;
    const uint32_t* counts =
        reinterpret_cast<const uint32_t*>(workspace + record_count);

    __shared__ int cursor[BLOCK_Q];
    if (tid < BLOCK_Q) {
        const int row = qblock * BLOCK_Q + tid;
        cursor[tid] = row < rows ? cand_cnt[row] : 0;
    }
    __syncthreads();

    for (int split = 0; split < num_kv_splits; ++ split) {
        for (int chunk = 0; chunk < chunks_per_split; ++ chunk) {
            const uint64_t chunk_linear =
                ((static_cast<uint64_t>(qblock) * num_kv_splits + split) *
                     chunks_per_split +
                 chunk) *
                    kMathWarps +
                source_warp;
            const uint32_t packed =
                counts[chunk_linear * 32u + lane];
            const uint64_t record_base =
                chunk_linear * (BLOCK_Q * DSA_EMIT_LANE_SLOTS * 32u);

            #pragma unroll
            for (int i = 0; i < BLOCK_Q; ++ i) {
                const int row = qblock * BLOCK_Q + i;
                if (row >= rows) continue;
                const uint32_t count = (packed >> (i * 8)) & 0xffu;
                const uint32_t total =
                    __reduce_add_sync(0xffffffffu, count);
                if (total == 0) continue;

                uint32_t offset = count;
                #pragma unroll
                for (int delta = 1; delta < 32; delta <<= 1) {
                    const uint32_t other =
                        __shfl_up_sync(0xffffffffu, offset, delta);
                    if (lane >= delta) offset += other;
                }
                offset -= count;

                int out_base = 0;
                if (lane == 0)
                    out_base = atomicAdd(cursor + i,
                                         static_cast<int>(total));
                out_base =
                    __shfl_sync(0xffffffffu, out_base, 0);
                const int copy_count = min(
                    static_cast<int>(count),
                    max(cap - out_base - static_cast<int>(offset), 0));
                for (int slot = 0; slot < copy_count; ++ slot) {
                    const uint64_t record =
                        workspace[
                            record_base +
                            (i * DSA_EMIT_LANE_SLOTS + slot) * 32u +
                            lane];
                    const int out =
                        out_base + static_cast<int>(offset) + slot;
                    const uint64_t out_pos =
                        static_cast<uint64_t>(row) * cap + out;
                    const uint32_t record_payload =
                        static_cast<uint32_t>(record);
                    uint32_t kvo = static_cast<uint32_t>(record >> 32);
                    if (probe_group != 0) {
                        const uint32_t sup =
                            static_cast<uint32_t>(
                                (static_cast<uint64_t>(kvo) *
                                 probe_magic) >>
                                42);
                        kvo += min((sup + 1) * 64u, probe_add_max);
                    }
                    DSA_ST_CANDIDATE_RECORD(
                        cand_val[out_pos],
                        cand_idx[out_pos],
                        record_payload,
                        kvo);
                }
            }
        }
    }

    __syncthreads();
    if (tid < BLOCK_Q) {
        const int row = qblock * BLOCK_Q + tid;
        if (row < rows) cand_cnt[row] = cursor[tid];
    }
}
#endif

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

#ifdef DSA_INPLACE_BOUNDARY_SELECT
// DSA specialization of the GitHub FlashTopK boundary-bucket strategy.
//
// The generic selector above logically restricts the radix set to bucket
// `th`, but every radix pass still rereads all `n` candidates and filters
// them. Sparse refresh normally leaves:
//
//     count(bucket < th) < K <= count(bucket <= th).
//
// Make that saving physical: one tiled pass writes bucket<th directly to the
// final output and compacts bucket==th in-place at the front of the candidate
// buffer. The four radix passes then read only that compact boundary. A tile
// is loaded completely before any write, and the compacted prefix can never
// extend beyond the end of the processed tile, so aliasing input/output is
// race-free and needs no second multi-GiB candidate slab.
//
// The two fallback modes mirror compact_topk_min_thr_litetopk_kernel:
//   * threshold too loose (lt >= K): compact/radix the lt set;
//   * threshold underfilled: compact/radix every finite buffered candidate.
__global__ void compact_topk_min_thr_inplace_idx_out_litetopk_kernel(
    CandidateValue* __restrict__ val,
    int32_t* __restrict__ idx,
    const int32_t* __restrict__ cnt,
    const int32_t* __restrict__ th_in,
    const int32_t* __restrict__ boundary_meta,
    int R,
    int CAP,
    int K,
    int NB,
    int32_t* __restrict__ out_idx) {
    constexpr int BT = 256;
    constexpr int RADIX = 256;
    const unsigned FULL = 0xffffffffu;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const unsigned lane_mask =
        lane == 0 ? 0u : ((1u << lane) - 1u);
    if (row >= R) return;

    CandidateValue* vrow =
        val + static_cast<size_t>(row) * CAP;
    int32_t* irow = idx + static_cast<size_t>(row) * CAP;
    int32_t* oi = out_idx + static_cast<size_t>(row) * K;
    const int raw_n = cnt[row];
    int n = raw_n;
    if (n > CAP) n = CAP;
    if (n < 0) n = 0;
    if (n == 0) {
        for (int j = tid; j < K; j += BT) {
            oi[j] = 0;
        }
        return;
    }

    const int th = th_in[row];
#if defined(DSA_CANDIDATE_U16) && \
    !defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8)
    // The packed boundary remains bit-exact only above its compile-time
    // lower bound.  Fail loudly instead of silently turning the exact path
    // into an approximation.
#ifdef DSA_CANDIDATE_U16_BUCKET_RESIDUAL
    constexpr int kPackedExactThreshold = 8;
#else
    constexpr int kPackedExactThreshold = 1;
#endif
    if (th < kPackedExactThreshold) {
        asm volatile("trap;");
        return;
    }
#endif

    // mode 0: standard boundary path; 1: loose threshold; 2: underfilled.
    __shared__ int s_count_lt;
    __shared__ int s_count_eq;
    __shared__ int s_count_valid;
    __shared__ int s_have_boundary_meta;
    __shared__ int s_mode;
    __shared__ int s_k_target;
    constexpr int BOUNDARY_SMEM_CAP = 256;
#ifdef DSA_CANDIDATE_U16
    __shared__ uint32_t s_boundary_val[BOUNDARY_SMEM_CAP];
#else
    __shared__ float s_boundary_val[BOUNDARY_SMEM_CAP];
#endif
    __shared__ int32_t s_boundary_idx[BOUNDARY_SMEM_CAP];
    __shared__ int s_fast_lt_cursor;
    __shared__ int s_fast_eq_cursor;
    __shared__ uint32_t s_fast_hist[RADIX];
    __shared__ uint32_t s_fast_desired;
    __shared__ uint32_t s_fast_kfind;
    __shared__ int s_fast_pivot_lt;
    __shared__ int s_fast_write_lt;
    __shared__ int s_fast_write_eq;
    if (tid == 0) {
        const int32_t* meta =
            boundary_meta + static_cast<size_t>(row) * NB;
        const int tag = meta[0];
        const int meta_th = ~tag;
        const int meta_lt = meta[1];
        const int meta_eq = meta[2];
        const int meta_need = K - meta_lt;
        s_have_boundary_meta =
            tag < 0 && meta_th == th &&
            meta_th >= 0 && meta_th < NB &&
            raw_n >= 0 && raw_n <= CAP &&
            meta_lt >= 0 && meta_eq >= 0 &&
            meta_lt < K && meta_need > 0 &&
            meta_need <= meta_eq &&
            meta_lt + meta_eq <= n;
        s_count_lt = s_have_boundary_meta ? meta_lt : 0;
        s_count_eq = s_have_boundary_meta ? meta_eq : 0;
        s_count_valid = 0;
    }
    __syncthreads();

#ifdef DSA_CANDIDATE_U16
    // The six-byte representation is conditionally exact for the certified
    // sparse-refresh boundary path.  A missing certificate could require a
    // top-K selection within collapsed bucket 0, so it is an explicit error.
    if (!s_have_boundary_meta) {
        asm volatile("trap;");
        return;
    }
#endif

    if (!s_have_boundary_meta) {
        int local_lt = 0;
        int local_eq = 0;
        int local_valid = 0;
        for (int j = tid; j < n; j += BT) {
            const float v =
                dsa_litetopk::candidate_decode_score(
                    vrow[j], irow[j]);
            if (!isfinite(v)) continue;
            ++local_valid;
            int braw = static_cast<int>(v);
            const int b =
                braw < 0 ? 0 :
                (braw > NB - 1 ? NB - 1 : braw);
            local_lt += b < th;
            local_eq += b == th;
        }
        atomicAdd(&s_count_lt, local_lt);
        atomicAdd(&s_count_eq, local_eq);
        atomicAdd(&s_count_valid, local_valid);
    }
    __syncthreads();
    if (tid == 0) {
        const int need = K - s_count_lt;
        if (s_count_lt < K &&
            need > 0 && need <= s_count_eq) {
            s_mode = 0;
            s_k_target = need;
        } else if (s_count_lt >= K) {
            s_mode = 1;
            s_k_target = K;
        } else {
            s_mode = 2;
            s_k_target = min(K, s_count_valid);
        }
    }
    __syncthreads();

    // Production sparse-refresh distribution (Q=8192, K=2048):
    // boundary E averages ~97 candidates, P99 ~163, max 212 on the 1M
    // corpus. Keep that boundary entirely in shared memory. Unlike the
    // generic in-place path below, this pass has no aliasing stores, so
    // warp-local shared-atomic reservations need no CTA barrier per tile.
    if (s_have_boundary_meta &&
        s_count_eq <= BOUNDARY_SMEM_CAP) {
        if (tid == 0) {
            s_fast_lt_cursor = 0;
            s_fast_eq_cursor = 0;
        }
        __syncthreads();
        for (int tile = 0; tile < n; tile += BT) {
            const int j = tile + tid;
#ifdef DSA_CANDIDATE_U16
            uint32_t score_code = 0u;
            bool valid = false;
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
            uint32_t bucket = 0u;
            if (j < n) {
                score_code =
                    static_cast<uint32_t>(vrow[j]);
                bucket = score_code >> 8;
                valid = true;
            }
            const bool is_lt =
                valid && bucket < static_cast<uint32_t>(th);
            const bool is_eq =
                valid && bucket == static_cast<uint32_t>(th);
#elif defined(DSA_CANDIDATE_FP16_NUMERIC)
            uint32_t bucket = 0u;
            if (j < n) {
                score_code =
                    static_cast<uint32_t>(vrow[j]);
                bucket =
                    dsa_litetopk::candidate_load_bucket(
                        irow[j]);
                valid = true;
            }
            const bool is_lt =
                valid && bucket < static_cast<uint32_t>(th);
            const bool is_eq =
                valid && bucket == static_cast<uint32_t>(th);
#elif defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL)
            uint32_t bucket = 0u;
            if (j < n) {
                score_code =
                    dsa_litetopk::candidate_load_residual(
                        vrow[j], irow[j]);
                bucket =
                    dsa_litetopk::candidate_load_bucket(
                        irow[j]);
                valid = true;
            }
            const bool is_lt =
                valid && bucket < static_cast<uint32_t>(th);
            const bool is_eq =
                valid && bucket == static_cast<uint32_t>(th);
#else
            if (j < n) {
                score_code =
                    dsa_litetopk::candidate_load_score_code(
                        vrow[j], irow[j]);
                valid = score_code != 0u;
            }
            const uint32_t th_code =
                dsa_litetopk::candidate_score_code(
                    static_cast<float>(th));
            const uint32_t next_th_code =
                dsa_litetopk::candidate_score_code(
                    static_cast<float>(th + 1));
            const bool is_lt =
                valid && score_code < th_code;
            const bool is_eq =
                valid && score_code >= th_code &&
                score_code < next_th_code;
#endif
#else
            float v = INFINITY;
            int b = NB;
            bool valid = false;
            if (j < n) {
                v = dsa_litetopk::candidate_decode_score(
                    vrow[j], irow[j]);
                valid = isfinite(v);
#ifndef DSA_BOUNDARY_FLOAT_CLASSIFY
                if (valid) {
                    int braw = static_cast<int>(v);
                    b = braw < 0 ? 0 :
                        (braw > NB - 1 ? NB - 1 : braw);
                }
#endif
            }
#ifdef DSA_BOUNDARY_FLOAT_CLASSIFY
            // Gate4 candidates are already in bucket space. For an interior
            // integer threshold, clamp(int(v), 0, NB-1) < th is exactly
            // v < th, and equality is the half-open interval [th, th+1).
            // The two edge buckets absorb the clamped tails.
            const float th_lo = static_cast<float>(th);
            const bool is_lt =
                valid && th > 0 && v < th_lo;
            const bool is_eq =
                valid &&
                (th == 0
                     ? v < 1.0f
                     : (th == NB - 1
                            ? v >= th_lo
                            : (v >= th_lo && v < th_lo + 1.0f)));
#else
            const bool is_lt = valid && b < th;
            const bool is_eq = valid && b == th;
#endif
#endif
            const unsigned lt_mask =
                __ballot_sync(FULL, is_lt);
            const unsigned eq_mask =
                __ballot_sync(FULL, is_eq);
            int warp_lt_base = 0;
            int warp_eq_base = 0;
            if (lane == 0) {
                const int lt_count = __popc(lt_mask);
                const int eq_count = __popc(eq_mask);
                if (lt_count != 0)
                    warp_lt_base =
                        atomicAdd(&s_fast_lt_cursor, lt_count);
                if (eq_count != 0)
                    warp_eq_base =
                        atomicAdd(&s_fast_eq_cursor, eq_count);
            }
            warp_lt_base =
                __shfl_sync(FULL, warp_lt_base, 0);
            warp_eq_base =
                __shfl_sync(FULL, warp_eq_base, 0);

            if (is_lt) {
                const int pos =
                    warp_lt_base + __popc(lt_mask & lane_mask);
                if (pos < K) {
                    const int32_t raw_idx = irow[j];
                    oi[pos] =
                        dsa_litetopk::candidate_decode_index(
                            raw_idx);
                }
            }
            if (is_eq) {
                const int pos =
                    warp_eq_base + __popc(eq_mask & lane_mask);
                if (pos < BOUNDARY_SMEM_CAP) {
#ifdef DSA_CANDIDATE_U16
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
                    s_boundary_val[pos] = score_code & 0xffu;
#else
                    s_boundary_val[pos] = score_code;
#endif
#else
                    s_boundary_val[pos] = v;
#endif
                    s_boundary_idx[pos] =
                        dsa_litetopk::candidate_decode_index(
                            irow[j]);
                }
            }
        }
        __syncthreads();

        const int boundary_n = s_fast_eq_cursor;
        const int output_base = s_fast_lt_cursor;
        const int k_target = K - output_base;
            if (boundary_n == k_target) {
                for (int j = tid;
                     j < boundary_n; j += BT) {
                    oi[output_base + j] = s_boundary_idx[j];
                }
                return;
            }

            if (tid == 0) {
#if defined(DSA_CANDIDATE_U16) && \
    !defined(DSA_CANDIDATE_U16_BUCKET_RESIDUAL) && \
    !defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8) && \
    !defined(DSA_CANDIDATE_FP16_NUMERIC)
                s_fast_desired =
                    s_boundary_val[0] & 0xff000000u;
#else
                s_fast_desired = 0u;
#endif
                s_fast_kfind =
                    static_cast<uint32_t>(k_target);
            }
            __syncthreads();
            uint32_t fast_mask = 0u;
            #pragma unroll
#ifdef DSA_CANDIDATE_U16_BUCKET8_FRAC8
            // The boundary bucket is fixed; the low byte is its quantized
            // fractional rank, so one radix pass is sufficient.
            for (int pass = 0; pass < 1; ++ pass) {
                const int shift = 0;
#elif defined(DSA_CANDIDATE_FP16_NUMERIC)
            // Positive IEEE FP16 bit patterns are monotonic. Two byte-wise
            // passes select exactly within the rounded FP16 boundary values.
            for (int pass = 0; pass < 2; ++ pass) {
                const int shift = 8 - pass * 8;
#elif defined(DSA_CANDIDATE_U16)
            // Every value is in one integer bucket. That interval spans at
            // most 23 varying FP32 mantissa bits. Select on those three bytes.
            for (int pass = 0; pass < 3; ++ pass) {
                const int shift = 16 - pass * 8;
#else
            for (int pass = 0; pass < 4; ++ pass) {
                const int shift = 24 - pass * 8;
#endif
                s_fast_hist[tid] = 0;
                __syncthreads();
                const uint32_t desired = s_fast_desired;
                if (tid < boundary_n) {
#ifdef DSA_CANDIDATE_U16
                    const uint32_t encoded =
                        s_boundary_val[tid];
#else
                    const uint32_t encoded =
                        compact_enc_float(s_boundary_val[tid]);
#endif
                    if ((encoded & fast_mask) ==
                        (desired & fast_mask)) {
                        atomicAdd(
                            &s_fast_hist[
                                (encoded >> shift) & 0xffu],
                            1u);
                    }
                }
                __syncthreads();
                if (tid == 0) {
                    uint32_t acc = 0;
                    const uint32_t kfind = s_fast_kfind;
                    for (int digit = 0;
                         digit < RADIX; ++ digit) {
                        const uint32_t count =
                            s_fast_hist[digit];
                        if (acc < kfind &&
                            kfind <= acc + count) {
                            s_fast_desired =
                                desired |
                                (static_cast<uint32_t>(digit)
                                 << shift);
                            s_fast_kfind = kfind - acc;
                            break;
                        }
                        acc += count;
                    }
                }
                __syncthreads();
                fast_mask |= 0xffu << shift;
            }
            const uint32_t pivot = s_fast_desired;

            if (tid == 0) {
                s_fast_pivot_lt = 0;
                s_fast_write_lt = 0;
                s_fast_write_eq = 0;
            }
            __syncthreads();
            if (tid < boundary_n &&
#ifdef DSA_CANDIDATE_U16
                s_boundary_val[tid] < pivot)
#else
                compact_enc_float(s_boundary_val[tid]) < pivot)
#endif
                atomicAdd(&s_fast_pivot_lt, 1);
            __syncthreads();
            const int eq_take =
                max(k_target - s_fast_pivot_lt, 0);
            if (tid < boundary_n) {
#ifdef DSA_CANDIDATE_U16
                const uint32_t encoded =
                    s_boundary_val[tid];
#else
                const float v = s_boundary_val[tid];
                const uint32_t encoded =
                    compact_enc_float(v);
#endif
                if (encoded < pivot) {
                    const int pos = atomicAdd(
                        &s_fast_write_lt, 1);
                    if (pos < k_target) {
                        oi[output_base + pos] =
                            s_boundary_idx[tid];
                    }
                } else if (encoded == pivot) {
                    const int equal_rank = atomicAdd(
                        &s_fast_write_eq, 1);
                    if (equal_rank < eq_take) {
                        const int pos =
                            output_base +
                            s_fast_pivot_lt +
                            equal_rank;
                        if (pos < K) {
                            oi[pos] = s_boundary_idx[tid];
                        }
                    }
                }
            }
            return;
    }

    // Tiled, alias-safe in-place compaction. In the standard mode, lt
    // candidates bypass the compact buffer and go straight to output.
    __shared__ int s_compact_base;
    __shared__ int s_direct_base;
    if (tid == 0) {
        s_compact_base = 0;
        s_direct_base = 0;
    }
    __syncthreads();

    for (int tile = 0; tile < n; tile += BT) {
        const int j = tile + tid;
        CandidateValue raw_value{};
        float v = INFINITY;
        int32_t raw_idx = 0;
        int b = NB;
        bool valid = false;
        if (j < n) {
            raw_value = vrow[j];
            raw_idx = irow[j];
            v = dsa_litetopk::candidate_decode_score(
                raw_value, raw_idx);
            valid = isfinite(v);
            if (valid) {
#ifdef DSA_CANDIDATE_FP16_NUMERIC
                // FP16 is used only to rank candidates inside the final
                // bucket. Classification must use the exact FP32 source
                // bucket packed by the scan, including this cold E>256
                // fallback path.
                b = min(
                    static_cast<int>(
                        dsa_litetopk::candidate_load_bucket(raw_idx)),
                    NB - 1);
#else
                int braw = static_cast<int>(v);
                b = braw < 0 ? 0 :
                    (braw > NB - 1 ? NB - 1 : braw);
#endif
            }
        }

        const bool is_lt = valid && b < th;
        bool selected = false;
        if (s_mode == 0)
            selected = valid && b == th;
        else if (s_mode == 1)
            selected = is_lt;
        else
            selected = valid;
        const bool direct = s_mode == 0 && is_lt;

        const unsigned selected_mask =
            __ballot_sync(FULL, selected);
        const unsigned direct_mask =
            __ballot_sync(FULL, direct);
        int warp_compact_base = 0;
        int warp_direct_base = 0;
        if (lane == 0) {
            const int selected_count = __popc(selected_mask);
            const int direct_count = __popc(direct_mask);
            if (selected_count != 0)
                warp_compact_base =
                    atomicAdd(&s_compact_base, selected_count);
            if (direct_count != 0)
                warp_direct_base =
                    atomicAdd(&s_direct_base, direct_count);
        }
        warp_compact_base =
            __shfl_sync(FULL, warp_compact_base, 0);
        warp_direct_base =
            __shfl_sync(FULL, warp_direct_base, 0);

        // One CTA barrier per tile is sufficient for alias safety: every
        // source element is already in a register and every warp has reserved
        // its compact ranges before any in-place store starts. Compact output
        // never reaches the next (unread) tile.
        __syncthreads();

        if (direct) {
            const int pos =
                warp_direct_base +
                __popc(direct_mask & lane_mask);
            if (pos < K) {
                oi[pos] =
                    dsa_litetopk::candidate_decode_index(
                        raw_idx);
            }
        }
        if (selected) {
            const int pos =
                warp_compact_base +
                __popc(selected_mask & lane_mask);
            vrow[pos] = raw_value;
            irow[pos] = raw_idx;
        }
    }
    __syncthreads();

    const int selected_n = s_compact_base;
    const int output_base = s_mode == 0 ? s_count_lt : 0;
    const int k_target = s_k_target;

    // Exact fallback with fewer than K finite buffered candidates.
    if (s_mode == 2 && selected_n <= K) {
        for (int j = tid; j < selected_n; j += BT) {
            oi[j] =
                dsa_litetopk::candidate_decode_index(
                    irow[j]);
        }
        for (int j = selected_n + tid; j < K; j += BT) {
            oi[j] = 0;
        }
        return;
    }
    if (selected_n == 0 || k_target == 0) {
        for (int j = output_base + tid; j < K; j += BT) {
            oi[j] = 0;
        }
        return;
    }

    // Radix-select only the compacted set. In the expected sparse-refresh
    // case this is exactly the threshold bucket and k_target == K-count_lt.
    __shared__ uint32_t hist[RADIX];
    __shared__ uint32_t desired;
    __shared__ uint32_t kfind;
    __shared__ int s_pivot_lt;
    __shared__ int s_write_lt;
    __shared__ int s_write_eq;
    if (tid == 0) {
        desired = 0u;
        kfind = static_cast<uint32_t>(k_target);
    }
    __syncthreads();

    uint32_t mask = 0u;
    #pragma unroll
    for (int pass = 0; pass < 4; ++ pass) {
        const int shift = 24 - pass * 8;
        hist[tid] = 0;
        __syncthreads();
        const uint32_t d = desired;
        for (int j = tid; j < selected_n; j += BT) {
            const uint32_t e = compact_enc_float(
                dsa_litetopk::candidate_decode_score(
                    vrow[j], irow[j]));
            if ((e & mask) == (d & mask))
                atomicAdd(&hist[(e >> shift) & 0xffu], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0;
            const uint32_t kf = kfind;
            for (int digit = 0; digit < RADIX; ++ digit) {
                const uint32_t h = hist[digit];
                if (acc < kf && kf <= acc + h) {
                    desired =
                        d | (static_cast<uint32_t>(digit) << shift);
                    kfind = kf - acc;
                    break;
                }
                acc += h;
            }
        }
        __syncthreads();
        mask |= 0xffu << shift;
    }
    const uint32_t pivot = desired;

    if (tid == 0) {
        s_pivot_lt = 0;
        s_write_lt = 0;
        s_write_eq = 0;
    }
    __syncthreads();
    int pivot_lt = 0;
    for (int j = tid; j < selected_n; j += BT)
        pivot_lt += compact_enc_float(
            dsa_litetopk::candidate_decode_score(
                vrow[j], irow[j])) < pivot;
    atomicAdd(&s_pivot_lt, pivot_lt);
    __syncthreads();
    const int eq_take = max(k_target - s_pivot_lt, 0);

    for (int j = tid; j < selected_n; j += BT) {
        const float v =
            dsa_litetopk::candidate_decode_score(
                vrow[j], irow[j]);
        const uint32_t e = compact_enc_float(v);
        if (e < pivot) {
            const int w = atomicAdd(&s_write_lt, 1);
            const int pos = output_base + w;
            if (pos < K) {
                oi[pos] =
                    dsa_litetopk::candidate_decode_index(
                        irow[j]);
            }
        } else if (e == pivot) {
            const int equal_rank = atomicAdd(&s_write_eq, 1);
            if (equal_rank < eq_take) {
                const int pos =
                    output_base + s_pivot_lt + equal_rank;
                if (pos < K) {
                    oi[pos] =
                        dsa_litetopk::candidate_decode_index(
                            irow[j]);
                }
            }
        }
    }
}
#endif

static int compute_smem_bytes() {
    const int esz_fp8 = 1, esz_f32 = 4;
    const int smem_q  = BLOCK_Q * NUM_HEADS * HEAD_DIM * esz_fp8;
    const int smem_w  = BLOCK_Q * NUM_HEADS * esz_f32;
    const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
    const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
    const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 + (MATH_THREADS / 128) * DSA_UMMA_STAGES * 2;
    const int smem_barriers = num_barriers * 8;
#ifdef DSA_PERSIST
    const int smem_slots = 8 * (int)sizeof(uint32_t);  // + done_qb, ack[2]
#else
    const int smem_slots = 4 * (int)sizeof(uint32_t);  // tmem ptr + daemon mailboxes
#endif
#ifdef DSA_CHUNKED_EMIT
#ifdef DSA_CANDIDATE_FP16_LOCAL32
    constexpr int emit_record_bytes = (int)sizeof(uint32_t);
#else
    constexpr int emit_record_bytes = (int)sizeof(uint64_t);
#endif
    const int smem_warpq =
        (MATH_THREADS / 32) * BLOCK_Q *
        ((int)sizeof(int32_t) +
         DSA_EMIT_LANE_SLOTS * 32 * emit_record_bytes);
#else
    const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                           ((int)sizeof(int32_t) + DSA_WARP_QUEUE_CAP * ((int)sizeof(float) + (int)sizeof(int32_t)));
#endif
#ifdef DSA_STATIC_GATE
    const int smem_hist = 0;
#else
    const int smem_hist = BLOCK_Q * 256 * (int)sizeof(int32_t);  // per-CTA refresh
                                                                  // histogram (NB<=256)
#endif
#ifdef DSA_HIST_DEFER
    const int smem_safe = BLOCK_Q * (int)sizeof(int32_t);  // deferred-feed watermark
#else
    const int smem_safe = 0;
#endif
    return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
           NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks +
           smem_barriers + smem_slots + smem_warpq + smem_hist + smem_safe;
}

#ifdef DSA_CHUNKED_EMIT
struct ChunkedEmitWorkspace {
    torch::Tensor storage;
    int chunks_per_split;
};

static ChunkedEmitWorkspace make_chunked_emit_workspace(
        int rows, int seq_len_kv, int num_kv_splits,
        const torch::TensorOptions& options) {
    const int num_q_blocks =
        (rows + BLOCK_Q - 1) / BLOCK_Q;
    const int total_blocks =
        (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
    const int blocks_per_split =
        (total_blocks + num_kv_splits - 1) / num_kv_splits;
    const int chunks_per_split =
        (blocks_per_split + DSA_EMIT_CHUNK_BLOCKS - 1) /
        DSA_EMIT_CHUNK_BLOCKS;
    const int64_t num_chunks =
        static_cast<int64_t>(num_q_blocks) * num_kv_splits *
        chunks_per_split * (MATH_THREADS / 32);
    const int64_t record_count =
        num_chunks * BLOCK_Q * DSA_EMIT_LANE_SLOTS * 32;
    const int64_t count_words = num_chunks * 32;
    const int64_t storage_words =
        record_count + (count_words + 1) / 2;
    return {
        torch::empty({storage_words}, options.dtype(torch::kLong)),
        chunks_per_split,
    };
}

static void compact_chunked_emit(
        const ChunkedEmitWorkspace& workspace,
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        int num_kv_splits,
        cudaStream_t stream,
        uint32_t probe_group,
        uint64_t probe_magic,
        uint32_t probe_add_max) {
    const int rows = static_cast<int>(cand_val.size(0));
    const int cap = static_cast<int>(cand_val.size(1));
    const int num_q_blocks =
        (rows + BLOCK_Q - 1) / BLOCK_Q;
    compact_chunked_emit_kernel<<<num_q_blocks, MATH_THREADS, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(
            workspace.storage.data_ptr<int64_t>()),
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        rows,
        cap,
        num_kv_splits,
        workspace.chunks_per_split,
        probe_group,
        probe_magic,
        probe_add_max);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

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
#if defined(DSA_CANDIDATE_U16) && \
    !defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8)
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "DSA_CANDIDATE_U16 supports at most 1M KV positions");
#endif
    const int topk = static_cast<int>(topk64);
    // Sparse-only: honor a caller-provided cap in [topk, S).
    const int cand_cap = (cand_cap64 >= topk && cand_cap64 < seq_len_kv)
                             ? static_cast<int>(cand_cap64) : seq_len_kv;
    TORCH_CHECK(num_heads == NUM_HEADS && head_dim == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(kv.size(1) == HEAD_DIM, "kv D mismatch");
    TORCH_CHECK(origin.numel() == seq_len && inv_delta.numel() == seq_len && th_bucket.numel() == seq_len, "bucket params must have Q elements");
    const int num_buckets = static_cast<int>(num_buckets64);
#ifdef DSA_STATIC_GATE
    TORCH_CHECK(refresh_every64 == 0,
                "DSA_STATIC_GATE requires refresh_every=0");
#endif
#ifdef DSA_SPARSE_REFRESH
    TORCH_CHECK(refresh_every64 > 0,
                "DSA_SPARSE_REFRESH requires refresh_every>0");
#endif
    const bool external_refresh = (refresh_every64 < 0);
    const int refresh_every = external_refresh ? 0x7fffffff : static_cast<int>(refresh_every64);
#ifdef DSA_INPLACE_BOUNDARY_SELECT
    TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
                "in-place boundary select requires 3 <= num_buckets <= 256");
#else
    TORCH_CHECK(num_buckets >= 2 && num_buckets <= 256,
                "num_buckets out of range (smem hist sized for 256)");
#endif
    TORCH_CHECK(topk >= 1 && topk <= cand_cap, "topk must be in [1, cand_cap]");
    TORCH_CHECK(refresh_every64 >= -1, "refresh_every must be >= -1");
    TORCH_CHECK(seed_val.dim() == 2 && seed_idx.dim() == 2, "seed tensors must be [Q, seed_k]");
    TORCH_CHECK(seed_val.size(0) == seq_len && seed_idx.size(0) == seq_len && seed_val.size(1) == seed_idx.size(1),
                "seed tensor shape mismatch");
    const int seed_k = static_cast<int>(seed_val.size(1));
    TORCH_CHECK(seed_k <= cand_cap, "seed_k must be <= cand_cap");
#ifdef DSA_SPARSE_REFRESH
    TORCH_CHECK(
        seed_k == 0,
        "DSA_SPARSE_REFRESH generic scan requires empty seeds; use the "
        "prepared ext API so sampled positions are not double-counted");
#endif

    auto cand_val = torch::empty(
        {seq_len, cand_cap}, candidate_options(q.options()));
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

    // V1 KV-split grid: x = q-blocks, y = KV splits (~4 CTA waves per SM).
    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
    int num_kv_splits;
    if (num_kv_splits_override > 0) {
        num_kv_splits = (int)num_kv_splits_override;
    } else {
        constexpr int kWaves = 4;
        const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
        num_kv_splits = (kWaves * NUM_SMS + qb - 1) / qb;
        const int max_useful_splits = total_kv_blocks > 0 ? (total_kv_blocks + 1) / 2 : 1;
        if (num_kv_splits > max_useful_splits) num_kv_splits = max_useful_splits;
    }
    if (num_kv_splits < 1) num_kv_splits = 1;
    if (num_kv_splits > total_kv_blocks) num_kv_splits = total_kv_blocks > 0 ? total_kv_blocks : 1;
#ifdef DSA_CANDIDATE_U16
    TORCH_CHECK(
        num_kv_splits == 1,
        "DSA_CANDIDATE_U16 requires a single KV split so sparse refresh can "
        "publish an exact boundary certificate");
#endif
#ifdef DSA_SPARSE_REFRESH
    auto kernel = num_kv_splits == 1
        ? &dsa_litetopk::sm100_dsa_litetopk<
              NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128, true>
        : &dsa_litetopk::sm100_dsa_litetopk<
              NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128, false>;
#else
    auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_SMS, SPEC_THREADS, MATH_THREADS>;
#endif
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
    auto emit_workspace = make_chunked_emit_workspace(
        seq_len, seq_len_kv, num_kv_splits, q.options());
#endif
    dim3 grid((unsigned)num_q_blocks, (unsigned)num_kv_splits, 1);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(), (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
        (uint32_t)num_kv_splits, 0u, 0ULL, 0u,
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
        reinterpret_cast<uint64_t*>(
            emit_workspace.storage.data_ptr<int64_t>()),
#endif
        candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(), (uint32_t)cand_cap,
        tm_q, tm_kv, tm_ks, tm_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
    compact_chunked_emit(
        emit_workspace,
        cand_val,
        cand_idx,
        cand_cnt,
        num_kv_splits,
        stream,
        0u,
        0ULL,
        0u);
#endif

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
    TORCH_CHECK(NB >= 2 && NB <= 4096, "num_buckets out of range");
    TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
    TORCH_CHECK(cand_val.size(0) >= Q && cand_val.size(1) == cap, "cand_val shape");
    check_candidate_dtype(cand_val);
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
#ifdef DSA_CANDIDATE_U16
    TORCH_CHECK(
        emit_limit == 0,
        "DSA_CANDIDATE_U16 supports the hot-only no-seed contract");
#endif
    const int probe_stride_tok = (int)probe_stride_tok64;
    const int hist_stride = hist_stride64 > 1 ? (int)hist_stride64 : 1;
    seed_prep_kernel<<<Q, 1024, seed_smem, stream>>>(
        slog.data_ptr<float>(), slog.stride(0), head, NB, K, cap, emit_limit,
        probe_stride_tok, hist_stride,
        (float)headroom,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(),
        candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
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
    auto cand_val = torch::empty({Q, cap}, candidate_options(opts_f));
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
#ifdef DSA_CANDIDATE_U16
    TORCH_CHECK(
        emit_limit == 0,
        "DSA_CANDIDATE_U16 supports the hot-only no-seed contract");
#endif
    const int probe_stride_tok = (int)probe_stride_tok64;
    const int hist_stride = hist_stride64 > 1 ? (int)hist_stride64 : 1;
    seed_prep_kernel<<<Q, 1024, seed_smem, stream>>>(
        slog.data_ptr<float>(), slog.stride(0), head, NB, K, cap, emit_limit,
        probe_stride_tok, hist_stride,
        (float)headroom,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(),
        candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
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
    check_candidate_dtype(cand_val);
    const int seq_len = (int)q.size(0);
    const int seq_len_kv = (int)kv.size(0);
#if defined(DSA_CANDIDATE_U16) && \
    !defined(DSA_CANDIDATE_U16_BUCKET8_FRAC8)
    TORCH_CHECK(
        seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
        "DSA_CANDIDATE_U16 supports at most 1M KV positions");
#endif
    const int cand_cap = (int)cand_val.size(1);
    const int num_buckets = (int)num_buckets64;
    const int topk = (int)topk64;
    TORCH_CHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM, "only GLM DSA H=32 D=128 is supported");
    TORCH_CHECK(num_buckets >= 3 && num_buckets <= 256,
                "prepared scan requires 3 <= num_buckets <= 256");
    TORCH_CHECK(topk >= 1 && topk <= cand_cap,
                "topk must be in [1, cand_cap]");
    TORCH_CHECK(refresh_every64 >= -1,
                "refresh_every must be >= -1");
    TORCH_CHECK(cand_val.size(0) == seq_len && cand_idx.sizes() == cand_val.sizes() &&
                cand_cnt.numel() == seq_len && bcount.size(0) == seq_len && bcount.size(1) == num_buckets,
                "prepared buffer shape mismatch");
#ifdef DSA_STATIC_GATE
    TORCH_CHECK(refresh_every64 == 0,
                "DSA_STATIC_GATE requires refresh_every=0");
#endif
#ifdef DSA_SPARSE_REFRESH
    TORCH_CHECK(refresh_every64 > 0,
                "DSA_SPARSE_REFRESH requires refresh_every>0");
#endif
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

    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
    int num_kv_splits;
    if (num_kv_splits_override > 0) {
        num_kv_splits = (int)num_kv_splits_override;
    } else {
        constexpr int kWaves = 4;
        const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
        num_kv_splits = (kWaves * NUM_SMS + qb - 1) / qb;
        const int max_useful_splits = total_kv_blocks > 0 ? (total_kv_blocks + 1) / 2 : 1;
        if (num_kv_splits > max_useful_splits) num_kv_splits = max_useful_splits;
    }
    if (num_kv_splits < 1) num_kv_splits = 1;
    if (num_kv_splits > total_kv_blocks) num_kv_splits = total_kv_blocks > 0 ? total_kv_blocks : 1;
#ifdef DSA_CANDIDATE_U16
    TORCH_CHECK(
        num_kv_splits == 1,
        "DSA_CANDIDATE_U16 requires a single KV split so sparse refresh can "
        "publish an exact boundary certificate");
#endif
#ifdef DSA_SPARSE_REFRESH
    auto kernel = num_kv_splits == 1
        ? &dsa_litetopk::sm100_dsa_litetopk<
              NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128, true>
        : &dsa_litetopk::sm100_dsa_litetopk<
              NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
              NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
              SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128, false>;
#else
    auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
        NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
        NUM_SMS, SPEC_THREADS, MATH_THREADS>;
#endif
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem));
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
    auto emit_workspace = make_chunked_emit_workspace(
        seq_len, seq_len_kv, num_kv_splits, q.options());
#endif
    int grid_q = num_q_blocks;
#ifdef DSA_PERSIST
    // persistent scheduling for the merge path: one CTA per SM loops
    // q-blocks (static stride).
    if (num_kv_splits == 1 && num_q_blocks > NUM_SMS)
        grid_q = NUM_SMS;
#endif
    dim3 grid((unsigned)grid_q, (unsigned)num_kv_splits, 1);
    kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
        (uint32_t)seq_len, (uint32_t)seq_len_kv,
        (uint32_t*)cu_start.data_ptr<int>(), (uint32_t*)cu_end.data_ptr<int>(),
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
        bcount.data_ptr<int32_t>(), (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
        (uint32_t)num_kv_splits, (uint32_t)probe_group64,
        probe_group64 > 0 ? (((1ULL << 42) + (uint64_t)probe_group64 - 1) / (uint64_t)probe_group64) : 0ULL,
        (uint32_t)probe_add_max64,
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
        reinterpret_cast<uint64_t*>(
            emit_workspace.storage.data_ptr<int64_t>()),
#endif
        candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(), (uint32_t)cand_cap,
        tm_q, tm_kv, tm_ks, tm_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#if defined(DSA_CHUNKED_EMIT) && \
    !defined(DSA_CHUNKED_FLUSH_DIRECT)
    compact_chunked_emit(
        emit_workspace,
        cand_val,
        cand_idx,
        cand_cnt,
        num_kv_splits,
        stream,
        static_cast<uint32_t>(probe_group64),
        probe_group64 > 0
            ? (((1ULL << 42) +
                static_cast<uint64_t>(probe_group64) - 1) /
               static_cast<uint64_t>(probe_group64))
            : 0ULL,
        static_cast<uint32_t>(probe_add_max64));
#endif

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
        std::optional<torch::Tensor> seed_base) {
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
    TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == R,
                "cand_cnt must have R elements");
    int K = static_cast<int>(k64);
    int NB = static_cast<int>(num_buckets64);
    TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1,CAP]");
    TORCH_CHECK(NB >= 2 && NB <= 4096, "num_buckets out of range");
    TORCH_CHECK(origin.numel() == R && inv_delta.numel() == R && th_bucket.numel() == R,
                "origin/inv_delta/th_bucket must have R elements");
    auto out_val = torch::empty({R, K}, cand_val.options());
    auto out_idx = torch::empty({R, K}, cand_idx.options());
    if (probe_group64 > 0) {
        TORCH_CHECK(seed_base.has_value() && seed_base->is_cuda() &&
                    seed_base->is_contiguous() &&
                    seed_base->scalar_type() == torch::kInt &&
                    seed_base->numel() >= R,
                    "seed_base [R] int32 required with probe_group");
    }
    compact_topk_min_thr_litetopk_kernel<<<
        R, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
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

#ifdef DSA_INPLACE_BOUNDARY_SELECT
// Destructive single-use selector for the fused indexer. The ordinary
// compact_topk_min_thr_litetopk binding above deliberately remains
// non-mutating: external callers may select from the same candidates more
// than once. This entry point consumes cand_val/cand_idx by compacting its
// selected subset in place. Candidate indices must already be in final corpus
// space, as they are for the current chunked-flush scan path. Gate4 candidate
// values are already in bucket space, and the caller owns the final idx
// output, so this specialization allocates and writes no discarded values or
// temporary index tensor.
void compact_topk_min_thr_inplace_idx_out_litetopk(
        torch::Tensor cand_val,
        torch::Tensor cand_idx,
        torch::Tensor cand_cnt,
        torch::Tensor th_bucket,
        torch::Tensor boundary_meta,
        int64_t num_buckets64,
        int64_t k64,
        torch::Tensor out_idx) {
    TORCH_CHECK(cand_val.is_cuda() && cand_idx.is_cuda() &&
                cand_cnt.is_cuda() && th_bucket.is_cuda() &&
                boundary_meta.is_cuda() && out_idx.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                cand_cnt.is_contiguous() && th_bucket.is_contiguous() &&
                boundary_meta.is_contiguous() && out_idx.is_contiguous(),
                "tensors must be contiguous");
    check_candidate_dtype(cand_val);
    TORCH_CHECK(cand_idx.scalar_type() == torch::kInt &&
                cand_cnt.scalar_type() == torch::kInt &&
                out_idx.scalar_type() == torch::kInt,
                "idx/cnt/out_idx must be int32");
    TORCH_CHECK(th_bucket.scalar_type() == torch::kInt,
                "th_bucket must be int32");
    TORCH_CHECK(boundary_meta.scalar_type() == torch::kInt,
                "boundary_meta must be int32");
    TORCH_CHECK(cand_val.dim() == 2 &&
                cand_idx.sizes() == cand_val.sizes(),
                "candidate tensors must be [R,CAP]");
    const int R = static_cast<int>(cand_val.size(0));
    const int CAP = static_cast<int>(cand_val.size(1));
    TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == R,
                "cand_cnt must have R elements");
    const int K = static_cast<int>(k64);
    const int NB = static_cast<int>(num_buckets64);
    TORCH_CHECK(K >= 1 && K <= CAP, "K must be in [1,CAP]");
    TORCH_CHECK(NB >= 3 && NB <= 256,
                "in-place boundary select requires 3 <= num_buckets <= 256");
    TORCH_CHECK(th_bucket.numel() == R,
                "th_bucket must have R elements");
    TORCH_CHECK(boundary_meta.dim() == 2 &&
                boundary_meta.size(0) == R &&
                boundary_meta.size(1) == NB,
                "boundary_meta must be [R,num_buckets]");
    TORCH_CHECK(out_idx.dim() == 2 &&
                out_idx.size(0) == R &&
                out_idx.size(1) == K,
                "out_idx must be [R,K]");

    compact_topk_min_thr_inplace_idx_out_litetopk_kernel<<<
        R, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        candidate_data_ptr(cand_val),
        cand_idx.data_ptr<int32_t>(),
        cand_cnt.data_ptr<int32_t>(),
        th_bucket.data_ptr<int32_t>(),
        boundary_meta.data_ptr<int32_t>(),
        R,
        CAP,
        K,
        NB,
        out_idx.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef DSA_CANDIDATE_U16
    m.def(
        "candidate_value_u16_litetopk",
        []() { return true; },
        "Reports the packed six-byte candidate ABI");
#endif
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
          "V3 scan into buffers prepared by seed_prep_litetopk",
          pybind11::arg("q"), pybind11::arg("kv"), pybind11::arg("kv_scales"),
          pybind11::arg("weights"), pybind11::arg("cu_start"), pybind11::arg("cu_end"),
          pybind11::arg("origin"), pybind11::arg("inv_delta"), pybind11::arg("th_bucket"),
          pybind11::arg("cand_val"), pybind11::arg("cand_idx"), pybind11::arg("cand_cnt"),
          pybind11::arg("bcount"), pybind11::arg("num_buckets"), pybind11::arg("topk"),
          pybind11::arg("refresh_every"), pybind11::arg("num_kv_splits")=-1,
          pybind11::arg("probe_group")=0, pybind11::arg("probe_add_max")=0);
    m.def("mqa_logits_dsa_litetopk", &mqa_logits_dsa_litetopk,
          "DSA ReLU-MQA scoring V3 hybrid (DeepGEMM-2.5 loop + V1 KV-split) with sparse epilogue",
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
          pybind11::arg("seed_base") = pybind11::none());
#ifdef DSA_INPLACE_BOUNDARY_SELECT
    m.def("compact_topk_min_thr_inplace_idx_out_litetopk",
          &compact_topk_min_thr_inplace_idx_out_litetopk,
          "Single-use Gate4 threshold top-k directly into caller idx output",
          pybind11::arg("cand_val"),
          pybind11::arg("cand_idx"),
          pybind11::arg("cand_cnt"),
          pybind11::arg("th_bucket"),
          pybind11::arg("boundary_meta"),
          pybind11::arg("num_buckets"),
          pybind11::arg("topk"),
          pybind11::arg("out_idx"));
#endif
}

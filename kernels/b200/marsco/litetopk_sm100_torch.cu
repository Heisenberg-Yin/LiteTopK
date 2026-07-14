// Host binding for the SM100 TidalDecode IP top-k (B200 port).
//
// Ops:
//   litetopk_sm100::score_sparse(...)            -- correctness probe: bucket-gated
//       sparse scan only (validates the fp16 UMMA IP numerics on B200).
//   litetopk_sm100::fused_ip_sparse_b200(...) -- full TidalDecode selection,
//       mirroring the Hopper fused_ip_gqa_sparse_paged_group_indexed pipeline:
//         1. tail-sample dense scores  (cuBLAS bmm, x = -IP)
//         2. flash_topk_min            (seed top-k of the sample)   [reused]
//         3. seed_from_sample          (origin/inv_delta/th/buf)    [reused]
//         4. SM100 UMMA/TMEM sparse scan over [0, sample_start)     [B200 kernel]
//         5. final threshold tighten — fused into 4: the last CTA per head
//            group refreshes th in-kernel (threadfence + completion counter)
//         6. flash_topk_select_thr_mb  (boundary radix select)      [reused]
//         7. negate + un-pad to [Hq, k]
//       Steps 2/3/5/6 are the architecture-agnostic Hopper kernels compiled
//       into the same extension; only the scan is Blackwell-specific
//       (TMA warp + UMMA warp + TMEM + spare-warp threshold refresh).
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

#include "sm100_litetopk_marsco.cuh"
#include "litetopk_topk.h"

namespace {

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
static CUtensorMap make_2d_half(void* ptr, int gmem_inner, long gmem_outer,
                                int smem_inner, int smem_outer, int swizzle_mode) {
    const int elem = 2; // fp16
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
    CUresult r = get_enc()(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, ptr, gdims, gstrides,
                           sdims, estrides, CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
    return tm;
}

// Fused sample-prep (ported from dsa_litetopk's seed_prep_kernel, params-only
// variant): one block per row derives the bucket params directly from the raw
// fp16 sample scores — pass 1 min/max (o = -max, hi = -min), pass 2 smem
// histogram -> th = bucket of the K-th best — and zeroes bcount/qcount in the
// same launch. Replaces neg_ + flash_topk_min + add_offset + seed_from_sample
// + two memsets (~6 passes, ~8 launches). No candidate prefill: in strided
// mode the scan re-emits sampled rows, so prefilled seeds would double.
__global__ void litetopk_seed_params_kernel(
    const __half* __restrict__ scores, const int64_t stride,
    const int S, const int NB, const int K,
    float* __restrict__ origin, float* __restrict__ inv_delta,
    int32_t* __restrict__ th_bucket,
    int32_t* __restrict__ bcount, int32_t* __restrict__ qcount) {
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
        th_bucket[row] = th;
        origin[row] = o;
        inv_delta[row] = inv;
        qcount[row] = 0;
    }
    for (int b = tid; b < NB; b += BT)
        bcount[(size_t)row * NB + b] = 0;
}

constexpr int QN = 8;
constexpr int BM = 128;
// KV pipeline depth. 3 stages (~104KB smem) allows 2 CTAs/SM; deeper pipelines
// (up to 6 within the 227KB budget) drop to 1 CTA/SM — tradeoff measured via
// the LITETOPK_KV_STAGES env of the loader.
#ifndef LITETOPK_KV_STAGES
#define LITETOPK_KV_STAGES 6
#endif
constexpr int NUM_KV_STAGES = LITETOPK_KV_STAGES;
constexpr int NUM_SMS = 148;       // B200
// SPEC_THREADS hardcoded to the optimal 64: TMA + UMMA, refresh inlined into the
// math warps (320-thread launch => 204-reg budget vs 168; +4-11% per cell). The
// legacy 128 path (2 spare refresh warps) is removed.
constexpr int SPEC_THREADS = 64;
constexpr int MATH_THREADS = 128;  // 1 math warpgroup, BM = 128 (one kv row per lane)

static int kv_stages_for(int head_dim) {
    // Large D uses chunked stages: 3 stages fit the smem budget.
    return head_dim <= 128 ? NUM_KV_STAGES : 3;
}

static int stages_for(int head_dim, int bm) {
    if (bm > 128) return 3;                 // 64KB chunk stages
    return head_dim <= 128 ? NUM_KV_STAGES : 3;
}

static int compute_smem_bytes(int head_dim, int qn, int bm, bool out_fp32 = false) {
    const int cand_b = out_fp32 ? 4 : 2;   // queue value entries follow cand_t
    // Mirrors the kernel's constexpr layout (CHUNK_D, warp queue, coeff tables).
    int chunk_d = qn > 8 ? (head_dim > 128 ? 128 : head_dim)
                         : (head_dim > 256 ? 256 : head_dim);
    const int cap = (qn > 8 ? 32768 : 65536) / (bm * 2);
    if (chunk_d > cap) chunk_d = cap;
    const int smem_q = qn * head_dim * 2;
    const int smem_kv = bm * chunk_d * 2;
    // KV full/empty per stage + UMMA full/empty per (warpgroup, acc buffer) + q_ready
    const int num_barriers = stages_for(head_dim, bm) * 2 + (bm / 128) * 2 * 2 + 1;
    const int smem_barriers = num_barriers * 8;
    const int smem_misc = 4 * (int)sizeof(uint32_t);  // tmem slot + flags
    // Warp-local candidate queue: QN <= 8 only (kUseWarpQueue in the kernel).
    const int smem_warpq = qn <= 8
        ? (bm / 32) * qn * ((int)sizeof(int32_t) + LITETOPK_WARP_QUEUE_CAP * (cand_b + (int)sizeof(int32_t)))
        : 0;
    const int smem_coeff = qn * (2 * (int)sizeof(float) + (int)sizeof(int32_t));
    return smem_q + stages_for(head_dim, bm) * smem_kv + smem_barriers + smem_misc + smem_warpq + smem_coeff;
}

#define LITETOPK_DISPATCH_D(head_dim, ...)                                           \
    do {                                                                          \
        if (head_dim == 64)  { constexpr int DD = 64;  __VA_ARGS__; }             \
        else if (head_dim == 128) { constexpr int DD = 128; __VA_ARGS__; }        \
        else if (head_dim == 256) { constexpr int DD = 256; __VA_ARGS__; }        \
        else if (head_dim == 512) { constexpr int DD = 512; __VA_ARGS__; }        \
        else if (head_dim == 768) { constexpr int DD = 768; __VA_ARGS__; }        \
        else TORCH_CHECK(false, "unsupported head_dim ", head_dim);               \
    } while (0)

// Launch the SM100 sparse scan: score kv rows [start_row, scan_end) of the
// contiguous [hkv, kv_row_stride, D] cache against xpad [n_head_groups*QN, D].
static void launch_sm100_scan(
    const at::Tensor& xpad, const at::Tensor& kv_cont,
    int n_head_groups, int q_group_size, int logical_rows,
    int scan_end, int start_row,
    const at::Tensor& origin, const at::Tensor& inv_delta, const at::Tensor& th,
    const at::Tensor& qcount, const at::Tensor& bcount,
    const at::Tensor& buf_val, const at::Tensor& buf_idx,
    int buf_cap, int num_buckets, int topk, int refresh_every, int num_ctas_x,
    cudaStream_t stream, int32_t* cta_done = nullptr, __half* dense_out = nullptr,
    int qn = 8, int bm = 128, bool out_fp32 = false, bool store_bucket_space = true) {
    const int head_dim = xpad.size(1);
    const int hkv = kv_cont.size(0);
    const long kv_row_stride = kv_cont.size(1);

    auto tm_q = make_2d_half(xpad.data_ptr(), head_dim, xpad.size(0), head_dim, qn, 128);
    auto tm_kv = make_2d_half(kv_cont.data_ptr(), head_dim, (long)hkv * kv_row_stride,
                              head_dim, bm, 128);

    if (num_ctas_x <= 0) {
        // Persistent tile-strided CTAs. Empirically the sparse-emit epilogue
        // wants heavy over-subscription (best measured at ~8 waves of 2 CTAs/SM
        // for Hkv=8): more CTAs overlap emit stalls with other CTAs' TMA/UMMA.
        num_ctas_x = std::max(1, 8 * NUM_SMS / std::max(1, n_head_groups));
    }
    const int scan_tiles = (scan_end - start_row + bm - 1) / bm;
    if (scan_tiles > 0 && num_ctas_x > scan_tiles) num_ctas_x = scan_tiles;
    if (num_ctas_x < 1) num_ctas_x = 1;

    LITETOPK_DISPATCH_D(head_dim, {
        constexpr int STAGES = DD <= 128 ? NUM_KV_STAGES : 3;
        auto dispatch = [&](auto* kernel, auto* buf_ptr) {
            int smem = compute_smem_bytes(DD, qn, bm, out_fp32);
            cudaError_t ae = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            // Force the MAX shared-memory carveout (minimal L1). Measured: a
            // ~193KB request lands in the 196KB carveout bucket, leaving ~32KB
            // MORE L1 than the 228KB bucket — and runs ~5% SLOWER at small k.
            // Plain-load re-reads of th_bucket / gate live longer in a bigger
            // (non-coherent) L1, so CTAs see STALER (looser) thresholds and
            // over-emit; a minimal L1 evicts those lines back to L2 quickly.
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
            TORCH_CHECK(ae == cudaSuccess, "cudaFuncSetAttribute failed (smem=", smem, "): ", cudaGetErrorString(ae));
            // x = head groups (co-resident groups share L2 on the same tiles), y = tile stride
            dim3 grid(n_head_groups, num_ctas_x, 1);
            kernel<<<grid, SPEC_THREADS + (bm == 256 ? 256 : MATH_THREADS), smem, stream>>>(
                (uint32_t)scan_end, (uint32_t)kv_row_stride, (uint32_t)q_group_size,
                (uint32_t)logical_rows, (uint32_t)start_row,
                origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
                qcount.data_ptr<int>(), bcount.data_ptr<int>(), cta_done, dense_out,
                buf_ptr, buf_idx.data_ptr<int>(),
                (uint32_t)buf_cap, (uint32_t)num_buckets, (uint32_t)topk, (uint32_t)refresh_every,
                store_bucket_space, tm_q, tm_kv);
            cudaError_t le = cudaGetLastError();
            TORCH_CHECK(le == cudaSuccess, "launch failed (smem=", smem, "): ", cudaGetErrorString(le));
        };
        if (out_fp32) {
            auto* bp = buf_val.data_ptr<float>();
            if (bm == 256) {
                if (qn == 64) dispatch(&litetopk_marsco::sm100_litetopk_ip<64, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, float>, bp);
                else          dispatch(&litetopk_marsco::sm100_litetopk_ip<QN, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, float>, bp);
            } else {
                if (qn == 64) dispatch(&litetopk_marsco::sm100_litetopk_ip<64, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, float>, bp);
                else          dispatch(&litetopk_marsco::sm100_litetopk_ip<QN, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, float>, bp);
            }
        } else {
            auto* bp = reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>());
            if (bm == 256) {
                if (qn == 64) dispatch(&litetopk_marsco::sm100_litetopk_ip<64, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, __half>, bp);
                else          dispatch(&litetopk_marsco::sm100_litetopk_ip<QN, DD, 256, 3, NUM_SMS, SPEC_THREADS, 256, __half>, bp);
            } else {
                if (qn == 64) dispatch(&litetopk_marsco::sm100_litetopk_ip<64, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, __half>, bp);
                else          dispatch(&litetopk_marsco::sm100_litetopk_ip<QN, DD, BM, STAGES, NUM_SMS, SPEC_THREADS, MATH_THREADS, __half>, bp);
            }
        }
    });
}

// ---------------- correctness probe op (scan only, open gate) ----------------
std::tuple<at::Tensor, at::Tensor, at::Tensor> litetopk_sm100_score_sparse(
    const at::Tensor& q_cont, const at::Tensor& kv_cont,
    const at::Tensor& origin, const at::Tensor& inv_delta, at::Tensor& th,
    int64_t q_group_size, int64_t logical_rows, int64_t start_row,
    int64_t num_buckets, int64_t topk, int64_t refresh_every, int64_t buf_cap,
    int64_t num_ctas_x) {
    TORCH_CHECK(q_cont.is_cuda() && kv_cont.is_cuda(), "cuda tensors required");
    TORCH_CHECK(q_cont.dim() == 2 && kv_cont.dim() == 3, "q_cont [Rwork,D], kv_cont [hkv,M,D]");
    TORCH_CHECK(q_cont.scalar_type() == at::kHalf && kv_cont.scalar_type() == at::kHalf, "fp16");
    const int Rwork = q_cont.size(0);
    const int M = kv_cont.size(1);
    TORCH_CHECK(kv_cont.size(2) == q_cont.size(1), "D mismatch");
    const int n_head_groups = Rwork / QN;

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

// ---------------- ours-dense ablation baseline ----------------
// Same TMA/UMMA/TMEM scan engine, dense [Hq, M] score output (the caller runs
// torch.topk). This is the "our engine + official baseline structure" ablation:
// it isolates the sparse-selection algorithm's contribution from the load/score
// engine's contribution.
at::Tensor litetopk_sm100_dense_scores(const at::Tensor& x, const at::Tensor& kv_cont,
                                    int64_t num_ctas_x) {
    TORCH_CHECK(x.is_cuda() && kv_cont.is_cuda(), "x/kv must be CUDA");
    TORCH_CHECK(x.dim() == 2 && kv_cont.dim() == 3, "x [Hq,D], kv [Hkv,M,D]");
    TORCH_CHECK(x.scalar_type() == at::kHalf && kv_cont.scalar_type() == at::kHalf, "fp16");
    const int64_t Hq = x.size(0);
    const int64_t D = x.size(1);
    const int64_t Hkv = kv_cont.size(0);
    const int64_t M = kv_cont.size(1);
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

// ---------------- full fused pipeline (h100 parity) ----------------
std::tuple<at::Tensor, at::Tensor> fused_ip_sparse_b200(
    const at::Tensor& x, const at::Tensor& kv_cont,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t sample_size,
    int64_t refresh_every, int64_t num_ctas_x, int64_t sample_mode,
    int64_t qn_arg, int64_t bm_arg, bool out_fp32) {
    TORCH_CHECK(x.is_cuda() && kv_cont.is_cuda(), "x/kv must be CUDA");
    TORCH_CHECK(x.dim() == 2 && kv_cont.dim() == 3, "x [Hq,D], kv [Hkv,M,D]");
    TORCH_CHECK(x.is_contiguous() && kv_cont.is_contiguous(), "x/kv must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf && kv_cont.scalar_type() == at::kHalf, "fp16 required");
    const int64_t Hq = x.size(0);
    const int64_t D = x.size(1);
    const int64_t Hkv = kv_cont.size(0);
    const int64_t M = kv_cont.size(1);
    TORCH_CHECK(kv_cont.size(2) == D, "head_dim mismatch");
    TORCH_CHECK(Hkv >= 1 && Hq >= Hkv && Hq % Hkv == 0, "Hq must be a multiple of Hkv");
    const int64_t group = Hq / Hkv;
    // Two layouts: GQA (group <= QN query rows per KV head) and flat batch
    // (Hkv == 1, N = Hq concurrent queries sharing one corpus, e.g. MS MARCO
    // batch retrieval) — the kernel maps every head group to kv head 0 via
    // q_group_size = n_groups.
    const bool flat_batch = (Hkv == 1 && group > QN);
    // Flat batch can run the QN=64 single-pass kernel (all 64 queries ride the
    // UMMA N dimension; the corpus is read once instead of relayed through L2
    // by 8 row groups). Measured: ~parity at low emission (k=128) but clearly
    // behind the QN=8 warp-queue path once the emit dominates (k >= 1024), so
    // it stays opt-in (LITETOPK_FLAT_QN=64) until the 64-row emit path gets a
    // queue / pipelined reservations.
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
    TORCH_CHECK(flat_batch, "flat-batch only (GQA path removed): require Hkv==1 and N (=Hq=", Hq,
                ") > QN=", QN);
    TORCH_CHECK(Hq % qn == 0, "flat batch requires N to be a multiple of ", qn);
    TORCH_CHECK(M % 64 == 0, "M must be a multiple of 64");
    TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
    TORCH_CHECK(num_buckets >= 2 && num_buckets <= 4096, "num_buckets out of range");

    sample_size = ((sample_size + 63) / 64) * 64;
    if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
    if (sample_size > M) sample_size = M;
    // sample_mode 0: contiguous tail window [M-S, M) — seeds prefilled, the
    //   scan skips the window (attention decode: recent tokens, exchangeable).
    // sample_mode 1: strided rows across the whole corpus — robust to corpus
    //   ordering (e.g. MS MARCO region structure that makes a tail window
    //   wildly unrepresentative). Seeds only set the gate; the scan covers
    //   [0, M) fully so sampled rows re-emit naturally (no duplicates:
    //   qcount is reset after seeding).
    const bool strided_sample = (sample_mode == 1);
    TORCH_CHECK(strided_sample,
                "litetopk MARCO select: only sample_mode=1 (strided) is supported "
                "(the legacy tail-window seed-prefill path was removed).");
    const int64_t sample_start = strided_sample ? M : (M - sample_size);
    const int64_t scan_end = strided_sample ? M : sample_start;
    if (buf_cap < k) buf_cap = k;
    if (buf_cap > M) buf_cap = M;
    const int64_t candidate_cap = buf_cap;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t n_groups = flat_batch ? (Hq / qn) : Hkv;
    const int64_t q_group_size = flat_batch ? n_groups : 1;
    const int64_t rows_valid = flat_batch ? qn : group;
    const int64_t Rwork = n_groups * qn;

    // flat-batch only (GQA path removed): every query row is real, no padding.
    at::Tensor xpad = x.contiguous();

    // 1) Sample dense scores, x = -IP (cuBLAS GEMM).
    at::Tensor kv_tail;
    at::Tensor samp_row_idx;   // strided mode: corpus row of each sample column
    if (strided_sample) {
        const int64_t stride = M / sample_size;
        samp_row_idx = at::arange(sample_size, x.options().dtype(at::kLong)) * stride;
        kv_tail = kv_cont.index_select(1, samp_row_idx);                 // [Hkv, sample, D]
    } else {
        kv_tail = kv_cont.slice(1, M - sample_size, M);                  // [Hkv, sample, D]
    }
    // raw scores (no neg): the fused prep kernel handles x = -score internally.
    at::Tensor sample_scores = at::matmul(xpad, kv_tail.select(0, 0).transpose(0, 1));
    auto sample_flat = sample_scores.view({Rwork, sample_size});         // GEMM output is contiguous

    auto opt_h = x.options();
    auto opt_i = x.options().dtype(at::kInt);
    auto opt_f = x.options().dtype(at::kFloat);
    // fp32 output path: candidates, select workspaces and outputs in fp32 —
    // also removes the fp16 bucket-rounding coupling entirely (scan histogram
    // and select both operate on the exact fp32 value).
    auto opt_c = out_fp32 ? opt_f : opt_h;
    auto sample_val = at::empty({Rwork, k}, opt_h);
    auto sample_idx = at::empty({Rwork, k}, opt_i);
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

    // 2+3) strided mode: fused sample prep (params only) replaces the
    // topk_min/seed/offset/memset chain — see litetopk_seed_params_kernel.
    if (strided_sample) {
        litetopk_seed_params_kernel<<<(int)Rwork, 256, num_buckets * sizeof(int), stream>>>(
            reinterpret_cast<const __half*>(sample_flat.data_ptr<at::Half>()),
            sample_size, (int)sample_size, (int)num_buckets, (int)k,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
            bcount.data_ptr<int>(), qcount.data_ptr<int>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Bucket-width floor: the select stage re-derives buckets from fp16
    // candidate values, so the bucket width (seed span / NB) must stay at
    // least a few fp16 ULPs of the score magnitude or scan(fp32)/select(fp16)
    // classifications diverge (observed on MS MARCO where |score| ~ 70 makes
    // ULP ~ 0.06). Clamping inv_delta DOWN only widens buckets = loosens the
    // gate — recall-safe.
    {
        auto ulp = origin.abs().clamp_min(1.0).mul(1.0 / 1024.0);
        auto inv_max = ulp.mul(4.0).reciprocal();   // width >= 4 ULP
        inv_delta.copy_(at::minimum(inv_delta, inv_max));
    }

    // 4) SM100 UMMA/TMEM sparse scan over [0, sample_start).
    //    refresh_every <= 0 selects the h100 default: bcount is histogrammed
    //    (0x7fffffff sentinel) and the threshold refreshed once post-scan.
    //    (The histogram is NOT optional: the thr select direct-copies the
    //    bucket<th set and is only exact when the post-scan refresh has
    //    tightened th so that cnt(bucket<th) < k.)
    const int scan_refresh = refresh_every > 0 ? (int)refresh_every : 0x7fffffff;
    // (Strided mode: qcount/bcount zeroing is fused into
    // litetopk_seed_params_kernel — the scan histograms every candidate exactly
    // once over the full range, so a seed histogram would double-count.)
    if (scan_end > 0) {
        // 5) is fused into the scan: the last CTA of each head group performs
        // the final threshold tighten in-kernel (threadfence + completion
        // counter), replacing the separate refresh kernel launch.
        auto cta_done = at::zeros({n_groups}, opt_i);
        // store_bucket_space = strided_sample: the tail-mode seed prefill
        // (launch_hopper_seed_from_sample_fp16, below) shares buf_val with the
        // scan and always writes raw scores, so bucket-space storage is only
        // safe when strided sampling skips that legacy prefill entirely.
        launch_sm100_scan(xpad, kv_cont, (int)n_groups, (int)q_group_size, (int)rows_valid,
                          (int)scan_end, /*start_row=*/0,
                          origin, inv_delta, th, qcount, bcount, buf_val, buf_idx,
                          (int)buf_cap, (int)num_buckets, (int)k, scan_refresh,
                          (int)num_ctas_x, stream, cta_done.data_ptr<int>(),
                          /*dense_out=*/nullptr, (int)qn, eff_bm, out_fp32, strided_sample);
    }

    // 6) boundary radix select over the compact candidates. buf_val / cand_val /
    // lt_val / val_pad all hold BUCKET-SPACE coordinates bq = (x-o)*inv end to
    // end (scan stores bq directly, select never reconstructs x — see the
    // scan epilogue and flashtopk_compact_thr_kernel). bq is an affine
    // (order-preserving) transform of x, so every comparison/sort inside the
    // select machinery is unaffected; only the FINAL K outputs per row need
    // converting back, done ONCE below instead of once per candidate scanned.
    if (out_fp32) {
        launch_flash_topk_select_thr_mb_idx_fp32(
            buf_val.data_ptr<float>(),
            buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)Rwork, (int)buf_cap, (int)k, (int)candidate_cap,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            lt_val.data_ptr<float>(), lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            val_pad.data_ptr<float>(), idx_pad.data_ptr<int>(), stream);
        // debucket + negate: bq -> x = bq*delta+o -> IP = -x, fused into one
        // host-side affine over [Rwork,k] (R*k elements, once per call).
        auto delta = inv_delta.reciprocal().unsqueeze(1);
        val_pad = val_pad.mul(delta).add(origin.unsqueeze(1)).neg_();
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
            /*bucket_space=*/strided_sample);
        // 7) tail-mode (!strided_sample) shares buf_val with the legacy
        // launch_hopper_seed_from_sample_fp16 seed prefill, which always
        // writes raw scores -- the scan then also stores raw x (see
        // store_bucket_space above), so val_pad already holds x and only
        // needs the sign flip. Strided mode: val_pad holds bq, needs the
        // full debucket + negate affine (fp32 math, cast back to half).
        if (strided_sample) {
            auto delta = inv_delta.reciprocal().unsqueeze(1);
            val_pad = val_pad.to(at::kFloat).mul(delta).add(origin.unsqueeze(1)).neg_().to(at::kHalf);
        } else {
            val_pad = val_pad.neg();
        }
    }
    auto val = val_pad.view({n_groups, qn, k}).slice(1, 0, rows_valid).reshape({Hq, k}).contiguous();
    auto idx = idx_pad.view({n_groups, qn, k}).slice(1, 0, rows_valid).reshape({Hq, k}).contiguous();
    return std::make_tuple(val, idx);
}

}  // namespace

TORCH_LIBRARY(litetopk_sm100, m) {
    m.def("score_sparse(Tensor q_cont, Tensor kv_cont, Tensor origin, Tensor inv_delta, Tensor th, "
          "int q_group_size, int logical_rows, int start_row, int num_buckets, int topk, "
          "int refresh_every, int buf_cap, int num_ctas_x=0) -> (Tensor, Tensor, Tensor)");
    m.def("fused_ip_sparse_b200(Tensor x, Tensor kv_cont, int k, int num_buckets, "
          "int buf_cap, int sample_size, int refresh_every, int num_ctas_x, int sample_mode=0, "
          "int qn=0, int bm=0, bool out_fp32=False) -> (Tensor, Tensor)");
    m.def("dense_scores(Tensor x, Tensor kv_cont, int num_ctas_x=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(litetopk_sm100, CUDA, m) {
    m.impl("score_sparse", TORCH_FN(litetopk_sm100_score_sparse));
    m.impl("fused_ip_sparse_b200", TORCH_FN(fused_ip_sparse_b200));
    m.impl("dense_scores", TORCH_FN(litetopk_sm100_dense_scores));
}

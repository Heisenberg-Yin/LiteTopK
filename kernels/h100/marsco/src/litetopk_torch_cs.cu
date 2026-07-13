// TORCH_LIBRARY 包装：把 litetopk_select.cu 的 CUDA launch 函数注册为 torch.ops.litetopk.*。
//
// 与 litetopk_select.cu 一起用 nvcc + gcc-12 编成单个 torch 扩展 .so（gcc>=9 满足 torch C++ 头要求）。
// 相比旧的 ctypes 路线：op 直接吃 torch tensor，stream 自动取 torch 当前 CUDA stream，
// 无需手动传 data_ptr / stream 句柄，也由 dispatcher 处理设备/dtype 派发。

#include "litetopk_select.h"

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

// 每行取 K 个最小值及列索引（与 torch.topk(largest=False, sorted=False) 等价）。
//   scores: [B, N] contiguous，fp32 或 fp16。返回 (val[B,K] 同 dtype, idx[B,K] int32)。
std::tuple<at::Tensor, at::Tensor> topk_min(const at::Tensor& scores, int64_t k) {
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(scores.dim() == 2, "scores must be 2-D [B, N]");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    const int64_t B = scores.size(0);
    const int64_t N = scores.size(1);
    TORCH_CHECK(k >= 1 && k <= N, "require 1 <= k <= N");

    auto val = at::empty({B, k}, scores.options());
    auto idx = at::empty({B, k}, scores.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (scores.scalar_type() == at::kFloat) {
        launch_flash_topk_min_fp32(
            scores.data_ptr<float>(), (int)B, (int)N, (int)k,
            val.data_ptr<float>(), idx.data_ptr<int>(), stream);
    } else if (scores.scalar_type() == at::kHalf) {
        launch_flash_topk_min_fp16(
            reinterpret_cast<const __half*>(scores.data_ptr<at::Half>()),
            (int)B, (int)N, (int)k,
            reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
            idx.data_ptr<int>(), stream);
    } else if (scores.scalar_type() == at::kBFloat16) {
        launch_flash_topk_min_bf16(
            reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr<at::BFloat16>()),
            (int)B, (int)N, (int)k,
            reinterpret_cast<__nv_bfloat16*>(val.data_ptr<at::BFloat16>()),
            idx.data_ptr<int>(), stream);
    } else {
        TORCH_CHECK(false, "scores dtype must be float32, float16 or bfloat16");
    }
    return std::make_tuple(val, idx);
}

// 每行取 K 个最大值及列索引（与 tf.math.top_k 等价，输出未排序）。
std::tuple<at::Tensor, at::Tensor> topk_max(const at::Tensor& scores, int64_t k) {
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(scores.dim() == 2, "scores must be 2-D [B, N]");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    const int64_t B = scores.size(0);
    const int64_t N = scores.size(1);
    TORCH_CHECK(k >= 1 && k <= N, "require 1 <= k <= N");

    auto val = at::empty({B, k}, scores.options());
    auto idx = at::empty({B, k}, scores.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (scores.scalar_type() == at::kFloat) {
        launch_flash_topk_max_fp32(
            scores.data_ptr<float>(), (int)B, (int)N, (int)k,
            val.data_ptr<float>(), idx.data_ptr<int>(), stream);
    } else if (scores.scalar_type() == at::kHalf) {
        launch_flash_topk_max_fp16(
            reinterpret_cast<const __half*>(scores.data_ptr<at::Half>()),
            (int)B, (int)N, (int)k,
            reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
            idx.data_ptr<int>(), stream);
    } else if (scores.scalar_type() == at::kBFloat16) {
        launch_flash_topk_max_bf16(
            reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr<at::BFloat16>()),
            (int)B, (int)N, (int)k,
            reinterpret_cast<__nv_bfloat16*>(val.data_ptr<at::BFloat16>()),
            idx.data_ptr<int>(), stream);
    } else {
        TORCH_CHECK(false, "scores dtype must be float32, float16 or bfloat16");
    }
    return std::make_tuple(val, idx);
}

// 阈值复用选择：配合融合 _flat_kernel，跳过 sample/hist/threshold，只跑 gather+boundary。
//   buf: [R, BUF] contiguous（尾部预填 +inf）；origin/inv_delta: [R] float32；th: [R] int32。
//   返回 (val[R,k] 同 buf dtype, idx[R,k] int32，idx 为 buffer 内列位置 [0,BUF))。
std::tuple<at::Tensor, at::Tensor> topk_select_thr(
        const at::Tensor& buf, const at::Tensor& origin,
        const at::Tensor& inv_delta, const at::Tensor& th, const at::Tensor& qcount,
        int64_t k, int64_t num_buckets) {
    TORCH_CHECK(buf.is_cuda(), "buf must be a CUDA tensor");
    TORCH_CHECK(buf.dim() == 2, "buf must be 2-D [R, BUF]");
    TORCH_CHECK(buf.is_contiguous(), "buf must be contiguous");
    TORCH_CHECK(origin.is_contiguous() && inv_delta.is_contiguous() &&
                th.is_contiguous() && qcount.is_contiguous(),
                "origin/inv_delta/th/qcount must be contiguous");
    TORCH_CHECK(origin.scalar_type() == inv_delta.scalar_type(),
                "origin/inv_delta must share dtype");
    TORCH_CHECK(origin.scalar_type() == at::kFloat || origin.scalar_type() == buf.scalar_type(),
                "origin/inv_delta dtype must be float32 or match buf dtype");
    TORCH_CHECK(th.scalar_type() == at::kInt, "th must be int32");
    TORCH_CHECK(qcount.scalar_type() == at::kInt, "qcount must be int32");
    const int64_t R = buf.size(0);
    const int64_t BUF = buf.size(1);
    TORCH_CHECK(k >= 1 && k <= BUF, "require 1 <= k <= BUF");
    TORCH_CHECK(origin.numel() == R && inv_delta.numel() == R &&
                th.numel() == R && qcount.numel() == R,
                "origin/inv_delta/th/qcount must have R elements");

    auto val = at::empty({R, k}, buf.options());
    auto idx = at::empty({R, k}, buf.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (buf.scalar_type() == at::kFloat) {
        TORCH_CHECK(origin.scalar_type() == at::kFloat,
                    "fp32 buf requires fp32 origin/inv_delta");
        launch_flash_topk_select_thr_fp32(
            buf.data_ptr<float>(), (int)R, (int)BUF, (int)k,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
            qcount.data_ptr<int>(), (int)num_buckets,
            val.data_ptr<float>(), idx.data_ptr<int>(), stream);
    } else if (buf.scalar_type() == at::kHalf) {
        bool coords_fp16 = origin.scalar_type() == at::kHalf;
        launch_flash_topk_select_thr_fp16(
            reinterpret_cast<const __half*>(buf.data_ptr<at::Half>()),
            (int)R, (int)BUF, (int)k,
            coords_fp16 ? (const void*)origin.data_ptr<at::Half>()
                        : (const void*)origin.data_ptr<float>(),
            coords_fp16 ? (const void*)inv_delta.data_ptr<at::Half>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
            idx.data_ptr<int>(), coords_fp16, stream);
    } else if (buf.scalar_type() == at::kBFloat16) {
        bool coords_bf16 = origin.scalar_type() == at::kBFloat16;
        launch_flash_topk_select_thr_bf16(
            reinterpret_cast<const __nv_bfloat16*>(buf.data_ptr<at::BFloat16>()),
            (int)R, (int)BUF, (int)k,
            coords_bf16 ? (const void*)origin.data_ptr<at::BFloat16>()
                        : (const void*)origin.data_ptr<float>(),
            coords_bf16 ? (const void*)inv_delta.data_ptr<at::BFloat16>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            reinterpret_cast<__nv_bfloat16*>(val.data_ptr<at::BFloat16>()),
            idx.data_ptr<int>(), coords_bf16, stream);
    } else {
        TORCH_CHECK(false, "buf dtype must be float32, float16 or bfloat16");
    }
    return std::make_tuple(val, idx);
}

// 阈值复用选择 + idx gather 融合：
//   buf_idx: [R, BUF] 候选 global idx（前 k 槽为 sample 段、其余为扫描段，均已是全局行号）。
//   返回 idx 已经是最终 corpus id，不再需要 Python 侧 torch.gather(buf_idx, topi_local)。
std::tuple<at::Tensor, at::Tensor> topk_select_thr_idx(
        const at::Tensor& buf, const at::Tensor& buf_idx,
        const at::Tensor& origin, const at::Tensor& inv_delta,
        const at::Tensor& th, const at::Tensor& qcount,
        int64_t k, int64_t num_buckets) {
    TORCH_CHECK(buf.is_cuda() && buf_idx.is_cuda(),
                "buf/buf_idx must be CUDA tensors");
    TORCH_CHECK(buf.dim() == 2 && buf_idx.dim() == 2,
                "buf/buf_idx must be 2-D");
    TORCH_CHECK(buf.is_contiguous() && buf_idx.is_contiguous(),
                "buf/buf_idx must be contiguous");
    TORCH_CHECK(origin.is_contiguous() && inv_delta.is_contiguous() &&
                th.is_contiguous() && qcount.is_contiguous(),
                "origin/inv_delta/th/qcount must be contiguous");
    TORCH_CHECK(origin.scalar_type() == inv_delta.scalar_type(),
                "origin/inv_delta must share dtype");
    TORCH_CHECK(origin.scalar_type() == at::kFloat || origin.scalar_type() == buf.scalar_type(),
                "origin/inv_delta dtype must be float32 or match buf dtype");
    TORCH_CHECK(buf_idx.scalar_type() == at::kInt,
                "buf_idx must be int32");
    TORCH_CHECK(th.scalar_type() == at::kInt, "th must be int32");
    TORCH_CHECK(qcount.scalar_type() == at::kInt, "qcount must be int32");
    const int64_t R = buf.size(0);
    const int64_t BUF = buf.size(1);
    TORCH_CHECK(buf_idx.size(0) == R && buf_idx.size(1) == BUF,
                "buf_idx shape must match buf");
    TORCH_CHECK(k >= 1 && k <= BUF, "require 1 <= k <= BUF");
    TORCH_CHECK(origin.numel() == R && inv_delta.numel() == R &&
                th.numel() == R && qcount.numel() == R,
                "origin/inv_delta/th/qcount must have R elements");

    auto val = at::empty({R, k}, buf.options());
    auto idx = at::empty({R, k}, buf.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (buf.scalar_type() == at::kFloat) {
        TORCH_CHECK(origin.scalar_type() == at::kFloat,
                    "fp32 buf requires fp32 origin/inv_delta");
        launch_flash_topk_select_thr_idx_fp32(
            buf.data_ptr<float>(), buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)R, (int)BUF, (int)k,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
            qcount.data_ptr<int>(), (int)num_buckets,
            val.data_ptr<float>(), idx.data_ptr<int>(), stream);
    } else if (buf.scalar_type() == at::kHalf) {
        bool coords_fp16 = origin.scalar_type() == at::kHalf;
        launch_flash_topk_select_thr_idx_fp16(
            reinterpret_cast<const __half*>(buf.data_ptr<at::Half>()),
            buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)R, (int)BUF, (int)k,
            coords_fp16 ? (const void*)origin.data_ptr<at::Half>()
                        : (const void*)origin.data_ptr<float>(),
            coords_fp16 ? (const void*)inv_delta.data_ptr<at::Half>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
            idx.data_ptr<int>(), coords_fp16, stream);
    } else if (buf.scalar_type() == at::kBFloat16) {
        bool coords_bf16 = origin.scalar_type() == at::kBFloat16;
        launch_flash_topk_select_thr_idx_bf16(
            reinterpret_cast<const __nv_bfloat16*>(buf.data_ptr<at::BFloat16>()),
            buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)R, (int)BUF, (int)k,
            coords_bf16 ? (const void*)origin.data_ptr<at::BFloat16>()
                        : (const void*)origin.data_ptr<float>(),
            coords_bf16 ? (const void*)inv_delta.data_ptr<at::BFloat16>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            reinterpret_cast<__nv_bfloat16*>(val.data_ptr<at::BFloat16>()),
            idx.data_ptr<int>(), coords_bf16, stream);
    } else {
        TORCH_CHECK(false, "buf dtype must be float32, float16 or bfloat16");
    }
    return std::make_tuple(val, idx);
}

// multi-block 版阈值复用选择：把整 BUF 扫描拆到多 block 并行分桶（b<th→lt、b==th→eq），再单
//   block finalize（lt 全 copy + eq 上小 radix）。大 BUF（阈值解耦放大 buffer）下显著缩短单
//   block 串行扫描。eq(cand) 与 lt 缓冲（各 [R,CAP] val/idx + [R] cnt）内部分配。CAP 由调用方给定（>=k）。
std::tuple<at::Tensor, at::Tensor> topk_select_thr_mb_idx(
        const at::Tensor& buf, const at::Tensor& buf_idx,
        const at::Tensor& origin, const at::Tensor& inv_delta,
        const at::Tensor& th, const at::Tensor& qcount,
        int64_t k, int64_t num_buckets, int64_t cap) {
    TORCH_CHECK(buf.is_cuda() && buf_idx.is_cuda(),
                "buf/buf_idx must be CUDA tensors");
    TORCH_CHECK(buf.dim() == 2 && buf_idx.dim() == 2,
                "buf/buf_idx must be 2-D");
    TORCH_CHECK(buf.is_contiguous() && buf_idx.is_contiguous(),
                "buf/buf_idx must be contiguous");
    TORCH_CHECK(origin.is_contiguous() && inv_delta.is_contiguous() &&
                th.is_contiguous() && qcount.is_contiguous(),
                "origin/inv_delta/th/qcount must be contiguous");
    TORCH_CHECK(origin.scalar_type() == inv_delta.scalar_type(),
                "origin/inv_delta must share dtype");
    TORCH_CHECK(origin.scalar_type() == at::kFloat || origin.scalar_type() == buf.scalar_type(),
                "origin/inv_delta dtype must be float32 or match buf dtype");
    TORCH_CHECK(buf_idx.scalar_type() == at::kInt, "buf_idx must be int32");
    TORCH_CHECK(th.scalar_type() == at::kInt, "th must be int32");
    TORCH_CHECK(qcount.scalar_type() == at::kInt, "qcount must be int32");
    const int64_t R = buf.size(0);
    const int64_t BUF = buf.size(1);
    TORCH_CHECK(buf_idx.size(0) == R && buf_idx.size(1) == BUF,
                "buf_idx shape must match buf");
    TORCH_CHECK(k >= 1 && k <= BUF, "require 1 <= k <= BUF");
    TORCH_CHECK(cap >= k, "require cap >= k");
    if (cap > BUF) cap = BUF;
    TORCH_CHECK(origin.numel() == R && inv_delta.numel() == R &&
                th.numel() == R && qcount.numel() == R,
                "origin/inv_delta/th/qcount must have R elements");

    auto val = at::empty({R, k}, buf.options());
    auto idx = at::empty({R, k}, buf.options().dtype(at::kInt));
    auto cand_val = at::empty({R, cap}, buf.options());
    auto cand_idx = at::empty({R, cap}, buf.options().dtype(at::kInt));
    auto cand_cnt = at::empty({R}, buf.options().dtype(at::kInt));
    // lt 缓冲（b<th 候选）与 cand（eq 缓冲，b==th）对称分配，均走 PyTorch caching
    //   allocator 复用、无需初始化（计数器在 kernel host wrapper 内清零）。
    auto lt_val = at::empty({R, cap}, buf.options());
    auto lt_idx = at::empty({R, cap}, buf.options().dtype(at::kInt));
    auto lt_cnt = at::empty({R}, buf.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (buf.scalar_type() == at::kFloat) {
        TORCH_CHECK(origin.scalar_type() == at::kFloat,
                    "fp32 buf requires fp32 origin/inv_delta");
        launch_flash_topk_select_thr_mb_idx_fp32(
            buf.data_ptr<float>(), buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)R, (int)BUF, (int)k, (int)cap,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
            qcount.data_ptr<int>(), (int)num_buckets,
            cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            lt_val.data_ptr<float>(), lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            val.data_ptr<float>(), idx.data_ptr<int>(), stream);
    } else if (buf.scalar_type() == at::kHalf) {
        bool coords_fp16 = origin.scalar_type() == at::kHalf;
        launch_flash_topk_select_thr_mb_idx_fp16(
            reinterpret_cast<const __half*>(buf.data_ptr<at::Half>()),
            buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)R, (int)BUF, (int)k, (int)cap,
            coords_fp16 ? (const void*)origin.data_ptr<at::Half>()
                        : (const void*)origin.data_ptr<float>(),
            coords_fp16 ? (const void*)inv_delta.data_ptr<at::Half>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
            cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
            lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
            idx.data_ptr<int>(), coords_fp16, stream);
    } else if (buf.scalar_type() == at::kBFloat16) {
        bool coords_bf16 = origin.scalar_type() == at::kBFloat16;
        launch_flash_topk_select_thr_mb_idx_bf16(
            reinterpret_cast<const __nv_bfloat16*>(buf.data_ptr<at::BFloat16>()),
            buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
            (int)R, (int)BUF, (int)k, (int)cap,
            coords_bf16 ? (const void*)origin.data_ptr<at::BFloat16>()
                        : (const void*)origin.data_ptr<float>(),
            coords_bf16 ? (const void*)inv_delta.data_ptr<at::BFloat16>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
            reinterpret_cast<__nv_bfloat16*>(cand_val.data_ptr<at::BFloat16>()),
            cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(lt_val.data_ptr<at::BFloat16>()),
            lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(val.data_ptr<at::BFloat16>()),
            idx.data_ptr<int>(), coords_bf16, stream);
    } else {
        TORCH_CHECK(false, "buf dtype must be float32, float16 or bfloat16");
    }
    return std::make_tuple(val, idx);
}

// DENSE 变体：buffer 为 [R, M] 稠密布局，buf[row, col]=score（col 即 corpus id），未通过
//   gate 的位置为 sentinel(NaN)。无 buf_idx（id=col=i）、无 qcount（扫满 M）。复用 mb 的
//   compact + boundary_radix（buf_idx/sample_idx/qcount 传 nullptr，compact kernel 内
//   id=i、limit=M、isfinite 跳过 NaN）。
std::tuple<at::Tensor, at::Tensor> topk_select_thr_mb_dense(
        const at::Tensor& buf,
        const at::Tensor& origin, const at::Tensor& inv_delta,
        const at::Tensor& th,
        int64_t k, int64_t num_buckets, int64_t cap) {
    TORCH_CHECK(buf.is_cuda(), "buf must be CUDA tensor");
    TORCH_CHECK(buf.dim() == 2 && buf.is_contiguous(), "buf must be 2-D contiguous");
    TORCH_CHECK(origin.is_contiguous() && inv_delta.is_contiguous() && th.is_contiguous(),
                "origin/inv_delta/th must be contiguous");
    TORCH_CHECK(origin.scalar_type() == inv_delta.scalar_type(),
                "origin/inv_delta must share dtype");
    TORCH_CHECK(origin.scalar_type() == at::kFloat || origin.scalar_type() == buf.scalar_type(),
                "origin/inv_delta dtype must be float32 or match buf dtype");
    TORCH_CHECK(th.scalar_type() == at::kInt, "th must be int32");
    const int64_t R = buf.size(0);
    const int64_t M = buf.size(1);
    TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
    TORCH_CHECK(cap >= k, "require cap >= k");
    if (cap > M) cap = M;
    TORCH_CHECK(origin.numel() == R && inv_delta.numel() == R && th.numel() == R,
                "origin/inv_delta/th must have R elements");

    auto val = at::empty({R, k}, buf.options());
    auto idx = at::empty({R, k}, buf.options().dtype(at::kInt));
    auto cand_val = at::empty({R, cap}, buf.options());
    auto cand_idx = at::empty({R, cap}, buf.options().dtype(at::kInt));
    auto cand_cnt = at::empty({R}, buf.options().dtype(at::kInt));
    auto lt_val = at::empty({R, cap}, buf.options());
    auto lt_idx = at::empty({R, cap}, buf.options().dtype(at::kInt));
    auto lt_cnt = at::empty({R}, buf.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (buf.scalar_type() == at::kFloat) {
        TORCH_CHECK(origin.scalar_type() == at::kFloat,
                    "fp32 buf requires fp32 origin/inv_delta");
        launch_flash_topk_select_thr_mb_idx_fp32(
            buf.data_ptr<float>(), /*buf_idx=*/nullptr, /*sample_idx=*/nullptr,
            (int)R, (int)M, (int)k, (int)cap,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
            /*qcount=*/nullptr, (int)num_buckets,
            cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            lt_val.data_ptr<float>(), lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            val.data_ptr<float>(), idx.data_ptr<int>(), stream);
    } else if (buf.scalar_type() == at::kHalf) {
        bool coords_fp16 = origin.scalar_type() == at::kHalf;
        launch_flash_topk_select_thr_mb_idx_fp16(
            reinterpret_cast<const __half*>(buf.data_ptr<at::Half>()),
            /*buf_idx=*/nullptr, /*sample_idx=*/nullptr,
            (int)R, (int)M, (int)k, (int)cap,
            coords_fp16 ? (const void*)origin.data_ptr<at::Half>()
                        : (const void*)origin.data_ptr<float>(),
            coords_fp16 ? (const void*)inv_delta.data_ptr<at::Half>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), /*qcount=*/nullptr, (int)num_buckets,
            reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
            cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
            lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
            idx.data_ptr<int>(), coords_fp16, stream);
    } else if (buf.scalar_type() == at::kBFloat16) {
        bool coords_bf16 = origin.scalar_type() == at::kBFloat16;
        launch_flash_topk_select_thr_mb_idx_bf16(
            reinterpret_cast<const __nv_bfloat16*>(buf.data_ptr<at::BFloat16>()),
            /*buf_idx=*/nullptr, /*sample_idx=*/nullptr,
            (int)R, (int)M, (int)k, (int)cap,
            coords_bf16 ? (const void*)origin.data_ptr<at::BFloat16>()
                        : (const void*)origin.data_ptr<float>(),
            coords_bf16 ? (const void*)inv_delta.data_ptr<at::BFloat16>()
                        : (const void*)inv_delta.data_ptr<float>(),
            th.data_ptr<int>(), /*qcount=*/nullptr, (int)num_buckets,
            reinterpret_cast<__nv_bfloat16*>(cand_val.data_ptr<at::BFloat16>()),
            cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(lt_val.data_ptr<at::BFloat16>()),
            lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(val.data_ptr<at::BFloat16>()),
            idx.data_ptr<int>(), coords_bf16, stream);
    } else {
        TORCH_CHECK(false, "buf dtype must be float32, float16 or bfloat16");
    }
    return std::make_tuple(val, idx);
}


// DENSE 直方图+阈值：对 [R,M] dense buffer 用 SMEM 直方图算每行阈值桶 th[R]（int32）。
//   替代 Triton _flat 的 global 原子直方图。origin/inv_delta [R]，float32 或与 buf 同 dtype。
at::Tensor dense_compute_threshold(
        const at::Tensor& buf,
        const at::Tensor& origin, const at::Tensor& inv_delta,
        int64_t k, int64_t num_buckets) {
    TORCH_CHECK(buf.is_cuda() && buf.dim() == 2 && buf.is_contiguous(),
                "buf must be 2-D contiguous CUDA tensor");
    TORCH_CHECK(origin.is_contiguous() && inv_delta.is_contiguous(),
                "origin/inv_delta must be contiguous");
    TORCH_CHECK(origin.scalar_type() == inv_delta.scalar_type(),
                "origin/inv_delta must share dtype");
    const int64_t R = buf.size(0);
    const int64_t M = buf.size(1);
    TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
    TORCH_CHECK(origin.numel() == R && inv_delta.numel() == R,
                "origin/inv_delta must have R elements");
    auto th = at::empty({R}, buf.options().dtype(at::kInt));
    auto bcount = at::empty({R, num_buckets}, buf.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (buf.scalar_type() == at::kFloat) {
        TORCH_CHECK(origin.scalar_type() == at::kFloat, "fp32 buf requires fp32 coords");
        launch_flash_topk_dense_threshold_fp32(
            buf.data_ptr<float>(), (int)R, (int)M, (int)k,
            origin.data_ptr<float>(), inv_delta.data_ptr<float>(), (int)num_buckets,
            bcount.data_ptr<int>(), th.data_ptr<int>(), stream);
    } else if (buf.scalar_type() == at::kHalf) {
        bool cf = origin.scalar_type() == at::kHalf;
        launch_flash_topk_dense_threshold_fp16(
            reinterpret_cast<const __half*>(buf.data_ptr<at::Half>()), (int)R, (int)M, (int)k,
            cf ? (const void*)origin.data_ptr<at::Half>() : (const void*)origin.data_ptr<float>(),
            cf ? (const void*)inv_delta.data_ptr<at::Half>() : (const void*)inv_delta.data_ptr<float>(),
            (int)num_buckets, bcount.data_ptr<int>(), th.data_ptr<int>(), cf, stream);
    } else if (buf.scalar_type() == at::kBFloat16) {
        bool cf = origin.scalar_type() == at::kBFloat16;
        launch_flash_topk_dense_threshold_bf16(
            reinterpret_cast<const __nv_bfloat16*>(buf.data_ptr<at::BFloat16>()), (int)R, (int)M, (int)k,
            cf ? (const void*)origin.data_ptr<at::BFloat16>() : (const void*)origin.data_ptr<float>(),
            cf ? (const void*)inv_delta.data_ptr<at::BFloat16>() : (const void*)inv_delta.data_ptr<float>(),
            (int)num_buckets, bcount.data_ptr<int>(), th.data_ptr<int>(), cf, stream);
    } else {
        TORCH_CHECK(false, "buf dtype must be float32, float16 or bfloat16");
    }
    return th;
}


// packed 分段 K-min select（fp16 专用）：直接吃 4B packed buffer + segcnt，
//   kernel 内 radix K-min + 解码 score/id，返回 (val[R,K] half, idx[R,K] int32 corpus id)。
std::tuple<at::Tensor, at::Tensor> topk_select_packed(
        const at::Tensor& buf_pack, const at::Tensor& segcnt,
        int64_t cap, int64_t blk, int64_t m, int64_t k) {
    TORCH_CHECK(buf_pack.is_cuda() && segcnt.is_cuda(), "buf_pack/segcnt must be CUDA tensors");
    TORCH_CHECK(buf_pack.dim() == 2 && segcnt.dim() == 2, "buf_pack/segcnt must be 2-D");
    TORCH_CHECK(buf_pack.is_contiguous() && segcnt.is_contiguous(),
                "buf_pack/segcnt must be contiguous");
    TORCH_CHECK(buf_pack.scalar_type() == at::kInt, "buf_pack must be int32 (packed)");
    TORCH_CHECK(segcnt.scalar_type() == at::kInt, "segcnt must be int32");
    const int64_t R = buf_pack.size(0);
    const int64_t NSEG = segcnt.size(1);
    TORCH_CHECK(segcnt.size(0) == R, "segcnt rows must match buf_pack");
    TORCH_CHECK(buf_pack.size(1) == NSEG * cap, "buf_pack cols must equal NSEG*CAP");
    TORCH_CHECK(NSEG <= 1024, "NSEG must be <= 1024 (shared segcnt cache)");
    TORCH_CHECK(k >= 1, "k must be >= 1");

    auto val = at::empty({R, k}, buf_pack.options().dtype(at::kHalf));
    auto idx = at::empty({R, k}, buf_pack.options().dtype(at::kInt));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_flash_topk_select_packed_fp16(
        buf_pack.data_ptr<int>(), (int)R, (int)cap, (int)NSEG, (int)blk, (int)m, (int)k,
        segcnt.data_ptr<int>(),
        reinterpret_cast<__half*>(val.data_ptr<at::Half>()), idx.data_ptr<int>(), stream);
    return std::make_tuple(val, idx);
}

// 就地把 [B, K] 的 (value, index) 按 value 降序排序（用于 sorted=True）。
void sort_desc_(at::Tensor& val, at::Tensor& idx) {
    TORCH_CHECK(val.is_cuda() && idx.is_cuda(), "val/idx must be CUDA tensors");
    TORCH_CHECK(val.dim() == 2 && idx.dim() == 2, "val/idx must be 2-D [B, K]");
    TORCH_CHECK(val.is_contiguous() && idx.is_contiguous(), "val/idx must be contiguous");
    TORCH_CHECK(idx.scalar_type() == at::kInt, "idx must be int32");
    const int64_t B = val.size(0);
    const int64_t K = val.size(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (val.scalar_type() == at::kFloat) {
        launch_topk_sort_desc_fp32(val.data_ptr<float>(), idx.data_ptr<int>(),
                                   (int)B, (int)K, stream);
    } else if (val.scalar_type() == at::kHalf) {
        launch_topk_sort_desc_fp16(reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
                                   idx.data_ptr<int>(), (int)B, (int)K, stream);
    } else if (val.scalar_type() == at::kBFloat16) {
        launch_topk_sort_desc_bf16(reinterpret_cast<__nv_bfloat16*>(val.data_ptr<at::BFloat16>()),
                                   idx.data_ptr<int>(), (int)B, (int)K, stream);
    } else {
        TORCH_CHECK(false, "val dtype must be float32, float16 or bfloat16");
    }
}

}  // namespace

TORCH_LIBRARY(litetopk_cs, m) {
    m.def("topk_min(Tensor scores, int k) -> (Tensor, Tensor)");
    m.def("topk_max(Tensor scores, int k) -> (Tensor, Tensor)");
    m.def("topk_select_thr(Tensor buf, Tensor origin, Tensor inv_delta, Tensor th, Tensor qcount, int k, int num_buckets) -> (Tensor, Tensor)");
    m.def("topk_select_thr_idx(Tensor buf, Tensor buf_idx, Tensor origin, Tensor inv_delta, Tensor th, Tensor qcount, int k, int num_buckets) -> (Tensor, Tensor)");
    m.def("topk_select_thr_mb_idx(Tensor buf, Tensor buf_idx, Tensor origin, Tensor inv_delta, Tensor th, Tensor qcount, int k, int num_buckets, int cap) -> (Tensor, Tensor)");
    m.def("topk_select_thr_mb_dense(Tensor buf, Tensor origin, Tensor inv_delta, Tensor th, int k, int num_buckets, int cap) -> (Tensor, Tensor)");
    m.def("dense_compute_threshold(Tensor buf, Tensor origin, Tensor inv_delta, int k, int num_buckets) -> Tensor");
    m.def("topk_select_packed(Tensor buf_pack, Tensor segcnt, int cap, int blk, int m, int k) -> (Tensor, Tensor)");
    m.def("sort_desc_(Tensor(a!) val, Tensor(b!) idx) -> ()");
}

TORCH_LIBRARY_IMPL(litetopk_cs, CUDA, m) {
    m.impl("topk_min", TORCH_FN(topk_min));
    m.impl("topk_max", TORCH_FN(topk_max));
    m.impl("topk_select_thr", TORCH_FN(topk_select_thr));
    m.impl("topk_select_thr_idx", TORCH_FN(topk_select_thr_idx));
    m.impl("topk_select_thr_mb_idx", TORCH_FN(topk_select_thr_mb_idx));
    m.impl("topk_select_thr_mb_dense", TORCH_FN(topk_select_thr_mb_dense));
    m.impl("dense_compute_threshold", TORCH_FN(dense_compute_threshold));
    m.impl("topk_select_packed", TORCH_FN(topk_select_packed));
    m.impl("sort_desc_", TORCH_FN(sort_desc_));
}

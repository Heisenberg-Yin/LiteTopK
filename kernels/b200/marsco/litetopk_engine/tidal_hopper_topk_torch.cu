// TORCH_LIBRARY 包装：把 hopper_src 的 D512 CUDA launchers 注册为 torch ops。

#include "hopper_topk.h"

#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cstdlib>

namespace {

#ifndef HOPPER_SPARSE_BCOUNT_SHARDS
#define HOPPER_SPARSE_BCOUNT_SHARDS 1
#endif
constexpr int BCOUNT_SHARDS = HOPPER_SPARSE_BCOUNT_SHARDS;
#ifndef HOPPER_SPARSE_REFRESH_FROM_BUF
#define HOPPER_SPARSE_REFRESH_FROM_BUF 0
#endif
constexpr int REFRESH_FROM_BUF = HOPPER_SPARSE_REFRESH_FROM_BUF;
#ifndef HOPPER_SPARSE_BUCKET_WRITE
#define HOPPER_SPARSE_BUCKET_WRITE 0
#endif
constexpr int BUCKET_WRITE = HOPPER_SPARSE_BUCKET_WRITE;
#ifndef HOPPER_SPARSE_DENSE_WRITE
#define HOPPER_SPARSE_DENSE_WRITE 0
#endif
constexpr int DENSE_WRITE = HOPPER_SPARSE_DENSE_WRITE;

void prepare_dpad_inputs(
    const at::Tensor& x, const at::Tensor& base,
    at::Tensor& x_work, at::Tensor& base_work, int64_t& dpad,
    const char* kernel_name, int64_t max_dim = 512) {
  TORCH_CHECK(x.size(1) == base.size(1),
              kernel_name, " requires x/base to have the same feature dimension");
  const int64_t d = x.size(1);
  TORCH_CHECK(d >= 1 && d <= max_dim, kernel_name, " supports feature dimension in range");
  dpad = ((d + 63) / 64) * 64;
  if (d == dpad) {
    x_work = x;
    base_work = base;
    return;
  }
  x_work = at::zeros({x.size(0), dpad}, x.options());
  base_work = at::zeros({base.size(0), dpad}, base.options());
  x_work.slice(1, 0, d).copy_(x);
  base_work.slice(1, 0, d).copy_(base);
}

void prepare_gqa_dpad_inputs(
    const at::Tensor& x, const at::Tensor& base3d,
    at::Tensor& x_work, at::Tensor& base_work, int64_t& dpad,
    const char* kernel_name, int64_t max_dim = 768) {
  TORCH_CHECK(x.size(1) == base3d.size(2),
              kernel_name, " requires x/base3d to have the same feature dimension");
  const int64_t d = x.size(1);
  TORCH_CHECK(d >= 1 && d <= max_dim, kernel_name, " supports feature dimension in range");
  dpad = ((d + 63) / 64) * 64;
  if (d == dpad) {
    x_work = x;
    base_work = base3d;
    return;
  }
  x_work = at::zeros({x.size(0), dpad}, x.options());
  base_work = at::zeros({base3d.size(0), base3d.size(1), dpad}, base3d.options());
  x_work.slice(1, 0, d).copy_(x);
  base_work.slice(2, 0, d).copy_(base3d);
}

// End-to-end C++/CUDA IP top-k path:
//   Hopper WGMMA score-to-dense(-IP) -> C++ dense threshold -> C++ multi-CTA select -> negate values.
// This is a correctness/ownership bridge that removes Triton/Python kernels from the hot path.
// It intentionally materializes dense [N,M] scores, so it is not the final 1.5ms target.
// RETIRED: experimental dense path (materializes [N,M]; slower than sparse).
#if 0
std::tuple<at::Tensor, at::Tensor> fused_ip_dense(
    const at::Tensor& x, const at::Tensor& base,
    int64_t k, int64_t num_buckets, int64_t cap) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_dense", /*max_dim=*/768);
  TORCH_CHECK(N % 64 == 0, "N must be a multiple of 64");
  TORCH_CHECK(M % 64 == 0, "M must be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");
  if (cap < k) cap = k;
  if (cap > M) cap = M;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto dense = at::empty({N, M}, x.options());
  auto origin = at::empty({N}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({N}, x.options().dtype(at::kFloat));
  auto th = at::empty({N}, x.options().dtype(at::kInt));
  auto bcount = at::empty({N, num_buckets}, x.options().dtype(at::kInt));
  auto val = at::empty({N, k}, x.options());
  auto idx = at::empty({N, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({N, cap}, x.options());
  auto cand_idx = at::empty({N, cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({N}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({N, cap}, x.options());
  auto lt_idx = at::empty({N, cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({N}, x.options().dtype(at::kInt));

  launch_hopper_score_to_dense_ip_fp16(
      reinterpret_cast<const __half*>(x_work.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)N, (int)M, (int)dpad,
      reinterpret_cast<__half*>(dense.data_ptr<at::Half>()), stream);
  launch_hopper_dense_bucket_coords_fp16(
      reinterpret_cast<const __half*>(dense.data_ptr<at::Half>()),
      (int)N, (int)M, origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      (int)num_buckets, stream);
  launch_flash_topk_dense_threshold_fp16(
      reinterpret_cast<const __half*>(dense.data_ptr<at::Half>()),
      (int)N, (int)M, (int)k,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), (int)num_buckets,
      bcount.data_ptr<int>(), th.data_ptr<int>(), false, stream);
  launch_flash_topk_select_thr_mb_idx_fp16(
      reinterpret_cast<const __half*>(dense.data_ptr<at::Half>()),
      /*buf_idx=*/nullptr, /*sample_idx=*/nullptr,
      (int)N, (int)M, (int)k, (int)cap,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th.data_ptr<int>(), /*qcount=*/nullptr, (int)num_buckets,
      reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
      cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
      lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
      idx.data_ptr<int>(), false, stream);
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
                            (int)(N * k), stream);
  return std::make_tuple(val, idx);
}
#endif  // fused_ip_dense

std::tuple<at::Tensor, at::Tensor> fused_ip_sparse(
    const at::Tensor& x, const at::Tensor& base,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t select_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_sparse", /*max_dim=*/768);
  // The D>512 K-chunk sparse kernel only implements the default refresh + plain
  // compacted-buffer write; the dense/bucket/refresh-from-buf variants are not
  // wired into the K-chunk path.
  TORCH_CHECK(dpad <= 512 || (!DENSE_WRITE && !BUCKET_WRITE && REFRESH_FROM_BUF == 0),
              "fused_ip_sparse with D>512 requires the default sparse write path "
              "(no DENSE_WRITE / BUCKET_WRITE / REFRESH_FROM_BUF)");
  TORCH_CHECK(N % 64 == 0, "N must be a multiple of 64");
  TORCH_CHECK(M % 64 == 0, "M must be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");
  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  if (select_cap < k) select_cap = k;
  if (select_cap > buf_cap) select_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto sample_scores = at::empty({N, sample_size}, x.options());
  auto sample_val = at::empty({N, k}, x.options());
  auto sample_idx = at::empty({N, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({N}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({N}, x.options().dtype(at::kFloat));
  auto th = at::empty({N}, x.options().dtype(at::kInt));
  auto qcount = at::empty({N}, x.options().dtype(at::kInt));
  auto bcount = at::empty({N, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto qprev = at::empty({REFRESH_FROM_BUF > 0 ? N : 1}, x.options().dtype(at::kInt));
  auto refresh_lock = at::empty({REFRESH_FROM_BUF > 0 ? N : 1}, x.options().dtype(at::kInt));
  // Dense-write variant: candidates scatter into a pre-zeroed [N, M] tensor at
  // their corpus column; select scans with column==id and skips 0 entries.
  const int64_t buf_stride = DENSE_WRITE ? M : buf_cap;
  auto buf_val = at::empty({N, buf_stride}, x.options());
  if (DENSE_WRITE) {
    cudaMemsetAsync(buf_val.data_ptr<at::Half>(), 0,
                    sizeof(at::Half) * (size_t)N * buf_stride, stream);
  }
  auto buf_idx = at::empty({DENSE_WRITE ? 1 : N, DENSE_WRITE ? 1 : buf_cap},
                           x.options().dtype(at::kInt));
  auto buf_bucket = at::empty({REFRESH_FROM_BUF == 2 ? N : 1,
                               REFRESH_FROM_BUF == 2 ? buf_cap : 1},
                              x.options().dtype(at::kByte));
  int64_t bucket_cap = BUCKET_WRITE ? k : 1;
  auto bucket_val = at::empty({BUCKET_WRITE ? N : 1, BUCKET_WRITE ? num_buckets : 1, bucket_cap},
                              x.options());
  auto bucket_idx = at::empty({BUCKET_WRITE ? N : 1, BUCKET_WRITE ? num_buckets : 1, bucket_cap},
                              x.options().dtype(at::kInt));
  auto val = at::empty({N, k}, x.options());
  auto idx = at::empty({N, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({N, select_cap}, x.options());
  auto cand_idx = at::empty({N, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({N}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({N, select_cap}, x.options());
  auto lt_idx = at::empty({N, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({N}, x.options().dtype(at::kInt));

  launch_hopper_sample_score_to_dense_ip_fp16(
      reinterpret_cast<const __half*>(x_work.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)N, (int)sample_size, (int)dpad,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)N, (int)sample_size, (int)k,
      reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), stream);
  launch_hopper_seed_from_sample_fp16(
      reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), (int)N, (int)k, (int)buf_stride, (int)num_buckets,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      REFRESH_FROM_BUF > 0 ? qprev.data_ptr<int>() : nullptr,
      REFRESH_FROM_BUF > 0 ? refresh_lock.data_ptr<int>() : nullptr,
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      DENSE_WRITE ? nullptr : buf_idx.data_ptr<int>(),
      REFRESH_FROM_BUF == 2 ? buf_bucket.data_ptr<uint8_t>() : nullptr,
      BUCKET_WRITE ? reinterpret_cast<__half*>(bucket_val.data_ptr<at::Half>()) : nullptr,
      BUCKET_WRITE ? bucket_idx.data_ptr<int>() : nullptr, (int)bucket_cap, stream);
  launch_hopper_score_to_sparse_ip_fp16(
      reinterpret_cast<const __half*>(x_work.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)N, (int)M, (int)dpad, (int)sample_size,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      DENSE_WRITE ? nullptr : qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      REFRESH_FROM_BUF > 0 ? qprev.data_ptr<int>() : nullptr,
      REFRESH_FROM_BUF > 0 ? refresh_lock.data_ptr<int>() : nullptr,
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      DENSE_WRITE ? nullptr : buf_idx.data_ptr<int>(),
      REFRESH_FROM_BUF == 2 ? buf_bucket.data_ptr<uint8_t>() : nullptr,
      BUCKET_WRITE ? reinterpret_cast<__half*>(bucket_val.data_ptr<at::Half>()) : nullptr,
      BUCKET_WRITE ? bucket_idx.data_ptr<int>() : nullptr, (int)bucket_cap,
      (int)buf_stride, (int)num_buckets, (int)k, stream);
  if (BUCKET_WRITE) {
    launch_flash_topk_select_bucket_idx_fp16(
        reinterpret_cast<const __half*>(bucket_val.data_ptr<at::Half>()),
        bucket_idx.data_ptr<int>(), bcount.data_ptr<int>(),
        (int)N, (int)num_buckets, (int)bucket_cap, (int)k, (int)select_cap,
        th.data_ptr<int>(),
        reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
        cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
        lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
        idx.data_ptr<int>(), stream);
  } else {
    launch_flash_topk_select_thr_mb_idx_fp16(
        reinterpret_cast<const __half*>(buf_val.data_ptr<at::Half>()),
        DENSE_WRITE ? nullptr : buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
        (int)N, (int)buf_stride, (int)k, (int)select_cap,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th.data_ptr<int>(), DENSE_WRITE ? nullptr : qcount.data_ptr<int>(), (int)num_buckets,
        reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
        cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
        lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
        idx.data_ptr<int>(), false, stream, /*skip_zero=*/DENSE_WRITE);
  }
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
                            (int)(N * k), stream);
  return std::make_tuple(val, idx);
}

std::tuple<at::Tensor, at::Tensor> fused_ip_sparse_fp32buf(
    const at::Tensor& x, const at::Tensor& base,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t select_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_sparse_fp32buf",
                      /*max_dim=*/768);
  TORCH_CHECK(N % 64 == 0, "N must be a multiple of 64");
  TORCH_CHECK(M % 64 == 0, "M must be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");
  TORCH_CHECK(dpad > 512, "fp32 sparse buffer variant currently targets D > 512 kchunk path");
  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  if (select_cap < k) select_cap = k;
  if (select_cap > buf_cap) select_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto sample_scores = at::empty({N, sample_size}, x.options());
  auto sample_val = at::empty({N, k}, x.options());
  auto sample_idx = at::empty({N, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({N}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({N}, x.options().dtype(at::kFloat));
  auto th = at::empty({N}, x.options().dtype(at::kInt));
  auto qcount = at::empty({N}, x.options().dtype(at::kInt));
  auto bcount = at::empty({N, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto buf_val = at::empty({N, buf_cap}, x.options().dtype(at::kFloat));
  auto buf_idx = at::empty({N, buf_cap}, x.options().dtype(at::kInt));
  auto val = at::empty({N, k}, x.options().dtype(at::kFloat));
  auto idx = at::empty({N, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({N, select_cap}, x.options().dtype(at::kFloat));
  auto cand_idx = at::empty({N, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({N}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({N, select_cap}, x.options().dtype(at::kFloat));
  auto lt_idx = at::empty({N, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({N}, x.options().dtype(at::kInt));

  launch_hopper_sample_score_to_dense_ip_fp16(
      reinterpret_cast<const __half*>(x_work.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)N, (int)sample_size, (int)dpad,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)N, (int)sample_size, (int)k,
      reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), stream);
  launch_hopper_seed_from_sample_fp32(
      reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), (int)N, (int)k, (int)buf_cap, (int)num_buckets,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      /*qprev=*/nullptr, /*refresh_lock=*/nullptr,
      buf_val.data_ptr<float>(), buf_idx.data_ptr<int>(), /*buf_bucket=*/nullptr,
      /*bucket_val=*/nullptr, /*bucket_idx=*/nullptr, /*bucket_cap=*/1, stream);
  launch_hopper_score_to_sparse_ip_fp32(
      reinterpret_cast<const __half*>(x_work.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)N, (int)M, (int)dpad, (int)sample_size,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      buf_val.data_ptr<float>(), buf_idx.data_ptr<int>(),
      (int)buf_cap, (int)num_buckets, (int)k, stream);
  launch_flash_topk_select_thr_mb_idx_fp32(
      buf_val.data_ptr<float>(), buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
      (int)N, (int)buf_cap, (int)k, (int)select_cap,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
      cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
      lt_val.data_ptr<float>(), lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
      val.data_ptr<float>(), idx.data_ptr<int>(), stream);
  return std::make_tuple(val.neg(), idx);
}

// RETIRED experimental paths (kept for reference, not registered):
//   fused_ip_sparse_timed  - per-stage event-timed profiling variant
//   fused_ip_smalln        - exact small-N CUDA path (slower than sparse)
//   fused_ip_smalln_tc     - tensor-core small-N path
#if 0
// Per-stage event-timed variant of fused_ip_sparse.  Returns a CPU float tensor
// [5] with milliseconds for: [0] sample WGMMA score, [1] sample top-k (min),
// [2] seed-from-sample, [3] sparse WGMMA tail scan, [4] multi-CTA select.
// Times are averaged over `iters` runs.  Used to break down where time goes
// without ncu perf-counter permission.
at::Tensor fused_ip_sparse_timed(
    const at::Tensor& x, const at::Tensor& base,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t select_cap,
    int64_t sample_size, int64_t iters) {
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_sparse_timed");
  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  if (select_cap < k) select_cap = k;
  if (select_cap > buf_cap) select_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto sample_scores = at::empty({N, sample_size}, x.options());
  auto sample_val = at::empty({N, k}, x.options());
  auto sample_idx = at::empty({N, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({N}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({N}, x.options().dtype(at::kFloat));
  auto th = at::empty({N}, x.options().dtype(at::kInt));
  auto qcount = at::empty({N}, x.options().dtype(at::kInt));
  auto bcount = at::empty({N, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto qprev = at::empty({REFRESH_FROM_BUF > 0 ? N : 1}, x.options().dtype(at::kInt));
  auto refresh_lock = at::empty({REFRESH_FROM_BUF > 0 ? N : 1}, x.options().dtype(at::kInt));
  auto buf_val = at::empty({N, buf_cap}, x.options());
  auto buf_idx = at::empty({N, buf_cap}, x.options().dtype(at::kInt));
  auto buf_bucket = at::empty({REFRESH_FROM_BUF == 2 ? N : 1,
                               REFRESH_FROM_BUF == 2 ? buf_cap : 1},
                              x.options().dtype(at::kByte));
  int64_t bucket_cap = BUCKET_WRITE ? k : 1;
  auto bucket_val = at::empty({BUCKET_WRITE ? N : 1, BUCKET_WRITE ? num_buckets : 1, bucket_cap},
                              x.options());
  auto bucket_idx = at::empty({BUCKET_WRITE ? N : 1, BUCKET_WRITE ? num_buckets : 1, bucket_cap},
                              x.options().dtype(at::kInt));
  auto val = at::empty({N, k}, x.options());
  auto idx = at::empty({N, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({N, select_cap}, x.options());
  auto cand_idx = at::empty({N, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({N}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({N, select_cap}, x.options());
  auto lt_idx = at::empty({N, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({N}, x.options().dtype(at::kInt));

  const __half* xp = reinterpret_cast<const __half*>(x_work.data_ptr<at::Half>());
  const __half* bp = reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>());

  const int NSTAGE = 6;  // 5 timed boundaries -> 5 intervals; +1 event
  cudaEvent_t ev[NSTAGE];
  for (int e = 0; e < NSTAGE; ++e) cudaEventCreate(&ev[e]);
  double acc[5] = {0, 0, 0, 0, 0};

  for (int it = 0; it < iters + 1; ++it) {
    cudaEventRecord(ev[0], stream);
    launch_hopper_sample_score_to_dense_ip_fp16(
        xp, bp, (int)N, (int)sample_size, (int)dpad,
        reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
    cudaEventRecord(ev[1], stream);
    launch_flash_topk_min_fp16(
        reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
        (int)N, (int)sample_size, (int)k,
        reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
        sample_idx.data_ptr<int>(), stream);
    cudaEventRecord(ev[2], stream);
    launch_hopper_seed_from_sample_fp16(
        reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
        sample_idx.data_ptr<int>(), (int)N, (int)k, (int)buf_cap, (int)num_buckets,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
        qcount.data_ptr<int>(), bcount.data_ptr<int>(),
        REFRESH_FROM_BUF > 0 ? qprev.data_ptr<int>() : nullptr,
        REFRESH_FROM_BUF > 0 ? refresh_lock.data_ptr<int>() : nullptr,
        reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()), buf_idx.data_ptr<int>(),
        REFRESH_FROM_BUF == 2 ? buf_bucket.data_ptr<uint8_t>() : nullptr,
        BUCKET_WRITE ? reinterpret_cast<__half*>(bucket_val.data_ptr<at::Half>()) : nullptr,
        BUCKET_WRITE ? bucket_idx.data_ptr<int>() : nullptr, (int)bucket_cap, stream);
    cudaEventRecord(ev[3], stream);
    launch_hopper_score_to_sparse_ip_fp16(
        xp, bp, (int)N, (int)M, (int)dpad, (int)sample_size,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
        qcount.data_ptr<int>(), bcount.data_ptr<int>(),
        REFRESH_FROM_BUF > 0 ? qprev.data_ptr<int>() : nullptr,
        REFRESH_FROM_BUF > 0 ? refresh_lock.data_ptr<int>() : nullptr,
        reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()), buf_idx.data_ptr<int>(),
        REFRESH_FROM_BUF == 2 ? buf_bucket.data_ptr<uint8_t>() : nullptr,
        BUCKET_WRITE ? reinterpret_cast<__half*>(bucket_val.data_ptr<at::Half>()) : nullptr,
        BUCKET_WRITE ? bucket_idx.data_ptr<int>() : nullptr, (int)bucket_cap,
        (int)buf_cap, (int)num_buckets, (int)k, stream);
    cudaEventRecord(ev[4], stream);
    launch_flash_topk_select_thr_mb_idx_fp16(
        reinterpret_cast<const __half*>(buf_val.data_ptr<at::Half>()),
        buf_idx.data_ptr<int>(), nullptr,
        (int)N, (int)buf_cap, (int)k, (int)select_cap,
        origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
        th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
        reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
        cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
        lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
        reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
        idx.data_ptr<int>(), false, stream);
    cudaEventRecord(ev[5], stream);
    cudaEventSynchronize(ev[5]);
    if (it == 0) continue;  // warmup
    float ms = 0;
    for (int s = 0; s < 5; ++s) {
      cudaEventElapsedTime(&ms, ev[s], ev[s + 1]);
      acc[s] += ms;
    }
  }
  for (int e = 0; e < NSTAGE; ++e) cudaEventDestroy(ev[e]);
  auto out = at::empty({5}, at::TensorOptions().dtype(at::kFloat));
  float* op = out.data_ptr<float>();
  for (int s = 0; s < 5; ++s) op[s] = (float)(acc[s] / (double)iters);
  return out;
}

std::tuple<at::Tensor, at::Tensor> fused_ip_smalln(
    const at::Tensor& x, const at::Tensor& base, int64_t k) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_smalln", /*max_dim=*/768);
  TORCH_CHECK(N >= 1 && N <= 32, "small-N kernel requires 1 <= N <= 32");
  TORCH_CHECK(k >= 1 && k <= M && k <= 128, "small-N kernel requires 1 <= k <= min(M, 128)");

  constexpr int SMALLN_CHUNK_HOST = 4096;
  int num_chunks = (int)((M + SMALLN_CHUNK_HOST - 1) / SMALLN_CHUNK_HOST);
  auto partial_val = at::empty({N, num_chunks, k}, x.options());
  auto partial_idx = at::empty({N, num_chunks, k}, x.options().dtype(at::kInt));
  auto val = at::empty({N, k}, x.options());
  auto idx = at::empty({N, k}, x.options().dtype(at::kInt));

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_hopper_smalln_ip_topk_fp16(
      reinterpret_cast<const __half*>(x_work.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)N, (int)M, (int)dpad, (int)k,
      reinterpret_cast<__half*>(partial_val.data_ptr<at::Half>()),
      partial_idx.data_ptr<int>(), num_chunks,
      reinterpret_cast<__half*>(val.data_ptr<at::Half>()),
      idx.data_ptr<int>(), stream);
  return std::make_tuple(val, idx);
}

std::tuple<at::Tensor, at::Tensor> fused_ip_smalln_tc(
    const at::Tensor& x, const at::Tensor& base, int64_t k) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_smalln_tc", /*max_dim=*/768);
  TORCH_CHECK(N >= 1 && N <= 32, "small-N TC kernel requires 1 <= N <= 32");
  TORCH_CHECK(M % 16 == 0, "small-N TC kernel requires M to be a multiple of 16");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");

  int64_t Npad = ((N + 15) / 16) * 16;
  // Tile real query rows into the padding slots (see fused_ip_smalln_sparse).
  at::Tensor xpad;
  if (N == Npad) {
    xpad = x_work;
  } else {
    int64_t reps = (Npad + N - 1) / N;
    xpad = x_work.repeat({reps, 1}).slice(0, 0, Npad).contiguous();
  }
  auto dense = at::empty({Npad, M}, x.options());
  auto val_pad = at::empty({Npad, k}, x.options());
  auto idx_pad = at::empty({Npad, k}, x.options().dtype(at::kInt));

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_hopper_smalln_score_dense_ip_gmma_m64n32_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)Npad, (int)Npad, (int)M, (int)dpad,
      reinterpret_cast<__half*>(dense.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(dense.data_ptr<at::Half>()),
      (int)Npad, (int)M, (int)k,
      reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
      idx_pad.data_ptr<int>(), stream);
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
                            (int)(Npad * k), stream);

  return std::make_tuple(val_pad.slice(0, 0, N).contiguous(),
                         idx_pad.slice(0, 0, N).contiguous());
}
#endif  // fused_ip_sparse_timed / fused_ip_smalln / fused_ip_smalln_tc

std::tuple<at::Tensor, at::Tensor> fused_ip_smalln_sparse(
    const at::Tensor& x, const at::Tensor& base,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t select_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_smalln_sparse", /*max_dim=*/768);
  TORCH_CHECK(N >= 1 && N <= 32, "small-N sparse kernel requires 1 <= N <= 32");
  TORCH_CHECK(M % 64 == 0, "small-N sparse kernel requires M to be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");

  const char* m8_env = std::getenv("HOPPER_SMALLBATCH_M8");
  const bool use_m8 = (m8_env == nullptr || m8_env[0] != '0');
  // m64n8 originally required dpad>512; with HOPPER_SMALLN8_LOW_D (default on)
  // it also serves D<=512, so pick Npad=8 for any N<=8, k<=128 with dpad<=768
  // (the m64n8 full kernel's max). dpad is padded up to a multiple of 64.
  const char* lowd_env = std::getenv("HOPPER_SMALLN8_LOW_D");
  const bool lowd = (lowd_env == nullptr || lowd_env[0] != '0');
  const bool m8_d_ok = (dpad > 512) || (lowd && dpad <= 768);
  const int64_t Npad = (use_m8 && N <= 8 && m8_d_ok && k <= 128) ? 8 : 32;
  const int64_t Rwork = (Npad == 8) ? N : Npad;
  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  if (select_cap < k) select_cap = k;
  if (select_cap > buf_cap) select_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // Fill the (Npad-N) padding rows with real query rows (tiled) instead of
  // zeros: an all-zero query scores 0 against every base row, and those zeros
  // pass the bucket gate, so zero padding rows flood the sparse buffer and make
  // both the tail scan and select ~7x slower. Tiling real rows keeps every lane
  // well-behaved; the padding outputs are dropped by the final slice.
  at::Tensor xpad;
  if (N == Npad) {
    xpad = x_work;
  } else {
    int64_t reps = (Npad + N - 1) / N;
    xpad = x_work.repeat({reps, 1}).slice(0, 0, Npad).contiguous();
  }
  auto sample_scores = at::empty({Rwork, sample_size}, x.options());
  auto sample_val = at::empty({Rwork, k}, x.options());
  auto sample_idx = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto th = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto qcount = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto bcount = at::empty({Rwork, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto buf_val = at::empty({Rwork, buf_cap}, x.options());
  auto buf_idx = at::empty({Rwork, buf_cap}, x.options().dtype(at::kInt));
  int64_t bucket_cap = BUCKET_WRITE ? k : 1;
  auto bucket_val = at::empty({BUCKET_WRITE ? Rwork : 1, BUCKET_WRITE ? num_buckets : 1, bucket_cap},
                              x.options());
  auto bucket_idx = at::empty({BUCKET_WRITE ? Rwork : 1, BUCKET_WRITE ? num_buckets : 1, bucket_cap},
                              x.options().dtype(at::kInt));
  auto val_pad = at::empty({Rwork, k}, x.options());
  auto idx_pad = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({Rwork, select_cap}, x.options());
  auto cand_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({Rwork, select_cap}, x.options());
  auto lt_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));

  launch_hopper_smalln_score_dense_ip_gmma_m64n32_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)Npad, (int)Rwork, (int)sample_size, (int)dpad,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)Rwork, (int)sample_size, (int)k,
      reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), stream);
  launch_hopper_seed_from_sample_fp16(
      reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), (int)Rwork, (int)k, (int)buf_cap, (int)num_buckets,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      /*qprev=*/nullptr, /*refresh_lock=*/nullptr,
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()), buf_idx.data_ptr<int>(),
      /*buf_bucket=*/nullptr,
      BUCKET_WRITE ? reinterpret_cast<__half*>(bucket_val.data_ptr<at::Half>()) : nullptr,
      BUCKET_WRITE ? bucket_idx.data_ptr<int>() : nullptr, (int)bucket_cap, stream);
  launch_hopper_smalln_score_to_sparse_ip_gmma_m64n32_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)Npad, (int)Rwork, (int)M, (int)dpad, (int)sample_size,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()), buf_idx.data_ptr<int>(),
      (int)buf_cap, (int)num_buckets, (int)k, stream);
  launch_flash_topk_select_thr_mb_idx_fp16(
      reinterpret_cast<const __half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
      (int)Rwork, (int)buf_cap, (int)k, (int)select_cap,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
      reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
      cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
      lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
      idx_pad.data_ptr<int>(), false, stream);
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
                            (int)(Rwork * k), stream);

  return std::make_tuple(val_pad.slice(0, 0, N).contiguous(),
                         idx_pad.slice(0, 0, N).contiguous());
}

std::tuple<at::Tensor, at::Tensor> fused_ip_smalln_sparse_fp32buf(
    const at::Tensor& x, const at::Tensor& base,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t select_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_smalln_sparse_fp32buf",
                      /*max_dim=*/768);
  TORCH_CHECK(N >= 1 && N <= 32, "small-N sparse fp32buf requires 1 <= N <= 32");
  TORCH_CHECK(M % 64 == 0, "small-N sparse fp32buf requires M to be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");
  TORCH_CHECK(dpad > 512, "small-N sparse fp32buf currently targets D > 512 kchunk path");

  const int64_t Npad = 32;
  const int64_t Rwork = Npad;
  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  if (select_cap < k) select_cap = k;
  if (select_cap > buf_cap) select_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  at::Tensor xpad;
  if (N == Npad) {
    xpad = x_work;
  } else {
    int64_t reps = (Npad + N - 1) / N;
    xpad = x_work.repeat({reps, 1}).slice(0, 0, Npad).contiguous();
  }

  auto sample_scores = at::empty({Rwork, sample_size}, x.options());
  auto sample_val = at::empty({Rwork, k}, x.options());
  auto sample_idx = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto th = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto qcount = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto bcount = at::empty({Rwork, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto buf_val = at::empty({Rwork, buf_cap}, x.options().dtype(at::kFloat));
  auto buf_idx = at::empty({Rwork, buf_cap}, x.options().dtype(at::kInt));
  auto val_pad = at::empty({Rwork, k}, x.options().dtype(at::kFloat));
  auto idx_pad = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({Rwork, select_cap}, x.options().dtype(at::kFloat));
  auto cand_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({Rwork, select_cap}, x.options().dtype(at::kFloat));
  auto lt_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));

  launch_hopper_smalln_score_dense_ip_gmma_m64n32_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)Npad, (int)Rwork, (int)sample_size, (int)dpad,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)Rwork, (int)sample_size, (int)k,
      reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), stream);
  launch_hopper_seed_from_sample_fp32(
      reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), (int)Rwork, (int)k, (int)buf_cap, (int)num_buckets,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      /*qprev=*/nullptr, /*refresh_lock=*/nullptr,
      buf_val.data_ptr<float>(), buf_idx.data_ptr<int>(), /*buf_bucket=*/nullptr,
      /*bucket_val=*/nullptr, /*bucket_idx=*/nullptr, /*bucket_cap=*/1, stream);
  launch_hopper_smalln_score_to_sparse_ip_gmma_m64n32_fp32(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)Npad, (int)Rwork, (int)M, (int)dpad, (int)sample_size,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      buf_val.data_ptr<float>(), buf_idx.data_ptr<int>(),
      (int)buf_cap, (int)num_buckets, (int)k, stream);
  launch_flash_topk_select_thr_mb_idx_fp32(
      buf_val.data_ptr<float>(), buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
      (int)Rwork, (int)buf_cap, (int)k, (int)select_cap,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
      cand_val.data_ptr<float>(), cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
      lt_val.data_ptr<float>(), lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
      val_pad.data_ptr<float>(), idx_pad.data_ptr<int>(), stream);

  return std::make_tuple(val_pad.slice(0, 0, N).contiguous().neg(),
                         idx_pad.slice(0, 0, N).contiguous());
}

// Fused full-store + in-GEMM histogram dense-bucket path (batch 1/8).
//   0. sample GEMM (first S cols) -> coords (origin/inv_delta) + seed gate from
//      the sample histogram (a subset, so its k-th bucket is a loose/safe gate).
//   1. main GEMM over base[S:M] writing the full dense matrix AND atomic-adding
//      the gated tail histogram into bcount (prefix [0,S) already counted).
//      dense prefix [0,S) is copied from the sample scores.
//   2. threshold_from_bcount over the full (prefix+tail) histogram. No extra
//      dense pass.
//   3. exact bucket select over the dense matrix (column == corpus id).
// RETIRED: materializes the [R,M] dense matrix; measured slower than sparse.
#if 0
std::tuple<at::Tensor, at::Tensor> fused_ip_smalln_dense_bucket_fused(
    const at::Tensor& x, const at::Tensor& base,
    int64_t k, int64_t num_buckets, int64_t select_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && base.is_cuda(), "x/base must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base.dim() == 2, "x/base must be 2-D");
  TORCH_CHECK(x.is_contiguous() && base.is_contiguous(), "x/base must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base.scalar_type() == at::kHalf,
              "x/base must be float16");
  const int64_t N = x.size(0);
  const int64_t M = base.size(0);
  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_dpad_inputs(x, base, x_work, base_work, dpad, "fused_ip_smalln_dense_bucket_fused",
                      /*max_dim=*/768);
  TORCH_CHECK(N >= 1 && N <= 8, "fused dense-bucket requires 1 <= N <= 8");
  TORCH_CHECK(M % 64 == 0, "M must be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2 && num_buckets <= 64, "num_buckets in [2,64]");
  TORCH_CHECK(dpad > 512 && k <= 128,
              "fused dense-bucket targets the m64n8 GEMM (D in (512,768], k<=128)");
  if (select_cap < k) select_cap = k;
  if (select_cap > M) select_cap = M;
  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;

  const int64_t Npad = 8;
  const int64_t Rwork = N;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  at::Tensor xpad;
  if (N == Npad) {
    xpad = x_work;
  } else {
    int64_t reps = (Npad + N - 1) / N;
    xpad = x_work.repeat({reps, 1}).slice(0, 0, Npad).contiguous();
  }

  auto dense = at::empty({Rwork, M}, x.options());
  auto origin = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto th = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto bcount = at::zeros({Rwork, num_buckets}, x.options().dtype(at::kInt));
  auto sample_scores = at::empty({Rwork, sample_size}, x.options());
  auto seed_th = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto val_pad = at::empty({Rwork, k}, x.options());
  auto idx_pad = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({Rwork, select_cap}, x.options());
  auto cand_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({Rwork, select_cap}, x.options());
  auto lt_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));

  const __half* xp = reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>());
  const __half* bp = reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>());

  // (0) Sample prefix -> coords (origin/inv_delta) + seed gate.
  launch_hopper_smalln_score_dense_ip_gmma_m64n32_fp16(
      xp, bp, (int)Npad, (int)Rwork, (int)sample_size, (int)dpad,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  // Coords from full-sample min/max. bcount is pre-zeroed via at::zeros above
  // (the sample histogram below accumulates into it with skip_bcount_zero).
  launch_hopper_dense_bucket_coords_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)Rwork, (int)sample_size, origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      (int)num_buckets, stream);
  // Single sample histogram into bcount: it is BOTH the gate seed AND the
  // prefix [0, sample_size) contribution to the final corpus histogram.
  launch_flash_topk_dense_threshold_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)Rwork, (int)sample_size, (int)k,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), (int)num_buckets,
      bcount.data_ptr<int>(), seed_th.data_ptr<int>(), false, stream,
      /*skip_bcount_zero=*/true);
  // (1) Fused GEMM: write dense AND atomic-add the tail [sample_size, M) into
  // bcount (gated by seed_th). The prefix dense scores come from the sample.
  dense.narrow(1, 0, sample_size).copy_(sample_scores);
  launch_hopper_smalln_score_dense_hist_ip_gmma_m64n8_fp16(
      xp, bp, (int)Npad, (int)Rwork, (int)M, (int)dpad, (int)num_buckets, (int)k,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), seed_th.data_ptr<int>(),
      bcount.data_ptr<int>(),
      reinterpret_cast<__half*>(dense.data_ptr<at::Half>()), stream,
      /*start_col=*/(int)sample_size);
  // (2) Final threshold from the full (prefix + tail) histogram in bcount.
  launch_flash_topk_threshold_from_bcount(
      bcount.data_ptr<int>(), (int)Rwork, (int)num_buckets, (int)k, th.data_ptr<int>(), stream);
  // (3) Exact bucket-based select over the dense matrix; column == corpus id.
  launch_flash_topk_select_thr_mb_idx_fp16(
      reinterpret_cast<const __half*>(dense.data_ptr<at::Half>()),
      /*buf_idx=*/nullptr, /*sample_idx=*/nullptr,
      (int)Rwork, (int)M, (int)k, (int)select_cap,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th.data_ptr<int>(), /*qcount=*/nullptr, (int)num_buckets,
      reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
      cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
      lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
      idx_pad.data_ptr<int>(), false, stream, /*skip_zero=*/0);
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
                            (int)(Rwork * k), stream);

  return std::make_tuple(val_pad.slice(0, 0, N).contiguous(),
                         idx_pad.slice(0, 0, N).contiguous());
}
#endif  // fused_ip_smalln_dense_bucket_fused

std::tuple<at::Tensor, at::Tensor> fused_ip_gqa_sparse(
    const at::Tensor& x, const at::Tensor& base3d,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t select_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && base3d.is_cuda(), "x/base3d must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && base3d.dim() == 3,
              "x must be [Hq,D], base3d must be [Hkv,M,D]");
  TORCH_CHECK(x.is_contiguous() && base3d.is_contiguous(), "x/base3d must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && base3d.scalar_type() == at::kHalf,
              "x/base3d must be float16");
  const int64_t Hq = x.size(0);
  const int64_t Hkv = base3d.size(0);
  const int64_t M = base3d.size(1);
  TORCH_CHECK(Hkv >= 1 && Hq >= Hkv && Hq % Hkv == 0,
              "Hq must be a positive multiple of Hkv");
  const int64_t group = Hq / Hkv;
  TORCH_CHECK(group >= 1 && group <= 8, "GQA group size must be in [1, 8]");
  TORCH_CHECK(M % 64 == 0, "GQA sparse kernel requires M to be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");
  TORCH_CHECK(!DENSE_WRITE && !BUCKET_WRITE && REFRESH_FROM_BUF == 0,
              "fused_ip_gqa_sparse currently supports only the default compact sparse path "
              "(no dense write / bucket write / refresh-from-buffer)");

  at::Tensor x_work;
  at::Tensor base_work;
  int64_t dpad = 0;
  prepare_gqa_dpad_inputs(x, base3d, x_work, base_work, dpad, "fused_ip_gqa_sparse",
                          /*max_dim=*/768);
  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  if (select_cap < k) select_cap = k;
  if (select_cap > buf_cap) select_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int qn8 = 8;
  const int64_t Rwork = Hkv * qn8;
  auto xpad = at::zeros({Rwork, dpad}, x.options());
  for (int64_t h = 0; h < Hkv; ++h) {
    xpad.slice(0, h * qn8, h * qn8 + group)
        .copy_(x_work.slice(0, h * group, (h + 1) * group));
  }

  auto sample_scores = at::zeros({Rwork, sample_size}, x.options());
  auto sample_val = at::empty({Rwork, k}, x.options());
  auto sample_idx = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto th = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto qcount = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto bcount = at::empty({Rwork, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto buf_val = at::empty({Rwork, buf_cap}, x.options());
  auto buf_idx = at::empty({Rwork, buf_cap}, x.options().dtype(at::kInt));
  auto val_pad = at::empty({Rwork, k}, x.options());
  auto idx_pad = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({Rwork, select_cap}, x.options());
  auto cand_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({Rwork, select_cap}, x.options());
  auto lt_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));

  launch_hopper_gqa_smalln_score_dense_ip_gmma_m64n8_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)Hkv, qn8, (int)group, (int)sample_size, (int)M, (int)dpad,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)Rwork, (int)sample_size, (int)k,
      reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), stream);
  launch_hopper_seed_from_sample_fp16(
      reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), (int)Rwork, (int)k, (int)buf_cap, (int)num_buckets,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      /*qprev=*/nullptr, /*refresh_lock=*/nullptr,
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), /*buf_bucket=*/nullptr,
      /*bucket_val=*/nullptr, /*bucket_idx=*/nullptr, /*bucket_cap=*/1, stream);
  launch_hopper_gqa_smalln_score_to_sparse_ip_gmma_m64n8_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(base_work.data_ptr<at::Half>()),
      (int)Hkv, qn8, (int)group, (int)M, (int)M, (int)dpad, (int)sample_size,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), (int)buf_cap, (int)num_buckets, (int)k, stream);
  launch_flash_topk_select_thr_mb_idx_fp16(
      reinterpret_cast<const __half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
      (int)Rwork, (int)buf_cap, (int)k, (int)select_cap,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
      reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
      cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
      lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
      idx_pad.data_ptr<int>(), false, stream);
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
                            (int)(Rwork * k), stream);

  auto val = val_pad.view({Hkv, qn8, k}).slice(1, 0, group).reshape({Hq, k}).contiguous();
  auto idx = idx_pad.view({Hkv, qn8, k}).slice(1, 0, group).reshape({Hq, k}).contiguous();
  return std::make_tuple(val, idx);
}

std::tuple<at::Tensor, at::Tensor> fused_ip_gqa_sparse_paged(
    const at::Tensor& x, const at::Tensor& kv_data,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t select_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && kv_data.is_cuda(), "x/kv_data must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && kv_data.dim() == 5,
              "x must be [Hq,D], kv_data must be [M,2,page_size,Hkv,D]");
  TORCH_CHECK(x.is_contiguous() && kv_data.is_contiguous(), "x/kv_data must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && kv_data.scalar_type() == at::kHalf,
              "x/kv_data must be float16");
  TORCH_CHECK(kv_data.size(1) == 2 && kv_data.size(2) == 1,
              "paged GQA path currently supports Tidal NHD page_size=1 only");
  const int64_t Hq = x.size(0);
  const int64_t M = kv_data.size(0);
  const int64_t Hkv = kv_data.size(3);
  const int64_t D = kv_data.size(4);
  TORCH_CHECK(x.size(1) == D, "x and kv_data head_dim mismatch");
  TORCH_CHECK((D % 64) == 0 && D <= 768,
              "paged GQA path currently requires D to be a multiple of 64 and <= 768");
  TORCH_CHECK(Hkv >= 1 && Hq >= Hkv && Hq % Hkv == 0,
              "Hq must be a positive multiple of Hkv");
  const int64_t group = Hq / Hkv;
  TORCH_CHECK(group >= 1 && group <= 8, "GQA group size must be in [1, 8]");
  TORCH_CHECK(M % 64 == 0, "paged GQA sparse kernel requires M to be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");
  TORCH_CHECK(!DENSE_WRITE && !BUCKET_WRITE && REFRESH_FROM_BUF == 0,
              "fused_ip_gqa_sparse_paged currently supports only the default compact sparse path");

  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  if (select_cap < k) select_cap = k;
  if (select_cap > buf_cap) select_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int qn8 = 8;
  const int64_t Rwork = Hkv * qn8;
  auto xpad = at::zeros({Rwork, D}, x.options());
  for (int64_t h = 0; h < Hkv; ++h) {
    xpad.slice(0, h * qn8, h * qn8 + group)
        .copy_(x.slice(0, h * group, (h + 1) * group));
  }

  auto sample_scores = at::zeros({Rwork, sample_size}, x.options());
  auto sample_val = at::empty({Rwork, k}, x.options());
  auto sample_idx = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto th = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto qcount = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto bcount = at::empty({Rwork, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto buf_val = at::empty({Rwork, buf_cap}, x.options());
  auto buf_idx = at::empty({Rwork, buf_cap}, x.options().dtype(at::kInt));
  auto val_pad = at::empty({Rwork, k}, x.options());
  auto idx_pad = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({Rwork, select_cap}, x.options());
  auto cand_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({Rwork, select_cap}, x.options());
  auto lt_idx = at::empty({Rwork, select_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));

  launch_hopper_gqa_paged_score_dense_ip_gmma_m64n8_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(kv_data.data_ptr<at::Half>()),
      (int)Hkv, qn8, (int)group, (int)sample_size, (int)D,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)Rwork, (int)sample_size, (int)k,
      reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), stream);
  launch_hopper_seed_from_sample_fp16(
      reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), (int)Rwork, (int)k, (int)buf_cap, (int)num_buckets,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      /*qprev=*/nullptr, /*refresh_lock=*/nullptr,
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), /*buf_bucket=*/nullptr,
      /*bucket_val=*/nullptr, /*bucket_idx=*/nullptr, /*bucket_cap=*/1, stream);
  launch_hopper_gqa_paged_score_to_sparse_ip_gmma_m64n8_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(kv_data.data_ptr<at::Half>()),
      (int)Hkv, qn8, (int)group, (int)M, (int)D, (int)sample_size,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), (int)buf_cap, (int)num_buckets, (int)k, stream);
  launch_flash_topk_select_thr_mb_idx_fp16(
      reinterpret_cast<const __half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), /*sample_idx=*/nullptr,
      (int)Rwork, (int)buf_cap, (int)k, (int)select_cap,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th.data_ptr<int>(), qcount.data_ptr<int>(), (int)num_buckets,
      reinterpret_cast<__half*>(cand_val.data_ptr<at::Half>()),
      cand_idx.data_ptr<int>(), cand_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(lt_val.data_ptr<at::Half>()),
      lt_idx.data_ptr<int>(), lt_cnt.data_ptr<int>(),
      reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
      idx_pad.data_ptr<int>(), false, stream);
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
                            (int)(Rwork * k), stream);

  auto val = val_pad.view({Hkv, qn8, k}).slice(1, 0, group).reshape({Hq, k}).contiguous();
  auto idx = idx_pad.view({Hkv, qn8, k}).slice(1, 0, group).reshape({Hq, k}).contiguous();
  return std::make_tuple(val, idx);
}

std::tuple<at::Tensor, at::Tensor> fused_ip_gqa_sparse_paged_group_indexed(
    const at::Tensor& x, const at::Tensor& kv_data, const at::Tensor& group_indices,
    int64_t k, int64_t num_buckets, int64_t buf_cap, int64_t sample_size) {
  TORCH_CHECK(x.is_cuda() && kv_data.is_cuda() && group_indices.is_cuda(),
              "x/kv_data/group_indices must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && kv_data.dim() == 5 && group_indices.dim() == 2,
              "x must be [Hq,D], kv_data [physical_pages,2,1,Hkv,D], group_indices [Hkv,M]");
  TORCH_CHECK(x.is_contiguous() && kv_data.is_contiguous() && group_indices.is_contiguous(),
              "x/kv_data/group_indices must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kHalf && kv_data.scalar_type() == at::kHalf,
              "x/kv_data must be float16");
  TORCH_CHECK(group_indices.scalar_type() == at::kInt, "group_indices must be int32");
  TORCH_CHECK(kv_data.size(1) == 2 && kv_data.size(2) == 1,
              "group-indexed paged path currently supports Tidal NHD page_size=1 only");
  const int64_t Hq = x.size(0);
  const int64_t D = x.size(1);
  const int64_t physical_pages = kv_data.size(0);
  const int64_t Hkv = kv_data.size(3);
  const int64_t M = group_indices.size(1);
  TORCH_CHECK(kv_data.size(4) == D, "x and kv_data head_dim mismatch");
  TORCH_CHECK(group_indices.size(0) == Hkv, "group_indices must have one row per KV head");
  TORCH_CHECK((D % 64) == 0 && D <= 768,
              "group-indexed path requires D to be a multiple of 64 and <= 768");
  TORCH_CHECK(Hkv >= 1 && Hq >= Hkv && Hq % Hkv == 0,
              "Hq must be a positive multiple of Hkv");
  const int64_t group = Hq / Hkv;
  TORCH_CHECK(group >= 1 && group <= 8, "GQA group size must be in [1, 8]");
  TORCH_CHECK(M % 64 == 0, "logical page count M must be a multiple of 64");
  TORCH_CHECK(k >= 1 && k <= M, "require 1 <= k <= M");
  TORCH_CHECK(num_buckets >= 2, "num_buckets must be >= 2");
  TORCH_CHECK(!DENSE_WRITE && !BUCKET_WRITE && REFRESH_FROM_BUF == 0,
              "fused_ip_gqa_sparse_paged_group_indexed supports only the default compact sparse path");

  sample_size = ((sample_size + 63) / 64) * 64;
  if (sample_size < k) sample_size = ((k + 63) / 64) * 64;
  if (sample_size > M) sample_size = M;
  const int64_t sample_start = M - sample_size;
  if (buf_cap < k) buf_cap = k;
  if (buf_cap > M) buf_cap = M;
  const int64_t candidate_cap = buf_cap;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int qn8 = 8;
  const int64_t Rwork = Hkv * qn8;
  auto xpad = at::zeros({Rwork, D}, x.options());
  for (int64_t h = 0; h < Hkv; ++h) {
    xpad.slice(0, h * qn8, h * qn8 + group)
        .copy_(x.slice(0, h * group, (h + 1) * group));
  }

  auto sample_scores = at::zeros({Rwork, sample_size}, x.options());
  auto sample_val = at::empty({Rwork, k}, x.options());
  auto sample_idx = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto origin = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto inv_delta = at::empty({Rwork}, x.options().dtype(at::kFloat));
  auto th = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto qcount = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto bcount = at::empty({Rwork, num_buckets * BCOUNT_SHARDS}, x.options().dtype(at::kInt));
  auto buf_val = at::empty({Rwork, buf_cap}, x.options());
  auto buf_idx = at::empty({Rwork, buf_cap}, x.options().dtype(at::kInt));
  auto val_pad = at::empty({Rwork, k}, x.options());
  auto idx_pad = at::empty({Rwork, k}, x.options().dtype(at::kInt));
  auto cand_val = at::empty({Rwork, candidate_cap}, x.options());
  auto cand_idx = at::empty({Rwork, candidate_cap}, x.options().dtype(at::kInt));
  auto cand_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));
  auto lt_val = at::empty({Rwork, candidate_cap}, x.options());
  auto lt_idx = at::empty({Rwork, candidate_cap}, x.options().dtype(at::kInt));
  auto lt_cnt = at::empty({Rwork}, x.options().dtype(at::kInt));

  launch_hopper_gqa_group_indexed_score_dense_ip_gmma_m64n8_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(kv_data.data_ptr<at::Half>()),
      group_indices.data_ptr<int>() + sample_start, (int)Hkv, (int)group, (int)M,
      (int)sample_size, (int)physical_pages, (int)D,
      reinterpret_cast<__half*>(sample_scores.data_ptr<at::Half>()), stream);
  launch_flash_topk_min_fp16(
      reinterpret_cast<const __half*>(sample_scores.data_ptr<at::Half>()),
      (int)Rwork, (int)sample_size, (int)k,
      reinterpret_cast<__half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), stream);
  launch_hopper_add_sample_idx_offset(
      sample_idx.data_ptr<int>(), (int)(Rwork * k), (int)sample_start, stream);
  launch_hopper_seed_from_sample_fp16(
      reinterpret_cast<const __half*>(sample_val.data_ptr<at::Half>()),
      sample_idx.data_ptr<int>(), (int)Rwork, (int)k, (int)buf_cap, (int)num_buckets,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      /*qprev=*/nullptr, /*refresh_lock=*/nullptr,
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), /*buf_bucket=*/nullptr,
      /*bucket_val=*/nullptr, /*bucket_idx=*/nullptr, /*bucket_cap=*/1, stream);
  launch_hopper_gqa_group_indexed_score_to_sparse_ip_gmma_m64n8_fp16(
      reinterpret_cast<const __half*>(xpad.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(kv_data.data_ptr<at::Half>()),
      group_indices.data_ptr<int>(), (int)Hkv, (int)group, (int)M,
      (int)sample_start, (int)physical_pages, (int)D, 0,
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(), th.data_ptr<int>(),
      qcount.data_ptr<int>(), bcount.data_ptr<int>(),
      reinterpret_cast<__half*>(buf_val.data_ptr<at::Half>()),
      buf_idx.data_ptr<int>(), (int)buf_cap, (int)num_buckets, (int)k, stream);
  launch_hopper_refresh_threshold_from_bcount(
      th.data_ptr<int>(), bcount.data_ptr<int>(),
      (int)Rwork, (int)num_buckets, (int)k, stream);
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
      idx_pad.data_ptr<int>(), false, stream);
  launch_hopper_negate_fp16(reinterpret_cast<__half*>(val_pad.data_ptr<at::Half>()),
                            (int)(Rwork * k), stream);

  auto val = val_pad.view({Hkv, qn8, k}).slice(1, 0, group).reshape({Hq, k}).contiguous();
  auto idx = idx_pad.view({Hkv, qn8, k}).slice(1, 0, group).reshape({Hq, k}).contiguous();
  return std::make_tuple(val, idx);
}

}  // namespace

TORCH_LIBRARY(tidal_hopper_topk, m) {
  // Retired experimental paths (measured slower than sparse; kept commented for
  // reference): fused_ip_dense, fused_ip_sparse_timed, fused_ip_smalln,
  // fused_ip_smalln_tc, fused_ip_smalln_dense_bucket_fused.
  // m.def("fused_ip_dense(Tensor x, Tensor base, int k, int num_buckets, int cap) -> (Tensor, Tensor)");
  // RETIRED for TidalDecode best path: generic 2D-base sparse op.
  // m.def("fused_ip_sparse(Tensor x, Tensor base, int k, int num_buckets, int buf_cap, int select_cap, int sample_size) -> (Tensor, Tensor)");
  // RETIRED for TidalDecode best path: generic 2D-base fp32-buffer sparse op.
  // m.def("fused_ip_sparse_fp32buf(Tensor x, Tensor base, int k, int num_buckets, int buf_cap, int select_cap, int sample_size) -> (Tensor, Tensor)");
  // m.def("fused_ip_sparse_timed(Tensor x, Tensor base, int k, int num_buckets, int buf_cap, int select_cap, int sample_size, int iters) -> Tensor");
  // m.def("fused_ip_smalln(Tensor x, Tensor base, int k) -> (Tensor, Tensor)");
  // m.def("fused_ip_smalln_tc(Tensor x, Tensor base, int k) -> (Tensor, Tensor)");
  // RETIRED for TidalDecode best path: generic small-N sparse op.
  // m.def("fused_ip_smalln_sparse(Tensor x, Tensor base, int k, int num_buckets, int buf_cap, int select_cap, int sample_size) -> (Tensor, Tensor)");
  // RETIRED for TidalDecode best path: generic small-N fp32-buffer sparse op.
  // m.def("fused_ip_smalln_sparse_fp32buf(Tensor x, Tensor base, int k, int num_buckets, int buf_cap, int select_cap, int sample_size) -> (Tensor, Tensor)");
  // RETIRED for TidalDecode best path: GQA non-paged KV op.
  // m.def("fused_ip_gqa_sparse(Tensor x, Tensor base3d, int k, int num_buckets, int buf_cap, int select_cap, int sample_size) -> (Tensor, Tensor)");
  // RETIRED for TidalDecode best path: paged KV op without group_indices.
  // m.def("fused_ip_gqa_sparse_paged(Tensor x, Tensor kv_data, int k, int num_buckets, int buf_cap, int select_cap, int sample_size) -> (Tensor, Tensor)");
  m.def("fused_ip_gqa_sparse_paged_group_indexed(Tensor x, Tensor kv_data, Tensor group_indices, int k, int num_buckets, int buf_cap, int sample_size) -> (Tensor, Tensor)");
  // m.def("fused_ip_smalln_dense_bucket_fused(Tensor x, Tensor base, int k, int num_buckets, int select_cap, int sample_size) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(tidal_hopper_topk, CUDA, m) {
  // m.impl("fused_ip_dense", TORCH_FN(fused_ip_dense));
  // m.impl("fused_ip_sparse", TORCH_FN(fused_ip_sparse));
  // m.impl("fused_ip_sparse_fp32buf", TORCH_FN(fused_ip_sparse_fp32buf));
  // m.impl("fused_ip_sparse_timed", TORCH_FN(fused_ip_sparse_timed));
  // m.impl("fused_ip_smalln", TORCH_FN(fused_ip_smalln));
  // m.impl("fused_ip_smalln_tc", TORCH_FN(fused_ip_smalln_tc));
  // m.impl("fused_ip_smalln_sparse", TORCH_FN(fused_ip_smalln_sparse));
  // m.impl("fused_ip_smalln_sparse_fp32buf", TORCH_FN(fused_ip_smalln_sparse_fp32buf));
  // m.impl("fused_ip_gqa_sparse", TORCH_FN(fused_ip_gqa_sparse));
  // m.impl("fused_ip_gqa_sparse_paged", TORCH_FN(fused_ip_gqa_sparse_paged));
  m.impl("fused_ip_gqa_sparse_paged_group_indexed", TORCH_FN(fused_ip_gqa_sparse_paged_group_indexed));
  // m.impl("fused_ip_smalln_dense_bucket_fused", TORCH_FN(fused_ip_smalln_dense_bucket_fused));
}

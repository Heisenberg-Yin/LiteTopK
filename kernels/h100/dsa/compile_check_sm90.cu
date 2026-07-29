// Compile-only check for the SM90 kernel template at the supported GLM-5
// shape (H=32, D=128, BLOCK_Q=4, BLOCK_KV=128, 4 KV
// stages, 132 SMs, 128 spec + 256 math threads) without the torch host.
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
//        -gencode=arch=compute_90a,code=sm_90a -I<DG>/deep_gemm/include \
//        -I<DG>/third-party/cutlass/include -c compile_check_sm90.cu
#include "sm90_dsa_litetopk.cuh"

void* dsa_litetopk_sm90_kernel_ptr() {
    return reinterpret_cast<void*>(
        &dsa_litetopk::sm90_dsa_litetopk<32u, 128u, 4u, 128u, 1u, 4u, 132u, 128u, 256u>);
}

// Build-time smem budget probe for the DSv4 bf16 kernel.
// Prints sizeof(SharedMemoryPlan) and every component so a regression in the
// layout atoms is visible before anything is launched.
//
//   nvcc -arch=sm_100a <include paths> probe_dsv4_smem.cu -o probe_dsv4_smem
#include <cstdio>
#include "litedsa_attention_sm100_dsv4.cuh"

int main() {
  using K = sm100::fwd::head128_dsv4::KernelTemplate<512>;
  using cute::cosize_v;
  printf("[dsv4-smem] D_QK=%d D_V=%d B_H=%d B_TOPK=%d NUM_BUFS=%d\n", K::D_Q,
         K::D_V, K::B_H, K::B_TOPK, K::NUM_BUFS);
  printf(
      "[dsv4-smem]   q_full = %8zu B\n",
      cosize_v<K::SmemLayoutQTiles<K::D_Q / 64>> * sizeof(cutlass::bfloat16_t));
  printf("[dsv4-smem]   k[2]   = %8zu B\n",
         2 * cosize_v<K::SmemLayoutKTiles<K::D_K / 64>> *
             sizeof(cutlass::bfloat16_t));
  printf("[dsv4-smem]   v[2]   = %8zu B\n",
         2 * cosize_v<K::SmemLayoutV> * sizeof(cutlass::bfloat16_t));
  printf("[dsv4-smem]   o      = %8zu B (union alternative)\n",
         cosize_v<K::SmemLayoutO> * sizeof(cutlass::bfloat16_t));
  printf("[dsv4-smem]   s[2]   = %8zu B\n",
         2 * cosize_v<K::SmemLayoutS> * sizeof(cutlass::bfloat16_t));
  printf(
      "[dsv4-smem] sizeof(SharedMemoryPlan) = %zu B (%.1f KiB), cap 232448 B\n",
      sizeof(K::SharedMemoryPlan), sizeof(K::SharedMemoryPlan) / 1024.0);
  if (sizeof(K::SharedMemoryPlan) > 232448) {
    printf("[dsv4-smem] *** OVER BUDGET ***\n");
    return 1;
  }
  printf("[dsv4-smem] headroom = %zu B\n",
         232448 - sizeof(K::SharedMemoryPlan));
  return 0;
}

# LiteTopK H100 源码与编译检查

本目录保存 B200 kernel 的 H100（SM90a）实现。当前主机只有 B200，因此这里仅
提供源码说明和 `sm_90a` 编译检查，不提供 H100 DSA 运行、正确性或性能复现。

## 代码

### DSA

| 文件 | 用途 |
|---|---|
| `dsa/sm90_dsa_litetopk.cuh` | 使用 WGMMA 和寄存器累加器的 SM90 scan kernel |
| `dsa/dsa_litetopk.cu` | H100 host wrapper、预处理和候选选择入口 |
| `dsa/compile_check_sm90.cu` | 不依赖 PyTorch 的生产 shape 模板实例化检查 |

H100 实现使用 WGMMA 和寄存器累加器；B200 实现使用 SM100 的 UMMA/TMEM。
两者不是同一个 cubin，也不能互换架构编译参数。

### MS MARCO

| 文件 | 用途 |
|---|---|
| `marsco/sm90_litetopk_marsco.cuh` | H100 inner-product scan kernel |
| `marsco/litetopk_sm90_torch.cu` | PyTorch host binding |
| `marsco/litetopk_select.cu`、`litetopk_select.h`、`litetopk_topk.h` | 候选选择 |
| `marsco/litetopk_ops.py` | SM90a JIT loader |
| `marsco/bench_marsco_h100.py` | H100 benchmark 入口 |

真实 H100 上的运行命令见
[`marsco/README.md`](marsco/README.md)。当前主机没有 H100，因此这里只完成
SM90a 编译，不把该命令列为运行验证。B200 MS MARCO 的可执行说明见
[`../b200/marsco/README.md`](../b200/marsco/README.md)。

## 依赖

编译检查需要：

- CUDA toolkit 中支持 `compute_90a`/`sm_90a` 的 NVCC；
- DeepGEMM 2.5 源码及其 CUTLASS 子模块；
- C++17 编译支持。

当前主机使用：

```text
容器：glm5-prefill
仓库：/data01/home/ziqi.yin/litetopk_github_clone
DeepGEMM：/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM
NVCC：/usr/local/cuda/bin/nvcc
```

编译检查不读取模型、DSA cache 或 MS MARCO 数据。

## `sm_90a` 编译检查

从宿主机运行：

```bash
docker exec \
  -w /data01/home/ziqi.yin/litetopk_github_clone/kernels/h100 \
  glm5-prefill bash -lc '
    DG=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM
    /usr/local/cuda/bin/nvcc \
      -O3 \
      -std=c++17 \
      --expt-relaxed-constexpr \
      --expt-extended-lambda \
      -gencode=arch=compute_90a,code=sm_90a \
      -I"$DG/deep_gemm/include" \
      -I"$DG/third-party/cutlass/include" \
      -Xptxas=-v \
      -c dsa/compile_check_sm90.cu \
      -o /tmp/litetopk_compile_check_sm90.o
  '
```

确认产物：

```bash
docker exec glm5-prefill \
  test -s /tmp/litetopk_compile_check_sm90.o
```

输出判据：

- 两条命令退出码均为 0；
- `/tmp/litetopk_compile_check_sm90.o` 存在且非空；
- NVCC 没有模板实例化、架构或 include 错误；
- `ptxas -v` 输出只用于检查编译资源信息，本文档不规定数值阈值。

该检查只证明源码能够生成 SM90a object，不证明 kernel 在 H100 上的数值正确性、
召回率或性能。

## H100 DSA 运行缺口

当前仓库的 vLLM Python 适配层以 B200 构建为默认路径，尚未提供经过 H100 实机
验证的 DSA loader 和运行命令。要建立 H100 DSA 复现，还需要：

1. 在 Python loader 中选择本目录源码和 `sm_90a` 架构；
2. 在真实 H100 上完成扩展加载；
3. 准备与
   [`../b200/dsa/README.md`](../b200/dsa/README.md)
   相同结构的 DSA cache；
4. 对照官方路径验证 top-k overlap 和数值边界；
5. 再单独记录性能。

上述步骤完成前，不应把编译检查描述为 H100 DSA 可运行验证。

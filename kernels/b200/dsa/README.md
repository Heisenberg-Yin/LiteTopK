# LiteTopK B200 DSA 复现手册

本目录提供 GLM-5 DSA prefill indexer 的 B200（SM100）CUDA 实现，以及直接使用
真实 cache 的正确性和计时入口。

## 代码

| 文件 | 用途 |
|---|---|
| `sm100_dsa_litetopk.cuh` | SM100 scan kernel |
| `dsa_litetopk.cu` | PyTorch CUDA 扩展、预处理、scan 和候选选择入口 |
| `bench_q8192.py` | Q=8192 的 indexer benchmark，对比 vLLM 官方 dense 路径 |
| `bench_whole_dsa.py` | indexer 与后续 sparse MLA attention 的组合 benchmark |
| `flashinfer_port/litedsa.so` | Whole-DSA 使用的预编译 LiteDSA 模块 |
| `../../../glm5_prefill/litetopk_vllm/litetopk_indexer.py` | JIT loader 和 vLLM 适配层 |

## 固定 B200 构建契约

本仓库的 B200 主路径只提供一套内建的 numerical-FP16 specialization，运行时
不需要传递编译 feature flags：

- global candidate 使用 16-bit IEEE FP16 score 和 32-bit packed
  bucket/index；
- shared-memory emit ring 使用 32-bit local record、256-KV-block window
  （W256）以及每 lane/row 18 个 slot（L18）；
- emit 使用 packed-prefix direct flush，局部 slot 耗尽时走 global
  fallback，不丢候选；
- accepted-score histogram 稀疏刷新 threshold；
- 最终 top-k 使用 in-place boundary selector。

刷新和 gate stride 也是源码内建配置，不是复现命令的可调参数。

### 6-byte 全局 candidate ABI

每个 candidate 由两个数组中的元素组成：

- `cand_val` 是 16-bit IEEE FP16 bucket-space score；
- `cand_idx` 是 32-bit packed index：低 20 bit 保存 KV 位置，bit 20--27
  保存从 FP32 score 计算出的精确 source bucket。

因此一个逻辑 candidate 占 6 bytes。精确 bucket 与 FP16 score 分开保存，
避免 FP16 在整数边界的舍入改变 threshold bucket 分类；最终 boundary bucket
内部的排序使用 FP16，所以 FP16 ties 可能选择与 FP32 reference 不同、但分数并列
的 index。KV 位置必须能放入 20 bit，即 `seq_len_kv <= 2^20`。

### local32 W256/L18 ring

scan kernel 的共享内存 ring 使用 32-bit local record：

```text
bits 15:0   FP16 score
bits 23:16  当前 256-block window 内的 block offset
bits 31:24  精确 source bucket
```

writer 的 warp/lane 提供 window 内剩余的 KV 坐标。每个
`(math warp, query row, lane)` 有 18 个 slot，window 为 256 个 KV block；
对应 ring 大小为 `8 * 4 * 32 * 18 * 4 = 73,728` bytes。lane 局部 slot
耗尽时走 direct global fallback，不因 local quota 丢 candidate；全局 candidate
buffer 仍必须足够大。

### sparse threshold 与适用范围

hot sample 只用于得到 recall-safe 的初始 threshold，不作为 seed 写入 candidate。
scan 随后覆盖完整 KV 范围，并用 accepted-score histogram 稀疏刷新 threshold。
内建的 zero-base refresh 路径依赖这个 hot-only、no-seed 契约：CTA histogram
直接从零开始，不读取无用的 global seed histogram。

最终 numerical FP16 路径要求：

- 单 KV split；CUDA binding 会检查 `num_kv_splits == 1`；
- merged query 至少有 8192 行；Python adapter 对 `Q < 8192` 返回 `False`，
  由 vLLM 执行官方 dense fallback；
- boundary selector 必须收到 sparse-refresh 发布的有效 boundary metadata；
- hot carry 尚未准备好时，adapter 返回 `False`，由上一段官方路径生成 carry 后
  再进入 LiteTopK。

## 依赖

- NVIDIA B200，计算能力 SM100a；
- 当前主机容器：`glm5-prefill`；
- CUDA toolkit 和 NVCC；
- `/opt/vllm-venv/bin/python` 环境中的 PyTorch、vLLM、DeepGEMM、
  safetensors、FlashInfer 和 TVM-FFI；
- DeepGEMM/CUTLASS headers。其他机器可设置 `DEEPGEMM_DIR` 指向包含
  `deep_gemm/include` 和 `third-party/cutlass/include` 的 DeepGEMM 源码树。

以下文档用 `REPO_DIR` 表示容器内可见的仓库路径，例如：

```text
REPO_DIR=/data01/home/ziqi.yin/litetopk_github_clone
```

JIT loader 通过以下变量定位 Python 模块、CUDA 源码和 DeepGEMM headers：

```text
LITETOPK_MODULE_DIR=$REPO_DIR/glm5_prefill/litetopk_vllm
LITETOPK_DSA_DIR=$REPO_DIR/kernels/b200/dsa
DEEPGEMM_DIR=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM
```

## 输入

`bench_q8192.py` 读取：

```text
/data/dsa_caches/glm5_256k_realtext_chunk8192.safetensors
/data/dsa_caches/glm5_512k_realtext_chunk8192.safetensors
/data/dsa_caches/glm5_768k_realtext_chunk8192.safetensors
/data/dsa_caches/glm5_1m_realtext_chunk8192.safetensors
```

这些文件至少需要包含：

```text
q_index
q_index_scale
idx_k_cache
idx_k_scale
gate_w
```

`bench_whole_dsa.py` 还需要：

```text
topk_idx
mla_q
mla_kv
metadata["mla"]
```

cache 不随仓库提供，仓库中也没有 cache 生成脚本。新机器必须自行准备相同结构的
safetensors 文件。

## 命令

以下命令从宿主机执行。第一次运行会 JIT 编译 CUDA 扩展。

先设置共用路径：

```bash
REPO_DIR=/data01/home/ziqi.yin/litetopk_github_clone
DEEPGEMM_DIR=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM
```

### B200 DSA indexer

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e PATH=/opt/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  -e DEEPGEMM_DIR="$DEEPGEMM_DIR" \
  -e DSA_CACHE_DIR=/data/dsa_caches \
  -e LITETOPK_MODULE_DIR="$REPO_DIR/glm5_prefill/litetopk_vllm" \
  -e LITETOPK_DSA_DIR="$REPO_DIR/kernels/b200/dsa" \
  -e VLLM_LITETOPK_OVF_LOG=1 \
  -w "$REPO_DIR/kernels/b200/dsa" \
  glm5-prefill /opt/vllm-venv/bin/python bench_q8192.py
```

脚本依次处理 256K、512K、768K 和 1M cache，并打印：

```text
S  Q  dense  litetopk  speedup  recall
```

输出判据：

- 四个输入都运行完成；
- 每个输入的 index recall 不低于 `99.9%`；
- 没有 extension build failure、fallback、candidate overflow、CUDA error
  或 boundary-metadata trap；
- numerical FP16 允许 boundary bucket 内的 ties 与 FP32 reference 选择不同，
  因此不要求 index byte-identical，也不要求 `recall` 恰好为 `100.00%`；
- `dense`、`litetopk` 和 `speedup` 仅用于解释本次运行，不设固定数值。

### Whole-DSA

Whole-DSA 在同一份 cache 上分别计时 indexer 和 sparse MLA attention，再将两段
时间相加。它不是一个融合 kernel，也不让 attention 直接消费本次 LiteTopK 生成的
`outi`；两条 attention 路径都使用 cache 中保存的 `topk_idx`，从而只比较
attention 实现本身。脚本报告：

- vLLM dense indexer + stock sparse attention；
- LiteTopK indexer + stock sparse attention；
- LiteTopK indexer + LiteDSA attention。

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=3 \
  -e PATH=/opt/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  -e DEEPGEMM_DIR="$DEEPGEMM_DIR" \
  -e TAGS="768k 1m" \
  -e DSA_CACHE_DIR=/data/dsa_caches \
  -e LITETOPK_MODULE_DIR="$REPO_DIR/glm5_prefill/litetopk_vllm" \
  -e LITETOPK_DSA_DIR="$REPO_DIR/kernels/b200/dsa" \
  -e LITEDSA_SO="$REPO_DIR/kernels/b200/dsa/flashinfer_port/litedsa.so" \
  -e VLLM_LITETOPK_OVF_LOG=1 \
  -w "$REPO_DIR/kernels/b200/dsa" \
  glm5-prefill /opt/vllm-venv/bin/python bench_whole_dsa.py
```

脚本打印 indexer overlap `idx%`、attention 最大绝对误差 `attn_maxdiff` 以及各
路径计时。当前脚本只报告这两个正确性字段，没有内置通过阈值，因此：

- 运行复现要求两个 TAG 均完成且没有 CUDA/加载错误；
- `idx%` 和 `attn_maxdiff` 必须随测试记录保存；
- 在用于回归测试前，应由调用方明确 overlap 和数值误差阈值；
- 延迟和 speedup 不作为正确性判据。

## 路径与构建问题

- 若扩展找不到源码，检查 `LITETOPK_DSA_DIR`；
- 若扩展找不到 DeepGEMM/CUTLASS headers，设置 `DEEPGEMM_DIR`；
- 若需要隔离 JIT cache，设置 `VLLM_LITETOPK_BUILD` 为新的可写目录；
- 修改 CUDA 源码或内建构建常量后，应使用新的 JIT build cache 并重新运行
  正确性检查；
- 若 Whole-DSA 无法加载模块，检查 `LITEDSA_SO` 是否存在，并确认其为 SM100
  构建。

LiteDSA 的源码与构建说明位于
[`flashinfer_port/README.md`](flashinfer_port/README.md)。

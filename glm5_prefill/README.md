# GLM FP8 prefill 复现

本目录把 B200 LiteTopK/LiteDSA 接到 vLLM 的 GLM 稀疏 prefill 路径。这里
不包含模型权重、输入语料或 vLLM 本体；运行前需要准备这些外部资产。

## 文件

- `build_fp8_truncated.py`：从分片 FP8 checkpoint 构造前 N 层模型。
- `run_prefill.py`：读取 Parquet 文本，执行一次预热和若干次 prefill。
- `run_e2e.sh`：从宿主机调用 Docker，设置 LiteTopK 的固定复现参数。
- `litetopk_vllm/litetopk_indexer.py`：vLLM indexer hook 调用的 B200 adapter。
- `litetopk_vllm/litedsa_attn.py`：可选的 LiteDSA grouped-attention adapter。

## 前置条件

当前主机的示例使用：

- 容器：`glm5-prefill`
- Python：`/opt/vllm-venv/bin/python`
- 仓库：`/data01/home/ziqi.yin/litetopk_github_clone`
- 文本：`/models/glm5/train-00000-of-00002.parquet`
- B200 GPU 和 CUDA 12.8

容器中当前可见的软件版本如下。仓库不包含容器镜像，迁移时应记录等价环境：

| 软件 | 版本 |
|---|---|
| Python | 3.12 |
| vLLM | 0.23.0 |
| PyTorch | 2.11.0+cu130 |
| DeepGEMM | 2.5.0 |
| FlashInfer | 0.6.12 |
| TVM-FFI | 0.1.9 |
| safetensors | 0.8.0 |
| NVCC | 12.8 |

此外还需要：

1. 一个 vLLM v0.23.0（commit `0fc695f`）环境，并应用
   [`vllm_patches`](../vllm_patches/README.md) 中与本次运行对应的补丁。
2. DeepGEMM 2.5 及其 CUDA/CUTLASS headers。
3. 容器内可见的 GLM FP8 checkpoint。
4. Parquet 文件包含字符串类型的 `text` 列。

当前 launcher 的 `DEEPGEMM_DIR` 默认值是
`/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM`。迁移到其他机器时，应把
它改为容器内可见、同时包含 `deep_gemm/include` 与
`third-party/cutlass/include` 的 DeepGEMM 源码树。

补丁应应用到干净的 vLLM 副本。当前 `glm5-prefill` 容器内的 vLLM 还含有
其他项目的本地修改，不应把它当作补丁基线。

## 可选：构造截断模型

目标目录应为空或专用于本次构建。完整保留的 shard 会使用符号链接，因此
构建完成后不能移动或删除源 checkpoint。

```bash
docker exec \
  -e MODEL_SRC=/models/glm5-fp8-official \
  -e MODEL_AUX=/models/glm5 \
  -w /data01/home/ziqi.yin/litetopk_github_clone/glm5_prefill \
  glm5-prefill /opt/vllm-venv/bin/python \
  build_fp8_truncated.py 16 /models/glm5-fp8-16l
```

脚本会生成 `config.json`、`model.safetensors.index.json`，重写包含被截断
layer 的混合 shard，并为 `shared` indexer layer 补齐参数名。

## 运行 stock vLLM

先用不启用 LiteTopK 的命令确认模型、语料和 vLLM 环境可用：

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e MODEL=/models/glm5-fp8-16l \
  -e PARQUET=/models/glm5/train-00000-of-00002.parquet \
  -e TP=4 \
  -e LENGTHS="4096 32768" \
  -e CHUNK=8192 \
  -e REPEATS=1 \
  -e VLLM_USE_DEEP_GEMM=1 \
  -w /data01/home/ziqi.yin/litetopk_github_clone/glm5_prefill \
  glm5-prefill /opt/vllm-venv/bin/python run_prefill.py
```

## 运行 LiteTopK

`run_e2e.sh` 是宿主机脚本。`MODEL` 必须是容器内可见路径；`REPO_DIR`
也表示容器内的仓库路径。

```bash
cd /data01/home/ziqi.yin/litetopk_github_clone/glm5_prefill
MODEL=/models/glm5-fp8-16l \
DEEPGEMM_DIR=/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM \
DEVICES=0,1,2,3 \
TP=4 \
LENGTHS="262144 524288" \
./run_e2e.sh
```

如果仓库在容器内挂载到另一位置：

```bash
MODEL=/models/glm5-fp8-16l \
REPO_DIR=/workspace/litetopk_github_clone \
./run_e2e.sh
```

launcher 还接受 `CONTAINER`（默认 `glm5-prefill`）、`DEVICES`（默认 8 张
GPU）、`TP`、`LOG`、`DEEPGEMM_DIR` 和下表中的 `run_prefill.py` 变量。
所有路径都必须是容器内可见路径。

脚本默认启用 LiteTopK、关闭 LiteDSA，并使用源码内建的 B200
numerical-FP16 specialization：6-byte global candidate、local32 W256/L18
emit ring、sparse refresh 和 in-place boundary selector。launcher 不传递
编译 feature selections。ABI 和适用范围见
[`kernels/b200/dsa/README.md`](../kernels/b200/dsa/README.md)。

启用 grouped attention：

```bash
MODEL=/models/glm5-fp8-16l \
VLLM_LITETOPK_LITEDSA=1 \
LITEDSA_SO=/data01/home/ziqi.yin/litetopk_github_clone/kernels/b200/dsa/flashinfer_port/litedsa.so \
./run_e2e.sh
```

单独验证 indexer adapter 使用
[`bench_q8192.py`](../kernels/b200/dsa/README.md#b200-dsa-indexer)。该入口会
直接导入 `litetopk_indexer.py`，并在当前支持的 merged query shape 上与稠密
参考路径比较。

## `run_prefill.py` 环境变量

| 变量 | 默认值 | 作用 |
|---|---:|---|
| `MODEL` | `/models/glm5-prefill-model` | checkpoint 目录 |
| `PARQUET` | `/models/glm5/train-00000-of-00002.parquet` | 文本输入 |
| `LENGTHS` | `4096 32768 131072 262144` | prompt token 数 |
| `TP` | `4` | tensor parallel 数 |
| `CHUNK` | `8192` | chunked-prefill token 上限 |
| `REPEATS` | `3` | 每个长度的计时次数 |
| `GPU_UTIL` | `0.90` | vLLM GPU memory utilization |
| `KVDTYPE` | `auto` | KV cache dtype |
| `MTP` | `0` | 设为 `1` 启用 MTP |
| `EAGER` | `0` | 设为 `1` 使用 eager execution |
| `ASYNC_SCHED` | `0` | 设为 `1` 启用 async scheduling |
| `OUTPUT` | `glm5_prefill/prefill_results.json` | JSON 输出路径 |

`run_prefill.py` 固定关闭 prefix caching，使重复 prompt 每次都实际执行
prefill。JSON 保存模型路径、TP、chunk 大小和各长度的最短 wall time。

## LiteTopK 关键变量

`run_e2e.sh` 给出一套固定默认值。需要改变内存占用或启用诊断时，可覆盖：

- `DEEPGEMM_DIR`：包含 DeepGEMM 与 CUTLASS headers 的源码树。
- `VLLM_LITETOPK_MIN_S`：进入 LiteTopK 的最小 KV 长度。
- `VLLM_LITETOPK_MERGE_CAP`：merged call 每行 candidate 容量。
- `VLLM_LITETOPK_HOTSAMPLE`：从前一份 top-k 结果保留的校准位置数。
- `VLLM_LITETOPK_CHECK=1`：同时生成稠密参考结果并打印 recall。
- `VLLM_LITETOPK_OVF_LOG=1`：打印 candidate 容量监控。
- `VLLM_LITETOPK_LITEDSA=1`：启用 LiteDSA attention hook。

若出现 `candidate overflow`，该次调用的 candidate buffer 已不足；应增大
对应 cap 后重新做正确性验证。

numerical FP16 保留精确 source-bucket 分类，但 boundary bucket 内按 FP16
排序，因此不要求与 FP32 reference 的 index byte-identical。独立 indexer
回归以四个 cache 均完成、每个 shape 的 recall 不低于 `99.9%`，且没有 build
failure、fallback、candidate overflow、CUDA error 或 boundary-metadata trap
为通过条件。E2E 运行还应确认日志中实际加载了 LiteTopK extension，而不是静默走
官方 dense fallback。

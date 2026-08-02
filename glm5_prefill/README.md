# GLM-5.2 / LongCat 端到端 A/B

本目录只保留 GLM-5.2 与 LongCat 的 native-vLLM prefill 端到端测试。

唯一入口是 `run_e2e.sh`。它依次运行相同配置的 raw 与 LiteTopK，校验生成
token 和运行配置，并输出端到端加速比。

## 文件

| 文件 | 用途 |
|---|---|
| `run_e2e.sh` | 端到端 raw/LiteTopK A/B 入口，并在容器中依次启动两条路径 |
| `run_prefill.py` | 加载模型、执行 prefill、计时并记录源码指纹 |
| `compare_results.py` | 校验两条路径并生成 speedup summary |

依赖关系：

```text
run_e2e.sh
├── run_prefill.py
└── compare_results.py
```

## 运行

```bash
cd /data01/home/ziqi.yin/litetopk/glm5_prefill

MODEL_FAMILY=glm5.2 REPEATS=2 ./run_e2e.sh
MODEL_FAMILY=longcat REPEATS=2 ./run_e2e.sh
```

默认配置为 TP8、chunk 8192、GPU util 0.90、FP8 KV、2 GiB indexer logits
budget，并开启 MTP。GLM-5.2 默认测试 1,048,512 tokens、MTP=5；LongCat
默认测试 974,848 tokens、MTP=3。

可用环境变量覆盖 `MODEL`、`LENGTHS`、`TP`、`REPEATS`、`GPU_UTIL`、
`KV_BLOCKS`、`MTP_K`、`RESULT_DIR` 等参数。默认加载：

- vLLM：`/data01/home/ziqi.yin/vllm-litetopk-longcat`
- CUDA kernel：`../kernels/b200/dsa`
- 容器：`glm5-prefill`

结果自动写入 `results/<UTC时间>-<模型>/`；该目录是运行产物，不纳入仓库。
若设置 `PROF_DIR`，指定 trial 会同时采集 Torch profiler，但正式 best/median
会排除该 trial。

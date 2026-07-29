# vLLM v0.23.0 补丁

本目录保存 LiteTopK/LiteDSA 对 vLLM 的三个 Python 补丁。基线是 vLLM
v0.23.0 commit `0fc695f`。不同版本或带有本地修改的文件不能直接假定兼容。

## 补丁内容

- `sparse_attn_indexer.diff`
  - 在 sparse indexer 中加入 LiteTopK hook。
  - 支持 merged prefill workspace 和跨调用的 top-k carry。
  - 每次写入 top-k buffer 时递增版本号，供 LiteDSA 缓存使用。
- `mla_indexer.diff`
  - 用 `VLLM_LITETOPK_RUNTIME_LOGITS_MB` 将运行时 logits chunk 上限与
    初始化时的 workspace 预留解耦。
- `flashinfer_mla_sparse.diff`
  - 在满足 B200、dtype 和 shape 条件时调用 LiteDSA grouped attention。
  - hook 不可用或拒绝该 shape 时继续执行 stock path。

## 应用到源码 checkout

从干净 checkout 开始。checkout 根目录本身包含 `vllm/` package，因此不剥离
补丁路径，使用 `-p0`：

```bash
VLLM_SRC=/path/to/vllm
PATCH_DIR=/data01/home/ziqi.yin/litetopk_github_clone/vllm_patches

git -C "$VLLM_SRC" checkout 0fc695f
git -C "$VLLM_SRC" status --short

for name in sparse_attn_indexer mla_indexer flashinfer_mla_sparse; do
  patch --dry-run --batch -d "$VLLM_SRC" -p0 < "$PATCH_DIR/$name.diff"
done

for name in sparse_attn_indexer mla_indexer flashinfer_mla_sparse; do
  patch --batch -d "$VLLM_SRC" -p0 < "$PATCH_DIR/$name.diff"
done

git -C "$VLLM_SRC" diff --check
```

`git status --short` 在应用前应为空。先完成全部 dry-run；不要在 dry-run
一部分成功后立即逐个应用，否则后续失败时目标树会处于半应用状态。

如需只运行 LiteTopK indexer，可只应用 `sparse_attn_indexer.diff`；merged
长序列还需要 `mla_indexer.diff`。只有启用 LiteDSA 时才需要
`flashinfer_mla_sparse.diff`。

## 应用到已安装的 wheel

先确认 wheel 是相同 vLLM 基线并备份环境。`SITE_PACKAGES` 的直接子目录应
包含 `vllm/`，此时不剥离路径：

```bash
SITE_PACKAGES=/opt/vllm-venv/lib/python3.12/site-packages
PATCH_DIR=/data01/home/ziqi.yin/litetopk_github_clone/vllm_patches

for name in sparse_attn_indexer mla_indexer flashinfer_mla_sparse; do
  patch --dry-run --batch -d "$SITE_PACKAGES" -p0 < "$PATCH_DIR/$name.diff"
done

for name in sparse_attn_indexer mla_indexer flashinfer_mla_sparse; do
  patch --batch -d "$SITE_PACKAGES" -p0 < "$PATCH_DIR/$name.diff"
done
```

不要把补丁覆盖到已经被其他项目修改的共享 wheel。若 dry-run 失败，先换成
干净 v0.23.0 环境，而不是使用模糊匹配强制应用。

## 运行环境

最少需要显式设置：

```bash
export LITETOPK_MODULE_DIR=/path/to/litetopk_github_clone/glm5_prefill/litetopk_vllm
export LITETOPK_DSA_DIR=/path/to/litetopk_github_clone/kernels/b200/dsa
export VLLM_LITETOPK=1
```

启用 LiteDSA 时再设置：

```bash
export VLLM_LITETOPK_LITEDSA=1
export LITEDSA_SO=/path/to/litetopk_github_clone/kernels/b200/dsa/flashinfer_port/litedsa.so
```

其他运行参数和完整命令见
[`glm5_prefill/README.md`](../glm5_prefill/README.md)。

## 反向移除

源码 checkout 推荐直接丢弃这三个已知文件的改动，或在确认没有其他改动后
反向应用：

```bash
for name in flashinfer_mla_sparse mla_indexer sparse_attn_indexer; do
  patch --dry-run --batch -R -d "$VLLM_SRC" -p0 < "$PATCH_DIR/$name.diff"
done
for name in flashinfer_mla_sparse mla_indexer sparse_attn_indexer; do
  patch --batch -R -d "$VLLM_SRC" -p0 < "$PATCH_DIR/$name.diff"
done
```

对 wheel 使用相同顺序，将目录改为 `$SITE_PACKAGES`；层级仍为 `-p0`。

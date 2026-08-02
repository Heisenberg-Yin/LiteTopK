#!/usr/bin/env python3
"""Run a reproducible long-context vLLM prefill benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REPO_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "LITETOPK_DSA_DIR", str(REPO_DIR / "kernels" / "b200" / "dsa")
)


def _positive_int(name: str, default: str) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _git_state(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    repo = Path(path)
    if not (repo / ".git").exists():
        return {"path": str(repo), "git": False}

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    try:
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
            text=False,
        )
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
        return {
            "path": str(repo),
            "git": True,
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(diff or untracked),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked": untracked,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"path": str(repo), "git": True, "error": str(exc)}


def _litetopk_source_state() -> dict[str, Any]:
    source_dir = Path(os.environ.get("LITETOPK_DSA_DIR", "")).resolve()
    files = (
        "dsa_litetopk.cu",
        "sm100_dsa_litetopk.cuh",
        "dense_topk_litetopk.cuh",
    )
    hashes = {}
    source_digest = hashlib.sha256()
    for name in files:
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing LiteTopK CUDA source: {path}")
        contents = path.read_bytes()
        hashes[name] = hashlib.sha256(contents).hexdigest()
        source_digest.update(name.encode())
        source_digest.update(contents)
    integration_hashes = {}
    vllm_source = Path(os.environ.get("LITETOPK_VLLM_SRC", "")).resolve()
    integration_files = (
        "vllm/model_executor/layers/litetopk_indexer.py",
        "vllm/model_executor/layers/dsa_litetopk.py",
        "vllm/model_executor/layers/litedsa.py",
        "vllm/model_executor/layers/sparse_attn_indexer.py",
        "vllm/model_executor/models/deepseek_v2.py",
        "vllm/model_executor/models/longcat_flash.py",
        "vllm/model_executor/models/longcat_flash_mtp.py",
        "vllm/model_executor/models/longcat_flash_ngram.py",
        "vllm/model_executor/models/registry.py",
        "vllm/transformers_utils/configs/longcat_flash.py",
        "vllm/v1/attention/backends/mla/indexer.py",
        "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
        "vllm/envs.py",
        "vllm/models/deepseek_v32/nvidia/attention.py",
        "vllm/models/deepseek_v32/nvidia/kernels.py",
    )
    for name in integration_files:
        path = vllm_source / name
        if not path.is_file():
            raise FileNotFoundError(f"missing native vLLM integration: {path}")
        integration_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    enabled = os.environ.get("VLLM_LITETOPK", "0") == "1"
    configured_headroom = float(
        os.environ.get("VLLM_LITETOPK_HEADROOM", "0.0")
    )
    family = os.environ.get("MODEL_FAMILY", "unknown").lower()
    effective_headroom = (
        max(configured_headroom, 0.5)
        if family == "longcat"
        else configured_headroom
    )
    return {
        "enabled": enabled,
        "source_dir": str(source_dir),
        "source_id": source_digest.hexdigest()[:12],
        "source_sha256": hashes,
        "integration_sha256": integration_hashes,
        "min_s": int(os.environ.get("VLLM_LITETOPK_MIN_S", "196608")),
        "dense_select": (
            os.environ.get("VLLM_LITETOPK_DENSE_SELECT", "1") == "1"
        ),
        "dense_select_min_s": int(
            os.environ.get("VLLM_LITETOPK_DENSE_SELECT_MIN_S", "40960")
        ),
        "dense_select_max_s": int(
            os.environ.get("VLLM_LITETOPK_DENSE_SELECT_MAX_S", "262144")
        ),
        "dense_select_bins": int(
            os.environ.get("VLLM_LITETOPK_DENSE_SELECT_BINS", "4096")
        ),
        "dense_select_min_logits_mb": int(
            os.environ.get(
                "VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB", "0"
            )
        ),
        "merge_cap": int(os.environ.get("VLLM_LITETOPK_MERGE_CAP", "24576")),
        "hot_sample": int(os.environ.get("VLLM_LITETOPK_HOTSAMPLE", "8192")),
        "hot_only": os.environ.get("VLLM_LITETOPK_HOTONLY", "1") == "1",
        "headroom": configured_headroom,
        "effective_headroom": effective_headroom if enabled else None,
        "prep_tile": int(os.environ.get("VLLM_LITETOPK_PREP_TILE", "2048")),
        "prep_untiled_max_mb": int(
            os.environ.get("VLLM_LITETOPK_PREP_UNTILED_MAX_MB", "512")
        ),
        "carry_row_stride": int(
            os.environ.get("VLLM_LITETOPK_CARRY_ROW_STRIDE", "8")
        ),
        "carry_stride16_max_nv": int(
            os.environ.get("VLLM_LITETOPK_CARRY_STRIDE16_MAX_NV", "131072")
        ),
        "carry_custom_topk": (
            os.environ.get("VLLM_LITETOPK_CARRY_CUSTOM_TOPK", "1") == "1"
        ),
        "cache_merged_ke": (
            os.environ.get("VLLM_LITETOPK_CACHE_MERGED_KE", "1") == "1"
        ),
        "dedup_carry_wait": (
            os.environ.get("VLLM_LITETOPK_DEDUP_CARRY_WAIT", "1") == "1"
        ),
        "fused_hot_gather": (
            os.environ.get("VLLM_LITETOPK_FUSED_HOT_GATHER", "1") == "1"
        ),
        "litedsa_reuse_output_buffers": (
            os.environ.get("VLLM_LITEDSA_REUSE_OUTPUT_BUFS", "0") == "1"
        ),
        "litedsa_dynamic_span": (
            os.environ.get("VLLM_LITEDSA_DYNAMIC_SPAN", "1") == "1"
        ),
        "litedsa_union_override": os.environ.get("VLLM_LITEDSA_UNION_SO"),
        "probe_every": int(
            os.environ.get("VLLM_LITETOPK_PROBE_EVERY", "8")
        ),
        "check": os.environ.get("VLLM_LITETOPK_CHECK", "0") == "1",
        "overflow_log": (
            os.environ.get("VLLM_LITETOPK_OVF_LOG", "1") == "1"
        ),
    }


def _load_model_config(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    expected = {
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"model is outside the native LiteTopK ABI: {details}")
    return config


def _extract_token_ids(output: Any) -> list[int]:
    if not output or not output[0].outputs:
        return []
    return list(output[0].outputs[0].token_ids)


MODEL = Path(os.environ["MODEL"]).resolve()
PARQUET = Path(
    os.environ.get(
        "PARQUET",
        "/data01/home/ziqi.yin/glm5/train-00000-of-00002.parquet",
    )
).resolve()
MODE = os.environ.get("VLLM_DSA_MODE", "raw").lower()
FAMILY = os.environ.get("MODEL_FAMILY", "unknown").lower()
LENGTHS = [
    int(value)
    for value in os.environ.get(
        "LENGTHS", "262144 524288 786432 1048512"
    ).split()
]
TP = _positive_int("TP", "8")
CHUNK = _positive_int("CHUNK", "8192")
REPEATS = _positive_int("REPEATS", "2")
MAX_SEQS = _positive_int("MAX_SEQS", "1")
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.90"))
MTP_ENABLED = os.environ.get("MTP", "0") == "1"
MTP_K = _positive_int("MTP_K", "3")
KV_BLOCKS = int(os.environ["KV_BLOCKS"]) if os.environ.get("KV_BLOCKS") else None
OUTPUT = Path(os.environ["OUTPUT"]).resolve()
VLLM_SRC = os.environ.get("LITETOPK_VLLM_SRC")
FLASHINFER_ALLREDUCE_BACKEND = os.environ.get(
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "trtllm"
)
PROFILER_ROOT = (
    Path(os.environ["PROF_DIR"]).resolve()
    if os.environ.get("PROF_DIR")
    else None
)
PROFILE_TRIAL = int(os.environ.get("PROFILE_TRIAL", str(REPEATS - 1)))


def main() -> None:
    if MODE not in {"raw", "litetopk", "litedsa"}:
        raise ValueError("VLLM_DSA_MODE must be raw, litetopk, or litedsa")
    if FAMILY not in {"glm5.2", "longcat"}:
        raise ValueError("MODEL_FAMILY must be glm5.2 or longcat")
    if not VLLM_SRC:
        raise ValueError("LITETOPK_VLLM_SRC must point to the native vLLM source")
    if not MODEL.is_dir():
        raise FileNotFoundError(f"MODEL directory does not exist: {MODEL}")
    if not PARQUET.is_file():
        raise FileNotFoundError(f"PARQUET file does not exist: {PARQUET}")
    if not LENGTHS or any(length < 1 for length in LENGTHS):
        raise ValueError("LENGTHS must contain positive integers")
    if not 0.0 < GPU_UTIL <= 1.0:
        raise ValueError("GPU_UTIL must be in (0, 1]")
    if PROFILER_ROOT is not None:
        if len(LENGTHS) != 1:
            raise ValueError("PROF_DIR requires exactly one prompt length")
        if REPEATS < 2:
            raise ValueError("PROF_DIR requires REPEATS>=2 for a clean trial")
        if PROFILE_TRIAL < 0 or PROFILE_TRIAL >= REPEATS:
            raise ValueError(
                f"PROFILE_TRIAL={PROFILE_TRIAL} is outside REPEATS={REPEATS}"
            )

    model_config = _load_model_config(MODEL)
    architectures = model_config.get("architectures") or []
    expected_architecture = {
        "glm5.2": "GlmMoeDsaForCausalLM",
        "longcat": "LongcatCausalLM",
    }[FAMILY]
    if expected_architecture not in architectures:
        raise ValueError(
            f"MODEL_FAMILY={FAMILY} expects {expected_architecture}, "
            f"but the checkpoint declares {architectures!r}"
        )
    actual_streaming_tokens = (
        int(model_config.get("index_init_tokens") or 0),
        int(model_config.get("index_local_tokens") or 0),
    )
    expected_streaming_tokens = {
        "glm5.2": (0, 0),
        "longcat": (16, 1024),
    }[FAMILY]
    if actual_streaming_tokens != expected_streaming_tokens:
        raise ValueError(
            f"MODEL_FAMILY={FAMILY} expects index_init_tokens/"
            f"index_local_tokens={expected_streaming_tokens}, but the "
            f"checkpoint declares {actual_streaming_tokens}"
        )
    model_limit = int(model_config.get("max_position_embeddings", 1048576))
    max_len = min(max(LENGTHS) + 64, model_limit)
    effective_lengths = [min(length, max_len - 64) for length in LENGTHS]
    if len(set(effective_lengths)) != len(effective_lengths):
        raise ValueError(
            f"LENGTHS collapse after applying model limit {model_limit}: "
            f"{effective_lengths}"
        )

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    loaded_vllm = Path(vllm.__file__).resolve()
    expected_vllm = Path(VLLM_SRC or "").resolve()
    if not loaded_vllm.is_relative_to(expected_vllm):
        raise RuntimeError(
            f"loaded vLLM from {loaded_vllm}, expected source under "
            f"{expected_vllm}"
        )

    litetopk_runtime = _litetopk_source_state()
    if MODE in {"litetopk", "litedsa"}:
        from vllm.model_executor.layers import litetopk_indexer
        from vllm.model_executor.layers.dsa_litetopk import (
            dsa_litetopk_latest_available,
        )

        if not dsa_litetopk_latest_available():
            raise RuntimeError(
                f"VLLM_DSA_MODE={MODE} requested, but the native "
                "high24-v1 LiteTopK adapter is unavailable"
            )
        extension = litetopk_indexer._ext()
        if (
            extension is None
            or not extension.candidate_value_u16_litetopk()
            or not extension.candidate_fp24_global_litetopk()
        ):
            raise RuntimeError(
                "the repository LiteTopK extension failed to load its "
                "U16 high24 baseline ABI"
            )
        loaded_source = Path(litetopk_indexer._DSA_DIR).resolve()
        configured_source = Path(litetopk_runtime["source_dir"])
        native_source_id = litetopk_indexer._dsa_source_id()
        if loaded_source != configured_source:
            raise RuntimeError(
                f"vLLM loaded LiteTopK source from {loaded_source}, "
                f"expected {configured_source}"
            )
        if native_source_id != litetopk_runtime["source_id"]:
            raise RuntimeError(
                f"vLLM source digest {native_source_id} does not match "
                f"runner digest {litetopk_runtime['source_id']}"
            )
        if native_source_id not in Path(extension.__file__).name:
            raise RuntimeError(
                f"extension {extension.__file__} does not match "
                f"source {native_source_id}"
            )
        if (
            litetopk_indexer.DENSE_SELECT
            and not hasattr(extension, "dense_topk_litetopk_")
        ):
            raise RuntimeError(
                "LiteTopK dense selector requested, but its CUDA ops "
                "are absent from the loaded extension"
            )
        litetopk_runtime["extension"] = extension.__file__
        litetopk_runtime["extension_sha256"] = hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest()
        litetopk_runtime["candidate_u16"] = True
        litetopk_runtime["candidate_high24"] = True
        smem_attr_once = getattr(extension, "smem_attr_once_litetopk", None)
        litetopk_runtime["smem_attr_once"] = (
            bool(smem_attr_once()) if smem_attr_once is not None else None
        )
        from vllm.v1.attention.backends.mla.indexer import (
            _configured_litetopk_fused_min_seq_len,
        )

        litetopk_runtime["planner_fused_min_seq_len"] = (
            _configured_litetopk_fused_min_seq_len()
        )
    if MODE == "litedsa":
        from vllm.model_executor.layers.litedsa import litedsa_available

        if not litedsa_available():
            raise RuntimeError(
                "VLLM_DSA_MODE=litedsa requested, but the grouped SM100 "
                "LiteDSA operators are unavailable"
            )
        litetopk_runtime["litedsa_available"] = True
        stable_extensions = sorted(
            Path(vllm.__file__).resolve().parent.glob(
                "_C_stable_libtorch*.so"
            )
        )
        if len(stable_extensions) != 1:
            raise RuntimeError(
                "expected exactly one vLLM stable-libtorch extension, got "
                f"{stable_extensions}"
            )
        stable_extension = stable_extensions[0]
        litetopk_runtime["litedsa_native_extension"] = str(
            stable_extension
        )
        litetopk_runtime["litedsa_native_extension_sha256"] = (
            hashlib.sha256(stable_extension.read_bytes()).hexdigest()
        )
        litedsa_sources = (
            "csrc/libtorch_stable/attention/dsa/litedsa.cu",
            "csrc/libtorch_stable/attention/dsa/include/flashinfer/"
            "litedsa/litedsa_union.cuh",
            "vllm/model_executor/layers/litedsa.py",
            "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
        )
        litetopk_runtime["litedsa_source_sha256"] = {
            name: hashlib.sha256(
                (Path(VLLM_SRC) / name).read_bytes()
            ).hexdigest()
            for name in litedsa_sources
        }
        union_override = os.environ.get("VLLM_LITEDSA_UNION_SO")
        if union_override:
            union_override_path = Path(union_override).resolve()
            if not union_override_path.is_file():
                raise FileNotFoundError(
                    f"LiteDSA union override not found: {union_override_path}"
                )
            litetopk_runtime["litedsa_union_override"] = str(
                union_override_path
            )
            litetopk_runtime["litedsa_union_override_sha256"] = (
                hashlib.sha256(union_override_path.read_bytes()).hexdigest()
            )

    model_layers = model_config.get(
        "num_hidden_layers", model_config.get("num_layers")
    )
    print(
        f"[runtime] family={FAMILY} mode={MODE} vllm={vllm.__version__} "
        f"module={vllm.__file__}",
        flush=True,
    )
    print(
        f"[model] architectures={model_config.get('architectures')} "
        f"layers={model_layers} "
        f"max_len={max_len}",
        flush=True,
    )

    llm_kwargs: dict[str, Any] = {
        "model": str(MODEL),
        "tensor_parallel_size": TP,
        "enable_expert_parallel": True,
        "max_model_len": max_len,
        "max_num_batched_tokens": CHUNK,
        "max_num_seqs": MAX_SEQS,
        "gpu_memory_utilization": GPU_UTIL,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "kv_cache_dtype": os.environ.get("KVDTYPE", "fp8"),
        "enforce_eager": os.environ.get("EAGER", "0") == "1",
        "async_scheduling": os.environ.get("ASYNC_SCHED", "0") == "1",
    }
    if KV_BLOCKS is not None:
        llm_kwargs["num_gpu_blocks_override"] = KV_BLOCKS
    if MTP_ENABLED:
        llm_kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": MTP_K,
        }
    attention_backend = os.environ.get("ATTN_BACKEND")
    if attention_backend:
        llm_kwargs["attention_config"] = {"backend": attention_backend}
    profiler_dir = PROFILER_ROOT / MODE if PROFILER_ROOT is not None else None
    if profiler_dir is not None:
        profiler_dir.mkdir(parents=True, exist_ok=True)
        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profiler_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": True,
            "torch_profiler_dump_cuda_time_total": True,
            "ignore_frontend": True,
        }

    load_start = time.perf_counter()
    llm = LLM(**llm_kwargs)
    load_seconds = time.perf_counter() - load_start
    print(f"[load] model ready in {load_seconds:.1f}s", flush=True)

    tokenizer = llm.get_tokenizer()
    texts: list[str] = []
    character_count = 0
    for text in pq.read_table(PARQUET, columns=["text"])["text"].to_pylist():
        if text and text.strip():
            texts.append(text)
            character_count += len(text)
        if character_count > max_len * 8:
            break
    corpus_ids = tokenizer.encode("\n".join(texts))
    required = max(effective_lengths)
    if len(corpus_ids) < required:
        raise RuntimeError(
            f"input corpus has {len(corpus_ids)} tokens, but {required} are required"
        )
    print(f"[data] corpus tokens available: {len(corpus_ids)}", flush=True)

    sampling = SamplingParams(max_tokens=1, temperature=0)
    measurements: dict[str, Any] = {}
    for length in effective_lengths:
        prompt = [{"prompt_token_ids": corpus_ids[:length]}]
        warmup_output = llm.generate(prompt, sampling, use_tqdm=False)
        trials: list[float] = []
        generated: list[list[int]] = []
        profiled_trials: list[bool] = []
        for trial_index in range(REPEATS):
            profile_this_trial = (
                profiler_dir is not None and trial_index == PROFILE_TRIAL
            )
            if profile_this_trial:
                llm.start_profile(
                    profile_prefix=f"{FAMILY}-{MODE}-{length}"
                )
            try:
                start = time.perf_counter()
                output = llm.generate(prompt, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
            finally:
                if profile_this_trial:
                    llm.stop_profile()
            trials.append(elapsed)
            generated.append(_extract_token_ids(output))
            profiled_trials.append(profile_this_trial)
        clean_trials = [
            seconds
            for seconds, profiled in zip(trials, profiled_trials)
            if not profiled
        ]
        profiled_seconds = [
            seconds
            for seconds, profiled in zip(trials, profiled_trials)
            if profiled
        ]
        best = min(clean_trials)
        median = statistics.median(clean_trials)
        measurements[str(length)] = {
            "seconds": trials,
            "clean_seconds": clean_trials,
            "profiled_seconds": profiled_seconds,
            "best_seconds": best,
            "median_seconds": median,
            "throughput_tokens_per_second": length / best,
            "warmup_token_ids": _extract_token_ids(warmup_output),
            "token_ids": generated,
            "profiled_trials": profiled_trials,
        }
        print(
            f"[prefill] tokens={length:>7} best={best:8.3f}s "
            f"median={median:8.3f}s throughput={length / best:9.0f} tok/s",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "family": FAMILY,
        "mode": MODE,
        "model": str(MODEL),
        "model_config": {
            "architectures": model_config.get("architectures"),
            "layers": model_config.get(
                "num_hidden_layers", model_config.get("num_layers")
            ),
            "index_n_heads": model_config.get("index_n_heads"),
            "index_head_dim": model_config.get("index_head_dim"),
            "index_topk": model_config.get("index_topk"),
            "index_init_tokens": int(
                model_config.get("index_init_tokens") or 0
            ),
            "index_local_tokens": int(
                model_config.get("index_local_tokens") or 0
            ),
            "max_position_embeddings": model_limit,
        },
        "runtime": {
            "vllm_version": vllm.__version__,
            "vllm_module": vllm.__file__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "use_deep_gemm": (
                os.environ.get("VLLM_USE_DEEP_GEMM", "1") == "1"
            ),
            "cuda_launch_blocking": os.environ.get(
                "CUDA_LAUNCH_BLOCKING", "0"
            ),
            "vllm_source": _git_state(VLLM_SRC),
            "litetopk": litetopk_runtime,
        },
        "benchmark": {
            "tensor_parallel_size": TP,
            "chunk": CHUNK,
            "repeats": REPEATS,
            "gpu_memory_utilization": GPU_UTIL,
            "kv_cache_dtype": llm_kwargs["kv_cache_dtype"],
            "max_num_seqs": MAX_SEQS,
            "num_gpu_blocks_override": KV_BLOCKS,
            "mtp_enabled": MTP_ENABLED,
            "mtp_tokens": MTP_K if MTP_ENABLED else 0,
            "flashinfer_allreduce_backend": FLASHINFER_ALLREDUCE_BACKEND,
            "sparse_indexer_max_logits_mb": int(
                os.environ.get("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "512")
            ),
            "prefix_caching": False,
            "enforce_eager": llm_kwargs["enforce_eager"],
            "async_scheduling": llm_kwargs["async_scheduling"],
            "attention_backend": attention_backend,
            "max_tokens": 1,
            "timing": (
                "wall clock, warmup once per length; best and median exclude "
                "profiled trials"
            ),
        },
        "load_seconds": load_seconds,
        "profiler_dir": str(profiler_dir) if profiler_dir is not None else None,
        "measurements": measurements,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"[done] {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

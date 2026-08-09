#!/usr/bin/env bash
set -euo pipefail

E2E_TMP_PAYLOAD=""

cleanup_e2e_payload() {
  if [[ -n "$E2E_TMP_PAYLOAD" ]]; then
    rm -f -- "$E2E_TMP_PAYLOAD"
  fi
}
trap cleanup_e2e_payload EXIT

run_prefill_payload() {
  E2E_TMP_PAYLOAD="$(mktemp /tmp/litetopk-e2e-prefill.XXXXXX.py)"
  command cat >"$E2E_TMP_PAYLOAD" <<'__LITETOPK_PREFILL_PY__'
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
    # Production LiteTopK is vendored in the vLLM checkout and the runtime
    # intentionally ignores external kernel trees.  Record and validate that
    # exact source of truth so the E2E runner cannot silently benchmark stale
    # standalone sources.
    vllm_source = Path(os.environ.get("LITETOPK_VLLM_SRC", "")).resolve()
    source_dir = (
        vllm_source
        / "csrc"
        / "libtorch_stable"
        / "attention"
        / "dsa"
        / "latest"
    )
    production_files = (
        "dsa_litetopk.cu",
        "sm100_dsa_litetopk.cuh",
        "dense_topk_litetopk.cuh",
    )
    hashes = {}

    def source_id(files: tuple[str, ...]) -> str:
        digest = hashlib.sha256()
        for name in files:
            path = source_dir / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing LiteTopK CUDA source: {path}"
                )
            contents = path.read_bytes()
            hashes[name] = hashlib.sha256(contents).hexdigest()
            digest.update(name.encode())
            digest.update(contents)
        return digest.hexdigest()[:12]

    production_source_id = source_id(production_files)
    integration_hashes = {}
    integration_files = (
        "vllm/model_executor/layers/litetopk_indexer.py",
        "vllm/model_executor/layers/dsa_litetopk.py",
        "vllm/model_executor/layers/litedsa.py",
        "vllm/model_executor/layers/sparse_attn_indexer.py",
        "vllm/model_executor/models/deepseek_v2.py",
        "vllm/model_executor/models/registry.py",
        "vllm/v1/attention/backends/mla/indexer.py",
        "vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
        "vllm/envs.py",
        "vllm/models/deepseek_v32/nvidia/attention.py",
        "vllm/models/deepseek_v32/nvidia/kernels.py",
        "vllm/models/deepseek_v4/nvidia/flashmla.py",
        "vllm/model_executor/layers/dsv4_packed_attn.py",
        "csrc/libtorch_stable/attention/dsa/litedsa.cu",
        (
            "csrc/libtorch_stable/attention/dsa/include/flashinfer/litedsa/"
            "litedsa_union.cuh"
        ),
        (
            "csrc/libtorch_stable/attention/dsa/vendor_fmla/sm100/prefill/"
            "sparse/fwd/head128_fp8/phase1.cuh"
        ),
        (
            "csrc/libtorch_stable/attention/dsa/dsv4_packed/"
            "litedsa_attention_sm100_dsv4.cuh"
        ),
        "csrc/libtorch_stable/attention/dsa/dsv4_packed/litedsa_dsv4.cu",
        (
            "csrc/libtorch_stable/attention/dsa/dsv4_packed/"
            "litedsa_dsv4_binding.cu"
        ),
        (
            "csrc/libtorch_stable/attention/dsa/dsv4_packed/"
            "litedsa_dsv4_atoms.cuh"
        ),
        "CMakeLists.txt",
        "csrc/libtorch_stable/ops.h",
        "csrc/libtorch_stable/torch_bindings.cpp",
    )
    for name in integration_files:
        path = vllm_source / name
        if not path.is_file():
            raise FileNotFoundError(f"missing vLLM integration source: {path}")
        integration_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    enabled = os.environ.get("VLLM_LITETOPK", "0") == "1"
    configured_headroom = float(
        os.environ.get("VLLM_LITETOPK_HEADROOM", "0.0")
    )
    effective_headroom = configured_headroom
    packed_so = os.environ.get("VLLM_DSV4_PACKED_SO")
    if packed_so:
        packed_so_path = Path(packed_so).resolve()
    else:
        packed_so_path = (
            vllm_source
            / "csrc/libtorch_stable/attention/dsa/dsv4_packed/dsa_dsv4.so"
        )
    packed_state: dict[str, Any] = {
        "enabled": os.environ.get("VLLM_DSV4_PACKED_ATTN", "0") == "1",
        "requested": False,
        "check": os.environ.get("VLLM_DSV4_PACKED_CHECK", "0") == "1",
        "path": str(packed_so_path),
        "exists": packed_so_path.is_file(),
    }
    if packed_so_path.is_file():
        packed_state["sha256"] = hashlib.sha256(
            packed_so_path.read_bytes()
        ).hexdigest()

    return {
        "enabled": enabled,
        "source_dir": str(source_dir),
        "source_id": production_source_id,
        "hot_prefix": 12288,
        "source_sha256": hashes,
        "integration_sha256": integration_hashes,
        "min_s": int(
            os.environ.get("VLLM_LITETOPK_PRODUCTION_MIN_S", "196608")
        ),
        "fp4_min_s": int(
            os.environ.get(
                "VLLM_LITETOPK_FP4_PRODUCTION_MIN_S",
                os.environ.get("VLLM_LITETOPK_PRODUCTION_MIN_S", "65536"),
            )
        ),
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
        "merge_cap": int(os.environ.get("VLLM_LITETOPK_MERGE_CAP", "196608")),
        "hot_sample": 12288,
        "hot_only": True,
        "headroom": configured_headroom,
        "effective_headroom": effective_headroom if enabled else None,
        "carry_recent_rows": 1536,
        "dedup_carry_wait": (
            os.environ.get("VLLM_LITETOPK_DEDUP_CARRY_WAIT", "1") == "1"
        ),
        "carry_io": os.environ.get("VLLM_LITETOPK_CARRY_IO", "1") == "1",
        "carry_every": int(
            os.environ.get("VLLM_LITETOPK_CARRY_EVERY", "1")
        ),
        "overflow_watermark": int(
            os.environ.get("VLLM_LITETOPK_OVF_WATERMARK", "65536")
        ),
        "path_timing": (
            os.environ.get("VLLM_LITETOPK_PATH_TIMING", "0") == "1"
        ),
        "host_timing": (
            os.environ.get("VLLM_LITETOPK_HOST_TIMING", "0") == "1"
        ),
        "carry_timing": (
            os.environ.get("VLLM_LITETOPK_CARRY_TIMING", "0") == "1"
        ),
        "carry_debug": (
            os.environ.get("VLLM_LITETOPK_CARRY_DEBUG", "0") == "1"
        ),
        "coldstart_identity": (
            os.environ.get("VLLM_LITETOPK_COLDSTART_IDENTITY", "0") == "1"
        ),
        "noop_indexer": (
            os.environ.get("VLLM_LITETOPK_NOOP_INDEXER", "0") == "1"
        ),
        "noop_mode": os.environ.get("VLLM_LITETOPK_NOOP_MODE", "arange"),
        "fp8_ring_policy": "fixed_warm_daemon_2048ns",
        "litedsa_reuse_output_buffers": (
            os.environ.get("VLLM_LITEDSA_REUSE_OUTPUT_BUFS", "0") == "1"
        ),
        "litedsa_dynamic_span": (
            os.environ.get("VLLM_LITEDSA_DYNAMIC_SPAN", "1") == "1"
        ),
        "litedsa_union_override": (
            str(Path(os.environ["VLLM_LITEDSA_UNION_SO"]).resolve())
            if os.environ.get("VLLM_LITEDSA_UNION_SO")
            else None
        ),
        "probe_every": int(
            os.environ.get("VLLM_LITETOPK_PROBE_EVERY", "8")
        ),
        "check": os.environ.get("VLLM_LITETOPK_CHECK", "0") == "1",
        "overflow_log": (
            os.environ.get("VLLM_LITETOPK_OVF_LOG", "1") == "1"
        ),
        "dsv4_packed": packed_state,
    }


def _load_model_config(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing model config: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    expected_by_family = {
        "dsv4": {
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
        },
    }
    expected = expected_by_family.get(FAMILY, {
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
    })
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
    # Concatenates every request's generated ids (request-major) so the
    # A/B token comparison covers all concurrent requests; with one request
    # this reduces to the historical single-list format.
    ids: list[int] = []
    for request_output in output or []:
        if request_output.outputs:
            ids.extend(request_output.outputs[0].token_ids)
    return ids


MODEL = Path(os.environ["MODEL"]).resolve()
PARQUET = Path(
    os.environ.get(
        "PARQUET",
        "/data01/home/ziqi.yin/glm5/train-00000-of-00002.parquet",
    )
).resolve()
FAMILY = os.environ.get("MODEL_FAMILY", "unknown").lower()
ARM = os.environ.get("BENCH_ARM", "unknown").lower()
DSA_MODE = os.environ.get("VLLM_DSA_MODE", "raw").lower()
LITETOPK_ENABLED = os.environ.get("VLLM_LITETOPK", "0") == "1"
DSV4_PACKED_ENABLED = os.environ.get("VLLM_DSV4_PACKED_ATTN", "0") == "1"
ARM_CONFIGS = {
    "glm5.2": {
        "raw": ("raw", False, False),
        "litetopk": ("litetopk", True, False),
        "litedsa": ("litedsa", False, False),
        "combo": ("litedsa", True, False),
    },
    "dsv4": {
        "raw": ("raw", False, False),
        "litetopk": ("litetopk", True, False),
        "litedsa": ("raw", False, True),
        "combo": ("litetopk", True, True),
    },
}
LENGTHS = [
    int(value)
    for value in os.environ.get(
        "LENGTHS", "262144 524288 786432 1048512"
    ).split()
]
TP = _positive_int("TP", "8")
CHUNK = _positive_int("CHUNK", "8192")
REPEATS = _positive_int("REPEATS", "2")
WARMUPS = _positive_int("WARMUPS", "1")
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
    if FAMILY not in ARM_CONFIGS:
        raise ValueError("MODEL_FAMILY must be glm5.2 or dsv4")
    expected_config = ARM_CONFIGS[FAMILY].get(ARM)
    if expected_config is None:
        raise ValueError(
            "BENCH_ARM must be raw, litetopk, litedsa, or combo"
        )
    actual_config = (DSA_MODE, LITETOPK_ENABLED, DSV4_PACKED_ENABLED)
    if actual_config != expected_config:
        raise ValueError(
            f"{FAMILY}/{ARM} requires "
            "(VLLM_DSA_MODE, VLLM_LITETOPK, VLLM_DSV4_PACKED_ATTN)="
            f"{expected_config!r}, got {actual_config!r}"
        )
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
        "dsv4": "DeepseekV4ForCausalLM",
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
        "dsv4": (0, 0),
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
    if LITETOPK_ENABLED:
        from vllm.model_executor.layers import litetopk_indexer
        from vllm.model_executor.layers.dsa_litetopk import (
            dsa_litetopk_latest_available,
        )

        if not dsa_litetopk_latest_available():
            raise RuntimeError(
                f"BENCH_ARM={ARM} requested LiteTopK, but the native "
                "high24-v1 LiteTopK adapter is unavailable"
            )
        extension = litetopk_indexer._ext()
        if (
            extension is None
            or not extension.candidate_value_u16_litetopk()
            or not extension.candidate_fp24_global_litetopk()
            or not hasattr(extension, "plan_and_permuted_paged_gather_out")
            or not hasattr(extension, "h2048_safe_topk_out_litetopk_")
        ):
            raise RuntimeError(
                "the vendored LiteTopK extension failed to load its "
                "U16 high24, fused planner/gather, and H2048 safe-selector ABI"
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
        extension_path = Path(extension.__file__).resolve()
        extension_sha256 = hashlib.sha256(
            extension_path.read_bytes()
        ).hexdigest()
        override_path = os.environ.get("VLLM_LITETOPK_SO", "")
        override_sha256 = os.environ.get("VLLM_LITETOPK_SO_SHA256", "")
        if bool(override_path) != bool(override_sha256):
            raise RuntimeError(
                "VLLM_LITETOPK_SO and VLLM_LITETOPK_SO_SHA256 must be "
                "set together"
            )
        if override_path:
            configured_override = Path(override_path).expanduser().resolve()
            if extension_path != configured_override:
                raise RuntimeError(
                    f"loaded LiteTopK extension {extension_path} does not "
                    f"match override {configured_override}"
                )
            if extension_sha256 != override_sha256.lower():
                raise RuntimeError(
                    "loaded LiteTopK extension SHA256 does not match "
                    "VLLM_LITETOPK_SO_SHA256"
                )
            litetopk_runtime["extension_override"] = True
        else:
            litetopk_runtime["extension_override"] = False
        litetopk_runtime["extension"] = str(extension_path)
        litetopk_runtime["extension_sha256"] = extension_sha256
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
            _configured_litetopk_fused_min_seq_len(
                os.environ.get("FP4_IDXCACHE") == "1"
            )
        )
    if FAMILY == "glm5.2" and DSA_MODE == "litedsa":
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
            (
                "csrc/libtorch_stable/attention/dsa/include/flashinfer/"
                "litedsa/litedsa_union.cuh"
            ),
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
        full_override = os.environ.get("VLLM_LITEDSA_SO")
        if full_override:
            full_override_path = Path(full_override).resolve()
            if not full_override_path.is_file():
                raise FileNotFoundError(
                    f"LiteDSA full override not found: {full_override_path}"
                )
            litetopk_runtime["litedsa_full_override"] = str(
                full_override_path
            )
            litetopk_runtime["litedsa_full_override_sha256"] = (
                hashlib.sha256(full_override_path.read_bytes()).hexdigest()
            )

    if FAMILY == "dsv4" and DSV4_PACKED_ENABLED:
        from vllm.model_executor.layers import dsv4_packed_attn

        packed_state = litetopk_runtime["dsv4_packed"]
        if not dsv4_packed_attn.enabled():
            raise RuntimeError(
                "DSV4 packed-attention arm requested, but its runtime flag "
                "is disabled"
            )
        if not packed_state["exists"]:
            raise FileNotFoundError(
                "DSV4 packed-attention shared library does not exist: "
                f"{packed_state['path']}"
            )
        packed_state["requested"] = True

    litetopk_runtime["arm_config"] = {
        "arm": ARM,
        "dsa_mode": DSA_MODE,
        "litetopk_enabled": LITETOPK_ENABLED,
        "generic_litedsa_enabled": (
            FAMILY == "glm5.2" and DSA_MODE == "litedsa"
        ),
        "dsv4_packed_enabled": DSV4_PACKED_ENABLED,
    }

    model_layers = model_config.get(
        "num_hidden_layers", model_config.get("num_layers")
    )
    print(
        f"[runtime] family={FAMILY} arm={ARM} dsa_mode={DSA_MODE} "
        f"litetopk={int(LITETOPK_ENABLED)} "
        f"dsv4_packed={int(DSV4_PACKED_ENABLED)} "
        f"vllm={vllm.__version__} "
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
        # The qualified launchers enable expert parallelism across the TP
        # workers (TP8+EP8 for GLM and TP4+EP4 for DSV4).
        "enable_expert_parallel": os.environ.get("EP", "1") == "1",
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
    if os.environ.get("MOE_BACKEND"):
        llm_kwargs["moe_backend"] = os.environ["MOE_BACKEND"]
    if os.environ.get("DP"):
        llm_kwargs["data_parallel_size"] = int(os.environ["DP"])
    attention_backend = os.environ.get("ATTN_BACKEND")
    attention_cfg: dict[str, Any] = {}
    if attention_backend:
        attention_cfg["backend"] = attention_backend
    if os.environ.get("FP4_IDXCACHE"):
        attention_cfg["use_fp4_indexer_cache"] = (
            os.environ["FP4_IDXCACHE"] == "1")
    if attention_cfg:
        llm_kwargs["attention_config"] = attention_cfg
    profiler_dir = PROFILER_ROOT / ARM if PROFILER_ROOT is not None else None
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
    # Throughput mode: submit CONCURRENCY equal-length requests in one
    # generate() call (offset slices keep the requests distinct) so the
    # scheduler spreads them across DP replicas; the recorded elapsed time
    # then measures aggregate machine throughput instead of one-request
    # latency. CONCURRENCY=1 preserves the historical latency protocol.
    concurrency = int(os.environ.get("CONCURRENCY", "1"))
    if concurrency < 1:
        raise ValueError("CONCURRENCY must be >= 1")
    for length in effective_lengths:
        if concurrency == 1:
            prompt = [{"prompt_token_ids": corpus_ids[:length]}]
        else:
            stride = max(
                1031,
                (len(corpus_ids) - length) // max(concurrency - 1, 1),
            )
            prompt = []
            for req in range(concurrency):
                begin = min(req * stride, max(len(corpus_ids) - length, 0))
                prompt.append(
                    {"prompt_token_ids": corpus_ids[begin:begin + length]}
                )
        warmup_seconds: list[float] = []
        warmup_tokens: list[list[int]] = []
        for warmup_index in range(WARMUPS):
            print(
                f"[warmup {warmup_index + 1}/{WARMUPS}] "
                f"tokens={length} start={time.strftime('%H:%M:%S', time.gmtime())}",
                flush=True,
            )
            start = time.perf_counter()
            warmup_output = llm.generate(prompt, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            warmup_seconds.append(elapsed)
            warmup_tokens.append(_extract_token_ids(warmup_output))
            print(
                f"[warmup {warmup_index + 1}/{WARMUPS}] "
                f"tokens={length} elapsed={elapsed:.6f}s",
                flush=True,
            )
        trials: list[float] = []
        generated: list[list[int]] = []
        profiled_trials: list[bool] = []
        for trial_index in range(REPEATS):
            profile_this_trial = (
                profiler_dir is not None and trial_index == PROFILE_TRIAL
            )
            if profile_this_trial:
                llm.start_profile(
                    profile_prefix=f"{FAMILY}-{ARM}-{length}"
                )
            try:
                print(
                    f"[trial {trial_index + 1}/{REPEATS}] "
                    f"tokens={length} start={time.strftime('%H:%M:%S', time.gmtime())}",
                    flush=True,
                )
                start = time.perf_counter()
                output = llm.generate(prompt, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
            finally:
                if profile_this_trial:
                    llm.stop_profile()
            print(
                f"[trial {trial_index + 1}/{REPEATS}] "
                f"tokens={length} elapsed={elapsed:.6f}s",
                flush=True,
            )
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
            "concurrency": concurrency,
            "throughput_tokens_per_second": concurrency * length / best,
            "warmup_seconds": warmup_seconds,
            "warmup_token_ids": warmup_tokens[-1],
            "warmup_token_ids_all": warmup_tokens,
            "token_ids": generated,
            "profiled_trials": profiled_trials,
        }
        print(
            f"[prefill] tokens={length:>7} best={best:8.3f}s "
            f"median={median:8.3f}s throughput={length / best:9.0f} tok/s",
            flush=True,
        )

    payload = {
        "schema_version": 2,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "family": FAMILY,
        "arm": ARM,
        "dsa_mode": DSA_MODE,
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
            "e2e_source_sha256": {
                "run_e2e.sh": hashlib.sha256(
                    Path(os.environ["E2E_RUNNER"]).resolve().read_bytes()
                ).hexdigest()
            },
            "litetopk": litetopk_runtime,
        },
        "benchmark": {
            "tensor_parallel_size": TP,
            "chunk": CHUNK,
            "repeats": REPEATS,
            "warmups": WARMUPS,
            "gpu_memory_utilization": GPU_UTIL,
            "kv_cache_dtype": llm_kwargs["kv_cache_dtype"],
            "max_num_seqs": MAX_SEQS,
            "num_gpu_blocks_override": KV_BLOCKS,
            "expert_parallel_enabled": llm_kwargs["enable_expert_parallel"],
            "moe_backend": llm_kwargs.get("moe_backend"),
            "data_parallel_size": llm_kwargs.get("data_parallel_size"),
            "fp4_indexer_cache": attention_cfg.get(
                "use_fp4_indexer_cache", False
            ),
            "concurrency": concurrency,
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
                "wall clock, configurable warmups per length; best and median exclude "
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
__LITETOPK_PREFILL_PY__
  "${PYBIN:-/opt/vllm-venv/bin/python}" "$E2E_TMP_PAYLOAD" "$@"
}

run_compare_payload() {
  E2E_TMP_PAYLOAD="$(mktemp /tmp/litetopk-e2e-compare.XXXXXX.py)"
  command cat >"$E2E_TMP_PAYLOAD" <<'__LITETOPK_COMPARE_PY__'
#!/usr/bin/env python3
"""Validate and summarize the four-arm LiteTopK/LiteDSA E2E matrix."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ARMS = ("raw", "litetopk", "litedsa", "combo")
ARM_CONFIGS = {
    "glm5.2": {
        "raw": ("raw", False, False, False),
        "litetopk": ("litetopk", True, False, False),
        "litedsa": ("litedsa", False, True, False),
        "combo": ("litedsa", True, True, False),
    },
    "dsv4": {
        "raw": ("raw", False, False, False),
        "litetopk": ("litetopk", True, False, False),
        "litedsa": ("raw", False, False, True),
        "combo": ("litetopk", True, False, True),
    },
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"result must be a JSON object: {path}")
    return result


def _different_keys(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> list[str]:
    return [
        key
        for key in sorted(set(left) | set(right))
        if key not in left or key not in right or left[key] != right[key]
    ]


def _validate_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _validate_source_hashes(config: dict[str, Any], arm: str) -> None:
    source_id = config.get("source_id")
    if (
        not isinstance(source_id, str)
        or len(source_id) != 12
        or any(character not in "0123456789abcdef" for character in source_id)
    ):
        raise ValueError(f"{arm} must record a 12-hex LiteTopK source_id")
    for field in ("source_sha256", "integration_sha256"):
        hashes = _require_dict(config.get(field), f"{arm} {field}")
        if not hashes:
            raise ValueError(f"{arm} {field} must not be empty")
        for name, digest in hashes.items():
            _validate_sha256(digest, f"{arm} {field}.{name}")


def _validate_vllm_source(runtime: dict[str, Any], arm: str) -> None:
    source = _require_dict(runtime.get("vllm_source"), f"{arm} vllm_source")
    if source.get("git") is not True:
        raise ValueError(f"{arm} must record a git-backed vLLM source")
    if not isinstance(source.get("head"), str) or not source["head"]:
        raise ValueError(f"{arm} must record the vLLM HEAD")
    _validate_sha256(
        source.get("tracked_diff_sha256"),
        f"{arm} vllm_source.tracked_diff_sha256",
    )


def _expected_arm_config(
    family: str,
    arm: str,
) -> tuple[str, bool, bool, bool]:
    try:
        return ARM_CONFIGS[family][arm]
    except KeyError as exc:
        raise ValueError(f"unsupported family/arm: {family}/{arm}") from exc


def _normalize_litetopk_runtime(
    result: dict[str, Any],
    arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = result["family"]
    dsa_mode, topk_enabled, generic_litedsa, packed_enabled = (
        _expected_arm_config(family, arm)
    )
    if result.get("dsa_mode") != dsa_mode:
        raise ValueError(
            f"{arm} dsa_mode={result.get('dsa_mode')!r}, expected {dsa_mode!r}"
        )

    runtime = _require_dict(result.get("runtime"), f"{arm} runtime")
    _validate_vllm_source(runtime, arm)
    config = _require_dict(runtime.get("litetopk"), f"{arm} litetopk")
    _validate_source_hashes(config, arm)

    if config.get("enabled") is not topk_enabled:
        raise ValueError(
            f"{arm} LiteTopK enabled={config.get('enabled')!r}, "
            f"expected {topk_enabled!r}"
        )
    headroom = config.get("headroom")
    if not isinstance(headroom, (int, float)):
        raise ValueError(f"{arm} must record numeric LiteTopK headroom")
    expected_effective = float(headroom) if topk_enabled else None
    if config.get("effective_headroom") != expected_effective:
        raise ValueError(
            f"{arm} effective_headroom does not match its enabled state"
        )

    expected_config = {
        "arm": arm,
        "dsa_mode": dsa_mode,
        "litetopk_enabled": topk_enabled,
        "generic_litedsa_enabled": generic_litedsa,
        "dsv4_packed_enabled": packed_enabled,
    }
    if config.get("arm_config") != expected_config:
        raise ValueError(
            f"{arm} runtime arm_config={config.get('arm_config')!r}, "
            f"expected {expected_config!r}"
        )

    topk_fields = (
        "extension",
        "extension_sha256",
        "extension_override",
        "candidate_u16",
        "candidate_high24",
        "smem_attr_once",
        "planner_fused_min_seq_len",
    )
    if topk_enabled:
        extension = config.get("extension")
        if not isinstance(extension, str) or not extension:
            raise ValueError(f"{arm} must record its LiteTopK extension")
        _validate_sha256(
            config.get("extension_sha256"),
            f"{arm} extension_sha256",
        )
        if config.get("extension_override") not in {True, False}:
            raise ValueError(f"{arm} must record extension_override")
        if config.get("candidate_u16") is not True:
            raise ValueError(f"{arm} did not validate the candidate-u16 ABI")
        if config.get("candidate_high24") is not True:
            raise ValueError(f"{arm} did not validate the high24 ABI")
        expected_min_s = (
            config.get("fp4_min_s") if family == "dsv4" else config.get("min_s")
        )
        if config.get("planner_fused_min_seq_len") != expected_min_s:
            raise ValueError(
                f"{arm} planner threshold does not match its production gate"
            )
    elif any(config.get(field) is not None for field in topk_fields):
        raise ValueError(f"{arm} unexpectedly loaded LiteTopK-only state")

    generic_fields = (
        "litedsa_available",
        "litedsa_native_extension",
        "litedsa_native_extension_sha256",
        "litedsa_source_sha256",
        "litedsa_union_override_sha256",
        "litedsa_full_override",
        "litedsa_full_override_sha256",
    )
    if generic_litedsa:
        if config.get("litedsa_available") is not True:
            raise ValueError(f"{arm} did not load the generic LiteDSA operators")
        native_extension = config.get("litedsa_native_extension")
        if not isinstance(native_extension, str) or not native_extension:
            raise ValueError(f"{arm} must record the LiteDSA native extension")
        _validate_sha256(
            config.get("litedsa_native_extension_sha256"),
            f"{arm} litedsa_native_extension_sha256",
        )
        sources = _require_dict(
            config.get("litedsa_source_sha256"),
            f"{arm} litedsa_source_sha256",
        )
        for name, digest in sources.items():
            _validate_sha256(digest, f"{arm} litedsa_source_sha256.{name}")
        if config.get("litedsa_union_override") is not None:
            _validate_sha256(
                config.get("litedsa_union_override_sha256"),
                f"{arm} litedsa_union_override_sha256",
            )
        if config.get("litedsa_full_override") is not None:
            _validate_sha256(
                config.get("litedsa_full_override_sha256"),
                f"{arm} litedsa_full_override_sha256",
            )
    elif any(config.get(field) is not None for field in generic_fields):
        raise ValueError(f"{arm} unexpectedly loaded generic LiteDSA state")

    packed = _require_dict(config.get("dsv4_packed"), f"{arm} dsv4_packed")
    if packed.get("enabled") is not packed_enabled:
        raise ValueError(f"{arm} packed-attention enabled state is incorrect")
    if packed.get("requested") is not packed_enabled:
        raise ValueError(f"{arm} packed-attention requested state is incorrect")
    if packed.get("check") is not False:
        raise ValueError(
            "four-arm performance summaries require VLLM_DSV4_PACKED_CHECK=0"
        )
    if not isinstance(packed.get("path"), str) or not packed["path"]:
        raise ValueError(f"{arm} must record the packed-attention SO path")
    if packed.get("exists") is True:
        _validate_sha256(packed.get("sha256"), f"{arm} dsv4_packed.sha256")
    if packed_enabled and packed.get("exists") is not True:
        raise ValueError(f"{arm} requested a missing packed-attention SO")

    normalized = dict(config)
    for field in topk_fields:
        normalized.pop(field, None)
    for field in generic_fields:
        normalized.pop(field, None)
    normalized.pop("enabled", None)
    normalized.pop("effective_headroom", None)
    normalized.pop("arm_config", None)
    normalized_packed = dict(packed)
    normalized_packed.pop("enabled", None)
    normalized_packed.pop("requested", None)
    normalized["dsv4_packed"] = normalized_packed

    artifacts = {
        "topk": {field: config.get(field) for field in topk_fields},
        "generic_litedsa": {
            field: config.get(field) for field in generic_fields
        },
        "packed": {
            "path": packed.get("path"),
            "exists": packed.get("exists"),
            "sha256": packed.get("sha256"),
        },
    }
    return normalized, artifacts


def _validate_identity(results: dict[str, dict[str, Any]]) -> str:
    raw = results["raw"]
    if raw.get("schema_version") != 2:
        raise ValueError("results must use schema_version=2")
    family = raw.get("family")
    if family not in ARM_CONFIGS:
        raise ValueError("family must be glm5.2 or dsv4")

    identity_fields = ("schema_version", "family", "model", "model_config")
    for arm, result in results.items():
        if result.get("arm") != arm:
            raise ValueError(
                f"{arm} file records arm={result.get('arm')!r}"
            )
        mismatches = [
            field
            for field in identity_fields
            if result.get(field) != raw.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"{arm} result identity differs at: {', '.join(mismatches)}"
            )
        benchmark = _require_dict(result.get("benchmark"), f"{arm} benchmark")
        if arm != "raw" and benchmark != raw["benchmark"]:
            keys = _different_keys(raw["benchmark"], benchmark)
            raise ValueError(
                f"{arm} benchmark differs at: {', '.join(keys)}"
            )

    normalized_configs: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    raw_runtime_base: dict[str, Any] | None = None
    for arm, result in results.items():
        runtime = _require_dict(result.get("runtime"), f"{arm} runtime")
        runtime_base = dict(runtime)
        runtime_base.pop("litetopk", None)
        if raw_runtime_base is None:
            raw_runtime_base = runtime_base
        elif runtime_base != raw_runtime_base:
            keys = _different_keys(raw_runtime_base, runtime_base)
            raise ValueError(
                f"{arm} runtime differs outside DSA state at: {', '.join(keys)}"
            )
        normalized, arm_artifacts = _normalize_litetopk_runtime(
            result, arm
        )
        normalized_configs[arm] = normalized
        artifacts[arm] = arm_artifacts

    raw_config = normalized_configs["raw"]
    for arm in ARMS[1:]:
        if normalized_configs[arm] != raw_config:
            keys = _different_keys(raw_config, normalized_configs[arm])
            raise ValueError(
                f"{arm} DSA source/config differs unexpectedly at: "
                + ", ".join(keys)
            )

    if artifacts["litetopk"]["topk"] != artifacts["combo"]["topk"]:
        raise ValueError("litetopk and combo loaded different LiteTopK artifacts")
    if family == "glm5.2" and (
        artifacts["litedsa"]["generic_litedsa"]
        != artifacts["combo"]["generic_litedsa"]
    ):
        raise ValueError("litedsa and combo loaded different LiteDSA artifacts")
    if family == "dsv4" and (
        artifacts["litedsa"]["packed"] != artifacts["combo"]["packed"]
    ):
        raise ValueError(
            "litedsa and combo loaded different DSV4 packed-attention artifacts"
        )
    return family


def _validate_measurements(
    results: dict[str, dict[str, Any]],
) -> list[str]:
    measurements = {
        arm: _require_dict(result.get("measurements"), f"{arm} measurements")
        for arm, result in results.items()
    }
    lengths = set(measurements["raw"])
    if not lengths:
        raise ValueError("results must contain measurements")
    for arm in ARMS[1:]:
        if set(measurements[arm]) != lengths:
            raise ValueError(f"{arm} token lengths do not match raw")

    repeats = results["raw"]["benchmark"].get("repeats")
    warmups = results["raw"]["benchmark"].get("warmups")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    if not isinstance(warmups, int) or warmups < 1:
        raise ValueError("benchmark warmups must be positive")

    for length in lengths:
        raw_measurement = measurements["raw"][length]
        raw_tokens = raw_measurement.get("token_ids")
        raw_warmups = raw_measurement.get("warmup_token_ids_all")
        for arm in ARMS:
            measurement = _require_dict(
                measurements[arm][length],
                f"{arm} measurement {length}",
            )
            if len(measurement.get("seconds", ())) != repeats:
                raise ValueError(
                    f"{arm} length {length} must contain {repeats} timings"
                )
            if len(measurement.get("token_ids", ())) != repeats:
                raise ValueError(
                    f"{arm} length {length} must contain {repeats} token outputs"
                )
            if len(measurement.get("warmup_token_ids_all", ())) != warmups:
                raise ValueError(
                    f"{arm} length {length} must contain {warmups} warmups"
                )
            if measurement.get("token_ids") != raw_tokens:
                raise ValueError(
                    f"generated trial tokens differ for {arm} at {length}"
                )
            if measurement.get("warmup_token_ids_all") != raw_warmups:
                raise ValueError(
                    f"generated warmup tokens differ for {arm} at {length}"
                )
            for field in ("best_seconds", "median_seconds"):
                seconds = measurement.get(field)
                if (
                    not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds)
                    or seconds <= 0
                ):
                    raise ValueError(
                        f"{arm} {length} {field} must be finite and positive"
                    )
    return sorted(lengths, key=int)


def _speedups(times: dict[str, float]) -> dict[str, float]:
    raw = times["raw"]
    return {
        "raw_over_litetopk": raw / times["litetopk"],
        "raw_over_litedsa": raw / times["litedsa"],
        "raw_over_combo": raw / times["combo"],
        "litetopk_over_combo": times["litetopk"] / times["combo"],
        "litedsa_over_combo": times["litedsa"] / times["combo"],
        "interaction": (
            times["litetopk"]
            * times["litedsa"]
            / (raw * times["combo"])
        ),
    }


def main() -> None:
    if len(sys.argv) not in {5, 6}:
        raise SystemExit(
            "usage: run_e2e.sh --compare RAW.json LITETOPK.json "
            "LITEDSA.json COMBO.json [SUMMARY.json]"
        )
    paths = dict(zip(ARMS, map(Path, sys.argv[1:5])))
    results = {arm: _load(path) for arm, path in paths.items()}
    family = _validate_identity(results)
    lengths = _validate_measurements(results)

    print(
        f"{'tokens':>9} {'raw(s)':>9} {'topk(s)':>9} {'dsa(s)':>9} "
        f"{'combo(s)':>9} {'raw/topk':>9} {'raw/dsa':>9} "
        f"{'raw/combo':>10} {'topk/combo':>11} {'dsa/combo':>10}"
    )
    rows = []
    for length in lengths:
        best_times = {
            arm: float(results[arm]["measurements"][length]["best_seconds"])
            for arm in ARMS
        }
        median_times = {
            arm: float(
                results[arm]["measurements"][length]["median_seconds"]
            )
            for arm in ARMS
        }
        best_speedups = _speedups(best_times)
        median_speedups = _speedups(median_times)
        rows.append(
            {
                "tokens": int(length),
                "best_seconds": best_times,
                "median_seconds": median_times,
                "best_speedups": best_speedups,
                "median_speedups": median_speedups,
                "generated_tokens_match": True,
            }
        )
        print(
            f"{int(length):>9} {median_times['raw']:>9.3f} "
            f"{median_times['litetopk']:>9.3f} "
            f"{median_times['litedsa']:>9.3f} "
            f"{median_times['combo']:>9.3f} "
            f"{median_speedups['raw_over_litetopk']:>8.3f}x "
            f"{median_speedups['raw_over_litedsa']:>8.3f}x "
            f"{median_speedups['raw_over_combo']:>9.3f}x "
            f"{median_speedups['litetopk_over_combo']:>10.3f}x "
            f"{median_speedups['litedsa_over_combo']:>9.3f}x"
        )

    summary = {
        "schema_version": 2,
        "family": family,
        "model": results["raw"]["model"],
        "results": {arm: str(path.resolve()) for arm, path in paths.items()},
        "generated_tokens_all_match": True,
        "primary_timing": "median_seconds",
        "rows": rows,
    }
    if len(sys.argv) == 6:
        output = Path(sys.argv[5])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"summary: {output}")


if __name__ == "__main__":
    main()
__LITETOPK_COMPARE_PY__
  "${PYTHON:-python3}" "$E2E_TMP_PAYLOAD" "$@"
}

if [[ "${1:-}" == "--prefill" ]]; then
  shift
  run_prefill_payload "$@"
  exit
fi
if [[ "${1:-}" == "--compare" ]]; then
  shift
  run_compare_payload "$@"
  exit
fi

HOST_SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# REPO_DIR is the repository path visible inside the container. By default the
# host and container use the same absolute mount.
REPO_DIR="${REPO_DIR:-$(dirname "$HOST_SCRIPT_DIR")}"
CONTAINER_SCRIPT_DIR="$REPO_DIR/e2e"
CONTAINER="${CONTAINER:-glm5-prefill}"
LITETOPK_VLLM_SRC="${LITETOPK_VLLM_SRC:-${VLLM_SRC:-/data01/home/ziqi.yin/vllm-v026}}"
MODEL_FAMILY="${MODEL_FAMILY:?set MODEL_FAMILY to glm5.2 or dsv4}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-$HOST_SCRIPT_DIR/results/$STAMP-$MODEL_FAMILY}"
ARMS="${ARMS:-raw litetopk litedsa combo}"
# Qualified E2E runs use vLLM asynchronous scheduling by default.
ASYNC_SCHED="${ASYNC_SCHED:-1}"

case "$MODEL_FAMILY" in
  glm5.2)
    MODEL="${MODEL:-/data07/glm5-fp8-official}"
    LENGTHS="${LENGTHS:-1048512}"
    TP="${TP:-8}"
    DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
    GPU_UTIL="${GPU_UTIL:-0.90}"
    MTP="${MTP:-1}"
    MTP_K="${MTP_K:-5}"
    EP="${EP:-1}"
    MOE_BACKEND="${MOE_BACKEND:-}"
    FP4_IDXCACHE="${FP4_IDXCACHE:-}"
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-2048}"
    MAX_SEQS="${MAX_SEQS:-32}"
    KV_BLOCKS="${KV_BLOCKS:-17000}"
    VLLM_LITETOPK_PRODUCTION_MIN_S="${VLLM_LITETOPK_PRODUCTION_MIN_S:-65536}"
    VLLM_LITETOPK_FP4_PRODUCTION_MIN_S="${VLLM_LITETOPK_FP4_PRODUCTION_MIN_S:-65536}"
    VLLM_LITETOPK_DENSE_SELECT_MIN_S="${VLLM_LITETOPK_DENSE_SELECT_MIN_S:-40960}"
    VLLM_LITETOPK_DENSE_SELECT_MAX_S="${VLLM_LITETOPK_DENSE_SELECT_MAX_S:-65536}"
    # The FP8 no-hist kernel has one fixed production policy: warm-started
    # ring refresh with its daemon paced at the compiled 2048 ns quantum.
    VLLM_LITETOPK_MERGE_CAP="${VLLM_LITETOPK_MERGE_CAP:-65536}"
    VLLM_LITETOPK_PROBE_EVERY="${VLLM_LITETOPK_PROBE_EVERY:-1}"
    VLLM_LITETOPK_OVF_WATERMARK="${VLLM_LITETOPK_OVF_WATERMARK:-49152}"
    ATTN_BACKEND="${ATTN_BACKEND:-FLASHINFER_MLA_SPARSE}"
    ;;
  dsv4)
    MODEL="${MODEL:-/data01/home/ziqi.yin/models/dsv4-flash-0731}"
    LENGTHS="${LENGTHS:-1048512}"
    TP="${TP:-4}"
    DEVICES="${DEVICES:-0,1,2,3}"
    GPU_UTIL="${GPU_UTIL:-0.90}"
    MTP="${MTP:-1}"
    MTP_K="${MTP_K:-2}"
    EP="${EP:-1}"
    MOE_BACKEND="${MOE_BACKEND:-deep_gemm_mega_moe}"
    FP4_IDXCACHE="${FP4_IDXCACHE:-1}"
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-2048}"
    MAX_SEQS="${MAX_SEQS:-32}"
    KV_BLOCKS="${KV_BLOCKS:-}"
    VLLM_LITETOPK_PRODUCTION_MIN_S="${VLLM_LITETOPK_PRODUCTION_MIN_S:-196608}"
    VLLM_LITETOPK_FP4_PRODUCTION_MIN_S="${VLLM_LITETOPK_FP4_PRODUCTION_MIN_S:-65536}"
    VLLM_LITETOPK_DENSE_SELECT_MIN_S="${VLLM_LITETOPK_DENSE_SELECT_MIN_S:-40960}"
    VLLM_LITETOPK_DENSE_SELECT_MAX_S="${VLLM_LITETOPK_DENSE_SELECT_MAX_S:-262144}"
    VLLM_LITETOPK_MERGE_CAP="${VLLM_LITETOPK_MERGE_CAP:-65536}"
    VLLM_LITETOPK_PROBE_EVERY="${VLLM_LITETOPK_PROBE_EVERY:-8}"
    VLLM_LITETOPK_OVF_WATERMARK="${VLLM_LITETOPK_OVF_WATERMARK:-65536}"
    ATTN_BACKEND="${ATTN_BACKEND:-FLASHMLA_SPARSE_DSV4}"
    ;;
  *)
    echo "MODEL_FAMILY must be glm5.2 or dsv4" >&2
    exit 2
    ;;
esac

VLLM_LITETOPK_HEADROOM="${VLLM_LITETOPK_HEADROOM:-0.0}"

configure_arm() {
  local arm="$1"
  case "$MODEL_FAMILY:$arm" in
    glm5.2:raw)
      DSA_MODE=raw
      LITETOPK_ENABLED=0
      DSV4_PACKED_ENABLED=0
      ;;
    glm5.2:litetopk)
      DSA_MODE=litetopk
      LITETOPK_ENABLED=1
      DSV4_PACKED_ENABLED=0
      ;;
    glm5.2:litedsa)
      DSA_MODE=litedsa
      LITETOPK_ENABLED=0
      DSV4_PACKED_ENABLED=0
      ;;
    glm5.2:combo)
      DSA_MODE=litedsa
      LITETOPK_ENABLED=1
      DSV4_PACKED_ENABLED=0
      ;;
    dsv4:raw)
      DSA_MODE=raw
      LITETOPK_ENABLED=0
      DSV4_PACKED_ENABLED=0
      ;;
    dsv4:litetopk)
      DSA_MODE=litetopk
      LITETOPK_ENABLED=1
      DSV4_PACKED_ENABLED=0
      ;;
    dsv4:litedsa)
      DSA_MODE=raw
      LITETOPK_ENABLED=0
      DSV4_PACKED_ENABLED=1
      ;;
    dsv4:combo)
      DSA_MODE=litetopk
      LITETOPK_ENABLED=1
      DSV4_PACKED_ENABLED=1
      ;;
    *)
      echo "unsupported E2E arm: $arm" >&2
      exit 2
      ;;
  esac
}

check_kernel_markers() {
  local arm="$1"
  local log="$2"
  local required_marker=""
  local forbidden_marker=""
  local expected_topk=0
  local topk_count=0
  local attention_count=0

  case "$arm" in
    litetopk|combo)
      expected_topk="$TP"
      ;;
  esac

  topk_count="$(grep -Fc "HOT12288 exact-once active" "$log" || true)"
  if (( topk_count != expected_topk )); then
    echo "$arm executed LiteTopK on $topk_count ranks; expected $expected_topk" >&2
    exit 1
  fi

  case "$MODEL_FAMILY:$arm" in
    glm5.2:litedsa|glm5.2:combo)
      required_marker="LITEDSA_KERNEL_EXECUTED"
      forbidden_marker="DSV4_PACKED_KERNEL_EXECUTED"
      ;;
    dsv4:litedsa|dsv4:combo)
      required_marker="DSV4_PACKED_KERNEL_EXECUTED"
      forbidden_marker="LITEDSA_KERNEL_EXECUTED"
      ;;
    *)
      forbidden_marker="LITEDSA_KERNEL_EXECUTED|DSV4_PACKED_KERNEL_EXECUTED"
      ;;
  esac

  if [[ -n "$required_marker" ]]; then
    attention_count="$(grep -Fc "$required_marker" "$log" || true)"
    if (( attention_count != TP )); then
      echo "$arm executed $required_marker on $attention_count ranks; expected $TP" >&2
      exit 1
    fi
  fi
  if [[ -n "$forbidden_marker" ]] && grep -Eq "$forbidden_marker" "$log"; then
    echo "$arm unexpectedly executed a different optimized attention kernel" >&2
    exit 1
  fi
}

run_arm() {
  local arm="$1"
  local output="$2"
  configure_arm "$arm"

  docker exec \
    -w "$CONTAINER_SCRIPT_DIR" \
    -e PATH=/opt/vllm-venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
    -e PYTHONPATH="$LITETOPK_VLLM_SRC" \
    -e LITETOPK_VLLM_SRC="$LITETOPK_VLLM_SRC" \
    -e CUDA_VISIBLE_DEVICES="$DEVICES" \
    -e BENCH_ARM="$arm" \
    -e MODEL_FAMILY="$MODEL_FAMILY" \
    -e MODEL="$MODEL" \
    -e PARQUET="${PARQUET:-/data01/home/ziqi.yin/glm5/train-00000-of-00002.parquet}" \
    -e OUTPUT="$output" \
    -e VLLM_DSA_MODE="$DSA_MODE" \
    -e VLLM_LITETOPK="$LITETOPK_ENABLED" \
    -e VLLM_DSV4_PACKED_ATTN="$DSV4_PACKED_ENABLED" \
    -e DEEPGEMM_DIR="${DEEPGEMM_DIR:-/data01/home/ziqi.yin/glm5_prefill_test/DeepGEMM}" \
    -e VLLM_LITETOPK_BUILD="${VLLM_LITETOPK_BUILD:-/root/.cache/vllm/litetopk_build}" \
    -e VLLM_LITETOPK_SO="${VLLM_LITETOPK_SO:-}" \
    -e VLLM_LITETOPK_SO_SHA256="${VLLM_LITETOPK_SO_SHA256:-}" \
    -e VLLM_LITETOPK_DENSE_SELECT="${VLLM_LITETOPK_DENSE_SELECT:-1}" \
    -e VLLM_LITETOPK_DENSE_SELECT_MIN_S="$VLLM_LITETOPK_DENSE_SELECT_MIN_S" \
    -e VLLM_LITETOPK_DENSE_SELECT_MAX_S="$VLLM_LITETOPK_DENSE_SELECT_MAX_S" \
    -e VLLM_LITETOPK_DENSE_SELECT_BINS="${VLLM_LITETOPK_DENSE_SELECT_BINS:-4096}" \
    -e VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB="${VLLM_LITETOPK_DENSE_SELECT_MIN_LOGITS_MB:-0}" \
    -e VLLM_LITETOPK_MERGE_CAP="$VLLM_LITETOPK_MERGE_CAP" \
    -e VLLM_LITETOPK_PRODUCTION_MIN_S="$VLLM_LITETOPK_PRODUCTION_MIN_S" \
    -e VLLM_LITETOPK_FP4_PRODUCTION_MIN_S="$VLLM_LITETOPK_FP4_PRODUCTION_MIN_S" \
    -e VLLM_LITETOPK_PATH_TIMING="${VLLM_LITETOPK_PATH_TIMING:-0}" \
    -e VLLM_LITETOPK_COLDSTART_IDENTITY="${VLLM_LITETOPK_COLDSTART_IDENTITY:-0}" \
    -e VLLM_LITETOPK_CARRY_DEBUG="${VLLM_LITETOPK_CARRY_DEBUG:-0}" \
    -e VLLM_LITETOPK_NOOP_INDEXER="${VLLM_LITETOPK_NOOP_INDEXER:-0}" \
    -e VLLM_LITETOPK_HOST_TIMING="${VLLM_LITETOPK_HOST_TIMING:-0}" \
    -e VLLM_LITETOPK_NOOP_MODE="${VLLM_LITETOPK_NOOP_MODE:-arange}" \
    -e VLLM_LITETOPK_CARRY_IO="${VLLM_LITETOPK_CARRY_IO:-1}" \
    -e VLLM_LITETOPK_CARRY_TIMING="${VLLM_LITETOPK_CARRY_TIMING:-0}" \
    -e VLLM_LITETOPK_CARRY_EVERY="${VLLM_LITETOPK_CARRY_EVERY:-1}" \
    -e VLLM_DSV4_PACKED_CHECK="${VLLM_DSV4_PACKED_CHECK:-0}" \
    -e VLLM_DSV4_PACKED_SO="${VLLM_DSV4_PACKED_SO:-}" \
    -e VLLM_LITETOPK_HEADROOM="$VLLM_LITETOPK_HEADROOM" \
    -e VLLM_LITETOPK_DEDUP_CARRY_WAIT="${VLLM_LITETOPK_DEDUP_CARRY_WAIT:-1}" \
    -e VLLM_LITEDSA_REUSE_OUTPUT_BUFS="${VLLM_LITEDSA_REUSE_OUTPUT_BUFS:-0}" \
    -e VLLM_LITEDSA_DYNAMIC_SPAN="${VLLM_LITEDSA_DYNAMIC_SPAN:-1}" \
    -e VLLM_LITEDSA_UNION_SO="${VLLM_LITEDSA_UNION_SO:-}" \
    -e VLLM_LITEDSA_SO="${VLLM_LITEDSA_SO:-}" \
    -e VLLM_LITETOPK_PROBE_EVERY="$VLLM_LITETOPK_PROBE_EVERY" \
    -e VLLM_LITETOPK_OVF_WATERMARK="$VLLM_LITETOPK_OVF_WATERMARK" \
    -e VLLM_LITETOPK_CHECK="${VLLM_LITETOPK_CHECK:-0}" \
    -e VLLM_LITETOPK_OVF_LOG="${VLLM_LITETOPK_OVF_LOG:-1}" \
    -e VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}" \
    -e VLLM_FLASHINFER_ALLREDUCE_BACKEND="${VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}" \
    -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="$VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" \
    -e TP="$TP" \
    -e CHUNK="${CHUNK:-8192}" \
    -e REPEATS="${REPEATS:-2}" \
    -e WARMUPS="${WARMUPS:-1}" \
    -e GPU_UTIL="$GPU_UTIL" \
    -e KVDTYPE="${KVDTYPE:-fp8}" \
    -e MAX_SEQS="$MAX_SEQS" \
    -e LENGTHS="$LENGTHS" \
    -e MTP="$MTP" \
    -e MTP_K="$MTP_K" \
    -e KV_BLOCKS="$KV_BLOCKS" \
    -e EAGER="${EAGER:-0}" \
    -e ASYNC_SCHED="$ASYNC_SCHED" \
    -e ATTN_BACKEND="$ATTN_BACKEND" \
    -e EP="$EP" \
    -e MOE_BACKEND="$MOE_BACKEND" \
    -e FP4_IDXCACHE="$FP4_IDXCACHE" \
    -e DP="${DP:-}" \
    -e CONCURRENCY="${CONCURRENCY:-1}" \
    -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
    -e PROF_DIR="${PROF_DIR:-}" \
    -e PROFILE_TRIAL="${PROFILE_TRIAL:-1}" \
    -e VLLM_DSA_DUMP_DIR="${VLLM_DSA_DUMP_DIR:-}" \
    -e VLLM_DSA_DUMP_MIN_S="${VLLM_DSA_DUMP_MIN_S:-245760}" \
    -e VLLM_DSA_DUMP_MAX="${VLLM_DSA_DUMP_MAX:-2}" \
    -e VLLM_DSA_DUMP_DEBUG="${VLLM_DSA_DUMP_DEBUG:-0}" \
    -e E2E_RUNNER="$CONTAINER_SCRIPT_DIR/run_e2e.sh" \
    -e PYBIN="${PYBIN:-/opt/vllm-venv/bin/python}" \
    "$CONTAINER" /usr/bin/env bash \
    "$CONTAINER_SCRIPT_DIR/run_e2e.sh" --prefill
}

if [[ -e "$RESULT_DIR" ]]; then
  echo "RESULT_DIR already exists; refusing to overwrite: $RESULT_DIR" >&2
  exit 2
fi
mkdir -p "$RESULT_DIR"
ran_raw=0
ran_litetopk=0
ran_litedsa=0
ran_combo=0
arm_index=0

for arm in ${ARMS//,/ }; do
  case "$arm" in
    raw)
      [[ "$ran_raw" == 0 ]] || { echo "duplicate arm: raw" >&2; exit 2; }
      ran_raw=1
      ;;
    litetopk)
      [[ "$ran_litetopk" == 0 ]] || { echo "duplicate arm: litetopk" >&2; exit 2; }
      ran_litetopk=1
      ;;
    litedsa)
      [[ "$ran_litedsa" == 0 ]] || { echo "duplicate arm: litedsa" >&2; exit 2; }
      ran_litedsa=1
      ;;
    combo)
      [[ "$ran_combo" == 0 ]] || { echo "duplicate arm: combo" >&2; exit 2; }
      ran_combo=1
      ;;
    *)
      echo "ARMS entries must be raw, litetopk, litedsa, or combo" >&2
      exit 2
      ;;
  esac

  if (( arm_index > 0 )); then
    sleep "${COOLDOWN_SECONDS:-10}"
  fi
  output="$RESULT_DIR/$arm.json"
  log="$RESULT_DIR/$arm.log"
  run_arm "$arm" "$output" 2>&1 | tee "$log"
  check_kernel_markers "$arm" "$log"
  arm_index=$((arm_index + 1))
done

if [[ "$ran_raw$ran_litetopk$ran_litedsa$ran_combo" == 1111 ]]; then
  run_compare_payload \
    "$RESULT_DIR/raw.json" \
    "$RESULT_DIR/litetopk.json" \
    "$RESULT_DIR/litedsa.json" \
    "$RESULT_DIR/combo.json" \
    "$RESULT_DIR/summary.json" | tee "$RESULT_DIR/summary.txt"
fi

echo "results: $RESULT_DIR"

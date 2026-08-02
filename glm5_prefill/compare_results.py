#!/usr/bin/env python3
"""Compare matched raw and LiteTopK vLLM E2E benchmark JSON files."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


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


def _normalized_litetopk_runtime(
    result: dict[str, Any],
    *,
    expected_enabled: bool,
    label: str,
) -> dict[str, Any]:
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} result must record runtime metadata")
    config = runtime.get("litetopk")
    if not isinstance(config, dict):
        raise ValueError(f"{label} result must record LiteTopK metadata")

    normalized = dict(config)
    enabled = normalized.pop("enabled", None)
    if enabled is not expected_enabled:
        raise ValueError(
            f"{label} LiteTopK enabled state is {enabled!r}, "
            f"expected {expected_enabled!r}"
        )

    headroom = normalized.get("headroom")
    effective_headroom = normalized.pop("effective_headroom", None)
    if expected_enabled:
        if not isinstance(headroom, (int, float)):
            raise ValueError(f"{label} must record numeric LiteTopK headroom")
        family = result.get("family")
        expected_headroom = (
            max(float(headroom), 0.5)
            if family == "longcat"
            else float(headroom)
        )
        if effective_headroom != expected_headroom:
            raise ValueError(
                f"{label} effective LiteTopK headroom is "
                f"{effective_headroom!r}, expected {expected_headroom!r}"
            )
    elif effective_headroom is not None:
        raise ValueError(
            "raw result must not report an effective LiteTopK headroom"
        )

    for source_field in ("source_sha256", "integration_sha256"):
        source_hashes = normalized.get(source_field)
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError(
                f"{label} must record non-empty {source_field} metadata"
            )
        for source_name, source_hash in source_hashes.items():
            _validate_sha256(
                source_hash,
                f"{label} {source_field}.{source_name}",
            )

    extension = normalized.pop("extension", None)
    extension_sha256 = normalized.pop("extension_sha256", None)
    candidate_u16 = normalized.pop("candidate_u16", None)
    candidate_high24 = normalized.pop("candidate_high24", None)
    normalized.pop("smem_attr_once", None)
    planner_gate = normalized.pop("planner_fused_min_seq_len", None)
    min_s = normalized.get("min_s")
    if expected_enabled:
        if not isinstance(extension, str) or not extension:
            raise ValueError(
                "litetopk result must record its loaded extension path"
            )
        _validate_sha256(
            extension_sha256,
            "litetopk extension_sha256",
        )
        if candidate_u16 is not True:
            raise ValueError(
                "litetopk result must confirm the candidate-u16 ABI"
            )
        if candidate_high24 is not True:
            raise ValueError(
                "litetopk result must confirm the high24 baseline ABI"
            )
        if not isinstance(min_s, int) or planner_gate != min_s:
            raise ValueError(
                "litetopk planner gate must equal its configured min_s: "
                f"{planner_gate!r} != {min_s!r}"
            )
    elif any(
        value is not None
        for value in (
            extension,
            extension_sha256,
            candidate_u16,
            candidate_high24,
            planner_gate,
        )
    ):
        raise ValueError(
            "raw result unexpectedly loaded LiteTopK-only runtime state"
        )

    return normalized


def _validate_measurements(
    raw: dict[str, Any],
    lite: dict[str, Any],
) -> None:
    raw_measurements = raw.get("measurements")
    lite_measurements = lite.get("measurements")
    if not isinstance(raw_measurements, dict) or not isinstance(
        lite_measurements, dict
    ):
        raise ValueError("both results must record measurements")
    if set(raw_measurements) != set(lite_measurements):
        raise ValueError("raw and LiteTopK token lengths do not match")

    repeats = raw["benchmark"].get("repeats")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("benchmark repeats must be a positive integer")
    token_mismatches: list[str] = []
    for key in raw_measurements:
        raw_measurement = raw_measurements[key]
        lite_measurement = lite_measurements[key]
        for label, measurement in (
            ("raw", raw_measurement),
            ("litetopk", lite_measurement),
        ):
            if len(measurement.get("seconds", ())) != repeats:
                raise ValueError(
                    f"{label} length {key} does not contain {repeats} timings"
                )
            if len(measurement.get("token_ids", ())) != repeats:
                raise ValueError(
                    f"{label} length {key} does not contain "
                    f"{repeats} trial token outputs"
                )
        if (
            raw_measurement.get("warmup_token_ids")
            != lite_measurement.get("warmup_token_ids")
            or raw_measurement.get("token_ids")
            != lite_measurement.get("token_ids")
        ):
            token_mismatches.append(key)
    if token_mismatches:
        raise ValueError(
            "generated warmup or trial tokens do not match at lengths: "
            + ", ".join(sorted(token_mismatches, key=int))
        )


def _validate_pair(raw: dict[str, Any], lite: dict[str, Any]) -> None:
    expected_modes = (raw.get("mode"), lite.get("mode"))
    if expected_modes != ("raw", "litetopk"):
        raise ValueError(
            f"expected raw then litetopk results, got {expected_modes}"
        )
    fields = (
        ("schema_version", raw.get("schema_version"), lite.get("schema_version")),
        ("family", raw.get("family"), lite.get("family")),
        ("model", raw.get("model"), lite.get("model")),
        ("model_config", raw.get("model_config"), lite.get("model_config")),
    )
    mismatches = [
        f"{name}: {left!r} != {right!r}"
        for name, left, right in fields
        if left != right
    ]
    if mismatches:
        raise ValueError("result identities do not match: " + "; ".join(mismatches))
    if not isinstance(raw.get("model_config"), dict) or not raw["model_config"]:
        raise ValueError("both results must record non-empty model_config metadata")

    raw_benchmark = raw.get("benchmark")
    lite_benchmark = lite.get("benchmark")
    if not isinstance(raw_benchmark, dict) or not isinstance(
        lite_benchmark, dict
    ):
        raise ValueError("both results must record benchmark configuration")
    benchmark_mismatches = _different_keys(raw_benchmark, lite_benchmark)
    if benchmark_mismatches:
        raise ValueError(
            "benchmark configurations differ at: "
            + ", ".join(benchmark_mismatches)
        )

    raw_runtime = raw.get("runtime")
    lite_runtime = lite.get("runtime")
    if not isinstance(raw_runtime, dict) or not isinstance(lite_runtime, dict):
        raise ValueError("both results must record runtime metadata")
    raw_runtime_base = dict(raw_runtime)
    lite_runtime_base = dict(lite_runtime)
    raw_runtime_base.pop("litetopk", None)
    lite_runtime_base.pop("litetopk", None)
    for label, runtime_base in (
        ("raw", raw_runtime_base),
        ("litetopk", lite_runtime_base),
    ):
        vllm_source = runtime_base.get("vllm_source")
        if (
            not isinstance(vllm_source, dict)
            or vllm_source.get("git") is not True
            or not isinstance(vllm_source.get("head"), str)
            or not vllm_source["head"]
        ):
            raise ValueError(
                f"{label} result must record a valid vLLM git state"
            )
        _validate_sha256(
            vllm_source.get("tracked_diff_sha256"),
            f"{label} vllm_source.tracked_diff_sha256",
        )
    runtime_mismatches = _different_keys(
        raw_runtime_base,
        lite_runtime_base,
    )
    if runtime_mismatches:
        raise ValueError(
            "runtime or vLLM source identity differs at: "
            + ", ".join(runtime_mismatches)
        )

    raw_litetopk = _normalized_litetopk_runtime(
        raw,
        expected_enabled=False,
        label="raw",
    )
    lite_litetopk = _normalized_litetopk_runtime(
        lite,
        expected_enabled=True,
        label="litetopk",
    )
    litetopk_mismatches = _different_keys(raw_litetopk, lite_litetopk)
    if litetopk_mismatches:
        raise ValueError(
            "LiteTopK configuration or source identity differs at: "
            + ", ".join(litetopk_mismatches)
        )

    _validate_measurements(raw, lite)


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: compare_results.py RAW.json LITETOPK.json [SUMMARY.json]"
        )
    raw_path, lite_path = map(Path, sys.argv[1:3])
    raw, lite = _load(raw_path), _load(lite_path)
    _validate_pair(raw, lite)

    rows = []
    generated_match = True
    print(
        f"{'tokens':>9} {'raw(s)':>10} {'litetopk(s)':>13} "
        f"{'best':>9} {'median':>9} {'raw tok/s':>11} "
        f"{'lite tok/s':>12} {'token':>7}"
    )
    for key in sorted(raw["measurements"], key=int):
        raw_measurement = raw["measurements"][key]
        lite_measurement = lite["measurements"][key]
        raw_seconds = float(raw_measurement["best_seconds"])
        lite_seconds = float(lite_measurement["best_seconds"])
        speedup = raw_seconds / lite_seconds
        raw_median_seconds = float(
            raw_measurement.get("median_seconds", raw_seconds)
        )
        lite_median_seconds = float(
            lite_measurement.get("median_seconds", lite_seconds)
        )
        median_speedup = raw_median_seconds / lite_median_seconds
        warmup_token_match = (
            raw_measurement["warmup_token_ids"]
            == lite_measurement["warmup_token_ids"]
        )
        trial_tokens_match = (
            raw_measurement["token_ids"]
            == lite_measurement["token_ids"]
        )
        token_match = warmup_token_match and trial_tokens_match
        generated_match &= token_match
        row = {
            "tokens": int(key),
            "raw_best_seconds": raw_seconds,
            "litetopk_best_seconds": lite_seconds,
            "speedup": speedup,
            "raw_median_seconds": raw_median_seconds,
            "litetopk_median_seconds": lite_median_seconds,
            "median_speedup": median_speedup,
            "raw_throughput_tokens_per_second": int(key) / raw_seconds,
            "litetopk_throughput_tokens_per_second": int(key) / lite_seconds,
            "warmup_token_match": warmup_token_match,
            "trial_tokens_match": trial_tokens_match,
            "generated_token_match": token_match,
        }
        rows.append(row)
        print(
            f"{int(key):>9} {raw_seconds:>10.3f} {lite_seconds:>13.3f} "
            f"{speedup:>8.3f}x {median_speedup:>8.3f}x "
            f"{row['raw_throughput_tokens_per_second']:>11.0f} "
            f"{row['litetopk_throughput_tokens_per_second']:>12.0f} "
            f"{str(token_match):>7}"
        )

    summary = {
        "schema_version": 1,
        "family": raw["family"],
        "model": raw["model"],
        "raw_result": str(raw_path.resolve()),
        "litetopk_result": str(lite_path.resolve()),
        "generated_tokens_all_match": generated_match,
        "rows": rows,
    }
    if len(sys.argv) == 4:
        output = Path(sys.argv[3])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        print(f"summary: {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass
import hashlib

from .backends import BackendName
from .inference import (
    DecodeVariant,
    Gemma4Engine,
    PrefillCachePolicy,
    PrefillStepSize,
    PrefillSyncPolicy,
)
from .stats import RunStats, median_stats, reset_peak_memory

PREFILL_CACHE_POLICIES: tuple[PrefillCachePolicy, ...] = ("clear", "retain")
PREFILL_SYNC_POLICIES: tuple[PrefillSyncPolicy, ...] = ("eval",)
PREFILL_STEP_SIZES: tuple[PrefillStepSize, ...] = ("auto",)
DECODE_BENCHMARK_VARIANTS: tuple[DecodeVariant, ...] = (
    "custom",
    "custom_no_async",
    "custom_eval_next",
    "custom_defer_ids",
    "mlx_lm_generate_step",
)


@dataclass(frozen=True)
class BenchConfig:
    prompt_lengths: list[int]
    decode_lengths: list[int]
    warmups: int = 1
    runs: int = 3
    prefill_step_size: PrefillStepSize = "auto"
    prefill_step_sizes: tuple[PrefillStepSize, ...] | None = None
    prefill_sync_policies: tuple[PrefillSyncPolicy, ...] = PREFILL_SYNC_POLICIES
    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = 0
    max_kv_size: int | None = None
    prefill_cache_policies: tuple[PrefillCachePolicy, ...] = PREFILL_CACHE_POLICIES
    decode_variants: tuple[DecodeVariant, ...] = DECODE_BENCHMARK_VARIANTS
    mlx_memory_limit_gb: float | None = None
    mlx_cache_limit_gb: float | None = None
    mlx_wired_limit_gb: float | None = None
    include_token_ids: bool = False


def synthetic_prompt(token_count: int) -> str:
    return " ".join(["Benchmark"] * token_count)


def run_benchmark(
    *,
    model_path: str,
    backend: BackendName,
    config: BenchConfig,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    decode_variants = _with_custom_baseline(config.decode_variants)
    prefill_cache_policies = _with_clear_baseline(config.prefill_cache_policies)
    prefill_sync_policies = _with_eval_baseline(config.prefill_sync_policies)
    prefill_step_sizes = _with_auto_baseline(config.prefill_step_sizes or (config.prefill_step_size,))
    engine = Gemma4Engine(
        model_path=model_path,
        backend=backend,
        mlx_memory_limit_gb=config.mlx_memory_limit_gb,
        mlx_cache_limit_gb=config.mlx_cache_limit_gb,
        mlx_wired_limit_gb=config.mlx_wired_limit_gb,
    )
    for prompt_length in config.prompt_lengths:
        for decode_length in config.decode_lengths:
            prompt = synthetic_prompt(prompt_length)
            baseline_token_ids: list[int] | None = None
            for prefill_step_size in prefill_step_sizes:
                for prefill_sync_policy in prefill_sync_policies:
                    for prefill_cache_policy in prefill_cache_policies:
                        for decode_variant in decode_variants:
                            for _ in range(config.warmups):
                                engine.infer(
                                    prompt,
                                    max_tokens=decode_length,
                                    prompt_mode="raw",
                                    prefill_step_size=prefill_step_size,
                                    prefill_cache_policy=prefill_cache_policy,
                                    prefill_sync_policy=prefill_sync_policy,
                                    kv_bits=config.kv_bits,
                                    kv_group_size=config.kv_group_size,
                                    quantized_kv_start=config.quantized_kv_start,
                                    max_kv_size=config.max_kv_size,
                                    _decode_variant=decode_variant,
                                )

                            measured: list[RunStats] = []
                            token_runs: list[list[int]] = []
                            for _ in range(config.runs):
                                reset_peak_memory()
                                result = engine.infer(
                                    prompt,
                                    max_tokens=decode_length,
                                    prompt_mode="raw",
                                    prefill_step_size=prefill_step_size,
                                    prefill_cache_policy=prefill_cache_policy,
                                    prefill_sync_policy=prefill_sync_policy,
                                    kv_bits=config.kv_bits,
                                    kv_group_size=config.kv_group_size,
                                    quantized_kv_start=config.quantized_kv_start,
                                    max_kv_size=config.max_kv_size,
                                    _decode_variant=decode_variant,
                                )
                                measured.append(result.stats)
                                token_runs.append(result.token_ids)

                            if baseline_token_ids is None:
                                baseline_token_ids = token_runs[0] if token_runs else []
                            tokens_match_baseline = all(
                                row == baseline_token_ids for row in token_runs
                            )
                            median = median_stats(measured)

                            cases.append(
                                {
                                    "prompt_length_hint": prompt_length,
                                    "decode_tokens_requested": decode_length,
                                    "prefill_step_size": prefill_step_size,
                                    "prefill_sync_policy": prefill_sync_policy,
                                    "prefill_cache_policy": prefill_cache_policy,
                                    "decode_variant": decode_variant,
                                    "max_kv_size": config.max_kv_size,
                                    "tokens_match_baseline": tokens_match_baseline,
                                    "generated_token_count": len(token_runs[0])
                                    if token_runs
                                    else 0,
                                    "generated_token_hash": _token_hash(
                                        token_runs[0] if token_runs else []
                                    ),
                                    "runs": [row.to_dict() for row in measured],
                                    "median": median,
                                    "best_decode_tokens_per_second": max(
                                        row.decode_tokens_per_second for row in measured
                                    ),
                                    "best_total_tokens_per_second": max(
                                        row.total_tokens_per_second for row in measured
                                    ),
                                }
                            )
                            if config.include_token_ids:
                                cases[-1]["generated_token_ids"] = (
                                    token_runs[0] if token_runs else []
                                )
    payload = {
        "model_path": model_path,
        "backend_requested": backend,
        "prefill_step_sizes": list(prefill_step_sizes),
        "prefill_sync_policies": list(prefill_sync_policies),
        "decode_variants": list(decode_variants),
        "prefill_cache_policies": list(prefill_cache_policies),
        "max_kv_size": config.max_kv_size,
        "mlx_memory": {
            "memory_limit_gb": config.mlx_memory_limit_gb,
            "cache_limit_gb": config.mlx_cache_limit_gb,
            "wired_limit_gb": config.mlx_wired_limit_gb,
        },
        "cases": cases,
    }
    return payload


def _with_custom_baseline(variants: tuple[DecodeVariant, ...]) -> tuple[DecodeVariant, ...]:
    return ("custom", *(variant for variant in variants if variant != "custom"))


def _with_auto_baseline(values: tuple[PrefillStepSize, ...]) -> tuple[PrefillStepSize, ...]:
    return ("auto", *(value for value in values if value != "auto"))


def _with_eval_baseline(values: tuple[PrefillSyncPolicy, ...]) -> tuple[PrefillSyncPolicy, ...]:
    return ("eval", *(value for value in values if value != "eval"))


def _with_clear_baseline(
    policies: tuple[PrefillCachePolicy, ...],
) -> tuple[PrefillCachePolicy, ...]:
    return ("clear", *(policy for policy in policies if policy != "clear"))


def _float_or_none(value: object) -> float | None:
    return float(value) if value is not None else None


def _token_hash(token_ids: list[int]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(len(token_ids).to_bytes(8, "little"))
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def benchmark_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def benchmark_summary(payload: dict[str, object]) -> str:
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return "No benchmark cases recorded."

    baselines = _baseline_cases(payload)
    rows: list[list[str]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        median = case.get("median", {})
        if not isinstance(median, dict):
            median = {}
        group_key = (
            int(case["prompt_length_hint"]),
            int(case["decode_tokens_requested"]),
        )
        baseline_median = baselines.get(group_key, {}).get("median", {})
        if not isinstance(baseline_median, dict):
            baseline_median = {}
        prefill_tps = _float_or_none(median.get("prefill_tokens_per_second_median"))
        decode_tps = _float_or_none(median.get("decode_tokens_per_second_median"))
        total_tps = _float_or_none(median.get("total_tokens_per_second_median"))
        ttft = _float_or_none(median.get("time_to_first_token_seconds_median"))
        peak_memory = _float_or_none(median.get("peak_memory_gb_median"))
        encode = _float_or_none(median.get("encode_seconds_median"))
        prefill_model = _float_or_none(median.get("prefill_model_seconds_median"))
        prefill_sync = _float_or_none(median.get("prefill_sync_seconds_median"))
        prefill_clear = _float_or_none(median.get("prefill_clear_cache_seconds_median"))
        decode_sync = _float_or_none(median.get("decode_sync_seconds_median"))
        decode_item = _float_or_none(median.get("decode_token_item_seconds_median"))
        decode_p95 = _float_or_none(median.get("decode_token_latency_p95_seconds_median"))
        baseline_prefill_tps = _float_or_none(
            baseline_median.get("prefill_tokens_per_second_median")
        )
        baseline_decode_tps = _float_or_none(
            baseline_median.get("decode_tokens_per_second_median")
        )
        rows.append(
            [
                str(group_key[0]),
                str(group_key[1]),
                str(case["prefill_step_size"]),
                str(case["prefill_sync_policy"]),
                str(case["prefill_cache_policy"]),
                str(case["decode_variant"]),
                "yes" if case.get("tokens_match_baseline") else "no",
                _format_float(prefill_tps, 1),
                _format_float(decode_tps, 1),
                _format_float(total_tps, 1),
                _format_float(ttft, 3),
                _format_float(peak_memory, 2),
                _format_float(encode, 3),
                _format_float(prefill_model, 3),
                _format_float(prefill_sync, 3),
                _format_float(prefill_clear, 3),
                _format_float(decode_sync, 3),
                _format_float(decode_item, 3),
                _format_float(decode_p95, 3),
                _format_ratio(_ratio(prefill_tps or 0.0, baseline_prefill_tps or 0.0)),
                _format_ratio(_ratio(decode_tps or 0.0, baseline_decode_tps or 0.0)),
            ]
        )

    lines = [
        f"Benchmark summary for {payload.get('model_path')} "
        f"(backend={payload.get('backend_requested')})",
        "",
        _format_table(
            [
                "prompt",
                "decode",
                "step",
                "sync",
                "cache",
                "variant",
                "match",
                "prefill tok/s",
                "decode tok/s",
                "total tok/s",
                "ttft s",
                "peak GB",
                "encode s",
                "prefill model s",
                "prefill sync s",
                "prefill clear s",
                "decode sync s",
                "decode item s",
                "decode p95 s",
                "prefill x",
                "decode x",
            ],
            rows,
        ),
    ]
    return "\n".join(lines)


def _baseline_cases(payload: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        return {}
    baselines = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        if (
            case.get("prefill_step_size") == "auto"
            and case.get("prefill_sync_policy") == "eval"
            and case.get("prefill_cache_policy") == "clear"
            and case.get("decode_variant") == "custom"
        ):
            baselines[
                (
                    int(case["prompt_length_hint"]),
                    int(case["decode_tokens_requested"]),
                )
            ] = case
    return baselines


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_float(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}x"

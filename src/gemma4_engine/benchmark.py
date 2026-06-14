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
    draft_model_path: str | None = None
    draft_tokens: int = 4
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
        draft_model_path=config.draft_model_path,
        draft_tokens=config.draft_tokens,
        mlx_memory_limit_gb=config.mlx_memory_limit_gb,
        mlx_cache_limit_gb=config.mlx_cache_limit_gb,
        mlx_wired_limit_gb=config.mlx_wired_limit_gb,
    )
    for prompt_length in config.prompt_lengths:
        for decode_length in config.decode_lengths:
            prompt = synthetic_prompt(prompt_length)
            baseline_token_ids: list[int] | None = None
            baseline_peak_memory_gb: float | None = None
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
                            if (
                                prefill_step_size == "auto"
                                and decode_variant == "custom"
                                and prefill_cache_policy == "clear"
                                and prefill_sync_policy == "eval"
                            ):
                                baseline_peak_memory_gb = _float_or_none(
                                    median.get("peak_memory_gb_median")
                                )
                            peak_memory_regression_gb = _memory_regression(
                                _float_or_none(median.get("peak_memory_gb_median")),
                                baseline_peak_memory_gb,
                            )

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
                                    "peak_memory_regression_gb": peak_memory_regression_gb,
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
        "promotion_gate": {
            "baseline_prefill_step_size": "auto",
            "baseline_prefill_sync_policy": "eval",
            "baseline_decode_variant": "custom",
            "baseline_prefill_cache_policy": "clear",
            "min_decode_tokens_per_second_improvement": 0.05,
            "max_peak_memory_regression_gb": 0.5,
            "requires_tokens_match_baseline": True,
        },
        "prefill_promotion_gate": {
            "baseline_prefill_step_size": "auto",
            "baseline_prefill_sync_policy": "eval",
            "baseline_decode_variant": "custom",
            "baseline_prefill_cache_policy": "clear",
            "min_prefill_tokens_per_second_improvement": 0.05,
            "max_decode_tokens_per_second_regression": 0.05,
            "max_peak_memory_regression_gb": 0.5,
            "requires_tokens_match_baseline": True,
        },
        "max_kv_size_feasibility": {
            "status": "enabled" if config.max_kv_size is not None else "not_enabled",
            "reason": (
                "passed to mlx_lm cache.make_prompt_cache/generate_step"
                if config.max_kv_size is not None
                else "not requested; default MLX prompt cache size is used"
            ),
        },
        "cases": cases,
    }
    payload["promotion_analysis"] = _promotion_analysis(payload)
    payload["prefill_promotion_analysis"] = _prefill_promotion_analysis(payload)
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


def _memory_regression(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _token_hash(token_ids: list[int]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(len(token_ids).to_bytes(8, "little"))
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def _promotion_analysis(payload: dict[str, object]) -> dict[str, object]:
    gate = payload["promotion_gate"]
    cases = payload["cases"]
    if not isinstance(gate, dict) or not isinstance(cases, list):
        return {"status": "invalid", "reason": "benchmark payload is malformed"}

    baseline_step_size = gate["baseline_prefill_step_size"]
    baseline_sync_policy = gate["baseline_prefill_sync_policy"]
    baseline_variant = gate["baseline_decode_variant"]
    baseline_policy = gate["baseline_prefill_cache_policy"]
    min_improvement = float(gate["min_decode_tokens_per_second_improvement"])
    max_memory_regression = float(gate["max_peak_memory_regression_gb"])
    grouped: dict[tuple[int, int], dict[tuple[str, str, str, str], dict[str, object]]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        key = (int(case["prompt_length_hint"]), int(case["decode_tokens_requested"]))
        candidate_key = (
            str(case["prefill_step_size"]),
            str(case["prefill_sync_policy"]),
            str(case["prefill_cache_policy"]),
            str(case["decode_variant"]),
        )
        grouped.setdefault(key, {})[candidate_key] = case

    candidate_keys = sorted(
        {
            candidate_key
            for group in grouped.values()
            for candidate_key in group
            if candidate_key
            != (baseline_step_size, baseline_sync_policy, baseline_policy, baseline_variant)
        }
    )
    candidate_results: list[dict[str, object]] = []
    for candidate_key in candidate_keys:
        failures: list[str] = []
        decode_speedups: list[float] = []
        prefill_speedups: list[float] = []
        memory_regressions: list[float] = []
        for group_key, group in grouped.items():
            baseline = group.get(
                (baseline_step_size, baseline_sync_policy, baseline_policy, baseline_variant)
            )
            candidate = group.get(candidate_key)
            if baseline is None or candidate is None:
                failures.append(f"missing case prompt={group_key[0]} decode={group_key[1]}")
                continue
            if not candidate.get("tokens_match_baseline"):
                failures.append(f"token mismatch prompt={group_key[0]} decode={group_key[1]}")
            baseline_median = baseline["median"]
            candidate_median = candidate["median"]
            if not isinstance(baseline_median, dict) or not isinstance(candidate_median, dict):
                failures.append(f"missing medians prompt={group_key[0]} decode={group_key[1]}")
                continue
            baseline_decode = float(baseline_median["decode_tokens_per_second_median"])
            candidate_decode = float(candidate_median["decode_tokens_per_second_median"])
            baseline_prefill = float(baseline_median["prefill_tokens_per_second_median"])
            candidate_prefill = float(candidate_median["prefill_tokens_per_second_median"])
            decode_speedup = _ratio(candidate_decode, baseline_decode)
            prefill_speedup = _ratio(candidate_prefill, baseline_prefill)
            if decode_speedup is None:
                failures.append(f"zero baseline decode speed prompt={group_key[0]} decode={group_key[1]}")
            else:
                decode_speedups.append(decode_speedup)
                if decode_speedup < 1.0 + min_improvement:
                    failures.append(
                        f"decode speedup {decode_speedup:.3f}x below gate "
                        f"prompt={group_key[0]} decode={group_key[1]}"
                    )
            if prefill_speedup is not None:
                prefill_speedups.append(prefill_speedup)
            memory_regression = candidate.get("peak_memory_regression_gb")
            if memory_regression is not None:
                memory_regression = float(memory_regression)
                memory_regressions.append(memory_regression)
                if memory_regression > max_memory_regression:
                    failures.append(
                        f"peak memory regression {memory_regression:.3f} GB above gate "
                        f"prompt={group_key[0]} decode={group_key[1]}"
                    )

        candidate_results.append(
            {
                "prefill_step_size": candidate_key[0],
                "prefill_sync_policy": candidate_key[1],
                "prefill_cache_policy": candidate_key[2],
                "decode_variant": candidate_key[3],
                "passes": not failures,
                "failures": failures,
                "min_decode_speedup": min(decode_speedups) if decode_speedups else None,
                "median_decode_speedup": _median_or_none(decode_speedups),
                "min_prefill_speedup": min(prefill_speedups) if prefill_speedups else None,
                "max_peak_memory_regression_gb": max(memory_regressions)
                if memory_regressions
                else None,
            }
        )

    passing = [candidate for candidate in candidate_results if candidate["passes"]]
    if not passing:
        return {
            "status": "no_candidate_passed",
            "baseline": {
                "prefill_step_size": baseline_step_size,
                "prefill_sync_policy": baseline_sync_policy,
                "prefill_cache_policy": baseline_policy,
                "decode_variant": baseline_variant,
            },
            "candidates": candidate_results,
        }
    best = max(
        passing,
        key=lambda candidate: float(candidate["median_decode_speedup"] or 0.0),
    )
    return {
        "status": "candidate_passed",
        "recommended_default": {
            "prefill_step_size": best["prefill_step_size"],
            "prefill_sync_policy": best["prefill_sync_policy"],
            "prefill_cache_policy": best["prefill_cache_policy"],
            "decode_variant": best["decode_variant"],
        },
        "baseline": {
            "prefill_step_size": baseline_step_size,
            "prefill_sync_policy": baseline_sync_policy,
            "prefill_cache_policy": baseline_policy,
            "decode_variant": baseline_variant,
        },
        "candidates": candidate_results,
    }


def _prefill_promotion_analysis(payload: dict[str, object]) -> dict[str, object]:
    gate = payload["prefill_promotion_gate"]
    cases = payload["cases"]
    if not isinstance(gate, dict) or not isinstance(cases, list):
        return {"status": "invalid", "reason": "benchmark payload is malformed"}

    baseline_step_size = gate["baseline_prefill_step_size"]
    baseline_sync_policy = gate["baseline_prefill_sync_policy"]
    baseline_variant = gate["baseline_decode_variant"]
    baseline_policy = gate["baseline_prefill_cache_policy"]
    min_prefill_improvement = float(gate["min_prefill_tokens_per_second_improvement"])
    max_decode_regression = float(gate["max_decode_tokens_per_second_regression"])
    max_memory_regression = float(gate["max_peak_memory_regression_gb"])
    grouped: dict[tuple[int, int], dict[tuple[str, str, str, str], dict[str, object]]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        key = (int(case["prompt_length_hint"]), int(case["decode_tokens_requested"]))
        candidate_key = (
            str(case["prefill_step_size"]),
            str(case["prefill_sync_policy"]),
            str(case["prefill_cache_policy"]),
            str(case["decode_variant"]),
        )
        grouped.setdefault(key, {})[candidate_key] = case

    baseline_key = (baseline_step_size, baseline_sync_policy, baseline_policy, baseline_variant)
    candidate_keys = sorted(
        {
            candidate_key
            for group in grouped.values()
            for candidate_key in group
            if candidate_key != baseline_key
        }
    )
    candidate_results: list[dict[str, object]] = []
    for candidate_key in candidate_keys:
        failures: list[str] = []
        prefill_speedups: list[float] = []
        decode_speedups: list[float] = []
        memory_regressions: list[float] = []
        for group_key, group in grouped.items():
            baseline = group.get(baseline_key)
            candidate = group.get(candidate_key)
            if baseline is None or candidate is None:
                failures.append(f"missing case prompt={group_key[0]} decode={group_key[1]}")
                continue
            if not candidate.get("tokens_match_baseline"):
                failures.append(f"token mismatch prompt={group_key[0]} decode={group_key[1]}")
            baseline_median = baseline["median"]
            candidate_median = candidate["median"]
            if not isinstance(baseline_median, dict) or not isinstance(candidate_median, dict):
                failures.append(f"missing medians prompt={group_key[0]} decode={group_key[1]}")
                continue
            baseline_prefill = float(baseline_median["prefill_tokens_per_second_median"])
            candidate_prefill = float(candidate_median["prefill_tokens_per_second_median"])
            baseline_decode = float(baseline_median["decode_tokens_per_second_median"])
            candidate_decode = float(candidate_median["decode_tokens_per_second_median"])
            prefill_speedup = _ratio(candidate_prefill, baseline_prefill)
            decode_speedup = _ratio(candidate_decode, baseline_decode)
            if prefill_speedup is None:
                failures.append(
                    f"zero baseline prefill speed prompt={group_key[0]} decode={group_key[1]}"
                )
            else:
                prefill_speedups.append(prefill_speedup)
                if prefill_speedup < 1.0 + min_prefill_improvement:
                    failures.append(
                        f"prefill speedup {prefill_speedup:.3f}x below gate "
                        f"prompt={group_key[0]} decode={group_key[1]}"
                    )
            if decode_speedup is not None:
                decode_speedups.append(decode_speedup)
                if decode_speedup < 1.0 - max_decode_regression:
                    failures.append(
                        f"decode speedup {decode_speedup:.3f}x below regression gate "
                        f"prompt={group_key[0]} decode={group_key[1]}"
                    )
            memory_regression = candidate.get("peak_memory_regression_gb")
            if memory_regression is not None:
                memory_regression = float(memory_regression)
                memory_regressions.append(memory_regression)
                if memory_regression > max_memory_regression:
                    failures.append(
                        f"peak memory regression {memory_regression:.3f} GB above gate "
                        f"prompt={group_key[0]} decode={group_key[1]}"
                    )

        candidate_results.append(
            {
                "prefill_step_size": candidate_key[0],
                "prefill_sync_policy": candidate_key[1],
                "prefill_cache_policy": candidate_key[2],
                "decode_variant": candidate_key[3],
                "passes": not failures,
                "failures": failures,
                "min_prefill_speedup": min(prefill_speedups) if prefill_speedups else None,
                "median_prefill_speedup": _median_or_none(prefill_speedups),
                "min_decode_speedup": min(decode_speedups) if decode_speedups else None,
                "max_peak_memory_regression_gb": max(memory_regressions)
                if memory_regressions
                else None,
            }
        )

    passing = [candidate for candidate in candidate_results if candidate["passes"]]
    if not passing:
        return {
            "status": "no_candidate_passed",
            "baseline": {
                "prefill_step_size": baseline_step_size,
                "prefill_sync_policy": baseline_sync_policy,
                "prefill_cache_policy": baseline_policy,
                "decode_variant": baseline_variant,
            },
            "candidates": candidate_results,
        }
    best = max(
        passing,
        key=lambda candidate: float(candidate["median_prefill_speedup"] or 0.0),
    )
    return {
        "status": "candidate_passed",
        "recommended_prefill_default": {
            "prefill_step_size": best["prefill_step_size"],
            "prefill_sync_policy": best["prefill_sync_policy"],
            "prefill_cache_policy": best["prefill_cache_policy"],
            "decode_variant": best["decode_variant"],
        },
        "baseline": {
            "prefill_step_size": baseline_step_size,
            "prefill_sync_policy": baseline_sync_policy,
            "prefill_cache_policy": baseline_policy,
            "decode_variant": baseline_variant,
        },
        "candidates": candidate_results,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def benchmark_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)

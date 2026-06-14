from __future__ import annotations

from dataclasses import dataclass

from .backends import BackendName
from .inference import Gemma4Engine, PrefillCachePolicy, PrefillStepSize, PrefillSyncPolicy
from .stats import RunStats


@dataclass
class CompareResult:
    baseline: str
    matches: bool
    engine_tokens: list[int]
    baseline_text: str
    engine_text: str
    engine_stats: RunStats
    baseline_stats: dict[str, float | int | str | None]
    speedup: dict[str, float | None]


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def compare_with_mlx_lm(
    *,
    prompt: str,
    model_path: str,
    max_tokens: int,
    backend: BackendName,
    prefill_step_size: PrefillStepSize = "auto",
    prefill_cache_policy: PrefillCachePolicy = "clear",
    prefill_sync_policy: PrefillSyncPolicy = "eval",
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: int | None = None,
    draft_model_path: str | None = None,
    draft_tokens: int = 4,
    mlx_memory_limit_gb: float | None = None,
    mlx_cache_limit_gb: float | None = None,
    mlx_wired_limit_gb: float | None = None,
) -> CompareResult:
    from mlx_lm import stream_generate

    engine = Gemma4Engine(
        model_path=model_path,
        backend=backend,
        draft_model_path=draft_model_path,
        draft_tokens=draft_tokens,
        mlx_memory_limit_gb=mlx_memory_limit_gb,
        mlx_cache_limit_gb=mlx_cache_limit_gb,
        mlx_wired_limit_gb=mlx_wired_limit_gb,
    )
    engine_result = engine.infer(
        prompt,
        max_tokens=max_tokens,
        prompt_mode="raw",
        prefill_step_size=prefill_step_size,
        prefill_cache_policy=prefill_cache_policy,
        prefill_sync_policy=prefill_sync_policy,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        max_kv_size=max_kv_size,
    )
    baseline_text = ""
    baseline_response = None
    for response in stream_generate(
        engine.loaded.model,
        engine.loaded.tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        baseline_text += response.text
        baseline_response = response

    if baseline_response is None:
        baseline_stats: dict[str, float | int | str | None] = {
            "backend": "mlx_lm",
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "prefill_tokens_per_second": 0.0,
            "decode_tokens_per_second": 0.0,
            "peak_memory_gb": None,
        }
    else:
        baseline_stats = {
            "backend": "mlx_lm",
            "prompt_tokens": baseline_response.prompt_tokens,
            "generated_tokens": baseline_response.generation_tokens,
            "prefill_tokens_per_second": baseline_response.prompt_tps,
            "decode_tokens_per_second": baseline_response.generation_tps,
            "peak_memory_gb": baseline_response.peak_memory,
            "finish_reason": baseline_response.finish_reason,
        }

    speedup = {
        "prefill": _ratio(
            engine_result.stats.prefill_tokens_per_second,
            float(baseline_stats["prefill_tokens_per_second"]),
        ),
        "decode": _ratio(
            engine_result.stats.decode_tokens_per_second,
            float(baseline_stats["decode_tokens_per_second"]),
        ),
    }

    return CompareResult(
        baseline="mlx_lm",
        matches=engine_result.text == baseline_text,
        engine_tokens=engine_result.token_ids,
        baseline_text=baseline_text,
        engine_text=engine_result.text,
        engine_stats=engine_result.stats,
        baseline_stats=baseline_stats,
        speedup=speedup,
    )

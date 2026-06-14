from __future__ import annotations

from dataclasses import dataclass

from .backends import BackendName
from .inference import infer
from .loader import load_model
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
) -> CompareResult:
    from mlx_lm import stream_generate

    engine = infer(
        prompt,
        model_path=model_path,
        max_tokens=max_tokens,
        backend=backend,
        prompt_mode="raw",
    )
    loaded = load_model(model_path)
    baseline_text = ""
    baseline_response = None
    for response in stream_generate(
        loaded.model,
        loaded.tokenizer,
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
            engine.stats.prefill_tokens_per_second,
            float(baseline_stats["prefill_tokens_per_second"]),
        ),
        "decode": _ratio(
            engine.stats.decode_tokens_per_second,
            float(baseline_stats["decode_tokens_per_second"]),
        ),
    }

    return CompareResult(
        baseline="mlx_lm",
        matches=engine.text == baseline_text,
        engine_tokens=engine.token_ids,
        baseline_text=baseline_text,
        engine_text=engine.text,
        engine_stats=engine.stats,
        baseline_stats=baseline_stats,
        speedup=speedup,
    )

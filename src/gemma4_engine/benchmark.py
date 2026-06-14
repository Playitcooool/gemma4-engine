from __future__ import annotations

import json
from dataclasses import dataclass

from .backends import BackendName
from .inference import Gemma4Engine, PrefillStepSize
from .stats import RunStats, median_stats


@dataclass(frozen=True)
class BenchConfig:
    prompt_lengths: list[int]
    decode_lengths: list[int]
    warmups: int = 1
    runs: int = 3
    prefill_step_size: PrefillStepSize = "auto"
    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = 0
    draft_model_path: str | None = None
    draft_tokens: int = 4


def synthetic_prompt(token_count: int) -> str:
    return " ".join(["Benchmark"] * token_count)


def run_benchmark(
    *,
    model_path: str,
    backend: BackendName,
    config: BenchConfig,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    engine = Gemma4Engine(
        model_path=model_path,
        backend=backend,
        draft_model_path=config.draft_model_path,
        draft_tokens=config.draft_tokens,
    )
    for prompt_length in config.prompt_lengths:
        for decode_length in config.decode_lengths:
            prompt = synthetic_prompt(prompt_length)
            for _ in range(config.warmups):
                engine.infer(
                    prompt,
                    max_tokens=decode_length,
                    prompt_mode="raw",
                    prefill_step_size=config.prefill_step_size,
                    kv_bits=config.kv_bits,
                    kv_group_size=config.kv_group_size,
                    quantized_kv_start=config.quantized_kv_start,
                )

            measured: list[RunStats] = []
            for _ in range(config.runs):
                measured.append(
                    engine.infer(
                        prompt,
                        max_tokens=decode_length,
                        prompt_mode="raw",
                        prefill_step_size=config.prefill_step_size,
                        kv_bits=config.kv_bits,
                        kv_group_size=config.kv_group_size,
                        quantized_kv_start=config.quantized_kv_start,
                    ).stats
                )

            cases.append(
                {
                    "prompt_length_hint": prompt_length,
                    "decode_tokens_requested": decode_length,
                    "runs": [row.to_dict() for row in measured],
                    "median": median_stats(measured),
                    "best_total_tokens_per_second": max(
                        row.total_tokens_per_second for row in measured
                    ),
                }
            )
    return {"model_path": model_path, "backend_requested": backend, "cases": cases}


def benchmark_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)

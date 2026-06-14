from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable


@dataclass
class RunStats:
    model_path: str
    backend: str
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    time_to_first_token_seconds: float
    peak_memory_gb: float | None = None
    active_memory_gb: float | None = None
    cache_memory_gb: float | None = None

    @property
    def prefill_tokens_per_second(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds > 0 else 0.0

    @property
    def decode_tokens_per_second(self) -> float:
        return self.generated_tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0

    @property
    def total_tokens_per_second(self) -> float:
        total_tokens = self.prompt_tokens + self.generated_tokens
        total_seconds = self.prefill_seconds + self.decode_seconds
        return total_tokens / total_seconds if total_seconds > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "prefill_tokens_per_second": self.prefill_tokens_per_second,
                "decode_tokens_per_second": self.decode_tokens_per_second,
                "total_tokens_per_second": self.total_tokens_per_second,
            }
        )
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def now() -> float:
    return time.perf_counter()


def memory_snapshot() -> dict[str, float | None]:
    try:
        import mlx.core as mx
    except Exception:
        return {"peak_memory_gb": None, "active_memory_gb": None, "cache_memory_gb": None}

    return {
        "peak_memory_gb": mx.get_peak_memory() / 1_000_000_000,
        "active_memory_gb": mx.get_active_memory() / 1_000_000_000,
        "cache_memory_gb": mx.get_cache_memory() / 1_000_000_000,
    }


def reset_peak_memory() -> None:
    try:
        import mlx.core as mx
    except Exception:
        return
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def median_stats(stats: Iterable[RunStats]) -> dict[str, float | None]:
    rows = list(stats)
    payload = {
        "prefill_tokens_per_second_median": median(row.prefill_tokens_per_second for row in rows),
        "decode_tokens_per_second_median": median(row.decode_tokens_per_second for row in rows),
        "total_tokens_per_second_median": median(row.total_tokens_per_second for row in rows),
        "time_to_first_token_seconds_median": median(
            row.time_to_first_token_seconds for row in rows
        ),
    }
    for field in ("peak_memory_gb", "active_memory_gb", "cache_memory_gb"):
        values = [getattr(row, field) for row in rows if getattr(row, field) is not None]
        payload[f"{field}_median"] = median(values) if values else None
    return payload

from types import SimpleNamespace

import gemma4_engine.benchmark as benchmark
from gemma4_engine.benchmark import BenchConfig, benchmark_summary, run_benchmark
from gemma4_engine.stats import RunStats


class FakeEngine:
    def __init__(self, **kwargs) -> None:
        self.model_path = kwargs["model_path"]

    def infer(self, prompt: str, **kwargs):
        token_ids = [1, 2, 3]
        stats = RunStats(
            model_path=self.model_path,
            backend="mlx",
            prompt_tokens=len(prompt.split()),
            generated_tokens=len(token_ids),
            prefill_seconds=0.25 if kwargs["prefill_step_size"] == "1024" else 0.5,
            decode_seconds=0.25 if kwargs["_decode_variant"] == "custom_no_async" else 0.5,
            time_to_first_token_seconds=0.6,
            peak_memory_gb=2.0,
            active_memory_gb=1.0,
            cache_memory_gb=0.5,
            speculative_acceptance_rate=0.5
            if kwargs["_decode_variant"] == "custom_speculative_ngram"
            else None,
        )
        return SimpleNamespace(token_ids=token_ids, stats=stats)


def test_benchmark_reports_cases_and_summary(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "Gemma4Engine", FakeEngine)

    payload = run_benchmark(
        model_path="fake",
        backend="mlx",
        config=BenchConfig(
            prompt_lengths=[128],
            decode_lengths=[64],
            warmups=0,
            runs=1,
            decode_variants=("custom_no_async", "custom_speculative_ngram", "custom"),
            prefill_cache_policies=("clear",),
            prefill_step_sizes=("1024", "auto"),
            max_kv_size=4096,
        ),
    )

    cases = payload["cases"]
    assert payload["prefill_step_sizes"] == ["auto", "1024"]
    assert payload["decode_variants"] == [
        "custom",
        "custom_no_async",
        "custom_speculative_ngram",
    ]
    assert [case["prefill_step_size"] for case in cases] == [
        "auto",
        "auto",
        "auto",
        "1024",
        "1024",
        "1024",
    ]
    assert [case["decode_variant"] for case in cases] == [
        "custom",
        "custom_no_async",
        "custom_speculative_ngram",
        "custom",
        "custom_no_async",
        "custom_speculative_ngram",
    ]
    assert cases[0]["tokens_match_baseline"] is True
    assert cases[0]["generated_token_count"] == 3
    assert cases[0]["generated_token_hash"]
    assert cases[0]["median"]["prefill_tokens_per_second_median"] == 256.0
    assert cases[1]["median"]["decode_tokens_per_second_median"] == 12.0
    assert cases[2]["median"]["speculative_acceptance_rate_median"] == 0.5

    summary = benchmark_summary(payload)
    assert "Benchmark summary for fake (backend=mlx)" in summary
    assert "prefill tok/s" in summary
    assert "decode tok/s" in summary
    assert "prefill model s" in summary
    assert "prefill sync s" in summary
    assert "decode sync s" in summary
    assert "decode p95 s" in summary
    assert "spec accept" in summary
    assert "custom_no_async" in summary
    assert "custom_speculative_ngram" in summary
    assert "2.000x" in summary


def test_benchmark_can_include_full_token_ids(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "Gemma4Engine", FakeEngine)

    payload = run_benchmark(
        model_path="fake",
        backend="mlx",
        config=BenchConfig(
            prompt_lengths=[128],
            decode_lengths=[64],
            warmups=0,
            runs=1,
            decode_variants=("custom",),
            prefill_cache_policies=("clear",),
            include_token_ids=True,
        ),
    )

    assert payload["cases"][0]["generated_token_ids"] == [1, 2, 3]

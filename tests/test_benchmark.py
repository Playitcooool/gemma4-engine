from types import SimpleNamespace

import gemma4_engine.benchmark as benchmark
from gemma4_engine.benchmark import (
    BenchConfig,
    BenchScenario,
    benchmark_summary,
    run_benchmark,
    single_user_latency_scenarios,
)
from gemma4_engine.stats import RunStats


class FakeEngine:
    def __init__(self, **kwargs) -> None:
        self.model_path = kwargs["model_path"]
        self.reset_sessions: list[str] = []

    def reset_session(self, session_id: str) -> None:
        self.reset_sessions.append(session_id)

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
    assert payload["benchmark_profile"] == "matrix"
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
    assert cases[0]["scenario"] == "synthetic_128_64"
    assert cases[0]["generated_token_count"] == 3
    assert cases[0]["generated_token_hash"]
    assert cases[0]["median"]["prefill_tokens_per_second_median"] == 256.0
    assert cases[1]["median"]["decode_tokens_per_second_median"] == 12.0
    assert cases[2]["median"]["speculative_acceptance_rate_median"] == 0.5

    summary = benchmark_summary(payload)
    assert "Benchmark summary for fake (backend=mlx)" in summary
    assert "scenario" in summary
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


def test_single_user_latency_profile_scenarios_cover_plan_cases() -> None:
    scenarios = single_user_latency_scenarios()
    names = {scenario.name for scenario in scenarios}

    assert "short_chat_128_64" in names
    assert "long_prompt_16384_128" in names
    assert "repeated_prefix_8192_512" in names
    assert "multi_turn_3" in names
    assert any(scenario.cache_prefix for scenario in scenarios)
    assert any(scenario.session_id for scenario in scenarios)


def test_benchmark_scenario_passes_prefix_and_session_setup(monkeypatch) -> None:
    engines: list[FakeEngine] = []

    class TrackingEngine(FakeEngine):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.calls: list[dict[str, object]] = []
            engines.append(self)

        def infer(self, prompt: str, **kwargs):
            self.calls.append({"prompt": prompt, **kwargs})
            return super().infer(prompt, **kwargs)

    monkeypatch.setattr(benchmark, "Gemma4Engine", TrackingEngine)

    payload = run_benchmark(
        model_path="fake",
        backend="mlx",
        config=BenchConfig(
            prompt_lengths=[],
            decode_lengths=[],
            warmups=0,
            runs=1,
            prefill_step_sizes=("auto",),
            prefill_cache_policies=("clear",),
            decode_variants=("custom",),
            scenarios=(
                BenchScenario(
                    name="scenario",
                    prompt="prefix suffix",
                    prompt_length_hint=2,
                    decode_length=1,
                    cache_prefix="prefix",
                    session_id="session",
                    append_to_session=True,
                    setup_prompts=("prefix",),
                ),
            ),
            benchmark_profile="single-user-latency",
        ),
    )

    engine = engines[0]
    assert payload["benchmark_profile"] == "single-user-latency"
    assert payload["cases"][0]["scenario"] == "scenario"
    assert payload["cases"][0]["cache_prefix_tokens"] == 1
    assert payload["cases"][0]["session_id"] == "session"
    assert engine.reset_sessions == ["session"]
    assert engine.calls[0]["prompt"] == "prefix"
    assert engine.calls[0]["append_to_session"] is True
    assert engine.calls[1]["cache_prefix"] == "prefix"
    assert engine.calls[1]["session_id"] == "session"

from types import SimpleNamespace

import gemma4_engine.benchmark as benchmark
from gemma4_engine.benchmark import BenchConfig, run_benchmark
from gemma4_engine.stats import RunStats


def test_benchmark_reports_decode_variant_metadata(monkeypatch) -> None:
    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.model_path = kwargs["model_path"]

        def infer(self, prompt: str, **kwargs):
            variant = kwargs["_decode_variant"]
            token_ids = [1, 2, 3]
            if variant == "custom_eval_next":
                token_ids = [1, 2, 4]
            stats = RunStats(
                model_path=self.model_path,
                backend="mlx",
                prompt_tokens=len(prompt.split()),
                generated_tokens=len(token_ids),
                prefill_seconds=0.5,
                decode_seconds=0.25 if variant == "custom_no_async" else 0.5,
                time_to_first_token_seconds=0.6,
                peak_memory_gb=2.2 if variant == "custom_no_async" else 2.0,
                active_memory_gb=1.0,
                cache_memory_gb=0.5,
            )
            return SimpleNamespace(token_ids=token_ids, stats=stats)

    monkeypatch.setattr(benchmark, "Gemma4Engine", FakeEngine)

    payload = run_benchmark(
        model_path="fake",
        backend="mlx",
        config=BenchConfig(
            prompt_lengths=[128],
            decode_lengths=[64],
            warmups=0,
            runs=1,
            decode_variants=("custom", "custom_no_async", "custom_eval_next"),
            prefill_cache_policies=("clear",),
            max_kv_size=4096,
            mlx_memory_limit_gb=48,
            mlx_cache_limit_gb=40,
            mlx_wired_limit_gb=32,
        ),
    )

    cases = payload["cases"]
    assert payload["prefill_step_sizes"] == ["auto"]
    assert payload["prefill_sync_policies"] == ["eval"]
    assert payload["decode_variants"] == ["custom", "custom_no_async", "custom_eval_next"]
    assert payload["prefill_cache_policies"] == ["clear"]
    assert payload["max_kv_size"] == 4096
    assert payload["mlx_memory"] == {
        "memory_limit_gb": 48,
        "cache_limit_gb": 40,
        "wired_limit_gb": 32,
    }
    assert payload["promotion_gate"]["max_peak_memory_regression_gb"] == 0.5
    assert payload["promotion_gate"]["baseline_prefill_sync_policy"] == "eval"
    assert payload["promotion_gate"]["baseline_prefill_cache_policy"] == "clear"
    assert payload["prefill_promotion_gate"]["min_prefill_tokens_per_second_improvement"] == 0.05
    assert payload["max_kv_size_feasibility"]["status"] == "enabled"
    assert [case["decode_variant"] for case in cases] == [
        "custom",
        "custom_no_async",
        "custom_eval_next",
    ]
    assert [case["prefill_step_size"] for case in cases] == ["auto", "auto", "auto"]
    assert [case["prefill_sync_policy"] for case in cases] == ["eval", "eval", "eval"]
    assert [case["prefill_cache_policy"] for case in cases] == ["clear", "clear", "clear"]
    assert [case["max_kv_size"] for case in cases] == [4096, 4096, 4096]
    assert cases[0]["tokens_match_baseline"] is True
    assert cases[1]["tokens_match_baseline"] is True
    assert round(cases[1]["peak_memory_regression_gb"], 3) == 0.2
    assert cases[2]["tokens_match_baseline"] is False
    assert cases[2]["generated_token_count"] == 3
    assert cases[2]["generated_token_hash"] != cases[0]["generated_token_hash"]
    assert "generated_token_ids" not in cases[2]
    assert payload["promotion_analysis"]["status"] == "candidate_passed"
    assert any(
        candidate["decode_variant"] == "custom_eval_next" and not candidate["passes"]
        for candidate in payload["promotion_analysis"]["candidates"]
    )
    assert payload["promotion_analysis"]["recommended_default"] == {
        "prefill_step_size": "auto",
        "prefill_sync_policy": "eval",
        "prefill_cache_policy": "clear",
        "decode_variant": "custom_no_async",
    }


def test_benchmark_clear_policy_is_always_baseline(monkeypatch) -> None:
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
                prefill_seconds=0.5,
                decode_seconds=0.5,
                time_to_first_token_seconds=0.6,
                peak_memory_gb=2.0,
            )
            return SimpleNamespace(token_ids=token_ids, stats=stats)

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
            prefill_cache_policies=("retain", "clear"),
        ),
    )

    assert payload["prefill_cache_policies"] == ["clear", "retain"]
    assert [case["prefill_cache_policy"] for case in payload["cases"]] == ["clear", "retain"]


def test_benchmark_promotion_analysis_recommends_candidate(monkeypatch) -> None:
    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.model_path = kwargs["model_path"]

        def infer(self, prompt: str, **kwargs):
            token_ids = [1, 2, 3]
            variant = kwargs["_decode_variant"]
            stats = RunStats(
                model_path=self.model_path,
                backend="mlx",
                prompt_tokens=len(prompt.split()),
                generated_tokens=len(token_ids),
                prefill_seconds=0.5,
                decode_seconds=0.25 if variant == "custom_no_async" else 0.5,
                time_to_first_token_seconds=0.6,
                peak_memory_gb=2.1 if variant == "custom_no_async" else 2.0,
            )
            return SimpleNamespace(token_ids=token_ids, stats=stats)

    monkeypatch.setattr(benchmark, "Gemma4Engine", FakeEngine)

    payload = run_benchmark(
        model_path="fake",
        backend="mlx",
        config=BenchConfig(
            prompt_lengths=[128, 512],
            decode_lengths=[64],
            warmups=0,
            runs=1,
            decode_variants=("custom", "custom_no_async"),
            prefill_cache_policies=("clear",),
        ),
    )

    analysis = payload["promotion_analysis"]
    assert analysis["status"] == "candidate_passed"
    assert analysis["recommended_default"] == {
        "prefill_step_size": "auto",
        "prefill_sync_policy": "eval",
        "prefill_cache_policy": "clear",
        "decode_variant": "custom_no_async",
    }


def test_benchmark_can_include_full_token_ids(monkeypatch) -> None:
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
                prefill_seconds=0.5,
                decode_seconds=0.5,
                time_to_first_token_seconds=0.6,
                peak_memory_gb=2.0,
            )
            return SimpleNamespace(token_ids=token_ids, stats=stats)

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


def test_benchmark_prefill_step_size_matrix_is_ordered_from_auto(monkeypatch) -> None:
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
                decode_seconds=0.5,
                time_to_first_token_seconds=0.6,
                peak_memory_gb=2.0,
            )
            return SimpleNamespace(token_ids=token_ids, stats=stats)

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
            prefill_step_sizes=("1024", "auto"),
        ),
    )

    assert payload["prefill_step_sizes"] == ["auto", "1024"]
    assert [case["prefill_step_size"] for case in payload["cases"]] == ["auto", "1024"]


def test_benchmark_prefill_sync_policy_matrix_is_ordered_from_eval(monkeypatch) -> None:
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
                prefill_seconds=0.25 if kwargs["prefill_sync_policy"] == "async" else 0.5,
                decode_seconds=0.5,
                time_to_first_token_seconds=0.6,
                peak_memory_gb=2.0,
            )
            return SimpleNamespace(token_ids=token_ids, stats=stats)

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
            prefill_sync_policies=("async", "eval"),
        ),
    )

    assert payload["prefill_sync_policies"] == ["eval", "async"]
    assert [case["prefill_sync_policy"] for case in payload["cases"]] == ["eval", "async"]


def test_benchmark_prefill_promotion_analysis_recommends_candidate(monkeypatch) -> None:
    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            self.model_path = kwargs["model_path"]

        def infer(self, prompt: str, **kwargs):
            token_ids = [1, 2, 3]
            is_async = kwargs["prefill_sync_policy"] == "async"
            stats = RunStats(
                model_path=self.model_path,
                backend="mlx",
                prompt_tokens=len(prompt.split()),
                generated_tokens=len(token_ids),
                prefill_seconds=0.25 if is_async else 0.5,
                decode_seconds=0.51 if is_async else 0.5,
                time_to_first_token_seconds=0.6,
                peak_memory_gb=2.1 if is_async else 2.0,
            )
            return SimpleNamespace(token_ids=token_ids, stats=stats)

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
            prefill_sync_policies=("eval", "async"),
        ),
    )

    analysis = payload["prefill_promotion_analysis"]
    assert analysis["status"] == "candidate_passed"
    assert analysis["recommended_prefill_default"] == {
        "prefill_step_size": "auto",
        "prefill_sync_policy": "async",
        "prefill_cache_policy": "clear",
        "decode_variant": "custom",
    }

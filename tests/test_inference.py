import types
from types import SimpleNamespace

import pytest

import gemma4_engine.inference as inference
from gemma4_engine.inference import (
    Gemma4Engine,
    GenerationTimings,
    PrefixCacheEntry,
    PrefixCacheBuildResult,
    _clear_mlx_cache,
    _clone_prompt_cache,
    _prefix_cache_key,
    _prefill_step_size,
    _sync_prompt_cache,
)


def test_auto_prefill_step_size_limits_long_prompt_chunks() -> None:
    assert _prefill_step_size("auto", 128) == 1024
    assert _prefill_step_size("auto", 512) == 1024
    assert _prefill_step_size("auto", 2048) == 2048
    assert _prefill_step_size("auto", 8192) == 2048
    assert _prefill_step_size("auto", 16384) == 4096
    assert _prefill_step_size("auto", 65536) == 8192
    assert _prefill_step_size("512", 65536) == 512


def test_periodic_prefill_sync_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    calls: list[str] = []
    prompt_cache = [SimpleNamespace(state="state")]
    monkeypatch.setattr(mx, "eval", lambda _states: calls.append("eval"))
    monkeypatch.setattr(mx, "async_eval", lambda _states: calls.append("async"))

    _sync_prompt_cache(
        prompt_cache,
        "periodic",
        chunk_index=0,
        is_last_chunk=False,
        sync_every=2,
    )
    _sync_prompt_cache(
        prompt_cache,
        "periodic",
        chunk_index=1,
        is_last_chunk=False,
        sync_every=2,
    )
    _sync_prompt_cache(
        prompt_cache,
        "periodic",
        chunk_index=2,
        is_last_chunk=True,
        sync_every=2,
    )

    assert calls == ["async", "eval", "eval"]


def test_threshold_prefill_cache_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    clear_calls = 0
    monkeypatch.setattr(mx, "get_active_memory", lambda: 2_000_000_000)

    def clear_cache() -> None:
        nonlocal clear_calls
        clear_calls += 1

    monkeypatch.setattr(mx, "clear_cache", clear_cache)

    _clear_mlx_cache("threshold", threshold_gb=3)
    _clear_mlx_cache("threshold", threshold_gb=1)

    assert clear_calls == 1


def test_prefix_cache_key_depends_on_token_sequence() -> None:
    assert _prefix_cache_key([1, 23]) == _prefix_cache_key([1, 23])
    assert _prefix_cache_key([1, 23]) != _prefix_cache_key([12, 3])
    assert _prefix_cache_key([1, 23]) != _prefix_cache_key([1, 23], max_kv_size=4096)


def test_clone_prompt_cache_clones_nested_mlx_arrays() -> None:
    import mlx.core as mx

    class FakeCacheEntry:
        def __init__(self, state, meta_state) -> None:
            self.state = state
            self.meta_state = meta_state

        @classmethod
        def from_state(cls, state, meta_state):
            return cls(state, meta_state)

    original_array = mx.array([1, 2, 3])
    original_meta_array = mx.array([4, 5])
    entry = FakeCacheEntry(
        state=[original_array, {"nested": (original_meta_array,)}],
        meta_state={"offset": 7},
    )

    cloned = _clone_prompt_cache([entry])
    original_array[0] = 99
    original_meta_array[1] = 88
    mx.eval(original_array, original_meta_array)

    assert cloned[0] is not entry
    assert cloned[0].state[0].tolist() == [1, 2, 3]
    assert cloned[0].state[1]["nested"][0].tolist() == [4, 5]
    assert cloned[0].meta_state == {"offset": 7}


def test_prefix_cache_suffix_prefill_uses_suffix_length(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(chr(token_id) for token_id in token_ids)

    seen: dict[str, object] = {}
    engine = object.__new__(Gemma4Engine)
    engine.model_path = "fake-model"
    engine.loaded = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        model=object(),
        warnings=[],
        config={},
    )
    engine.argmax_backend = SimpleNamespace(name="mlx")
    engine.backend_status = SimpleNamespace(selected="mlx", reason="test")
    engine._token_cache = inference.HierarchicalTokenCache(disk_dir=None)

    def get_or_create_prefix_cache(
        self,
        prefix_ids: list[int],
        *,
        prefill_step_size: int,
        prefill_cache_policy: str,
        prefill_sync_policy: str,
        prefill_sync_every: int,
        prefill_cache_clear_every: int,
        prefill_cache_threshold_gb: float | None,
        max_kv_size: int | None,
    ):
        seen["prefix_ids"] = prefix_ids
        seen["prefix_prefill_step_size"] = prefill_step_size
        seen["prefix_prefill_cache_policy"] = prefill_cache_policy
        seen["prefix_prefill_sync_policy"] = prefill_sync_policy
        seen["prefix_prefill_sync_every"] = prefill_sync_every
        seen["prefix_prefill_cache_clear_every"] = prefill_cache_clear_every
        seen["prefix_prefill_cache_threshold_gb"] = prefill_cache_threshold_gb
        seen["prefix_max_kv_size"] = max_kv_size
        return PrefixCacheBuildResult(
            PrefixCacheEntry(token_ids=prefix_ids, cache=[]),
            True,
            0.0,
            GenerationTimings(),
        )

    def greedy_generate_tokens(**kwargs):
        seen["suffix_ids"] = kwargs["prompt_ids"]
        seen["suffix_prefill_step_size"] = kwargs["prefill_step_size"]
        return [33], 0.1, 0.2, 0.3, GenerationTimings(decode_token_latencies=[0.2])

    monkeypatch.setattr(engine, "_get_or_create_prefix_cache", types.MethodType(
        get_or_create_prefix_cache,
        engine,
    ))
    monkeypatch.setattr(inference, "_prefill_step_size", lambda _value, prompt_tokens: prompt_tokens)
    monkeypatch.setattr(inference, "_greedy_generate_tokens", greedy_generate_tokens)

    result = engine.infer(
        "ab12345",
        max_tokens=1,
        prompt_mode="raw",
        prefill_cache_policy="retain",
        prefill_sync_policy="async",
        prefill_sync_every=3,
        prefill_cache_clear_every=5,
        prefill_cache_threshold_gb=12,
        max_kv_size=4096,
        cache_prefix="ab",
        cache_prefix_mode="raw",
    )

    assert result.text == "!"
    assert result.stats.encode_seconds is not None
    assert result.stats.prefix_token_cache_seconds is not None
    assert result.stats.prefix_kv_cache_lookup_seconds is not None
    assert result.stats.prefix_kv_cache_clone_seconds is not None
    assert result.stats.decode_token_latency_p50_seconds == 0.2
    assert seen["prefix_ids"] == [97, 98]
    assert seen["prefix_prefill_step_size"] == 2
    assert seen["prefix_prefill_cache_policy"] == "retain"
    assert seen["prefix_prefill_sync_policy"] == "async"
    assert seen["prefix_prefill_sync_every"] == 3
    assert seen["prefix_prefill_cache_clear_every"] == 5
    assert seen["prefix_prefill_cache_threshold_gb"] == 12
    assert seen["prefix_max_kv_size"] == 4096
    assert seen["suffix_ids"] == [49, 50, 51, 52, 53]
    assert seen["suffix_prefill_step_size"] == 5


def test_internal_decode_variant_is_passed_to_greedy_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        eos_token_id = 2

        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(chr(token_id) for token_id in token_ids)

    seen: dict[str, object] = {}
    engine = object.__new__(Gemma4Engine)
    engine.model_path = "fake-model"
    engine.loaded = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        model=object(),
        warnings=[],
        config={},
    )
    engine.argmax_backend = SimpleNamespace(name="mlx")
    engine.backend_status = SimpleNamespace(selected="mlx", reason="test")
    engine._token_cache = inference.HierarchicalTokenCache(disk_dir=None)

    def greedy_generate_tokens(**kwargs):
        seen["decode_variant"] = kwargs["decode_variant"]
        seen["eos_token_ids"] = kwargs["eos_token_ids"]
        seen["prefill_cache_policy"] = kwargs["prefill_cache_policy"]
        seen["prefill_sync_policy"] = kwargs["prefill_sync_policy"]
        seen["max_kv_size"] = kwargs["max_kv_size"]
        return [33], 0.1, 0.2, 0.3, GenerationTimings(
            prefill_model_seconds=0.08,
            decode_sync_seconds=0.02,
            decode_token_item_seconds=0.01,
            decode_token_latencies=[0.2],
        )

    monkeypatch.setattr(inference, "_greedy_generate_tokens", greedy_generate_tokens)

    result = engine.infer(
        "ab",
        max_tokens=1,
        prompt_mode="raw",
        prefill_cache_policy="retain",
        prefill_sync_policy="none",
        max_kv_size=4096,
        _decode_variant="custom_no_async",
    )

    assert result.text == "!"
    assert result.stats.prefill_model_seconds == 0.08
    assert result.stats.decode_sync_seconds == 0.02
    assert result.stats.decode_token_item_seconds == 0.01
    assert seen["decode_variant"] == "custom_no_async"
    assert seen["eos_token_ids"] == {2}
    assert seen["prefill_cache_policy"] == "retain"
    assert seen["prefill_sync_policy"] == "none"
    assert seen["max_kv_size"] == 4096


def test_generation_oom_retries_with_smaller_prefill_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(chr(token_id) for token_id in token_ids)

    engine = object.__new__(Gemma4Engine)
    engine.model_path = "fake-model"
    engine.loaded = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        model=object(),
        warnings=[],
        config={},
    )
    engine.argmax_backend = SimpleNamespace(name="mlx")
    engine.backend_status = SimpleNamespace(selected="mlx", reason="test")
    engine._token_cache = inference.HierarchicalTokenCache(disk_dir=None)
    seen_step_sizes: list[int] = []
    cleared = False

    def greedy_generate_tokens(**kwargs):
        seen_step_sizes.append(kwargs["prefill_step_size"])
        if kwargs["prefill_step_size"] == 8192:
            raise RuntimeError("out of memory")
        return [33], 0.1, 0.2, 0.3, GenerationTimings()

    def clear_cache():
        nonlocal cleared
        cleared = True

    monkeypatch.setattr(inference, "_greedy_generate_tokens", greedy_generate_tokens)
    monkeypatch.setattr(inference, "_clear_mlx_runtime_cache", clear_cache)

    result = engine.infer(
        "ab",
        max_tokens=1,
        prompt_mode="raw",
        prefill_step_size="8192",
    )

    assert result.text == "!"
    assert seen_step_sizes == [8192, 4096]
    assert cleared is True
    assert "retried with smaller prefill chunks" in result.config_warnings[0]


def test_prefix_token_cache_uses_memory_then_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeTokenizer:
        def __init__(self) -> None:
            self.encode_calls = 0

        def encode(self, text: str) -> list[int]:
            self.encode_calls += 1
            return [ord(char) for char in text]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(chr(token_id) for token_id in token_ids)

    def make_engine(tokenizer: FakeTokenizer) -> Gemma4Engine:
        engine = object.__new__(Gemma4Engine)
        engine.model_path = "fake-model"
        engine.loaded = SimpleNamespace(
            tokenizer=tokenizer,
            model=object(),
            warnings=[],
            config={},
        )
        engine.argmax_backend = SimpleNamespace(name="mlx")
        engine.backend_status = SimpleNamespace(selected="mlx", reason="test")
        engine._prefix_cache = {}
        engine._token_cache = inference.HierarchicalTokenCache(disk_dir=tmp_path)
        return engine

    def get_or_create_prefix_cache(self, prefix_ids: list[int], **_kwargs):
        return PrefixCacheBuildResult(
            PrefixCacheEntry(token_ids=prefix_ids, cache=[]),
            False,
            0.0,
            GenerationTimings(),
        )

    monkeypatch.setattr(Gemma4Engine, "_get_or_create_prefix_cache", get_or_create_prefix_cache)
    monkeypatch.setattr(
        inference,
        "_greedy_generate_tokens",
        lambda **_kwargs: ([33], 0.1, 0.2, 0.3, GenerationTimings()),
    )

    first_tokenizer = FakeTokenizer()
    first_engine = make_engine(first_tokenizer)
    first = first_engine.infer(
        "ab123",
        max_tokens=1,
        prompt_mode="raw",
        cache_prefix="ab",
        cache_prefix_mode="raw",
    )
    second = first_engine.infer(
        "ab456",
        max_tokens=1,
        prompt_mode="raw",
        cache_prefix="ab",
        cache_prefix_mode="raw",
    )

    second_tokenizer = FakeTokenizer()
    second_engine = make_engine(second_tokenizer)
    third = second_engine.infer(
        "ab789",
        max_tokens=1,
        prompt_mode="raw",
        cache_prefix="ab",
        cache_prefix_mode="raw",
    )

    assert first.prefix_token_cache_source == "miss"
    assert second.prefix_token_cache_source == "memory"
    assert third.prefix_token_cache_source == "disk"
    assert first_tokenizer.encode_calls == 3
    assert second_tokenizer.encode_calls == 1

import types
from types import SimpleNamespace

import pytest

import gemma4_engine.inference as inference
from gemma4_engine.inference import (
    Gemma4Engine,
    PrefixCacheEntry,
    _clone_prompt_cache,
    _prefix_cache_key,
    _prefill_step_size,
)


def test_auto_prefill_step_size_limits_long_prompt_chunks() -> None:
    assert _prefill_step_size("auto", 128) == 512
    assert _prefill_step_size("auto", 512) == 512
    assert _prefill_step_size("auto", 2048) == 512
    assert _prefill_step_size("auto", 8192) == 512


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
        max_kv_size: int | None,
    ):
        seen["prefix_ids"] = prefix_ids
        seen["prefix_prefill_step_size"] = prefill_step_size
        seen["prefix_prefill_cache_policy"] = prefill_cache_policy
        seen["prefix_prefill_sync_policy"] = prefill_sync_policy
        seen["prefix_max_kv_size"] = max_kv_size
        return PrefixCacheEntry(token_ids=prefix_ids, cache=[]), True, 0.0

    def greedy_generate_tokens(**kwargs):
        seen["suffix_ids"] = kwargs["prompt_ids"]
        seen["suffix_prefill_step_size"] = kwargs["prefill_step_size"]
        return [33], 0.1, 0.2, 0.3

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
        max_kv_size=4096,
        cache_prefix="ab",
        cache_prefix_mode="raw",
    )

    assert result.text == "!"
    assert seen["prefix_ids"] == [97, 98]
    assert seen["prefix_prefill_step_size"] == 2
    assert seen["prefix_prefill_cache_policy"] == "retain"
    assert seen["prefix_prefill_sync_policy"] == "async"
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
        return [33], 0.1, 0.2, 0.3

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
    assert seen["decode_variant"] == "custom_no_async"
    assert seen["eos_token_ids"] == {2}
    assert seen["prefill_cache_policy"] == "retain"
    assert seen["prefill_sync_policy"] == "none"
    assert seen["max_kv_size"] == 4096


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
        return PrefixCacheEntry(token_ids=prefix_ids, cache=[]), False, 0.0

    monkeypatch.setattr(Gemma4Engine, "_get_or_create_prefix_cache", get_or_create_prefix_cache)
    monkeypatch.setattr(
        inference,
        "_greedy_generate_tokens",
        lambda **_kwargs: ([33], 0.1, 0.2, 0.3),
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

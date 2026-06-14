import sys
import types
from types import SimpleNamespace

import pytest

import gemma4_engine.inference as inference
from gemma4_engine.inference import (
    SPECULATIVE_INSTALL_MESSAGE,
    Gemma4Engine,
    PrefixCacheEntry,
    _create_speculative_runtime,
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
    engine.draft_model_path = None
    engine.speculative_runtime = None

    def get_or_create_prefix_cache(self, prefix_ids: list[int], *, prefill_step_size: int):
        seen["prefix_ids"] = prefix_ids
        seen["prefix_prefill_step_size"] = prefill_step_size
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
        cache_prefix="ab",
        cache_prefix_mode="raw",
    )

    assert result.text == "!"
    assert seen["prefix_ids"] == [97, 98]
    assert seen["prefix_prefill_step_size"] == 2
    assert seen["suffix_ids"] == [49, 50, 51, 52, 53]
    assert seen["suffix_prefill_step_size"] == 5


def test_create_speculative_runtime_reports_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("gemma4_engine.speculative")

    class MissingSpeculativeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "No module named 'mlx_vlm'",
                name="mlx_vlm",
            )

    module.SpeculativeRuntime = MissingSpeculativeRuntime
    monkeypatch.setitem(sys.modules, "gemma4_engine.speculative", module)

    with pytest.raises(RuntimeError, match="Speculative decoding is experimental") as exc_info:
        _create_speculative_runtime("target", "draft", draft_tokens=4)

    assert str(exc_info.value) == SPECULATIVE_INSTALL_MESSAGE

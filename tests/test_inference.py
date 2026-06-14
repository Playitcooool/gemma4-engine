from gemma4_engine.inference import _prefix_cache_key, _prefill_step_size


def test_auto_prefill_step_size_limits_long_prompt_chunks() -> None:
    assert _prefill_step_size("auto", 128) == 512
    assert _prefill_step_size("auto", 512) == 512
    assert _prefill_step_size("auto", 2048) == 1024
    assert _prefill_step_size("auto", 8192) == 1024


def test_prefix_cache_key_depends_on_token_sequence() -> None:
    assert _prefix_cache_key([1, 23]) == _prefix_cache_key([1, 23])
    assert _prefix_cache_key([1, 23]) != _prefix_cache_key([12, 3])

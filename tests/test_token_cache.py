from gemma4_engine.token_cache import HierarchicalTokenCache, token_cache_key


def test_hierarchical_token_cache_uses_memory_before_encoder(tmp_path) -> None:
    calls = 0
    cache = HierarchicalTokenCache(disk_dir=tmp_path)

    def encode() -> list[int]:
        nonlocal calls
        calls += 1
        return [1, 2, 3]

    first = cache.get_or_encode(key="abc", encode=encode)
    second = cache.get_or_encode(key="abc", encode=encode)

    assert first.source == "miss"
    assert second.source == "memory"
    assert second.token_ids == [1, 2, 3]
    assert calls == 1


def test_hierarchical_token_cache_uses_disk_across_instances(tmp_path) -> None:
    first_cache = HierarchicalTokenCache(disk_dir=tmp_path)
    first_cache.get_or_encode(key="abc", encode=lambda: [4, 5, 6])

    second_cache = HierarchicalTokenCache(disk_dir=tmp_path)
    result = second_cache.get_or_encode(
        key="abc",
        encode=lambda: (_ for _ in ()).throw(AssertionError("encoder should not run")),
    )

    assert result.source == "disk"
    assert result.token_ids == [4, 5, 6]


def test_hierarchical_token_cache_ignores_corrupt_disk_entry(tmp_path) -> None:
    (tmp_path / "abc.g4tokens").write_bytes(b"not a cache")
    cache = HierarchicalTokenCache(disk_dir=tmp_path)

    result = cache.get_or_encode(key="abc", encode=lambda: [7])

    assert result.source == "miss"
    assert result.token_ids == [7]


def test_token_cache_key_includes_model_mode_and_text() -> None:
    base = token_cache_key(model_path="model-a", prompt_mode="raw", text="hello")

    assert base == token_cache_key(model_path="model-a", prompt_mode="raw", text="hello")
    assert base != token_cache_key(model_path="model-b", prompt_mode="raw", text="hello")
    assert base != token_cache_key(model_path="model-a", prompt_mode="chat", text="hello")
    assert base != token_cache_key(model_path="model-a", prompt_mode="raw", text="bye")

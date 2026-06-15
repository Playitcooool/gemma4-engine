from types import SimpleNamespace

from gemma4_engine import cli
from gemma4_engine.cli import build_parser
from gemma4_engine.constants import DEFAULT_MODEL_PATH
from gemma4_engine.stats import RunStats


def test_default_model_path() -> None:
    parser = build_parser()
    args = parser.parse_args(["infer", "--prompt", "hello"])
    cli._resolve_profile_defaults(args)

    assert args.model == DEFAULT_MODEL_PATH


def test_bench_csv_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["bench"])

    assert args.profile == "matrix"
    assert args.prompt_tokens == "128,512,2048"
    assert args.decode_tokens == "64,128,256"
    assert args.prefill_step_sizes == ("auto", "512", "1024", "2048", "4096", "8192")
    assert args.prefill_sync_policies is None
    assert args.prefill_cache_policy == "both"


def test_bench_prefill_step_sizes_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--profile",
            "single-user-latency",
            "--prefill-step-sizes",
            "auto,1024,2048",
            "--prefill-sync-policies",
            "eval,async,none",
            "--decode-variants",
            "custom,custom_speculative_ngram,custom_blockwise_16,mlx_lm_generate_step",
            "--include-token-ids",
        ]
    )

    assert args.profile == "single-user-latency"
    assert args.prefill_step_sizes == ("auto", "1024", "2048")
    assert args.prefill_sync_policies == ("eval", "async", "none")
    assert args.decode_variants == (
        "custom",
        "custom_speculative_ngram",
        "custom_blockwise_16",
        "mlx_lm_generate_step",
    )
    assert args.include_token_ids is True


def test_serve_simple_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve"])
    cli._resolve_profile_defaults(args)

    assert args.model == DEFAULT_MODEL_PATH
    assert args.backend == "auto"
    assert args.port == 8000
    assert args.prefill_cache_policy == "clear"
    assert args.prefill_sync_policy == "eval"
    assert args.max_sessions == 8


def test_chat_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["chat"])
    cli._resolve_profile_defaults(args)

    assert args.model == DEFAULT_MODEL_PATH
    assert args.profile == "single_user_fast"
    assert args.prefill_cache_policy == "retain"
    assert args.prefill_sync_policy == "async"
    assert args.session_id == "chat"
    assert args.max_sessions == 4


def test_single_user_fast_profile_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve", "--profile", "single_user_fast"])
    cli._resolve_profile_defaults(args)

    assert args.prefill_step_size == "auto"
    assert args.prefill_cache_policy == "retain"
    assert args.prefill_sync_policy == "async"
    assert args.prefill_sync_every == 4
    assert args.prefill_cache_clear_every == 8
    assert args.decode_variant == "custom"
    assert args.stream is True
    assert args.non_stream_decode_variant == "custom_blockwise_16"
    assert args.max_sessions == 4


def test_profile_does_not_override_explicit_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--profile",
            "single_user_fast",
            "--prefill-cache-policy",
            "clear",
            "--prefill-sync-policy",
            "periodic",
            "--max-sessions",
            "9",
        ]
    )
    cli._resolve_profile_defaults(args)

    assert args.prefill_cache_policy == "clear"
    assert args.prefill_sync_policy == "periodic"
    assert args.max_sessions == 9


def test_chat_loop_uses_session_cache(monkeypatch, capsys) -> None:
    seen = {}

    class FakeEngine:
        def __init__(self, **kwargs) -> None:
            seen["init"] = kwargs

        def reset_session(self, session_id: str) -> None:
            seen["reset"] = session_id

        def infer(self, prompt: str, **kwargs):
            seen["prompt"] = prompt
            seen["infer"] = kwargs
            return SimpleNamespace(
                text="ok",
                stats=RunStats(
                    model_path="fake",
                    backend="mlx",
                    prompt_tokens=2,
                    generated_tokens=1,
                    prefill_seconds=0.5,
                    decode_seconds=0.25,
                    time_to_first_token_seconds=0.6,
                    session_cache_hit=True,
                    session_tokens_reused=3,
                ),
            )

    inputs = iter(["hello", "/stats", "/reset", "/exit"])
    monkeypatch.setattr(cli, "Gemma4Engine", FakeEngine)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    args = build_parser().parse_args(["chat", "--session-id", "main"])
    cli._resolve_profile_defaults(args)

    assert cli._run_chat(args) == 0

    captured = capsys.readouterr()
    assert "ok" in captured.out
    assert "session_hit=True" in captured.out
    assert seen["infer"]["session_id"] == "main"
    assert seen["infer"]["append_to_session"] is True
    assert seen["reset"] == "main"


def test_infer_advanced_flags_still_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--prompt",
            "hello",
            "--backend",
            "mlx",
            "--prefill-step-size",
            "4096",
            "--prefill-cache-policy",
            "threshold",
            "--prefill-sync-policy",
            "periodic",
            "--prefill-sync-every",
            "3",
            "--prefill-cache-clear-every",
            "5",
            "--prefill-cache-threshold-gb",
            "12",
            "--kv-bits",
            "4",
            "--max-kv-size",
            "4096",
            "--max-sliding-kv-size",
            "1024",
            "--max-global-kv-size",
            "8192",
            "--max-prefix-cache-mb",
            "256",
            "--decode-variant",
            "custom_speculative_ngram",
            "--no-stream",
            "--non-stream-decode-variant",
            "custom_blockwise_32",
            "--speculative-ngram-min",
            "2",
            "--speculative-ngram-max",
            "5",
            "--speculative-draft-tokens",
            "7",
            "--mlx-memory-limit-gb",
            "48",
            "--mlx-cache-limit-gb",
            "40",
            "--mlx-wired-limit-gb",
            "32",
            "--cache-prefix",
            "shared",
            "--cache-prefix-mode",
            "raw",
            "--token-cache-dir",
            "/tmp/gemma4-token-cache",
            "--token-cache-max-disk-mb",
            "123",
            "--json",
        ]
    )
    cli._resolve_profile_defaults(args)

    assert args.backend == "mlx"
    assert args.prefill_step_size == "4096"
    assert args.prefill_cache_policy == "threshold"
    assert args.prefill_sync_policy == "periodic"
    assert args.prefill_sync_every == 3
    assert args.prefill_cache_clear_every == 5
    assert args.prefill_cache_threshold_gb == 12
    assert args.kv_bits == 4
    assert args.max_kv_size == 4096
    assert args.max_sliding_kv_size == 1024
    assert args.max_global_kv_size == 8192
    assert args.max_prefix_cache_mb == 256
    assert args.decode_variant == "custom_speculative_ngram"
    assert args.stream is False
    assert args.non_stream_decode_variant == "custom_blockwise_32"
    assert args.speculative_ngram_min == 2
    assert args.speculative_ngram_max == 5
    assert args.speculative_draft_tokens == 7
    assert args.mlx_memory_limit_gb == 48
    assert args.mlx_cache_limit_gb == 40
    assert args.mlx_wired_limit_gb == 32
    assert args.cache_prefix == "shared"
    assert args.token_cache_dir == "/tmp/gemma4-token-cache"
    assert args.token_cache_max_disk_mb == 123
    assert args.json is True


def test_token_cache_dir_empty_string_disables_disk_cache() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--prompt",
            "hello",
            "--token-cache-dir",
            "",
        ]
    )
    cli._resolve_profile_defaults(args)

    assert args.token_cache_dir is None


def test_serve_token_cache_max_disk_mb_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--token-cache-max-disk-mb",
            "250",
            "--max-prefix-cache-mb",
            "128",
            "--enable-sessions",
            "--max-sessions",
            "4",
            "--max-session-tokens",
            "4096",
        ]
    )
    cli._resolve_profile_defaults(args)

    assert args.token_cache_max_disk_mb == 250
    assert args.max_prefix_cache_mb == 128
    assert args.enable_sessions is True
    assert args.max_sessions == 4
    assert args.max_session_tokens == 4096

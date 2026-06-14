from gemma4_engine import cli
from gemma4_engine.cli import build_parser
from gemma4_engine.constants import DEFAULT_MODEL_PATH
from gemma4_engine.inference import SPECULATIVE_INSTALL_MESSAGE


def test_default_model_path() -> None:
    parser = build_parser()
    args = parser.parse_args(["infer", "--prompt", "hello"])

    assert args.model == DEFAULT_MODEL_PATH


def test_bench_csv_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["bench"])

    assert args.prompt_tokens == "128,512,2048"
    assert args.decode_tokens == "64,128,256"
    assert args.prefill_step_sizes is None
    assert args.prefill_sync_policies is None
    assert args.prefill_cache_policy == "both"


def test_bench_prefill_step_sizes_parse() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "--prefill-step-sizes",
            "auto,1024,2048",
            "--prefill-sync-policies",
            "eval,async,none",
            "--decode-variants",
            "custom,custom_defer_ids,mlx_lm_generate_step",
            "--include-token-ids",
        ]
    )

    assert args.prefill_step_sizes == ("auto", "1024", "2048")
    assert args.prefill_sync_policies == ("eval", "async", "none")
    assert args.decode_variants == ("custom", "custom_defer_ids", "mlx_lm_generate_step")
    assert args.include_token_ids is True


def test_infer_draft_model_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--prompt",
            "hello",
            "--draft-model",
            "/tmp/draft",
            "--draft-tokens",
            "3",
        ]
    )

    assert args.draft_model == "/tmp/draft"
    assert args.draft_tokens == 3


def test_serve_simple_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve"])

    assert args.model == DEFAULT_MODEL_PATH
    assert args.backend == "auto"
    assert args.port == 8000


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
            "retain",
            "--prefill-sync-policy",
            "async",
            "--kv-bits",
            "4",
            "--max-kv-size",
            "4096",
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
            "--json",
        ]
    )

    assert args.backend == "mlx"
    assert args.prefill_step_size == "4096"
    assert args.prefill_cache_policy == "retain"
    assert args.prefill_sync_policy == "async"
    assert args.kv_bits == 4
    assert args.max_kv_size == 4096
    assert args.mlx_memory_limit_gb == 48
    assert args.mlx_cache_limit_gb == 40
    assert args.mlx_wired_limit_gb == 32
    assert args.cache_prefix == "shared"
    assert args.token_cache_dir == "/tmp/gemma4-token-cache"
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

    assert args.token_cache_dir is None


def test_missing_speculative_extra_is_clear_cli_error(
    monkeypatch,
    capsys,
) -> None:
    def raise_missing_extra(*args, **kwargs):
        raise RuntimeError(SPECULATIVE_INSTALL_MESSAGE)

    monkeypatch.setattr(cli, "infer", raise_missing_extra)

    exit_code = cli.main(
        [
            "infer",
            "--prompt",
            "hello",
            "--draft-model",
            "/tmp/draft",
        ]
    )

    assert exit_code == 1
    assert "uv sync --extra speculative" in capsys.readouterr().err

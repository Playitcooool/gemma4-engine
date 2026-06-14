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

    assert args.prompt_tokens == "128,512,2048,8192"
    assert args.decode_tokens == "128,512"


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
            "--kv-bits",
            "4",
            "--cache-prefix",
            "shared",
            "--cache-prefix-mode",
            "raw",
            "--json",
        ]
    )

    assert args.backend == "mlx"
    assert args.prefill_step_size == "4096"
    assert args.kv_bits == 4
    assert args.cache_prefix == "shared"
    assert args.json is True


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

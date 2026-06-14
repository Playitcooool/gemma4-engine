from gemma4_engine.cli import build_parser
from gemma4_engine.constants import DEFAULT_MODEL_PATH


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

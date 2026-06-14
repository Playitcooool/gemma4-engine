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

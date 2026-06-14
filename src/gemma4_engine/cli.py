from __future__ import annotations

import argparse
import sys

from .benchmark import BenchConfig, benchmark_json, run_benchmark
from .compare import compare_with_mlx_lm
from .constants import DEFAULT_MODEL_PATH
from .inference import infer


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma4")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    infer_parser.add_argument("--prompt", required=True)
    infer_parser.add_argument("--max-tokens", type=int, default=128)
    infer_parser.add_argument("--backend", choices=["mlx", "rust-metal", "auto"], default="auto")
    infer_parser.add_argument("--prompt-mode", choices=["chat", "raw"], default="chat")
    infer_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default="auto",
    )
    infer_parser.add_argument("--json", action="store_true")

    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    bench_parser.add_argument("--backend", choices=["mlx", "rust-metal", "auto"], default="auto")
    bench_parser.add_argument("--prompt-tokens", default="128,512,2048,8192")
    bench_parser.add_argument("--decode-tokens", default="128,512")
    bench_parser.add_argument("--warmups", type=int, default=1)
    bench_parser.add_argument("--runs", type=int, default=3)
    bench_parser.add_argument("--json", action="store_true")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    compare_parser.add_argument("--baseline", choices=["mlx_lm"], default="mlx_lm")
    compare_parser.add_argument("--prompt", required=True)
    compare_parser.add_argument("--max-tokens", type=int, default=64)
    compare_parser.add_argument("--backend", choices=["mlx", "rust-metal", "auto"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "infer":
        result = infer(
            args.prompt,
            model_path=args.model,
            max_tokens=args.max_tokens,
            backend=args.backend,
            prompt_mode=args.prompt_mode,
            prefill_step_size=args.prefill_step_size,
        )
        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "text": result.text,
                        "token_ids": result.token_ids,
                        "stats": result.stats.to_dict(),
                        "backend_reason": result.backend_reason,
                        "config_warnings": result.config_warnings,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(result.text)
            print()
            print(f"backend: {result.stats.backend} ({result.backend_reason})", file=sys.stderr)
            print(
                f"prefill: {result.stats.prefill_tokens_per_second:.2f} tok/s, "
                f"decode: {result.stats.decode_tokens_per_second:.2f} tok/s, "
                f"ttft: {result.stats.time_to_first_token_seconds:.3f}s",
                file=sys.stderr,
            )
            for warning in result.config_warnings:
                print(f"config warning: {warning}", file=sys.stderr)
        return 0

    if args.command == "bench":
        payload = run_benchmark(
            model_path=args.model,
            backend=args.backend,
            config=BenchConfig(
                prompt_lengths=_csv_ints(args.prompt_tokens),
                decode_lengths=_csv_ints(args.decode_tokens),
                warmups=args.warmups,
                runs=args.runs,
            ),
        )
        print(benchmark_json(payload) if args.json else benchmark_json(payload))
        return 0

    if args.command == "compare":
        result = compare_with_mlx_lm(
            prompt=args.prompt,
            model_path=args.model,
            max_tokens=args.max_tokens,
            backend=args.backend,
        )
        print(f"baseline: {result.baseline}")
        print(f"matches: {result.matches}")
        if not result.matches:
            print("engine:")
            print(result.engine_text)
            print("baseline:")
            print(result.baseline_text)
        return 0 if result.matches else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

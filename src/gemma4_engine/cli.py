from __future__ import annotations

import argparse
import sys

from .benchmark import BenchConfig, benchmark_json, run_benchmark
from .compare import compare_with_mlx_lm
from .constants import DEFAULT_MODEL_PATH
from .inference import infer
from .server import ServerConfig, run_server


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _read_optional_text(value: str | None, file_path: str | None) -> str | None:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()
    return value


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
    infer_parser.add_argument("--kv-bits", type=int, choices=[2, 4, 8], default=None)
    infer_parser.add_argument("--kv-group-size", type=int, default=64)
    infer_parser.add_argument("--quantized-kv-start", type=int, default=0)
    infer_parser.add_argument("--cache-prefix", default=None)
    infer_parser.add_argument("--cache-prefix-file", default=None)
    infer_parser.add_argument("--cache-prefix-mode", choices=["chat", "raw"], default="raw")
    infer_parser.add_argument("--json", action="store_true")

    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    bench_parser.add_argument("--backend", choices=["mlx", "rust-metal", "auto"], default="auto")
    bench_parser.add_argument("--prompt-tokens", default="128,512,2048,8192")
    bench_parser.add_argument("--decode-tokens", default="128,512")
    bench_parser.add_argument("--warmups", type=int, default=1)
    bench_parser.add_argument("--runs", type=int, default=3)
    bench_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default="auto",
    )
    bench_parser.add_argument("--kv-bits", type=int, choices=[2, 4, 8], default=None)
    bench_parser.add_argument("--kv-group-size", type=int, default=64)
    bench_parser.add_argument("--quantized-kv-start", type=int, default=0)
    bench_parser.add_argument("--json", action="store_true")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    compare_parser.add_argument("--baseline", choices=["mlx_lm"], default="mlx_lm")
    compare_parser.add_argument("--prompt", required=True)
    compare_parser.add_argument("--max-tokens", type=int, default=64)
    compare_parser.add_argument("--backend", choices=["mlx", "rust-metal", "auto"], default="auto")
    compare_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default="auto",
    )
    compare_parser.add_argument("--kv-bits", type=int, choices=[2, 4, 8], default=None)
    compare_parser.add_argument("--kv-group-size", type=int, default=64)
    compare_parser.add_argument("--quantized-kv-start", type=int, default=0)
    compare_parser.add_argument("--json", action="store_true")

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    serve_parser.add_argument("--backend", choices=["mlx", "rust-metal", "auto"], default="auto")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--max-tokens", type=int, default=128)
    serve_parser.add_argument("--prompt-mode", choices=["chat", "raw"], default="chat")
    serve_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default="auto",
    )
    serve_parser.add_argument("--kv-bits", type=int, choices=[2, 4, 8], default=None)
    serve_parser.add_argument("--kv-group-size", type=int, default=64)
    serve_parser.add_argument("--quantized-kv-start", type=int, default=0)
    serve_parser.add_argument("--cache-prefix", default=None)
    serve_parser.add_argument("--cache-prefix-file", default=None)
    serve_parser.add_argument("--cache-prefix-mode", choices=["chat", "raw"], default="raw")
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
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
            cache_prefix=_read_optional_text(args.cache_prefix, args.cache_prefix_file),
            cache_prefix_mode=args.cache_prefix_mode,
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
                        "prefix_cache_hit": result.prefix_cache_hit,
                        "prefix_tokens": result.prefix_tokens,
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
            if result.prefix_tokens:
                print(
                    f"prefix cache: hit={result.prefix_cache_hit}, "
                    f"tokens={result.prefix_tokens}",
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
                prefill_step_size=args.prefill_step_size,
                kv_bits=args.kv_bits,
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
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
            prefill_step_size=args.prefill_step_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            quantized_kv_start=args.quantized_kv_start,
        )
        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "baseline": result.baseline,
                        "matches": result.matches,
                        "engine": {
                            "text": result.engine_text,
                            "token_ids": result.engine_tokens,
                            "stats": result.engine_stats.to_dict(),
                        },
                        "baseline_stats": result.baseline_stats,
                        "speedup": result.speedup,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if result.matches else 1

        print(f"baseline: {result.baseline}")
        print(f"matches: {result.matches}")
        print(
            "engine: "
            f"backend={result.engine_stats.backend}, "
            f"prefill={result.engine_stats.prefill_tokens_per_second:.2f} tok/s, "
            f"decode={result.engine_stats.decode_tokens_per_second:.2f} tok/s, "
            f"ttft={result.engine_stats.time_to_first_token_seconds:.3f}s"
        )
        print(
            "baseline: "
            f"backend={result.baseline_stats['backend']}, "
            f"prefill={float(result.baseline_stats['prefill_tokens_per_second']):.2f} tok/s, "
            f"decode={float(result.baseline_stats['decode_tokens_per_second']):.2f} tok/s"
        )
        if result.speedup["prefill"] is not None and result.speedup["decode"] is not None:
            print(
                "speedup: "
                f"prefill={result.speedup['prefill']:.2f}x, "
                f"decode={result.speedup['decode']:.2f}x"
            )
        if not result.matches:
            print("engine:")
            print(result.engine_text)
            print("baseline:")
            print(result.baseline_text)
        return 0 if result.matches else 1

    if args.command == "serve":
        run_server(
            ServerConfig(
                model_path=args.model,
                backend=args.backend,
                host=args.host,
                port=args.port,
                default_max_tokens=args.max_tokens,
                default_prompt_mode=args.prompt_mode,
                default_prefill_step_size=args.prefill_step_size,
                default_kv_bits=args.kv_bits,
                default_kv_group_size=args.kv_group_size,
                default_quantized_kv_start=args.quantized_kv_start,
                default_cache_prefix=_read_optional_text(
                    args.cache_prefix,
                    args.cache_prefix_file,
                ),
                default_cache_prefix_mode=args.cache_prefix_mode,
            )
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys

from dataclasses import dataclass

from .benchmark import (
    DECODE_BENCHMARK_VARIANTS,
    PREFILL_STEP_SIZES,
    PREFILL_SYNC_POLICIES,
    BenchConfig,
    benchmark_json,
    benchmark_summary,
    run_benchmark,
    single_user_latency_scenarios,
)
from .constants import DEFAULT_MODEL_PATH
from .inference import DecodeVariant, Gemma4Engine, infer
from .server import ServerConfig, run_server
from .token_cache import DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES, DEFAULT_TOKEN_CACHE_DIR


@dataclass(frozen=True)
class RuntimeProfile:
    prefill_step_size: str = "auto"
    prefill_cache_policy: str = "clear"
    prefill_sync_policy: str = "eval"
    prefill_sync_every: int = 4
    prefill_cache_clear_every: int = 8
    decode_variant: DecodeVariant = "custom"
    stream: bool = True
    non_stream_decode_variant: DecodeVariant = "custom_blockwise_16"
    max_sessions: int = 8


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "default": RuntimeProfile(),
    "single_user_fast": RuntimeProfile(
        prefill_cache_policy="retain",
        prefill_sync_policy="async",
        max_sessions=4,
    ),
}


def _csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _csv_prefill_step_sizes(value: str) -> tuple[str, ...]:
    choices = {"auto", "512", "1024", "2048", "4096", "8192"}
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [part for part in values if part not in choices]
    if invalid:
        raise argparse.ArgumentTypeError(
            "prefill step sizes must be one or more of: auto, 512, 1024, 2048, 4096, 8192"
        )
    if not values:
        raise argparse.ArgumentTypeError("must include at least one prefill step size")
    return values


def _csv_prefill_sync_policies(value: str) -> tuple[str, ...]:
    choices = {"eval", "async", "none", "periodic"}
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [part for part in values if part not in choices]
    if invalid:
        raise argparse.ArgumentTypeError(
            "prefill sync policies must be one or more of: eval, async, none, periodic"
        )
    if not values:
        raise argparse.ArgumentTypeError("must include at least one prefill sync policy")
    return values


def _csv_decode_variants(value: str) -> tuple[str, ...]:
    choices = {
        "custom",
        "custom_no_async",
        "custom_eval_next",
        "custom_defer_ids",
        "custom_blockwise_8",
        "custom_blockwise_16",
        "custom_blockwise_32",
        "custom_speculative_ngram",
        "mlx_lm_generate_step",
    }
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [part for part in values if part not in choices]
    if invalid:
        raise argparse.ArgumentTypeError(
            "decode variants must be one or more of: "
            "custom, custom_no_async, custom_eval_next, custom_defer_ids, "
            "custom_blockwise_8, custom_blockwise_16, custom_blockwise_32, "
            "custom_speculative_ngram, mlx_lm_generate_step"
        )
    if not values:
        raise argparse.ArgumentTypeError("must include at least one decode variant")
    return values


def _decode_variant_choices() -> tuple[DecodeVariant, ...]:
    return (
        "custom",
        "custom_no_async",
        "custom_eval_next",
        "custom_defer_ids",
        "custom_blockwise_8",
        "custom_blockwise_16",
        "custom_blockwise_32",
        "custom_speculative_ngram",
        "mlx_lm_generate_step",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _read_optional_text(value: str | None, file_path: str | None) -> str | None:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()
    return value


def _optional_cache_dir(value: str) -> str | None:
    return value or None


def _mb_to_bytes(value: int) -> int:
    return value * 1_000_000


def _bench_prefill_cache_policies(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("clear", "retain")
    return (value,)


def _add_profile_argument(parser: argparse.ArgumentParser, *, default: str = "default") -> None:
    parser.add_argument(
        "--profile",
        choices=tuple(RUNTIME_PROFILES),
        default=default,
        help="runtime preset for local inference defaults",
    )


def _resolve_profile_defaults(args: argparse.Namespace) -> None:
    profile = RUNTIME_PROFILES[getattr(args, "profile", "default")]
    for name in (
        "prefill_step_size",
        "prefill_cache_policy",
        "prefill_sync_policy",
        "prefill_sync_every",
        "prefill_cache_clear_every",
        "decode_variant",
        "stream",
        "non_stream_decode_variant",
        "max_sessions",
    ):
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, getattr(profile, name))


def _print_chat_stats(result) -> None:
    stats = result.stats
    print(
        "stats: "
        f"prefill={stats.prefill_tokens_per_second:.2f} tok/s, "
        f"decode={stats.decode_tokens_per_second:.2f} tok/s, "
        f"ttft={stats.time_to_first_token_seconds:.3f}s, "
        f"session_hit={stats.session_cache_hit}, "
        f"session_reused={stats.session_tokens_reused}"
    )


def _run_chat(args) -> int:
    _resolve_profile_defaults(args)
    try:
        engine = Gemma4Engine(
            model_path=args.model,
            backend=args.backend,
            token_cache_dir=args.token_cache_dir,
            max_sessions=args.max_sessions,
            mlx_memory_limit_gb=args.mlx_memory_limit_gb,
            mlx_cache_limit_gb=args.mlx_cache_limit_gb,
            mlx_wired_limit_gb=args.mlx_wired_limit_gb,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    last_result = None
    print("gemma4 chat ready. Type /reset, /stats, or /exit.", file=sys.stderr)
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            print()
            return 0
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/reset":
            engine.reset_session(args.session_id)
            last_result = None
            print("session reset")
            continue
        if prompt == "/stats":
            if last_result is None:
                print("stats: no generation yet")
            else:
                _print_chat_stats(last_result)
            continue

        try:
            last_result = engine.infer(
                prompt,
                max_tokens=args.max_tokens,
                prompt_mode=args.prompt_mode,
                prefill_step_size=args.prefill_step_size,
                prefill_cache_policy=args.prefill_cache_policy,
                prefill_sync_policy=args.prefill_sync_policy,
                prefill_sync_every=args.prefill_sync_every,
                prefill_cache_clear_every=args.prefill_cache_clear_every,
                prefill_cache_threshold_gb=args.prefill_cache_threshold_gb,
                max_kv_size=args.max_kv_size,
                session_id=args.session_id,
                append_to_session=True,
                speculative_ngram_min=args.speculative_ngram_min,
                speculative_ngram_max=args.speculative_ngram_max,
                speculative_draft_tokens=args.speculative_draft_tokens,
                stream=args.stream,
                non_stream_decode_variant=args.non_stream_decode_variant,
                _decode_variant=args.decode_variant,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(last_result.text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma4",
        description=(
            "Small MLX Gemma 4 runner. Start with `gemma4 serve` or "
            "`gemma4 infer --prompt \"Say hi.\"`."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser(
        "infer",
        description='Run one prompt. Example: gemma4 infer --prompt "Say hi."',
    )
    _add_profile_argument(infer_parser)
    infer_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    infer_parser.add_argument("--prompt", required=True)
    infer_parser.add_argument("--max-tokens", type=int, default=128)
    infer_parser.add_argument("--backend", choices=["mlx", "auto"], default="auto")
    infer_parser.add_argument("--prompt-mode", choices=["chat", "raw"], default="chat")
    infer_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default=None,
    )
    infer_parser.add_argument(
        "--prefill-cache-policy",
        choices=["clear", "retain", "periodic", "threshold"],
        default=None,
        help="clear MLX allocator cache after prefill chunks, or retain it for high-memory speed tests",
    )
    infer_parser.add_argument(
        "--prefill-sync-policy",
        choices=["eval", "async", "none", "periodic"],
        default=None,
        help="synchronize MLX prompt-cache states after each prefill chunk",
    )
    infer_parser.add_argument("--prefill-sync-every", type=_positive_int, default=None)
    infer_parser.add_argument("--prefill-cache-clear-every", type=_positive_int, default=None)
    infer_parser.add_argument("--prefill-cache-threshold-gb", type=_positive_float, default=None)
    infer_parser.add_argument("--kv-bits", type=int, choices=[2, 4, 8], default=None)
    infer_parser.add_argument("--kv-group-size", type=int, default=64)
    infer_parser.add_argument("--quantized-kv-start", type=int, default=0)
    infer_parser.add_argument("--max-kv-size", type=_positive_int, default=None)
    infer_parser.add_argument("--decode-variant", choices=_decode_variant_choices(), default=None)
    infer_parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use streaming-safe per-token decode; --no-stream uses the non-stream decode variant",
    )
    infer_parser.add_argument(
        "--non-stream-decode-variant",
        choices=_decode_variant_choices(),
        default=None,
    )
    infer_parser.add_argument("--speculative-ngram-min", type=_positive_int, default=3)
    infer_parser.add_argument("--speculative-ngram-max", type=_positive_int, default=6)
    infer_parser.add_argument("--speculative-draft-tokens", type=_positive_int, default=4)
    infer_parser.add_argument("--mlx-memory-limit-gb", type=_positive_float, default=None)
    infer_parser.add_argument("--mlx-cache-limit-gb", type=_positive_float, default=None)
    infer_parser.add_argument("--mlx-wired-limit-gb", type=_positive_float, default=None)
    infer_parser.add_argument("--cache-prefix", default=None)
    infer_parser.add_argument("--cache-prefix-file", default=None)
    infer_parser.add_argument("--cache-prefix-mode", choices=["chat", "raw"], default="raw")
    infer_parser.add_argument(
        "--token-cache-dir",
        default=DEFAULT_TOKEN_CACHE_DIR,
        type=_optional_cache_dir,
        help="disk cache directory for tokenized prefixes; pass an empty string to disable",
    )
    infer_parser.add_argument(
        "--token-cache-max-disk-mb",
        type=_nonnegative_int,
        default=DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES // 1_000_000,
        help="maximum token cache disk usage in decimal MB",
    )
    infer_parser.add_argument("--json", action="store_true")

    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument(
        "--profile",
        choices=["matrix", "single-user-latency"],
        default="matrix",
        help="benchmark preset; matrix uses the explicit prompt/decode grids",
    )
    bench_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    bench_parser.add_argument("--backend", choices=["mlx", "auto"], default="auto")
    bench_parser.add_argument("--prompt-tokens", default="128,512,2048")
    bench_parser.add_argument("--decode-tokens", default="64,128,256")
    bench_parser.add_argument("--warmups", type=int, default=1)
    bench_parser.add_argument("--runs", type=int, default=3)
    bench_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default="auto",
    )
    bench_parser.add_argument(
        "--prefill-step-sizes",
        type=_csv_prefill_step_sizes,
        default=PREFILL_STEP_SIZES,
        help="comma-separated prefill chunk-size matrix for benchmark runs",
    )
    bench_parser.add_argument(
        "--prefill-cache-policy",
        choices=["clear", "retain", "periodic", "threshold", "both"],
        default="both",
        help="prefill allocator-cache policy matrix for benchmark runs",
    )
    bench_parser.add_argument("--prefill-cache-clear-every", type=_positive_int, default=8)
    bench_parser.add_argument("--prefill-cache-threshold-gb", type=_positive_float, default=None)
    bench_parser.add_argument(
        "--prefill-sync-policies",
        type=_csv_prefill_sync_policies,
        default=None,
        help="comma-separated prefill cache-state sync matrix for benchmark runs",
    )
    bench_parser.add_argument("--prefill-sync-every", type=_positive_int, default=4)
    bench_parser.add_argument(
        "--decode-variants",
        type=_csv_decode_variants,
        default=None,
        help="comma-separated decode variant matrix for benchmark runs",
    )
    bench_parser.add_argument("--kv-bits", type=int, choices=[2, 4, 8], default=None)
    bench_parser.add_argument("--kv-group-size", type=int, default=64)
    bench_parser.add_argument("--quantized-kv-start", type=int, default=0)
    bench_parser.add_argument("--max-kv-size", type=_positive_int, default=None)
    bench_parser.add_argument("--speculative-ngram-min", type=_positive_int, default=3)
    bench_parser.add_argument("--speculative-ngram-max", type=_positive_int, default=6)
    bench_parser.add_argument("--speculative-draft-tokens", type=_positive_int, default=4)
    bench_parser.add_argument("--mlx-memory-limit-gb", type=_positive_float, default=None)
    bench_parser.add_argument("--mlx-cache-limit-gb", type=_positive_float, default=None)
    bench_parser.add_argument("--mlx-wired-limit-gb", type=_positive_float, default=None)
    bench_parser.add_argument(
        "--include-token-ids",
        action="store_true",
        help="include full generated token IDs in benchmark JSON instead of only count/hash",
    )
    bench_parser.add_argument("--json", action="store_true")

    serve_parser = subparsers.add_parser(
        "serve",
        description="Start the persistent /generate JSON service. Example: gemma4 serve",
    )
    _add_profile_argument(serve_parser)
    serve_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    serve_parser.add_argument("--backend", choices=["mlx", "auto"], default="auto")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--enable-sessions", action="store_true")
    serve_parser.add_argument("--max-sessions", type=_positive_int, default=None)
    serve_parser.add_argument("--max-tokens", type=int, default=128)
    serve_parser.add_argument("--prompt-mode", choices=["chat", "raw"], default="chat")
    serve_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default=None,
    )
    serve_parser.add_argument(
        "--prefill-cache-policy",
        choices=["clear", "retain", "periodic", "threshold"],
        default=None,
        help="clear MLX allocator cache after prefill chunks, or retain it for high-memory speed tests",
    )
    serve_parser.add_argument(
        "--prefill-sync-policy",
        choices=["eval", "async", "none", "periodic"],
        default=None,
        help="synchronize MLX prompt-cache states after each prefill chunk",
    )
    serve_parser.add_argument("--prefill-sync-every", type=_positive_int, default=None)
    serve_parser.add_argument("--prefill-cache-clear-every", type=_positive_int, default=None)
    serve_parser.add_argument("--prefill-cache-threshold-gb", type=_positive_float, default=None)
    serve_parser.add_argument("--kv-bits", type=int, choices=[2, 4, 8], default=None)
    serve_parser.add_argument("--kv-group-size", type=int, default=64)
    serve_parser.add_argument("--quantized-kv-start", type=int, default=0)
    serve_parser.add_argument("--max-kv-size", type=_positive_int, default=None)
    serve_parser.add_argument("--decode-variant", choices=_decode_variant_choices(), default=None)
    serve_parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="default stream mode for /generate requests",
    )
    serve_parser.add_argument(
        "--non-stream-decode-variant",
        choices=_decode_variant_choices(),
        default=None,
    )
    serve_parser.add_argument("--speculative-ngram-min", type=_positive_int, default=3)
    serve_parser.add_argument("--speculative-ngram-max", type=_positive_int, default=6)
    serve_parser.add_argument("--speculative-draft-tokens", type=_positive_int, default=4)
    serve_parser.add_argument("--mlx-memory-limit-gb", type=_positive_float, default=None)
    serve_parser.add_argument("--mlx-cache-limit-gb", type=_positive_float, default=None)
    serve_parser.add_argument("--mlx-wired-limit-gb", type=_positive_float, default=None)
    serve_parser.add_argument("--cache-prefix", default=None)
    serve_parser.add_argument("--cache-prefix-file", default=None)
    serve_parser.add_argument("--cache-prefix-mode", choices=["chat", "raw"], default="raw")
    serve_parser.add_argument(
        "--token-cache-dir",
        default=DEFAULT_TOKEN_CACHE_DIR,
        type=_optional_cache_dir,
        help="disk cache directory for tokenized prefixes; pass an empty string to disable",
    )
    serve_parser.add_argument(
        "--token-cache-max-disk-mb",
        type=_nonnegative_int,
        default=DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES // 1_000_000,
        help="maximum token cache disk usage in decimal MB",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        description="Interactive single-user chat. Commands: /reset, /stats, /exit",
    )
    _add_profile_argument(chat_parser, default="single_user_fast")
    chat_parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    chat_parser.add_argument("--backend", choices=["mlx", "auto"], default="auto")
    chat_parser.add_argument("--max-tokens", type=int, default=128)
    chat_parser.add_argument("--prompt-mode", choices=["chat", "raw"], default="chat")
    chat_parser.add_argument(
        "--prefill-step-size",
        choices=["auto", "512", "1024", "2048", "4096", "8192"],
        default=None,
    )
    chat_parser.add_argument(
        "--prefill-cache-policy",
        choices=["clear", "retain", "periodic", "threshold"],
        default=None,
    )
    chat_parser.add_argument(
        "--prefill-sync-policy",
        choices=["eval", "async", "none", "periodic"],
        default=None,
    )
    chat_parser.add_argument("--prefill-sync-every", type=_positive_int, default=None)
    chat_parser.add_argument("--prefill-cache-clear-every", type=_positive_int, default=None)
    chat_parser.add_argument("--prefill-cache-threshold-gb", type=_positive_float, default=None)
    chat_parser.add_argument("--max-kv-size", type=_positive_int, default=None)
    chat_parser.add_argument("--decode-variant", choices=_decode_variant_choices(), default=None)
    chat_parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    chat_parser.add_argument(
        "--non-stream-decode-variant",
        choices=_decode_variant_choices(),
        default=None,
    )
    chat_parser.add_argument("--speculative-ngram-min", type=_positive_int, default=3)
    chat_parser.add_argument("--speculative-ngram-max", type=_positive_int, default=6)
    chat_parser.add_argument("--speculative-draft-tokens", type=_positive_int, default=4)
    chat_parser.add_argument("--session-id", default="chat")
    chat_parser.add_argument("--max-sessions", type=_positive_int, default=None)
    chat_parser.add_argument("--token-cache-dir", default=DEFAULT_TOKEN_CACHE_DIR, type=_optional_cache_dir)
    chat_parser.add_argument("--mlx-memory-limit-gb", type=_positive_float, default=None)
    chat_parser.add_argument("--mlx-cache-limit-gb", type=_positive_float, default=None)
    chat_parser.add_argument("--mlx-wired-limit-gb", type=_positive_float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_profile_defaults(args)

    if args.command == "infer":
        try:
            result = infer(
                args.prompt,
                model_path=args.model,
                max_tokens=args.max_tokens,
                backend=args.backend,
                prompt_mode=args.prompt_mode,
                prefill_step_size=args.prefill_step_size,
                prefill_cache_policy=args.prefill_cache_policy,
                prefill_sync_policy=args.prefill_sync_policy,
                prefill_sync_every=args.prefill_sync_every,
                prefill_cache_clear_every=args.prefill_cache_clear_every,
                prefill_cache_threshold_gb=args.prefill_cache_threshold_gb,
                kv_bits=args.kv_bits,
                kv_group_size=args.kv_group_size,
                quantized_kv_start=args.quantized_kv_start,
                max_kv_size=args.max_kv_size,
                cache_prefix=_read_optional_text(args.cache_prefix, args.cache_prefix_file),
                cache_prefix_mode=args.cache_prefix_mode,
                token_cache_dir=args.token_cache_dir,
                max_token_cache_disk_bytes=_mb_to_bytes(args.token_cache_max_disk_mb),
                speculative_ngram_min=args.speculative_ngram_min,
                speculative_ngram_max=args.speculative_ngram_max,
                speculative_draft_tokens=args.speculative_draft_tokens,
                stream=args.stream,
                non_stream_decode_variant=args.non_stream_decode_variant,
                _decode_variant=args.decode_variant,
                mlx_memory_limit_gb=args.mlx_memory_limit_gb,
                mlx_cache_limit_gb=args.mlx_cache_limit_gb,
                mlx_wired_limit_gb=args.mlx_wired_limit_gb,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
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
                        "prefix_token_cache_source": result.prefix_token_cache_source,
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
                    f"tokens={result.prefix_tokens}, "
                    f"token_cache={result.prefix_token_cache_source}",
                    file=sys.stderr,
                )
            for warning in result.config_warnings:
                print(f"config warning: {warning}", file=sys.stderr)
        return 0

    if args.command == "bench":
        try:
            payload = run_benchmark(
                model_path=args.model,
                backend=args.backend,
                config=BenchConfig(
                    prompt_lengths=_csv_ints(args.prompt_tokens),
                    decode_lengths=_csv_ints(args.decode_tokens),
                    warmups=args.warmups,
                    runs=args.runs,
                    prefill_step_size=args.prefill_step_size,
                    prefill_step_sizes=args.prefill_step_sizes,
                    prefill_sync_policies=args.prefill_sync_policies
                    if args.prefill_sync_policies is not None
                    else PREFILL_SYNC_POLICIES,
                    prefill_sync_every=args.prefill_sync_every,
                    prefill_cache_policies=_bench_prefill_cache_policies(
                        args.prefill_cache_policy
                    ),
                    prefill_cache_clear_every=args.prefill_cache_clear_every,
                    prefill_cache_threshold_gb=args.prefill_cache_threshold_gb,
                    kv_bits=args.kv_bits,
                    kv_group_size=args.kv_group_size,
                    quantized_kv_start=args.quantized_kv_start,
                    max_kv_size=args.max_kv_size,
                    speculative_ngram_min=args.speculative_ngram_min,
                    speculative_ngram_max=args.speculative_ngram_max,
                    speculative_draft_tokens=args.speculative_draft_tokens,
                    decode_variants=args.decode_variants
                    if args.decode_variants is not None
                    else DECODE_BENCHMARK_VARIANTS,
                    mlx_memory_limit_gb=args.mlx_memory_limit_gb,
                    mlx_cache_limit_gb=args.mlx_cache_limit_gb,
                    mlx_wired_limit_gb=args.mlx_wired_limit_gb,
                    include_token_ids=args.include_token_ids,
                    scenarios=single_user_latency_scenarios()
                    if args.profile == "single-user-latency"
                    else None,
                    benchmark_profile=args.profile,
                ),
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(benchmark_json(payload) if args.json else benchmark_summary(payload))
        return 0

    if args.command == "chat":
        return _run_chat(args)

    if args.command == "serve":
        try:
            run_server(
                ServerConfig(
                    model_path=args.model,
                    backend=args.backend,
                    host=args.host,
                    port=args.port,
                    default_max_tokens=args.max_tokens,
                    default_prompt_mode=args.prompt_mode,
                    default_prefill_step_size=args.prefill_step_size,
                    default_prefill_cache_policy=args.prefill_cache_policy,
                    default_prefill_sync_policy=args.prefill_sync_policy,
                    default_prefill_sync_every=args.prefill_sync_every,
                    default_prefill_cache_clear_every=args.prefill_cache_clear_every,
                    default_prefill_cache_threshold_gb=args.prefill_cache_threshold_gb,
                    default_kv_bits=args.kv_bits,
                    default_kv_group_size=args.kv_group_size,
                    default_quantized_kv_start=args.quantized_kv_start,
                    default_max_kv_size=args.max_kv_size,
                    default_decode_variant=args.decode_variant,
                    default_stream=args.stream,
                    default_non_stream_decode_variant=args.non_stream_decode_variant,
                    default_speculative_ngram_min=args.speculative_ngram_min,
                    default_speculative_ngram_max=args.speculative_ngram_max,
                    default_speculative_draft_tokens=args.speculative_draft_tokens,
                    default_cache_prefix=_read_optional_text(
                        args.cache_prefix,
                        args.cache_prefix_file,
                    ),
                    default_cache_prefix_mode=args.cache_prefix_mode,
                    token_cache_dir=args.token_cache_dir,
                    max_token_cache_disk_bytes=_mb_to_bytes(args.token_cache_max_disk_mb),
                    max_sessions=args.max_sessions,
                    mlx_memory_limit_gb=args.mlx_memory_limit_gb,
                    mlx_cache_limit_gb=args.mlx_cache_limit_gb,
                    mlx_wired_limit_gb=args.mlx_wired_limit_gb,
                )
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

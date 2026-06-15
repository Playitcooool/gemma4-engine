from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .backends import BackendName
from .constants import DEFAULT_MODEL_PATH
from .inference import (
    Gemma4Engine,
    PrefillCachePolicy,
    PrefillStepSize,
    PrefillSyncPolicy,
    PromptMode,
)
from .token_cache import DEFAULT_TOKEN_CACHE_DIR
from .token_cache import DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES


@dataclass(frozen=True)
class ServerConfig:
    model_path: str = DEFAULT_MODEL_PATH
    backend: BackendName = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    default_max_tokens: int = 128
    default_prompt_mode: PromptMode = "chat"
    default_prefill_step_size: PrefillStepSize = "auto"
    default_prefill_cache_policy: PrefillCachePolicy = "clear"
    default_prefill_sync_policy: PrefillSyncPolicy = "eval"
    default_kv_bits: int | None = None
    default_kv_group_size: int = 64
    default_quantized_kv_start: int = 0
    default_max_kv_size: int | None = None
    default_cache_prefix: str | None = None
    default_cache_prefix_mode: PromptMode = "raw"
    token_cache_dir: str | None = DEFAULT_TOKEN_CACHE_DIR
    max_token_cache_disk_bytes: int | None = DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES
    mlx_memory_limit_gb: float | None = None
    mlx_cache_limit_gb: float | None = None
    mlx_wired_limit_gb: float | None = None


class EngineService:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.engine = Gemma4Engine(
            model_path=config.model_path,
            backend=config.backend,
            token_cache_dir=config.token_cache_dir,
            max_token_cache_disk_bytes=config.max_token_cache_disk_bytes,
            mlx_memory_limit_gb=config.mlx_memory_limit_gb,
            mlx_cache_limit_gb=config.mlx_cache_limit_gb,
            mlx_wired_limit_gb=config.mlx_wired_limit_gb,
        )
        self._lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": self.config.model_path,
            "backend_requested": self.config.backend,
            "backend_selected": self.engine.backend_status.selected,
            "backend_reason": self.engine.backend_status.reason,
            "config_warnings": self.engine.loaded.warnings,
            "token_cache_dir": self.config.token_cache_dir,
            "max_token_cache_disk_bytes": self.config.max_token_cache_disk_bytes,
            "default_prefill_cache_policy": self.config.default_prefill_cache_policy,
            "default_prefill_sync_policy": self.config.default_prefill_sync_policy,
            "default_max_kv_size": self.config.default_max_kv_size,
            "mlx_memory": {
                "memory_limit_gb": self.config.mlx_memory_limit_gb,
                "cache_limit_gb": self.config.mlx_cache_limit_gb,
                "wired_limit_gb": self.config.mlx_wired_limit_gb,
            },
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("request JSON must include a non-empty string field: prompt")

        max_tokens = int(payload.get("max_tokens", self.config.default_max_tokens))
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        prompt_mode = payload.get("prompt_mode", self.config.default_prompt_mode)
        if prompt_mode not in ("chat", "raw"):
            raise ValueError("prompt_mode must be 'chat' or 'raw'")

        prefill_step_size = payload.get(
            "prefill_step_size",
            self.config.default_prefill_step_size,
        )
        if prefill_step_size not in ("auto", "512", "1024", "2048", "4096", "8192"):
            raise ValueError("prefill_step_size must be auto, 512, 1024, 2048, 4096, or 8192")

        prefill_cache_policy = payload.get(
            "prefill_cache_policy",
            self.config.default_prefill_cache_policy,
        )
        if prefill_cache_policy not in ("clear", "retain"):
            raise ValueError("prefill_cache_policy must be 'clear' or 'retain'")

        prefill_sync_policy = payload.get(
            "prefill_sync_policy",
            self.config.default_prefill_sync_policy,
        )
        if prefill_sync_policy not in ("eval", "async", "none"):
            raise ValueError("prefill_sync_policy must be 'eval', 'async', or 'none'")

        kv_bits = payload.get("kv_bits", self.config.default_kv_bits)
        if kv_bits is not None:
            kv_bits = int(kv_bits)
            if kv_bits not in (2, 4, 8):
                raise ValueError("kv_bits must be one of 2, 4, or 8")

        kv_group_size = int(payload.get("kv_group_size", self.config.default_kv_group_size))
        quantized_kv_start = int(
            payload.get("quantized_kv_start", self.config.default_quantized_kv_start)
        )
        max_kv_size = payload.get("max_kv_size", self.config.default_max_kv_size)
        if max_kv_size is not None:
            max_kv_size = int(max_kv_size)
            if max_kv_size < 1:
                raise ValueError("max_kv_size must be >= 1")

        cache_prefix = payload.get("cache_prefix", self.config.default_cache_prefix)
        if cache_prefix is not None and not isinstance(cache_prefix, str):
            raise ValueError("cache_prefix must be a string when provided")

        cache_prefix_mode = payload.get(
            "cache_prefix_mode",
            self.config.default_cache_prefix_mode,
        )
        if cache_prefix_mode not in ("chat", "raw"):
            raise ValueError("cache_prefix_mode must be 'chat' or 'raw'")

        with self._lock:
            result = self.engine.infer(
                prompt,
                max_tokens=max_tokens,
                prompt_mode=prompt_mode,
                prefill_step_size=prefill_step_size,
                prefill_cache_policy=prefill_cache_policy,
                prefill_sync_policy=prefill_sync_policy,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                max_kv_size=max_kv_size,
                cache_prefix=cache_prefix,
                cache_prefix_mode=cache_prefix_mode,
            )

        return {
            "text": result.text,
            "token_ids": result.token_ids,
            "stats": result.stats.to_dict(),
            "backend_reason": result.backend_reason,
            "config_warnings": result.config_warnings,
            "prefix_cache_hit": result.prefix_cache_hit,
            "prefix_tokens": result.prefix_tokens,
            "prefix_token_cache_source": result.prefix_token_cache_source,
        }


def make_handler(service: EngineService) -> type[BaseHTTPRequestHandler]:
    class Gemma4Handler(BaseHTTPRequestHandler):
        server_version = "gemma4-engine/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, service.health())
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                payload = self._read_json()
                response = service.generate(payload)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            self._send_json(HTTPStatus.OK, response)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Gemma4Handler


def run_server(config: ServerConfig) -> None:
    service = EngineService(config)
    server = HTTPServer((config.host, config.port), make_handler(service))
    print(
        f"gemma4 serve listening on http://{config.host}:{config.port} "
        f"backend={service.engine.backend_status.selected}",
        flush=True,
    )
    server.serve_forever()

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .backends import BackendName
from .constants import DEFAULT_MODEL_PATH
from .inference import Gemma4Engine, PrefillStepSize, PromptMode


@dataclass(frozen=True)
class ServerConfig:
    model_path: str = DEFAULT_MODEL_PATH
    backend: BackendName = "auto"
    host: str = "127.0.0.1"
    port: int = 8000
    default_max_tokens: int = 128
    default_prompt_mode: PromptMode = "chat"
    default_prefill_step_size: PrefillStepSize = "auto"
    default_kv_bits: int | None = None
    default_kv_group_size: int = 64
    default_quantized_kv_start: int = 0
    default_cache_prefix: str | None = None
    default_cache_prefix_mode: PromptMode = "raw"
    draft_model_path: str | None = None
    draft_tokens: int = 4


class EngineService:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.engine = Gemma4Engine(
            model_path=config.model_path,
            backend=config.backend,
            draft_model_path=config.draft_model_path,
            draft_tokens=config.draft_tokens,
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
            "draft_model_path": self.config.draft_model_path,
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

        kv_bits = payload.get("kv_bits", self.config.default_kv_bits)
        if kv_bits is not None:
            kv_bits = int(kv_bits)
            if kv_bits not in (2, 4, 8):
                raise ValueError("kv_bits must be one of 2, 4, or 8")

        kv_group_size = int(payload.get("kv_group_size", self.config.default_kv_group_size))
        quantized_kv_start = int(
            payload.get("quantized_kv_start", self.config.default_quantized_kv_start)
        )
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
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
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
            "draft_model_path": result.draft_model_path,
            "speculative_acceptance_rate": result.speculative_acceptance_rate,
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

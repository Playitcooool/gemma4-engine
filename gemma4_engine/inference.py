from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from .backends import ArgmaxBackend, BackendName, select_backend
from .loader import load_model
from .stats import RunStats, memory_snapshot, now
from .token_cache import (
    DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES,
    DEFAULT_TOKEN_CACHE_DIR,
    HierarchicalTokenCache,
    token_cache_key,
)

PromptMode = Literal["chat", "raw"]
PrefillStepSize = Literal["auto", "512", "1024", "2048", "4096", "8192"]
PrefillCachePolicy = Literal["clear", "retain", "periodic", "threshold"]
PrefillSyncPolicy = Literal["eval", "async", "none", "periodic"]
DecodeVariant = Literal[
    "custom",
    "custom_no_async",
    "custom_eval_next",
    "custom_defer_ids",
    "mlx_lm_generate_step",
]


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    stats: RunStats
    backend_reason: str
    config_warnings: list[str]
    prefix_cache_hit: bool = False
    prefix_tokens: int = 0
    prefix_token_cache_source: str | None = None


@dataclass
class PrefixCacheEntry:
    token_ids: list[int]
    cache: list[object]


@dataclass
class SessionState:
    token_ids: list[int]
    prompt_cache: list[object]
    generated_token_ids: list[int]
    last_access_time: float


@dataclass
class GenerationTimings:
    prefill_model_seconds: float = 0.0
    prefill_sync_seconds: float = 0.0
    prefill_clear_cache_seconds: float = 0.0
    first_token_eval_seconds: float = 0.0
    decode_model_seconds: float = 0.0
    decode_sync_seconds: float = 0.0
    decode_token_item_seconds: float = 0.0
    decode_token_latencies: list[float] | None = None


@dataclass
class PrefixCacheBuildResult:
    entry: PrefixCacheEntry | None
    hit: bool
    seconds: float
    timings: GenerationTimings


@dataclass
class Gemma4Engine:
    model_path: str
    backend: BackendName = "auto"
    max_prefix_cache_entries: int = 4
    token_cache_dir: str | None = DEFAULT_TOKEN_CACHE_DIR
    max_token_cache_entries: int = 128
    max_token_cache_disk_bytes: int | None = DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES
    max_sessions: int = 8
    mlx_memory_limit_gb: float | None = None
    mlx_cache_limit_gb: float | None = None
    mlx_wired_limit_gb: float | None = None

    def __post_init__(self) -> None:
        memory_warnings = _configure_mlx_memory(
            memory_limit_gb=self.mlx_memory_limit_gb,
            cache_limit_gb=self.mlx_cache_limit_gb,
            wired_limit_gb=self.mlx_wired_limit_gb,
        )
        self.loaded = load_model(self.model_path)
        self.loaded.warnings.extend(memory_warnings)
        self.argmax_backend, self.backend_status = select_backend(self.backend)
        self._prefix_cache: dict[str, PrefixCacheEntry] = {}
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._token_cache = HierarchicalTokenCache(
            disk_dir=self.token_cache_dir,
            max_memory_entries=self.max_token_cache_entries,
            max_disk_bytes=self.max_token_cache_disk_bytes,
        )

    def infer(
        self,
        prompt: str,
        *,
        max_tokens: int,
        prompt_mode: PromptMode = "chat",
        prefill_step_size: PrefillStepSize = "auto",
        prefill_cache_policy: PrefillCachePolicy = "clear",
        prefill_sync_policy: PrefillSyncPolicy = "eval",
        prefill_sync_every: int = 4,
        prefill_cache_clear_every: int = 8,
        prefill_cache_threshold_gb: float | None = None,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        max_kv_size: int | None = None,
        eos_token_id: int | None = None,
        cache_prefix: str | None = None,
        cache_prefix_mode: PromptMode = "raw",
        session_id: str | None = None,
        reset_session: bool = False,
        append_to_session: bool = False,
        _decode_variant: DecodeVariant = "custom",
    ) -> GenerationResult:
        encode_start = now()
        prompt_text = _format_prompt(self.loaded.tokenizer, prompt, prompt_mode)
        prompt_ids = _encode(self.loaded.tokenizer, prompt_text)
        encode_seconds = now() - encode_start
        prefix_cache_hit = False
        prefix_tokens = 0
        prefix_token_cache_source = None
        prompt_cache = None
        prefill_ids = prompt_ids
        session_cache_hit = False
        session_tokens_reused = 0
        session_state: SessionState | None = None
        prefix_cache_build_seconds = 0.0
        prefix_token_cache_seconds = 0.0
        prefix_kv_cache_lookup_seconds = 0.0
        prefix_kv_cache_clone_seconds = 0.0
        prefix_timings = GenerationTimings()
        cached_prefix_entry = None

        if not hasattr(self, "_sessions"):
            self._sessions = OrderedDict()

        if session_id and reset_session:
            self._sessions.pop(session_id, None)

        if session_id and not reset_session:
            session_state = self._sessions.get(session_id)
            if session_state is not None:
                self._sessions.move_to_end(session_id)
                session_state.last_access_time = now()
                session_cache_hit = True
                session_tokens_reused = len(session_state.token_ids)
                prompt_cache = session_state.prompt_cache

        if cache_prefix and not session_cache_hit:
            prefix_text = _format_prompt(self.loaded.tokenizer, cache_prefix, cache_prefix_mode)
            prefix_token_cache_start = now()
            prefix_token_cache = self._token_cache.get_or_encode(
                key=token_cache_key(
                    model_path=self.model_path,
                    prompt_mode=cache_prefix_mode,
                    text=prefix_text,
                ),
                encode=lambda: _encode(self.loaded.tokenizer, prefix_text),
            )
            prefix_token_cache_seconds = now() - prefix_token_cache_start
            prefix_ids = prefix_token_cache.token_ids
            prefix_token_cache_source = prefix_token_cache.source
            if prompt_ids[: len(prefix_ids)] == prefix_ids:
                suffix_ids = prompt_ids[len(prefix_ids) :]
            else:
                suffix_ids = prompt_ids
                prompt_ids = prefix_ids + suffix_ids

            prefix_lookup_start = now()
            prefix_result = self._get_or_create_prefix_cache(
                prefix_ids,
                prefill_step_size=_prefill_step_size(prefill_step_size, len(prefix_ids)),
                prefill_cache_policy=prefill_cache_policy,
                prefill_sync_policy=prefill_sync_policy,
                prefill_sync_every=prefill_sync_every,
                prefill_cache_clear_every=prefill_cache_clear_every,
                prefill_cache_threshold_gb=prefill_cache_threshold_gb,
                max_kv_size=max_kv_size,
            )
            prefix_kv_cache_lookup_seconds = now() - prefix_lookup_start
            cached = prefix_result.entry
            cached_prefix_entry = cached
            prefix_cache_hit = prefix_result.hit
            prefix_cache_build_seconds = prefix_result.seconds
            prefix_timings = prefix_result.timings
            if cached is not None:
                clone_start = now()
                prompt_cache = _clone_prompt_cache(cached.cache)
                prefix_kv_cache_clone_seconds = now() - clone_start
                prefill_ids = suffix_ids
                prefix_tokens = len(prefix_ids)

        if session_id and append_to_session and prompt_cache is None:
            prompt_cache = _make_prompt_cache(self.loaded.model, max_kv_size=max_kv_size)

        eos_ids = set(getattr(self.loaded.tokenizer, "eos_token_ids", []) or [])
        if eos_token_id is None:
            eos_token_id = getattr(self.loaded.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_ids.add(int(eos_token_id))

        config_warnings = list(self.loaded.warnings)
        if kv_bits is not None and self.loaded.config.get("model_type") == "gemma4":
            if self.loaded.config.get("text_config", {}).get("num_kv_shared_layers", 0) > 0:
                kv_bits = None
                config_warnings.append(
                    "KV cache quantization disabled because current mlx_lm Gemma 4 "
                    "shared-KV caches are incompatible with quantized KV entries"
                )

        generation_prefill_step_size = _prefill_step_size(prefill_step_size, len(prefill_ids))
        generated, prefill_seconds, decode_seconds, first_token_seconds, generation_timings = (
            self._generate_with_oom_retry(
                prompt_ids=prefill_ids,
                max_tokens=max_tokens,
                initial_prefill_step_size=generation_prefill_step_size,
                eos_token_ids=eos_ids,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
                max_kv_size=max_kv_size,
                prompt_cache=prompt_cache,
                cached_prefix_entry=cached_prefix_entry,
                decode_variant=_decode_variant,
                prefill_cache_policy=prefill_cache_policy,
                prefill_sync_policy=prefill_sync_policy,
                prefill_sync_every=prefill_sync_every,
                prefill_cache_clear_every=prefill_cache_clear_every,
                prefill_cache_threshold_gb=prefill_cache_threshold_gb,
                config_warnings=config_warnings,
            )
        )
        prefill_model_seconds = (
            generation_timings.prefill_model_seconds
            + prefix_timings.prefill_model_seconds
        )
        prefill_sync_seconds = (
            generation_timings.prefill_sync_seconds
            + prefix_timings.prefill_sync_seconds
        )
        prefill_clear_cache_seconds = (
            generation_timings.prefill_clear_cache_seconds
            + prefix_timings.prefill_clear_cache_seconds
        )
        latency_p50, latency_p95, latency_max = _decode_latency_stats(
            generation_timings.decode_token_latencies or []
        )

        stats = RunStats(
            model_path=self.model_path,
            backend=self.backend_status.selected,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            prefill_seconds=prefill_seconds + prefix_cache_build_seconds,
            decode_seconds=decode_seconds,
            time_to_first_token_seconds=first_token_seconds + prefix_cache_build_seconds,
            encode_seconds=encode_seconds,
            prefix_token_cache_seconds=prefix_token_cache_seconds,
            prefix_kv_cache_lookup_seconds=prefix_kv_cache_lookup_seconds,
            prefix_kv_cache_build_seconds=prefix_cache_build_seconds,
            prefix_kv_cache_clone_seconds=prefix_kv_cache_clone_seconds,
            prefill_model_seconds=prefill_model_seconds,
            prefill_sync_seconds=prefill_sync_seconds,
            prefill_clear_cache_seconds=prefill_clear_cache_seconds,
            first_token_eval_seconds=generation_timings.first_token_eval_seconds,
            decode_model_seconds=generation_timings.decode_model_seconds,
            decode_sync_seconds=generation_timings.decode_sync_seconds,
            decode_token_item_seconds=generation_timings.decode_token_item_seconds,
            decode_token_latency_p50_seconds=latency_p50,
            decode_token_latency_p95_seconds=latency_p95,
            decode_token_latency_max_seconds=latency_max,
            session_cache_hit=session_cache_hit,
            session_tokens_reused=session_tokens_reused,
            session_count=len(self._sessions),
            **memory_snapshot(),
        )
        if session_id and append_to_session and prompt_cache is not None:
            previous_tokens = session_state.token_ids if session_state is not None else []
            previous_generated = (
                session_state.generated_token_ids if session_state is not None else []
            )
            self._remember_session(
                session_id,
                SessionState(
                    token_ids=[*previous_tokens, *prefill_ids, *generated],
                    prompt_cache=prompt_cache,
                    generated_token_ids=[*previous_generated, *generated],
                    last_access_time=now(),
                ),
            )
            stats.session_count = len(self._sessions)
        return GenerationResult(
            text=_decode(self.loaded.tokenizer, generated),
            token_ids=generated,
            stats=stats,
            backend_reason=self.backend_status.reason,
            config_warnings=config_warnings,
            prefix_cache_hit=prefix_cache_hit,
            prefix_tokens=prefix_tokens,
            prefix_token_cache_source=prefix_token_cache_source,
        )

    def clear_prefix_cache(self) -> None:
        self._prefix_cache.clear()

    def clear_token_memory_cache(self) -> None:
        self._token_cache.clear_memory()

    def clear_sessions(self) -> None:
        self._sessions.clear()

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[dict[str, object]]:
        return [
            {
                "session_id": session_id,
                "tokens": len(state.token_ids),
                "generated_tokens": len(state.generated_token_ids),
                "last_access_time": state.last_access_time,
            }
            for session_id, state in self._sessions.items()
        ]

    def _remember_session(self, session_id: str, state: SessionState) -> None:
        self._sessions[session_id] = state
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def _generate_with_oom_retry(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        initial_prefill_step_size: int,
        eos_token_ids: set[int],
        kv_bits: int | None,
        kv_group_size: int,
        quantized_kv_start: int,
        max_kv_size: int | None,
        prompt_cache: list[object] | None,
        cached_prefix_entry: PrefixCacheEntry | None,
        decode_variant: DecodeVariant,
        prefill_cache_policy: PrefillCachePolicy,
        prefill_sync_policy: PrefillSyncPolicy,
        prefill_sync_every: int,
        prefill_cache_clear_every: int,
        prefill_cache_threshold_gb: float | None,
        config_warnings: list[str],
    ) -> tuple[list[int], float, float, float, GenerationTimings]:
        prefill_step_size = initial_prefill_step_size
        current_prompt_cache = prompt_cache
        warned = False
        while True:
            try:
                return _greedy_generate_tokens(
                    model=self.loaded.model,
                    prompt_ids=prompt_ids,
                    max_tokens=max_tokens,
                    backend=self.argmax_backend,
                    prefill_step_size=prefill_step_size,
                    eos_token_ids=eos_token_ids,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    quantized_kv_start=quantized_kv_start,
                    max_kv_size=max_kv_size,
                    prompt_cache=current_prompt_cache,
                    decode_variant=decode_variant,
                    prefill_cache_policy=prefill_cache_policy,
                    prefill_sync_policy=prefill_sync_policy,
                    prefill_sync_every=prefill_sync_every,
                    prefill_cache_clear_every=prefill_cache_clear_every,
                    prefill_cache_threshold_gb=prefill_cache_threshold_gb,
                )
            except RuntimeError as exc:
                next_step_size = _smaller_prefill_step_size(prefill_step_size)
                if next_step_size is None or not _is_mlx_memory_error(exc):
                    raise
                _clear_mlx_runtime_cache()
                prefill_step_size = next_step_size
                current_prompt_cache = (
                    _clone_prompt_cache(cached_prefix_entry.cache)
                    if cached_prefix_entry is not None
                    else None
                )
                if not warned:
                    config_warnings.append(
                        "MLX memory pressure during generation; retried with smaller "
                        f"prefill chunks starting at {prefill_step_size}"
                    )
                    warned = True

    def _get_or_create_prefix_cache(
        self,
        prefix_ids: list[int],
        *,
        prefill_step_size: int,
        prefill_cache_policy: PrefillCachePolicy = "clear",
        prefill_sync_policy: PrefillSyncPolicy = "eval",
        prefill_sync_every: int = 4,
        prefill_cache_clear_every: int = 8,
        prefill_cache_threshold_gb: float | None = None,
        max_kv_size: int | None = None,
    ) -> PrefixCacheBuildResult:
        timings = GenerationTimings()
        if len(prefix_ids) < 2:
            return PrefixCacheBuildResult(None, False, 0.0, timings)

        key = _prefix_cache_key(prefix_ids, max_kv_size=max_kv_size)
        existing = self._prefix_cache.get(key)
        if existing is not None:
            return PrefixCacheBuildResult(existing, True, 0.0, timings)

        import mlx.core as mx
        from mlx_lm.models import cache

        prompt_cache = cache.make_prompt_cache(self.loaded.model, max_kv_size=max_kv_size)
        prompt = mx.array(prefix_ids)
        processed = 0
        chunk_index = 0
        build_start = now()
        while len(prompt) - processed > 0:
            count = min(prefill_step_size, len(prompt) - processed)
            model_start = now()
            self.loaded.model(prompt[processed : processed + count][None], cache=prompt_cache)
            timings.prefill_model_seconds += now() - model_start
            is_last_chunk = len(prompt) - (processed + count) <= 0
            sync_start = now()
            _sync_prompt_cache(
                prompt_cache,
                prefill_sync_policy,
                chunk_index=chunk_index,
                is_last_chunk=is_last_chunk,
                sync_every=prefill_sync_every,
            )
            timings.prefill_sync_seconds += now() - sync_start
            processed += count
            clear_start = now()
            _clear_mlx_cache(
                prefill_cache_policy,
                chunk_index=chunk_index,
                clear_every=prefill_cache_clear_every,
                threshold_gb=prefill_cache_threshold_gb,
            )
            timings.prefill_clear_cache_seconds += now() - clear_start
            chunk_index += 1
        build_seconds = now() - build_start

        entry = PrefixCacheEntry(token_ids=list(prefix_ids), cache=_clone_prompt_cache(prompt_cache))
        if len(self._prefix_cache) >= self.max_prefix_cache_entries:
            self._prefix_cache.pop(next(iter(self._prefix_cache)))
        self._prefix_cache[key] = entry
        return PrefixCacheBuildResult(entry, False, build_seconds, timings)


def _prefix_cache_key(token_ids: list[int], *, max_kv_size: int | None = None) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(len(token_ids).to_bytes(8, "little"))
    digest.update(
        (-1 if max_kv_size is None else int(max_kv_size)).to_bytes(
            8,
            "little",
            signed=True,
        )
    )
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def _clone_prompt_cache(prompt_cache: list[object]) -> list[object]:
    cloned = []
    for entry in prompt_cache:
        cloned.append(
            type(entry).from_state(
                _clone_cache_state(entry.state),
                _clone_cache_state(entry.meta_state),
            )
        )
    return cloned


def _clone_cache_state(value: object) -> object:
    if _is_mlx_array(value):
        return value.__copy__()
    if isinstance(value, list):
        return [_clone_cache_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache_state(item) for item in value)
    if isinstance(value, dict):
        return {
            _clone_cache_state(key): _clone_cache_state(item)
            for key, item in value.items()
        }
    return value


def _is_mlx_array(value: object) -> bool:
    return type(value).__module__.startswith("mlx.") and hasattr(value, "__copy__")


def _make_prompt_cache(model: object, *, max_kv_size: int | None = None) -> list[object]:
    from mlx_lm.models import cache

    return cache.make_prompt_cache(model, max_kv_size=max_kv_size)


def _format_prompt(tokenizer: object, prompt: str, mode: PromptMode) -> str:
    if mode == "raw":
        return prompt
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _encode(tokenizer: object, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return list(encoded)


def _decode(tokenizer: object, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids)


def _prefill_step_size(value: PrefillStepSize, prompt_tokens: int) -> int:
    if value != "auto":
        return int(value)
    if prompt_tokens <= 1024:
        return 1024
    if prompt_tokens <= 8192:
        return 2048
    if prompt_tokens <= 32768:
        return 4096
    return 8192


def _smaller_prefill_step_size(value: int) -> int | None:
    for candidate in (4096, 2048, 1024, 512):
        if value > candidate:
            return candidate
    return None


def _is_mlx_memory_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in (
            "out of memory",
            "oom",
            "failed to allocate",
            "memory limit",
            "metal",
        )
    )


def _clear_mlx_runtime_cache() -> None:
    try:
        import mlx.core as mx
    except Exception:
        return
    mx.clear_cache()


def _decode_latency_stats(latencies: list[float]) -> tuple[float | None, float | None, float | None]:
    if not latencies:
        return None, None, None
    sorted_latencies = sorted(latencies)
    return (
        _percentile(sorted_latencies, 0.50),
        _percentile(sorted_latencies, 0.95),
        max(sorted_latencies),
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(
        len(sorted_values) - 1,
        max(0, round((len(sorted_values) - 1) * percentile)),
    )
    return sorted_values[index]


def _clear_mlx_cache(
    policy: PrefillCachePolicy,
    *,
    chunk_index: int = 0,
    clear_every: int = 8,
    threshold_gb: float | None = None,
) -> None:
    if policy == "retain":
        return
    if policy == "periodic":
        if clear_every < 1:
            raise ValueError("prefill cache clear interval must be >= 1")
        if (chunk_index + 1) % clear_every != 0:
            return
    elif policy == "threshold":
        if threshold_gb is None:
            return
        import mlx.core as mx

        if mx.get_active_memory() <= _gb_to_bytes(threshold_gb):
            return
    elif policy != "clear":
        raise ValueError(f"unknown prefill cache policy: {policy}")
    import mlx.core as mx

    mx.clear_cache()


def _sync_prompt_cache(
    prompt_cache: list[object],
    policy: PrefillSyncPolicy,
    *,
    chunk_index: int = 0,
    is_last_chunk: bool = True,
    sync_every: int = 4,
) -> None:
    if policy == "none":
        return
    import mlx.core as mx

    states = [entry.state for entry in prompt_cache]
    if policy == "eval":
        mx.eval(states)
        return
    if policy == "async":
        mx.async_eval(states)
        return
    if policy == "periodic":
        if sync_every < 1:
            raise ValueError("prefill sync interval must be >= 1")
        if is_last_chunk or (chunk_index + 1) % sync_every == 0:
            mx.eval(states)
        else:
            mx.async_eval(states)
        return
    raise ValueError(f"unknown prefill sync policy: {policy}")


def _gb_to_bytes(value: float) -> int:
    return int(value * 1_000_000_000)


def _configure_mlx_memory(
    *,
    memory_limit_gb: float | None = None,
    cache_limit_gb: float | None = None,
    wired_limit_gb: float | None = None,
) -> list[str]:
    if memory_limit_gb is None and cache_limit_gb is None and wired_limit_gb is None:
        return []

    import mlx.core as mx

    configured: list[str] = []
    try:
        if memory_limit_gb is not None:
            previous = mx.set_memory_limit(_gb_to_bytes(memory_limit_gb))
            configured.append(
                f"MLX memory limit set to {memory_limit_gb:g} GB "
                f"(previous {previous / 1_000_000_000:.3g} GB)"
            )
        if cache_limit_gb is not None:
            previous = mx.set_cache_limit(_gb_to_bytes(cache_limit_gb))
            configured.append(
                f"MLX cache limit set to {cache_limit_gb:g} GB "
                f"(previous {previous / 1_000_000_000:.3g} GB)"
            )
        if wired_limit_gb is not None:
            previous = mx.set_wired_limit(_gb_to_bytes(wired_limit_gb))
            configured.append(
                f"MLX wired limit set to {wired_limit_gb:g} GB "
                f"(previous {previous / 1_000_000_000:.3g} GB)"
            )
    except Exception as exc:
        raise RuntimeError(f"failed to configure MLX memory limits: {exc}") from exc
    return configured


def _greedy_generate_tokens(
    *,
    model: object,
    prompt_ids: list[int],
    max_tokens: int,
    backend: ArgmaxBackend,
    prefill_step_size: int,
    eos_token_ids: set[int],
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: int | None = None,
    prompt_cache: list[object] | None = None,
    decode_variant: DecodeVariant = "custom",
    prefill_cache_policy: PrefillCachePolicy = "clear",
    prefill_sync_policy: PrefillSyncPolicy = "eval",
    prefill_sync_every: int = 4,
    prefill_cache_clear_every: int = 8,
    prefill_cache_threshold_gb: float | None = None,
) -> tuple[list[int], float, float, float, GenerationTimings]:
    if backend.name == "mlx":
        return _greedy_generate_tokens_mlx(
            model=model,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
            eos_token_ids=eos_token_ids,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            max_kv_size=max_kv_size,
            prompt_cache=prompt_cache,
            decode_variant=decode_variant,
            prefill_cache_policy=prefill_cache_policy,
            prefill_sync_policy=prefill_sync_policy,
            prefill_sync_every=prefill_sync_every,
            prefill_cache_clear_every=prefill_cache_clear_every,
            prefill_cache_threshold_gb=prefill_cache_threshold_gb,
        )
    if decode_variant != "custom":
        raise ValueError(f"decode variant {decode_variant!r} requires the MLX backend")

    import mlx.core as mx
    from mlx_lm.models import cache

    prompt = mx.array(prompt_ids)
    prompt_cache = prompt_cache or cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    processed = 0

    timings = GenerationTimings(decode_token_latencies=[])
    prefill_start = now()
    chunk_index = 0
    while len(prompt) - processed > 1:
        remaining = (len(prompt) - processed) - 1
        count = min(prefill_step_size, remaining)
        model_start = now()
        model(prompt[processed : processed + count][None], cache=prompt_cache)
        timings.prefill_model_seconds += now() - model_start
        is_last_chunk = (len(prompt) - (processed + count)) <= 1
        sync_start = now()
        _sync_prompt_cache(
            prompt_cache,
            prefill_sync_policy,
            chunk_index=chunk_index,
            is_last_chunk=is_last_chunk,
            sync_every=prefill_sync_every,
        )
        timings.prefill_sync_seconds += now() - sync_start
        processed += count
        clear_start = now()
        _clear_mlx_cache(
            prefill_cache_policy,
            chunk_index=chunk_index,
            clear_every=prefill_cache_clear_every,
            threshold_gb=prefill_cache_threshold_gb,
        )
        timings.prefill_clear_cache_seconds += now() - clear_start
        chunk_index += 1

    model_start = now()
    logits = model(prompt[processed:][None], cache=prompt_cache)[:, -1, :]
    timings.prefill_model_seconds += now() - model_start
    sync_start = now()
    mx.eval(logits)
    timings.prefill_sync_seconds += now() - sync_start
    prefill_seconds = now() - prefill_start

    generated: list[int] = []
    decode_start = now()
    first_token_seconds = 0.0
    next_logits = logits

    for index in range(max_tokens):
        token_start = now()
        next_token = backend.argmax(next_logits[0])
        if index == 0:
            first_token_seconds = now() - prefill_start
            timings.first_token_eval_seconds = now() - token_start
        if next_token in eos_token_ids:
            timings.decode_token_latencies.append(now() - token_start)
            break
        generated.append(next_token)
        model_start = now()
        next_logits = model(mx.array([[next_token]]), cache=prompt_cache)[:, -1, :]
        timings.decode_model_seconds += now() - model_start
        sync_start = now()
        mx.eval(next_logits)
        timings.decode_sync_seconds += now() - sync_start
        timings.decode_token_latencies.append(now() - token_start)

    decode_seconds = now() - decode_start
    return generated, prefill_seconds, decode_seconds, first_token_seconds, timings


def _greedy_generate_tokens_mlx(
    *,
    model: object,
    prompt_ids: list[int],
    max_tokens: int,
    prefill_step_size: int,
    eos_token_ids: set[int],
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: int | None = None,
    prompt_cache: list[object] | None = None,
    decode_variant: DecodeVariant = "custom",
    prefill_cache_policy: PrefillCachePolicy = "clear",
    prefill_sync_policy: PrefillSyncPolicy = "eval",
    prefill_sync_every: int = 4,
    prefill_cache_clear_every: int = 8,
    prefill_cache_threshold_gb: float | None = None,
) -> tuple[list[int], float, float, float, GenerationTimings]:
    if decode_variant == "mlx_lm_generate_step":
        return _greedy_generate_tokens_mlx_generate_step(
            model=model,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
            eos_token_ids=eos_token_ids,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
            max_kv_size=max_kv_size,
            prompt_cache=prompt_cache,
        )
    if decode_variant not in {
        "custom",
        "custom_no_async",
        "custom_eval_next",
        "custom_defer_ids",
    }:
        raise ValueError(f"unknown decode variant: {decode_variant}")

    import mlx.core as mx
    from mlx_lm.generate import generation_stream, wired_limit
    from mlx_lm.models import cache

    prompt = mx.array(prompt_ids)
    prompt_cache = prompt_cache or cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    processed = 0
    timings = GenerationTimings(decode_token_latencies=[])

    def quantize_supported_cache_entries() -> None:
        if kv_bits is None:
            return
        for index, entry in enumerate(prompt_cache):
            if not hasattr(entry, "to_quantized") or entry.offset < quantized_kv_start:
                continue
            try:
                prompt_cache[index] = entry.to_quantized(
                    group_size=kv_group_size,
                    bits=kv_bits,
                )
            except NotImplementedError:
                continue

    def model_call(input_tokens: object) -> object:
        return model(input_tokens[None], cache=prompt_cache)

    def step(input_tokens: object) -> object:
        with mx.stream(generation_stream):
            logits = model_call(input_tokens)[:, -1, :]
            return mx.argmax(logits, axis=-1)

    with wired_limit(model, [generation_stream]):
        prefill_start = now()
        with mx.stream(generation_stream):
            chunk_index = 0
            while len(prompt) - processed > 1:
                remaining = (len(prompt) - processed) - 1
                count = min(prefill_step_size, remaining)
                model_start = now()
                model_call(prompt[processed : processed + count])
                timings.prefill_model_seconds += now() - model_start
                is_last_chunk = (len(prompt) - (processed + count)) <= 1
                sync_start = now()
                _sync_prompt_cache(
                    prompt_cache,
                    prefill_sync_policy,
                    chunk_index=chunk_index,
                    is_last_chunk=is_last_chunk,
                    sync_every=prefill_sync_every,
                )
                timings.prefill_sync_seconds += now() - sync_start
                processed += count
                clear_start = now()
                _clear_mlx_cache(
                    prefill_cache_policy,
                    chunk_index=chunk_index,
                    clear_every=prefill_cache_clear_every,
                    threshold_gb=prefill_cache_threshold_gb,
                )
                timings.prefill_clear_cache_seconds += now() - clear_start
                chunk_index += 1

            first_token_start = now()
            token = step(prompt[processed:])
            timings.prefill_model_seconds += now() - first_token_start
            quantize_supported_cache_entries()
            sync_start = now()
            mx.async_eval(token)
            mx.eval(token)
            timings.prefill_sync_seconds += now() - sync_start
            timings.first_token_eval_seconds = now() - first_token_start

        prefill_seconds = now() - prefill_start
        if decode_variant == "custom_defer_ids":
            generated: list[int] = []
            decode_start = now()
            token_arrays = []
            for index in range(max_tokens):
                token_arrays.append(token)
                if index + 1 < max_tokens:
                    model_start = now()
                    token = step(token)
                    timings.decode_model_seconds += now() - model_start
                    sync_start = now()
                    mx.async_eval(token)
                    timings.decode_sync_seconds += now() - sync_start

            if token_arrays:
                sync_start = now()
                tokens = mx.concatenate(token_arrays, axis=0)
                mx.eval(tokens)
                timings.decode_sync_seconds += now() - sync_start
                item_start = now()
                token_ids = [int(token_id) for token_id in tokens.tolist()]
                timings.decode_token_item_seconds += now() - item_start
                for token_id in token_ids:
                    if token_id in eos_token_ids:
                        break
                    generated.append(token_id)
            decode_seconds = now() - decode_start
            first_token_seconds = prefill_seconds + decode_seconds
            if generated:
                per_token = decode_seconds / len(generated)
                timings.decode_token_latencies.extend([per_token] * len(generated))
            return generated, prefill_seconds, decode_seconds, first_token_seconds, timings

        generated: list[int] = []
        decode_start = now()
        first_token_seconds = 0.0

        for index in range(max_tokens):
            token_start = now()
            next_token = None
            if decode_variant != "custom_no_async" and index + 1 < max_tokens:
                model_start = now()
                next_token = step(token)
                timings.decode_model_seconds += now() - model_start
                sync_start = now()
                mx.async_eval(next_token)
                timings.decode_sync_seconds += now() - sync_start

            sync_start = now()
            mx.eval(token)
            timings.decode_sync_seconds += now() - sync_start
            item_start = now()
            token_id = int(token.item())
            timings.decode_token_item_seconds += now() - item_start
            if index == 0:
                first_token_seconds = prefill_seconds + (now() - decode_start)
            if token_id in eos_token_ids:
                timings.decode_token_latencies.append(now() - token_start)
                break

            generated.append(token_id)
            if next_token is None:
                if decode_variant == "custom_no_async" and index + 1 < max_tokens:
                    model_start = now()
                    token = step(token)
                    timings.decode_model_seconds += now() - model_start
                    sync_start = now()
                    mx.eval(token)
                    timings.decode_sync_seconds += now() - sync_start
                    timings.decode_token_latencies.append(now() - token_start)
                    continue
                break
            if decode_variant == "custom_eval_next":
                sync_start = now()
                mx.eval(next_token)
                timings.decode_sync_seconds += now() - sync_start
            token = next_token
            timings.decode_token_latencies.append(now() - token_start)

        decode_seconds = now() - decode_start

    return generated, prefill_seconds, decode_seconds, first_token_seconds, timings


def _greedy_generate_tokens_mlx_generate_step(
    *,
    model: object,
    prompt_ids: list[int],
    max_tokens: int,
    prefill_step_size: int,
    eos_token_ids: set[int],
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: int | None = None,
    prompt_cache: list[object] | None = None,
) -> tuple[list[int], float, float, float, GenerationTimings]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    prompt = mx.array(prompt_ids)
    prefill_start = now()
    prefill_seconds: float | None = None
    timings = GenerationTimings(decode_token_latencies=[])

    def prompt_progress(processed: int, total: int) -> None:
        nonlocal prefill_seconds
        if processed == total and prefill_seconds is None:
            prefill_seconds = now() - prefill_start

    generated: list[int] = []
    generator = generate_step(
        prompt,
        model,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        max_kv_size=max_kv_size,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        prompt_progress_callback=prompt_progress,
    )

    decode_start: float | None = None
    first_token_seconds = 0.0
    for token_id, _logprobs in generator:
        token_start = now()
        if prefill_seconds is None:
            prefill_seconds = now() - prefill_start
        if decode_start is None:
            decode_start = now()
            first_token_seconds = prefill_seconds
            timings.first_token_eval_seconds = 0.0
        item_start = now()
        token_id = int(token_id)
        timings.decode_token_item_seconds += now() - item_start
        if token_id in eos_token_ids:
            timings.decode_token_latencies.append(now() - token_start)
            break
        generated.append(token_id)
        timings.decode_token_latencies.append(now() - token_start)

    if prefill_seconds is None:
        prefill_seconds = now() - prefill_start
    timings.prefill_model_seconds = prefill_seconds
    decode_seconds = now() - decode_start if decode_start is not None else 0.0
    return generated, prefill_seconds, decode_seconds, first_token_seconds, timings


def infer(
    prompt: str,
    *,
    model_path: str,
    max_tokens: int,
    backend: BackendName = "auto",
    prompt_mode: PromptMode = "chat",
    prefill_step_size: PrefillStepSize = "auto",
    prefill_cache_policy: PrefillCachePolicy = "clear",
    prefill_sync_policy: PrefillSyncPolicy = "eval",
    prefill_sync_every: int = 4,
    prefill_cache_clear_every: int = 8,
    prefill_cache_threshold_gb: float | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: int | None = None,
    eos_token_id: int | None = None,
    cache_prefix: str | None = None,
    cache_prefix_mode: PromptMode = "raw",
    session_id: str | None = None,
    reset_session: bool = False,
    append_to_session: bool = False,
    token_cache_dir: str | None = DEFAULT_TOKEN_CACHE_DIR,
    max_token_cache_disk_bytes: int | None = DEFAULT_MAX_TOKEN_CACHE_DISK_BYTES,
    mlx_memory_limit_gb: float | None = None,
    mlx_cache_limit_gb: float | None = None,
    mlx_wired_limit_gb: float | None = None,
) -> GenerationResult:
    engine = Gemma4Engine(
        model_path=model_path,
        backend=backend,
        token_cache_dir=token_cache_dir,
        max_token_cache_disk_bytes=max_token_cache_disk_bytes,
        mlx_memory_limit_gb=mlx_memory_limit_gb,
        mlx_cache_limit_gb=mlx_cache_limit_gb,
        mlx_wired_limit_gb=mlx_wired_limit_gb,
    )
    return engine.infer(
        prompt,
        max_tokens=max_tokens,
        prompt_mode=prompt_mode,
        prefill_step_size=prefill_step_size,
        prefill_cache_policy=prefill_cache_policy,
        prefill_sync_policy=prefill_sync_policy,
        prefill_sync_every=prefill_sync_every,
        prefill_cache_clear_every=prefill_cache_clear_every,
        prefill_cache_threshold_gb=prefill_cache_threshold_gb,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        max_kv_size=max_kv_size,
        eos_token_id=eos_token_id,
        cache_prefix=cache_prefix,
        cache_prefix_mode=cache_prefix_mode,
        session_id=session_id,
        reset_session=reset_session,
        append_to_session=append_to_session,
    )

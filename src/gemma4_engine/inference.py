from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Literal

from .backends import ArgmaxBackend, BackendName, select_backend
from .loader import load_model
from .stats import RunStats, memory_snapshot, now
from .token_cache import DEFAULT_TOKEN_CACHE_DIR, HierarchicalTokenCache, token_cache_key

PromptMode = Literal["chat", "raw"]
PrefillStepSize = Literal["auto", "512", "1024", "2048", "4096", "8192"]
PrefillCachePolicy = Literal["clear", "retain"]
PrefillSyncPolicy = Literal["eval", "async", "none"]
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
    draft_model_path: str | None = None
    speculative_acceptance_rate: float | None = None


@dataclass
class PrefixCacheEntry:
    token_ids: list[int]
    cache: list[object]


SPECULATIVE_INSTALL_MESSAGE = (
    "Speculative decoding is experimental and requires optional dependencies. "
    "Install them with `uv sync --extra speculative` or "
    "`pip install 'gemma4-engine[speculative]'`."
)


@dataclass
class Gemma4Engine:
    model_path: str
    backend: BackendName = "auto"
    max_prefix_cache_entries: int = 4
    token_cache_dir: str | None = DEFAULT_TOKEN_CACHE_DIR
    max_token_cache_entries: int = 128
    draft_model_path: str | None = None
    draft_tokens: int = 4
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
        self._token_cache = HierarchicalTokenCache(
            disk_dir=self.token_cache_dir,
            max_memory_entries=self.max_token_cache_entries,
        )
        self.speculative_runtime = (
            _create_speculative_runtime(
                self.model_path,
                self.draft_model_path,
                draft_tokens=self.draft_tokens,
            )
            if self.draft_model_path
            else None
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
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        max_kv_size: int | None = None,
        eos_token_id: int | None = None,
        cache_prefix: str | None = None,
        cache_prefix_mode: PromptMode = "raw",
        _decode_variant: DecodeVariant = "custom",
    ) -> GenerationResult:
        prompt_text = _format_prompt(self.loaded.tokenizer, prompt, prompt_mode)
        prompt_ids = _encode(self.loaded.tokenizer, prompt_text)
        prefix_cache_hit = False
        prefix_tokens = 0
        prefix_token_cache_source = None
        prompt_cache = None
        prefill_ids = prompt_ids
        prefix_cache_build_seconds = 0.0

        if cache_prefix:
            prefix_text = _format_prompt(self.loaded.tokenizer, cache_prefix, cache_prefix_mode)
            prefix_token_cache = self._token_cache.get_or_encode(
                key=token_cache_key(
                    model_path=self.model_path,
                    prompt_mode=cache_prefix_mode,
                    text=prefix_text,
                ),
                encode=lambda: _encode(self.loaded.tokenizer, prefix_text),
            )
            prefix_ids = prefix_token_cache.token_ids
            prefix_token_cache_source = prefix_token_cache.source
            if prompt_ids[: len(prefix_ids)] == prefix_ids:
                suffix_ids = prompt_ids[len(prefix_ids) :]
            else:
                suffix_ids = prompt_ids
                prompt_ids = prefix_ids + suffix_ids

            cached, prefix_cache_hit, prefix_cache_build_seconds = (
                self._get_or_create_prefix_cache(
                    prefix_ids,
                    prefill_step_size=_prefill_step_size(prefill_step_size, len(prefix_ids)),
                    prefill_cache_policy=prefill_cache_policy,
                    prefill_sync_policy=prefill_sync_policy,
                    max_kv_size=max_kv_size,
                )
            )
            if cached is not None:
                prompt_cache = _clone_prompt_cache(cached.cache)
                prefill_ids = suffix_ids
                prefix_tokens = len(prefix_ids)

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

        speculative_acceptance_rate = None
        if (
            self.speculative_runtime is not None
            and prompt_cache is None
            and self.argmax_backend.name == "mlx"
        ):
            spec = self.speculative_runtime.generate(
                prompt_ids,
                max_tokens=max_tokens,
                eos_token_ids=eos_ids,
                prefill_step_size=_prefill_step_size(prefill_step_size, len(prompt_ids)),
            )
            generated = spec.token_ids
            prefill_seconds = spec.prefill_seconds
            decode_seconds = spec.decode_seconds
            first_token_seconds = spec.time_to_first_token_seconds
            total_draft = sum(spec.draft_lengths)
            speculative_acceptance_rate = (
                sum(spec.accept_lengths) / total_draft if total_draft else None
            )
        else:
            if self.speculative_runtime is not None and prompt_cache is not None:
                config_warnings.append(
                    "draft model disabled for this request because prefix-cache reuse "
                    "is not wired into the MTP speculative runtime yet"
                )
            generated, prefill_seconds, decode_seconds, first_token_seconds = (
                _greedy_generate_tokens(
                    model=self.loaded.model,
                    prompt_ids=prefill_ids,
                    max_tokens=max_tokens,
                    backend=self.argmax_backend,
                    prefill_step_size=_prefill_step_size(prefill_step_size, len(prefill_ids)),
                    eos_token_ids=eos_ids,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    quantized_kv_start=quantized_kv_start,
                    max_kv_size=max_kv_size,
                    prompt_cache=prompt_cache,
                    decode_variant=_decode_variant,
                    prefill_cache_policy=prefill_cache_policy,
                    prefill_sync_policy=prefill_sync_policy,
                )
            )

        stats = RunStats(
            model_path=self.model_path,
            backend=self.backend_status.selected,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            prefill_seconds=prefill_seconds + prefix_cache_build_seconds,
            decode_seconds=decode_seconds,
            time_to_first_token_seconds=first_token_seconds + prefix_cache_build_seconds,
            **memory_snapshot(),
        )
        return GenerationResult(
            text=_decode(self.loaded.tokenizer, generated),
            token_ids=generated,
            stats=stats,
            backend_reason=self.backend_status.reason,
            config_warnings=config_warnings,
            prefix_cache_hit=prefix_cache_hit,
            prefix_tokens=prefix_tokens,
            prefix_token_cache_source=prefix_token_cache_source,
            draft_model_path=self.draft_model_path,
            speculative_acceptance_rate=speculative_acceptance_rate,
        )

    def clear_prefix_cache(self) -> None:
        self._prefix_cache.clear()

    def clear_token_memory_cache(self) -> None:
        self._token_cache.clear_memory()

    def _get_or_create_prefix_cache(
        self,
        prefix_ids: list[int],
        *,
        prefill_step_size: int,
        prefill_cache_policy: PrefillCachePolicy = "clear",
        prefill_sync_policy: PrefillSyncPolicy = "eval",
        max_kv_size: int | None = None,
    ) -> tuple[PrefixCacheEntry | None, bool, float]:
        if len(prefix_ids) < 2:
            return None, False, 0.0

        key = _prefix_cache_key(prefix_ids, max_kv_size=max_kv_size)
        existing = self._prefix_cache.get(key)
        if existing is not None:
            return existing, True, 0.0

        import mlx.core as mx
        from mlx_lm.models import cache

        prompt_cache = cache.make_prompt_cache(self.loaded.model, max_kv_size=max_kv_size)
        prompt = mx.array(prefix_ids)
        processed = 0
        build_start = now()
        while len(prompt) - processed > 0:
            count = min(prefill_step_size, len(prompt) - processed)
            self.loaded.model(prompt[processed : processed + count][None], cache=prompt_cache)
            _sync_prompt_cache(prompt_cache, prefill_sync_policy)
            processed += count
            _clear_mlx_cache(prefill_cache_policy)
        build_seconds = now() - build_start

        entry = PrefixCacheEntry(token_ids=list(prefix_ids), cache=_clone_prompt_cache(prompt_cache))
        if len(self._prefix_cache) >= self.max_prefix_cache_entries:
            self._prefix_cache.pop(next(iter(self._prefix_cache)))
        self._prefix_cache[key] = entry
        return entry, False, build_seconds


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
    from mlx_lm.models import cache

    cloned = []
    for entry in prompt_cache:
        cloned.append(type(entry).from_state(copy.deepcopy(entry.state), copy.deepcopy(entry.meta_state)))
    return cloned


def _create_speculative_runtime(
    target_model_path: str,
    draft_model_path: str,
    *,
    draft_tokens: int,
) -> object:
    try:
        from .speculative import SpeculativeRuntime

        return SpeculativeRuntime(
            target_model_path,
            draft_model_path,
            draft_tokens=draft_tokens,
        )
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == "mlx_vlm" or exc.name.startswith("mlx_vlm.")):
            raise RuntimeError(SPECULATIVE_INSTALL_MESSAGE) from exc
        raise


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
    if value == "auto":
        return 512
    return int(value)


def _clear_mlx_cache(policy: PrefillCachePolicy) -> None:
    if policy == "retain":
        return
    if policy != "clear":
        raise ValueError(f"unknown prefill cache policy: {policy}")
    import mlx.core as mx

    mx.clear_cache()


def _sync_prompt_cache(prompt_cache: list[object], policy: PrefillSyncPolicy) -> None:
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
) -> tuple[list[int], float, float, float]:
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
        )
    if decode_variant != "custom":
        raise ValueError(f"decode variant {decode_variant!r} requires the MLX backend")

    import mlx.core as mx
    from mlx_lm.models import cache

    prompt = mx.array(prompt_ids)
    prompt_cache = prompt_cache or cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    processed = 0

    prefill_start = now()
    while len(prompt) - processed > 1:
        remaining = (len(prompt) - processed) - 1
        count = min(prefill_step_size, remaining)
        model(prompt[processed : processed + count][None], cache=prompt_cache)
        _sync_prompt_cache(prompt_cache, prefill_sync_policy)
        processed += count
        _clear_mlx_cache(prefill_cache_policy)

    logits = model(prompt[processed:][None], cache=prompt_cache)[:, -1, :]
    mx.eval(logits)
    prefill_seconds = now() - prefill_start

    generated: list[int] = []
    decode_start = now()
    first_token_seconds = 0.0
    next_logits = logits

    for index in range(max_tokens):
        next_token = backend.argmax(next_logits[0])
        if index == 0:
            first_token_seconds = now() - prefill_start
        if next_token in eos_token_ids:
            break
        generated.append(next_token)
        next_logits = model(mx.array([[next_token]]), cache=prompt_cache)[:, -1, :]
        mx.eval(next_logits)

    decode_seconds = now() - decode_start
    return generated, prefill_seconds, decode_seconds, first_token_seconds


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
) -> tuple[list[int], float, float, float]:
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
            while len(prompt) - processed > 1:
                remaining = (len(prompt) - processed) - 1
                count = min(prefill_step_size, remaining)
                model_call(prompt[processed : processed + count])
                _sync_prompt_cache(prompt_cache, prefill_sync_policy)
                processed += count
                _clear_mlx_cache(prefill_cache_policy)

            token = step(prompt[processed:])
            quantize_supported_cache_entries()
            mx.async_eval(token)
            mx.eval(token)

        prefill_seconds = now() - prefill_start
        if decode_variant == "custom_defer_ids":
            generated: list[int] = []
            decode_start = now()
            token_arrays = []
            for index in range(max_tokens):
                token_arrays.append(token)
                if index + 1 < max_tokens:
                    token = step(token)
                    mx.async_eval(token)

            if token_arrays:
                tokens = mx.concatenate(token_arrays, axis=0)
                mx.eval(tokens)
                token_ids = [int(token_id) for token_id in tokens.tolist()]
                for token_id in token_ids:
                    if token_id in eos_token_ids:
                        break
                    generated.append(token_id)
            decode_seconds = now() - decode_start
            first_token_seconds = prefill_seconds + decode_seconds
            return generated, prefill_seconds, decode_seconds, first_token_seconds

        generated: list[int] = []
        decode_start = now()
        first_token_seconds = 0.0

        for index in range(max_tokens):
            next_token = None
            if decode_variant != "custom_no_async" and index + 1 < max_tokens:
                next_token = step(token)
                mx.async_eval(next_token)

            mx.eval(token)
            token_id = int(token.item())
            if index == 0:
                first_token_seconds = prefill_seconds + (now() - decode_start)
            if token_id in eos_token_ids:
                break

            generated.append(token_id)
            if next_token is None:
                if decode_variant == "custom_no_async" and index + 1 < max_tokens:
                    token = step(token)
                    mx.eval(token)
                    continue
                break
            if decode_variant == "custom_eval_next":
                mx.eval(next_token)
            token = next_token

        decode_seconds = now() - decode_start

    return generated, prefill_seconds, decode_seconds, first_token_seconds


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
) -> tuple[list[int], float, float, float]:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    prompt = mx.array(prompt_ids)
    prefill_start = now()
    prefill_seconds: float | None = None

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
        if prefill_seconds is None:
            prefill_seconds = now() - prefill_start
        if decode_start is None:
            decode_start = now()
            first_token_seconds = prefill_seconds
        token_id = int(token_id)
        if token_id in eos_token_ids:
            break
        generated.append(token_id)

    if prefill_seconds is None:
        prefill_seconds = now() - prefill_start
    decode_seconds = now() - decode_start if decode_start is not None else 0.0
    return generated, prefill_seconds, decode_seconds, first_token_seconds


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
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    max_kv_size: int | None = None,
    eos_token_id: int | None = None,
    cache_prefix: str | None = None,
    cache_prefix_mode: PromptMode = "raw",
    token_cache_dir: str | None = DEFAULT_TOKEN_CACHE_DIR,
    draft_model_path: str | None = None,
    draft_tokens: int = 4,
    mlx_memory_limit_gb: float | None = None,
    mlx_cache_limit_gb: float | None = None,
    mlx_wired_limit_gb: float | None = None,
) -> GenerationResult:
    engine = Gemma4Engine(
        model_path=model_path,
        backend=backend,
        token_cache_dir=token_cache_dir,
        draft_model_path=draft_model_path,
        draft_tokens=draft_tokens,
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
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        max_kv_size=max_kv_size,
        eos_token_id=eos_token_id,
        cache_prefix=cache_prefix,
        cache_prefix_mode=cache_prefix_mode,
    )

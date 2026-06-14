from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .backends import ArgmaxBackend, BackendName, select_backend
from .loader import load_model
from .stats import RunStats, memory_snapshot, now

PromptMode = Literal["chat", "raw"]
PrefillStepSize = Literal["auto", "512", "1024", "2048", "4096", "8192"]


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    stats: RunStats
    backend_reason: str
    config_warnings: list[str]


@dataclass
class Gemma4Engine:
    model_path: str
    backend: BackendName = "auto"

    def __post_init__(self) -> None:
        self.loaded = load_model(self.model_path)
        self.argmax_backend, self.backend_status = select_backend(self.backend)

    def infer(
        self,
        prompt: str,
        *,
        max_tokens: int,
        prompt_mode: PromptMode = "chat",
        prefill_step_size: PrefillStepSize = "auto",
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        eos_token_id: int | None = None,
    ) -> GenerationResult:
        prompt_text = _format_prompt(self.loaded.tokenizer, prompt, prompt_mode)
        prompt_ids = _encode(self.loaded.tokenizer, prompt_text)
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

        generated, prefill_seconds, decode_seconds, first_token_seconds = (
            _greedy_generate_tokens(
                model=self.loaded.model,
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                backend=self.argmax_backend,
                prefill_step_size=_prefill_step_size(prefill_step_size, len(prompt_ids)),
                eos_token_ids=eos_ids,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                quantized_kv_start=quantized_kv_start,
            )
        )

        stats = RunStats(
            model_path=self.model_path,
            backend=self.backend_status.selected,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            time_to_first_token_seconds=first_token_seconds,
            **memory_snapshot(),
        )
        return GenerationResult(
            text=_decode(self.loaded.tokenizer, generated),
            token_ids=generated,
            stats=stats,
            backend_reason=self.backend_status.reason,
            config_warnings=config_warnings,
        )


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
        if prompt_tokens >= 2048:
            return 1024
        return 512
    return int(value)


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
        )

    import mlx.core as mx
    from mlx_lm.models import cache

    prompt = mx.array(prompt_ids)
    prompt_cache = cache.make_prompt_cache(model)
    processed = 0

    prefill_start = now()
    while len(prompt) - processed > 1:
        remaining = (len(prompt) - processed) - 1
        count = min(prefill_step_size, remaining)
        model(prompt[processed : processed + count][None], cache=prompt_cache)
        mx.eval([entry.state for entry in prompt_cache])
        processed += count
        mx.clear_cache()

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
) -> tuple[list[int], float, float, float]:
    import mlx.core as mx
    from mlx_lm.generate import generation_stream, wired_limit
    from mlx_lm.models import cache

    prompt = mx.array(prompt_ids)
    prompt_cache = cache.make_prompt_cache(model)
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
                mx.eval([entry.state for entry in prompt_cache])
                processed += count
                mx.clear_cache()

            token = step(prompt[processed:])
            quantize_supported_cache_entries()
            mx.async_eval(token)
            mx.eval(token)

        prefill_seconds = now() - prefill_start
        generated: list[int] = []
        decode_start = now()
        first_token_seconds = 0.0

        for index in range(max_tokens):
            next_token = None
            if index + 1 < max_tokens:
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
                break
            token = next_token

        decode_seconds = now() - decode_start

    return generated, prefill_seconds, decode_seconds, first_token_seconds


def infer(
    prompt: str,
    *,
    model_path: str,
    max_tokens: int,
    backend: BackendName = "auto",
    prompt_mode: PromptMode = "chat",
    prefill_step_size: PrefillStepSize = "auto",
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    eos_token_id: int | None = None,
) -> GenerationResult:
    engine = Gemma4Engine(model_path=model_path, backend=backend)
    return engine.infer(
        prompt,
        max_tokens=max_tokens,
        prompt_mode=prompt_mode,
        prefill_step_size=prefill_step_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        quantized_kv_start=quantized_kv_start,
        eos_token_id=eos_token_id,
    )

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
        if prompt_tokens >= 8192:
            return 8192
        if prompt_tokens >= 2048:
            return 2048
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
) -> tuple[list[int], float, float, float]:
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
        generated.append(next_token)
        if index == 0:
            first_token_seconds = now() - prefill_start
        if next_token in eos_token_ids:
            break
        next_logits = model(mx.array([[next_token]]), cache=prompt_cache)[:, -1, :]
        mx.eval(next_logits)

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
    eos_token_id: int | None = None,
) -> GenerationResult:
    loaded = load_model(model_path)
    argmax_backend, backend_status = select_backend(backend)

    prompt_text = _format_prompt(loaded.tokenizer, prompt, prompt_mode)
    prompt_ids = _encode(loaded.tokenizer, prompt_text)
    eos_ids = set(getattr(loaded.tokenizer, "eos_token_ids", []) or [])
    if eos_token_id is None:
        eos_token_id = getattr(loaded.tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_ids.add(int(eos_token_id))

    generated, prefill_seconds, decode_seconds, first_token_seconds = _greedy_generate_tokens(
        model=loaded.model,
        prompt_ids=prompt_ids,
        max_tokens=max_tokens,
        backend=argmax_backend,
        prefill_step_size=_prefill_step_size(prefill_step_size, len(prompt_ids)),
        eos_token_ids=eos_ids,
    )

    stats = RunStats(
        model_path=model_path,
        backend=backend_status.selected,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        time_to_first_token_seconds=first_token_seconds,
        **memory_snapshot(),
    )
    return GenerationResult(
        text=_decode(loaded.tokenizer, generated),
        token_ids=generated,
        stats=stats,
        backend_reason=backend_status.reason,
        config_warnings=loaded.warnings,
    )

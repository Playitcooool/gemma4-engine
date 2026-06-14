from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

from .stats import now


@dataclass
class SpeculativeResult:
    token_ids: list[int]
    prefill_seconds: float
    decode_seconds: float
    time_to_first_token_seconds: float
    accept_lengths: list[int]
    draft_lengths: list[int]


class SpeculativeRuntime:
    def __init__(self, target_model_path: str, draft_model_path: str, *, draft_tokens: int = 4):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.draft_tokens = draft_tokens
        self.target_model, self.processor = _load_vlm_target_non_strict(target_model_path)
        self.draft_model, self.draft_kind = _load_drafter(draft_model_path)
        self.draft_model.bind(self.target_model)
        _patch_quantized_gemma4_assistant_sparse_head(self.draft_model)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        eos_token_ids: set[int],
        prefill_step_size: int,
    ) -> SpeculativeResult:
        import mlx.core as mx
        from mlx_vlm.models import cache
        from mlx_vlm.speculative.utils import run_speculative_rounds

        prompt = mx.array(prompt_ids, dtype=mx.int32)
        prompt_cache = cache.make_prompt_cache(self.target_model.language_model)
        processed = 0

        prefill_start = now()
        while len(prompt) - processed > 1:
            remaining = (len(prompt) - processed) - 1
            count = min(prefill_step_size, remaining)
            self.target_model.language_model(
                prompt[processed : processed + count][None],
                cache=prompt_cache,
            )
            mx.eval([entry.state for entry in prompt_cache])
            processed += count
            mx.clear_cache()

        outputs = self.target_model.language_model(
            prompt[processed:][None],
            cache=prompt_cache,
            return_hidden=True,
            return_shared_kv=True,
        )
        logits = outputs.logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_token = mx.argmax(logprobs, axis=-1)
        mx.eval(first_token)
        prefill_seconds = now() - prefill_start

        def sampler(log_probs: Any) -> Any:
            return mx.argmax(log_probs, axis=-1)

        generated: list[int] = []
        decode_start = now()
        first_token_seconds = prefill_seconds
        for token, _ in run_speculative_rounds(
            self.target_model.language_model,
            self.draft_model,
            prompt_cache,
            prompt[None],
            first_token,
            logprobs.squeeze(0),
            outputs,
            draft_kind=self.draft_kind,
            max_tokens=max_tokens,
            sampler=sampler,
            draft_block_size=self.draft_tokens,
            sampler_is_greedy=True,
        ):
            if hasattr(token, "item"):
                token = int(token.item())
            token = int(token)
            if not generated:
                first_token_seconds = prefill_seconds + (now() - decode_start)
            if token in eos_token_ids:
                break
            generated.append(token)
            if len(generated) >= max_tokens:
                break

        return SpeculativeResult(
            token_ids=generated,
            prefill_seconds=prefill_seconds,
            decode_seconds=now() - decode_start,
            time_to_first_token_seconds=first_token_seconds,
            accept_lengths=list(getattr(self.draft_model, "accept_lens", [])),
            draft_lengths=list(getattr(self.draft_model, "draft_lens", [])),
        )


def _load_vlm_target_non_strict(model_path: str):
    import mlx.nn as nn
    from mlx_vlm import load

    original = nn.Module.load_weights

    def load_weights_non_strict(self, weights, strict=True):
        return original(self, weights, strict=False)

    nn.Module.load_weights = load_weights_non_strict
    try:
        return load(model_path)
    finally:
        nn.Module.load_weights = original


def _load_drafter(draft_model_path: str):
    from mlx_vlm.speculative.drafters import load_drafter

    return load_drafter(draft_model_path)


def _patch_quantized_gemma4_assistant_sparse_head(draft_model: object) -> None:
    masked = getattr(draft_model, "masked_embedding", None)
    embedding = getattr(getattr(draft_model, "model", None), "embed_tokens", None)
    if masked is None or embedding is None or not hasattr(embedding, "scales"):
        return

    def selected_logits(self, hidden_states, lm_head_weight):
        import mlx.core as mx

        batch, length = hidden_states.shape[:2]
        centroid_logits = self.centroids(hidden_states)
        topk_idx = mx.argpartition(centroid_logits, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        ordering = self.token_ordering.reshape(
            self.num_centroids,
            self.vocab_size_per_centroid,
        )
        selected_canonical = ordering[topk_idx]
        flat_idx = selected_canonical.reshape(-1)

        if lm_head_weight.dtype == mx.uint32:
            selected_emb = mx.dequantize(
                embedding.weight[flat_idx],
                embedding.scales[flat_idx],
                embedding.biases[flat_idx],
                group_size=embedding.group_size,
                bits=embedding.bits,
                mode=embedding.mode,
                dtype=hidden_states.dtype,
            )
        else:
            selected_emb = lm_head_weight[flat_idx]

        selected_emb = selected_emb.reshape(
            batch,
            length,
            self.top_k * self.vocab_size_per_centroid,
            self.hidden_size,
        )
        selected_logits = mx.matmul(
            hidden_states[..., None, :],
            selected_emb.swapaxes(-1, -2),
        ).squeeze(-2)
        return selected_canonical, selected_logits

    masked._selected_logits = types.MethodType(selected_logits, masked)

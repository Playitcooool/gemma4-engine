from __future__ import annotations

from dataclasses import dataclass

from .backends import BackendName
from .inference import infer
from .loader import load_model


@dataclass
class CompareResult:
    baseline: str
    matches: bool
    engine_tokens: list[int]
    baseline_text: str
    engine_text: str


def compare_with_mlx_lm(
    *,
    prompt: str,
    model_path: str,
    max_tokens: int,
    backend: BackendName,
) -> CompareResult:
    from mlx_lm import generate

    engine = infer(
        prompt,
        model_path=model_path,
        max_tokens=max_tokens,
        backend=backend,
        prompt_mode="raw",
    )
    loaded = load_model(model_path)
    baseline_text = generate(
        loaded.model,
        loaded.tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return CompareResult(
        baseline="mlx_lm",
        matches=engine.text == baseline_text,
        engine_tokens=engine.token_ids,
        baseline_text=baseline_text,
        engine_text=engine.text,
    )

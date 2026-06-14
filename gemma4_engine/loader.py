from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import read_config, validate_gemma4_e4b_config


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    config: dict
    warnings: list[str]


def _generation_eos_token_ids(model_path: str) -> set[int] | None:
    generation_config_path = Path(model_path) / "generation_config.json"
    if not generation_config_path.exists():
        return None

    with generation_config_path.open("r", encoding="utf-8") as handle:
        generation_config = json.load(handle)

    eos = generation_config.get("eos_token_id")
    if eos is None:
        return None
    if isinstance(eos, int):
        return {eos}
    return {int(token_id) for token_id in eos}


def load_model(model_path: str, *, trust_remote_code: bool = True) -> LoadedModel:
    from mlx_lm import load
    from mlx_lm.utils import load_model as mlx_load_model
    from mlx_lm.utils import load_tokenizer

    config = read_config(model_path)
    warnings = validate_gemma4_e4b_config(config)
    tokenizer_config = {"trust_remote_code": trust_remote_code}

    try:
        model, tokenizer = load(model_path, tokenizer_config=tokenizer_config)
    except ValueError as exc:
        if (
            "parameters not in model" not in str(exc)
            or config.get("model_type") != "gemma4"
            or not config.get("text_config", {}).get("num_kv_shared_layers")
        ):
            raise

        model, _ = mlx_load_model(Path(model_path), strict=False)
        tokenizer = load_tokenizer(
            Path(model_path),
            tokenizer_config_extra=tokenizer_config,
            eos_token_ids=_generation_eos_token_ids(model_path),
        )
        warnings.append(
            "loaded non-strict because checkpoint contains extra per-layer K/V "
            "weights for shared-KV layers"
        )

    return LoadedModel(model=model, tokenizer=tokenizer, config=config, warnings=warnings)

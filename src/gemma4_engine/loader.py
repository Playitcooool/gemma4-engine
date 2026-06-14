from __future__ import annotations

from dataclasses import dataclass

from .config import read_config, validate_gemma4_e4b_config


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    config: dict
    warnings: list[str]


def load_model(model_path: str, *, trust_remote_code: bool = True) -> LoadedModel:
    from mlx_lm import load

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

        patched_text_config = dict(config["text_config"])
        patched_text_config["num_kv_shared_layers"] = 0
        model, tokenizer = load(
            model_path,
            tokenizer_config=tokenizer_config,
            model_config={"text_config": patched_text_config},
        )
        warnings.append(
            "loaded with text_config.num_kv_shared_layers=0 because checkpoint "
            "contains per-layer K/V weights for shared-KV layers"
        )

    return LoadedModel(model=model, tokenizer=tokenizer, config=config, warnings=warnings)

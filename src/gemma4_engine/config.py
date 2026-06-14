from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import EXPECTED_CONFIG


def read_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("text_config") or config.get("text_model") or {})


def validate_gemma4_e4b_config(config: dict[str, Any]) -> list[str]:
    text = text_config(config)
    checks = {
        "model_type": config.get("model_type"),
        "text_model_type": text.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "sliding_window": text.get("sliding_window"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "vocab_size": config.get("vocab_size") or text.get("vocab_size"),
    }

    mismatches: list[str] = []
    for key, expected in EXPECTED_CONFIG.items():
        if checks.get(key) != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {checks.get(key)!r}")
    return mismatches

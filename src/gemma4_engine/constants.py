from __future__ import annotations

DEFAULT_MODEL_PATH = (
    "/Volumes/Samsung/lmstudio/lmstudio-community/unsloth:gemma-4-E4B-it-UD-MLX-4bit"
)

EXPECTED_CONFIG = {
    "model_type": "gemma4",
    "text_model_type": "gemma4_text",
    "num_hidden_layers": 42,
    "hidden_size": 2560,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "sliding_window": 512,
    "max_position_embeddings": 131072,
    "vocab_size": 262144,
}

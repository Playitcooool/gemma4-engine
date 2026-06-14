from gemma4_engine.config import validate_gemma4_e4b_config


def test_validate_expected_gemma4_config() -> None:
    config = {
        "model_type": "gemma4",
        "vocab_size": 262144,
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 42,
            "hidden_size": 2560,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "sliding_window": 512,
            "max_position_embeddings": 131072,
        },
    }

    assert validate_gemma4_e4b_config(config) == []


def test_validate_reports_mismatches() -> None:
    config = {"model_type": "gemma3", "text_config": {"hidden_size": 1024}}

    mismatches = validate_gemma4_e4b_config(config)

    assert "model_type: expected 'gemma4', got 'gemma3'" in mismatches
    assert "hidden_size: expected 2560, got 1024" in mismatches

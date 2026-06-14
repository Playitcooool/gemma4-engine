from types import SimpleNamespace
import threading

import pytest

from gemma4_engine.server import EngineService, ServerConfig
from gemma4_engine.stats import RunStats


class FakeService(EngineService):
    def __init__(self) -> None:
        self.config = ServerConfig(model_path="fake-model")
        self.engine = SimpleNamespace(
            backend_status=SimpleNamespace(
                selected="mlx",
                reason="test",
            ),
            loaded=SimpleNamespace(warnings=[]),
            infer=self._infer,
        )
        self._lock = threading.Lock()
        self._seen = None

    def _infer(self, prompt: str, **kwargs):
        self._seen = {"prompt": prompt, **kwargs}
        return SimpleNamespace(
            text="ok",
            token_ids=[1, 2],
            stats=RunStats(
                model_path="fake-model",
                backend="mlx",
                prompt_tokens=3,
                generated_tokens=2,
                prefill_seconds=0.5,
                decode_seconds=0.25,
                time_to_first_token_seconds=0.6,
            ),
            backend_reason="test",
            config_warnings=[],
            prefix_cache_hit=False,
            prefix_tokens=0,
            draft_model_path=None,
            speculative_acceptance_rate=None,
        )


def test_generate_validates_prompt() -> None:
    service = FakeService()

    with pytest.raises(ValueError, match="prompt"):
        service.generate({})


def test_generate_returns_text_stats_and_uses_overrides() -> None:
    service = FakeService()

    response = service.generate(
        {
            "prompt": "hello",
            "max_tokens": 4,
            "prompt_mode": "raw",
            "prefill_step_size": "1024",
        }
    )

    assert response["text"] == "ok"
    assert response["stats"]["decode_tokens_per_second"] == 8.0
    assert response["prefix_cache_hit"] is False
    assert response["prefix_tokens"] == 0
    assert service._seen["prompt"] == "hello"
    assert service._seen["max_tokens"] == 4
    assert service._seen["prompt_mode"] == "raw"
    assert service._seen["prefill_step_size"] == "1024"


def test_health_reports_loaded_backend() -> None:
    service = FakeService()

    assert service.health()["backend_selected"] == "mlx"

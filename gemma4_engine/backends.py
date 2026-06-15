from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

BackendName = Literal["mlx", "auto"]


class ArgmaxBackend(Protocol):
    name: str

    def argmax(self, logits: object) -> int:
        ...


@dataclass(frozen=True)
class BackendStatus:
    requested: BackendName
    selected: str
    reason: str


class MlxBackend:
    name = "mlx"

    def argmax(self, logits: object) -> int:
        import mlx.core as mx

        return int(mx.argmax(logits, axis=-1).item())


def select_backend(requested: BackendName) -> tuple[ArgmaxBackend, BackendStatus]:
    if requested == "mlx":
        return MlxBackend(), BackendStatus(requested, "mlx", "MLX requested")

    return MlxBackend(), BackendStatus(requested, "mlx", "auto selects MLX")

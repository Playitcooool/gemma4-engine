from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

BackendName = Literal["mlx", "rust-metal", "auto"]


class ArgmaxBackend(Protocol):
    name: str

    def argmax(self, logits: object) -> int:
        ...


@dataclass(frozen=True)
class BackendStatus:
    requested: BackendName
    selected: str
    rust_available: bool
    rust_correct: bool
    reason: str


class MlxBackend:
    name = "mlx"

    def argmax(self, logits: object) -> int:
        import mlx.core as mx

        return int(mx.argmax(logits, axis=-1).item())


class RustMetalBackend:
    name = "rust-metal"

    def __init__(self) -> None:
        import gemma4_kernels

        self._kernels = gemma4_kernels

    def argmax(self, logits: object) -> int:
        import mlx.core as mx

        return int(self._kernels.greedy_argmax(np.array(logits.astype(mx.float32))))


def _rust_self_test() -> tuple[bool, str]:
    try:
        import gemma4_kernels
    except Exception as exc:  # pragma: no cover - depends on optional extension
        return False, f"Rust extension unavailable: {exc}"

    values = np.array([-3.0, 10.0, 5.0, 10.5, 0.0], dtype=np.float32)
    try:
        got = int(gemma4_kernels.greedy_argmax(values))
    except Exception as exc:  # pragma: no cover - depends on optional extension
        return False, f"Rust argmax self-test failed: {exc}"

    expected = int(np.argmax(values))
    if got != expected:
        return False, f"Rust argmax mismatch: got {got}, expected {expected}"
    return True, "Rust argmax self-test passed"


def select_backend(requested: BackendName) -> tuple[ArgmaxBackend, BackendStatus]:
    if requested == "mlx":
        return MlxBackend(), BackendStatus(requested, "mlx", False, False, "MLX requested")

    rust_ok, reason = _rust_self_test()
    if requested == "rust-metal":
        if not rust_ok:
            return MlxBackend(), BackendStatus(requested, "mlx", False, False, reason)
        return RustMetalBackend(), BackendStatus(requested, "rust-metal", True, True, reason)

    if rust_ok:
        return MlxBackend(), BackendStatus(
            requested,
            "mlx",
            True,
            True,
            f"{reason}; using MLX until Rust median speed beats fallback",
        )
    return MlxBackend(), BackendStatus(requested, "mlx", False, False, reason)

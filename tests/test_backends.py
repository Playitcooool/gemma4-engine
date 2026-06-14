from gemma4_engine.backends import MlxBackend, select_backend


def test_mlx_backend_selected_explicitly() -> None:
    backend, status = select_backend("mlx")

    assert isinstance(backend, MlxBackend)
    assert status.selected == "mlx"


def test_rust_backend_falls_back_when_extension_missing() -> None:
    backend, status = select_backend("rust-metal")

    assert backend.name in {"mlx", "rust-metal"}
    if backend.name == "mlx":
        assert "Rust" in status.reason

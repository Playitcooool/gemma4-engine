from gemma4_engine.backends import MlxBackend, select_backend


def test_mlx_backend_selected_explicitly() -> None:
    backend, status = select_backend("mlx")

    assert isinstance(backend, MlxBackend)
    assert status.selected == "mlx"


def test_auto_backend_selects_mlx() -> None:
    backend, status = select_backend("auto")

    assert isinstance(backend, MlxBackend)
    assert status.selected == "mlx"
    assert status.reason == "auto selects MLX"

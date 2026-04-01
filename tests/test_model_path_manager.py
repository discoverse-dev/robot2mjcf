"""Tests for model path manager helpers."""

from robot2mjcf.model_path_manager import get_current_model_paths, is_description_package, set_model_paths


def test_is_description_package_requires_expected_structure(tmp_path) -> None:
    package = tmp_path / "robot_description"
    package.mkdir()
    (package / "package.xml").write_text("<package />")
    (package / "urdf").mkdir()

    assert is_description_package(package) is True
    assert is_description_package(tmp_path / "not_a_package") is False


def test_set_model_paths_round_trip_posix(tmp_path, monkeypatch) -> None:
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    path_a.mkdir()
    path_b.mkdir()

    monkeypatch.setattr("robot2mjcf.model_path_manager.platform.system", lambda: "Linux")

    export_cmd = set_model_paths([path_a, path_b])

    assert export_cmd == f'export URDF2MJCF_MODEL_PATH="{path_a}:{path_b}"'
    assert get_current_model_paths() == [path_a.resolve(), path_b.resolve()]


def test_set_model_paths_round_trip_windows(tmp_path, monkeypatch) -> None:
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    path_a.mkdir()
    path_b.mkdir()

    monkeypatch.setattr("robot2mjcf.model_path_manager.platform.system", lambda: "Windows")

    export_cmd = set_model_paths([path_a, path_b])

    assert export_cmd == f'set URDF2MJCF_MODEL_PATH="{path_a};{path_b}"'
    assert get_current_model_paths() == [path_a.resolve(), path_b.resolve()]

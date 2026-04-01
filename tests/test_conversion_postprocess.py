"""Tests for conversion post-processing helpers."""

from robot2mjcf.conversion_postprocess import maybe_capture_robot_images
from robot2mjcf.model_path_manager import find_description_packages, scan_and_add


def test_maybe_capture_robot_images_is_disabled_by_default(tmp_path, monkeypatch) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("capture should not run")

    monkeypatch.setattr("robot2mjcf.postprocess.capture.capture_robot_images", fail_if_called)

    maybe_capture_robot_images(tmp_path / "robot.xml", capture_images=False)

    assert called is False


def test_find_description_packages_respects_max_depth(tmp_path) -> None:
    shallow = tmp_path / "ws" / "robot_description"
    shallow.mkdir(parents=True)
    (shallow / "package.xml").write_text("<package />")
    (shallow / "urdf").mkdir()

    deep = tmp_path / "ws" / "nested" / "deeper" / "deep_description"
    deep.mkdir(parents=True)
    (deep / "package.xml").write_text("<package />")
    (deep / "meshes").mkdir()

    shallow_found = find_description_packages(tmp_path / "ws", max_depth=2)
    deep_found = find_description_packages(tmp_path / "ws", max_depth=4)

    assert shallow.resolve() in shallow_found
    assert deep.resolve() not in shallow_found
    assert deep.resolve() in deep_found


def test_scan_and_add_respects_max_depth(tmp_path, monkeypatch, capsys) -> None:
    deep = tmp_path / "root" / "a" / "b" / "deep_description"
    deep.mkdir(parents=True)
    (deep / "package.xml").write_text("<package />")
    (deep / "urdf").mkdir()

    monkeypatch.delenv("URDF2MJCF_MODEL_PATH", raising=False)

    scan_and_add([tmp_path / "root"], append=False, quiet=True, max_depth=2)
    assert str(deep.resolve()) not in capsys.readouterr().out

    scan_and_add([tmp_path / "root"], append=False, quiet=True, max_depth=3)
    assert str(deep.resolve()) in capsys.readouterr().out

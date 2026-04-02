"""Tests for conversion post-processing helpers."""

from robot2mjcf.postprocess import PostprocessOptions, apply_postprocess_pipeline, maybe_capture_robot_images
from robot2mjcf.core.model import ConversionMetadata
from robot2mjcf.cli.model_path import find_description_packages, scan_and_add


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


def test_apply_postprocess_pipeline_can_skip_heavy_mesh_steps(tmp_path, monkeypatch) -> None:
    mjcf_path = tmp_path / "robot.xml"
    mjcf_path.write_text("<mujoco><worldbody><body name='base' /></worldbody></mujoco>")

    called: list[str] = []

    def record(name: str):
        def _inner(*args, **kwargs):
            called.append(name)

        return _inner

    for name in [
        "add_light",
        "add_floor",
        "remove_redundancies",
        "fix_base_joint",
        "maybe_capture_robot_images",
    ]:
        monkeypatch.setattr(f"robot2mjcf.postprocess.{name}", record(name))

    for name in [
        "convex_decomposition",
        "convex_collision",
        "collision_to_stl",
        "split_obj_by_materials",
        "update_mesh",
        "move_mesh_scale",
        "check_shell_meshes",
        "deduplicate_meshes",
    ]:
        monkeypatch.setattr(f"robot2mjcf.postprocess.{name}", record(name))

    apply_postprocess_pipeline(
        mjcf_path,
        options=PostprocessOptions(
            metadata=ConversionMetadata(cameras=[]),
            collision_only=False,
            collision_type="decomposition",
            max_vertices=123,
            appendix_files=None,
            capture_images=False,
            run_mesh_postprocess=False,
        ),
    )

    assert "add_light" in called
    assert "fix_base_joint" in called
    assert "add_floor" in called
    assert "remove_redundancies" in called
    assert "collision_to_stl" not in called
    assert "split_obj_by_materials" not in called
    assert "update_mesh" not in called
    assert "convex_decomposition" not in called

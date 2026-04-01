"""Basic import and CLI smoke tests."""

import importlib
import subprocess
import sys

import robot2mjcf
from robot2mjcf.urdf_format import format_urdf_file


def test_version() -> None:
    assert robot2mjcf.__version__
    assert isinstance(robot2mjcf.__version__, str)


def test_run_exported() -> None:
    assert callable(robot2mjcf.run)


def test_import_urdf_format_module() -> None:
    module = importlib.import_module("robot2mjcf.urdf_format")
    assert hasattr(module, "format_urdf_file")


def test_format_urdf_file_in_place(tmp_path) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='demo'><link name='base'/></robot>")

    format_urdf_file(urdf_path)

    assert urdf_path.exists()
    assert "<robot" in urdf_path.read_text()
    assert not (tmp_path / "robot_tmp.urdf").exists()


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "robot2mjcf.convert", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Convert a URDF file to an MJCF file" in result.stdout
    assert "--capture-images" in result.stdout

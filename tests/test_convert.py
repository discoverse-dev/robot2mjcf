"""End-to-end conversion tests using example robots."""

import tempfile
from pathlib import Path
import pytest

from robot2mjcf.convert import convert_urdf_to_mjcf

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def test_convert_piper_basic(tmp_dir: Path) -> None:
    robot_dir = EXAMPLES_DIR / "agilex-piper"
    urdf_path = robot_dir / "piper.urdf"
    metadata_path = robot_dir / "metadata" / "metadata.json"
    appendix_path = robot_dir / "metadata" / "appendix.xml"

    out_path = tmp_dir / "piper.xml"

    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=out_path,
        metadata_file=metadata_path,
        appendix_files=[appendix_path] if appendix_path.exists() else None,
        max_vertices=200000,
    )

    assert out_path.exists()
    content = out_path.read_text()
    assert "<mujoco" in content
    assert "</mujoco>" in content
    assert out_path.stat().st_size > 0


def test_convert_rm65_with_metadata(tmp_dir: Path) -> None:
    robot_dir = EXAMPLES_DIR / "realman-rm65"
    urdf_path = robot_dir / "rm65b_eg24c2_description.urdf"
    metadata_path = robot_dir / "metadata" / "metadata.json"

    out_path = tmp_dir / "rm65.xml"

    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=out_path,
        metadata_file=metadata_path,
        max_vertices=200000,
    )

    assert out_path.exists()
    content = out_path.read_text()
    assert "<mujoco" in content
    assert "</mujoco>" in content


def test_convert_missing_urdf(tmp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        convert_urdf_to_mjcf(urdf_path=tmp_dir / "nonexistent.urdf")


def test_default_output_path() -> None:
    """When no output path is given, output goes to output_mjcf/robot.xml."""
    robot_dir = EXAMPLES_DIR / "agilex-piper"
    urdf_path = robot_dir / "piper.urdf"
    metadata_path = robot_dir / "metadata" / "metadata.json"

    expected_output = robot_dir / "output_mjcf" / "robot.xml"

    try:
        convert_urdf_to_mjcf(
            urdf_path=urdf_path,
            mjcf_path=None,
            metadata_file=metadata_path,
            max_vertices=200000,
        )

        assert expected_output.exists()
        content = expected_output.read_text()
        assert "<mujoco" in content
    finally:
        # Clean up generated output_mjcf directory
        import shutil

        output_dir = robot_dir / "output_mjcf"
        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_reject_same_directory_as_urdf(capsys: pytest.CaptureFixture) -> None:
    """When output path is in the same directory as the URDF, fall back to default."""
    robot_dir = EXAMPLES_DIR / "agilex-piper"
    urdf_path = robot_dir / "piper.urdf"
    metadata_path = robot_dir / "metadata" / "metadata.json"

    # Request output in the same directory as the URDF
    same_dir_output = robot_dir / "output.xml"
    expected_output = robot_dir / "output_mjcf" / "robot.xml"

    try:
        convert_urdf_to_mjcf(
            urdf_path=urdf_path,
            mjcf_path=same_dir_output,
            metadata_file=metadata_path,
            max_vertices=200000,
        )

        # Should NOT have created file in the URDF directory
        assert not same_dir_output.exists()
        # Should have used the default path
        assert expected_output.exists()

        # Should have printed a yellow warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "same directory" in captured.out
    finally:
        import shutil

        output_dir = robot_dir / "output_mjcf"
        if output_dir.exists():
            shutil.rmtree(output_dir)

"""End-to-end conversion tests using example robots."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import pytest

from robot2mjcf.convert import convert_urdf_to_mjcf

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

EXPECTED_SIGNATURES: dict[str, dict[str, Any]] = {
    "agilex-piper": {
        "model": "piper_description",
        "body_names": [
            "base_link",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "link8",
        ],
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "joint8",
        ],
        "actuator_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "joint8",
        ],
        "equality_pairs": [("joint7", "joint8")],
        "xml_counts": {
            "body": 9,
            "geom": 38,
            "mesh": 34,
            "material": 9,
            "actuator": 8,
            "equality": 1,
        },
        "model_counts": {
            "nbody": 10,
            "njnt": 8,
            "ngeom": 35,
            "nmesh": 34,
        },
    },
    "realman-rm65": {
        "model": "RM65B_EG24C2_description",
        "body_names": [
            "base_link",
            "camera_base_link",
            "camera_link",
            "gripper_base_link",
            "link1",
            "link10",
            "link11",
            "link12",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "link8",
            "link9",
        ],
        "joint_names": [
            "gripper_joint1",
            "gripper_joint2",
            "gripper_joint3",
            "gripper_joint4",
            "gripper_joint5",
            "gripper_joint6",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
        "actuator_names": [
            "gripper_joint1",
            "gripper_joint2",
            "gripper_joint3",
            "gripper_joint4",
            "gripper_joint5",
            "gripper_joint6",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ],
        "equality_pairs": [
            ("gripper_joint1", "gripper_joint2"),
            ("gripper_joint1", "gripper_joint3"),
            ("gripper_joint1", "gripper_joint4"),
            ("gripper_joint1", "gripper_joint5"),
            ("gripper_joint1", "gripper_joint6"),
        ],
        "xml_counts": {
            "body": 16,
            "geom": 44,
            "mesh": 40,
            "material": 7,
            "actuator": 12,
            "equality": 5,
        },
        "model_counts": {
            "nbody": 17,
            "njnt": 12,
            "ngeom": 41,
            "nmesh": 40,
        },
    },
}


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def assert_example_signature(out_path: Path, example_name: str) -> None:
    expected = EXPECTED_SIGNATURES[example_name]
    root = ET.parse(out_path).getroot()

    actuator = root.find("actuator")
    equality = root.find("equality")

    assert root.attrib.get("model") == expected["model"]
    assert sorted(elem.attrib["name"] for elem in root.iter("body") if "name" in elem.attrib) == expected["body_names"]
    assert sorted(elem.attrib["name"] for elem in root.iter("joint") if "name" in elem.attrib) == expected["joint_names"]
    assert sorted(
        elem.attrib["name"] for elem in (list(actuator) if actuator is not None else []) if "name" in elem.attrib
    ) == expected["actuator_names"]
    assert sorted(
        (elem.attrib.get("joint1"), elem.attrib.get("joint2"))
        for elem in (list(equality) if equality is not None else [])
        if elem.tag == "joint"
    ) == expected["equality_pairs"]

    assert len(list(root.iter("body"))) == expected["xml_counts"]["body"]
    assert len(list(root.iter("geom"))) == expected["xml_counts"]["geom"]
    assert len(list(root.iter("mesh"))) == expected["xml_counts"]["mesh"]
    assert len(list(root.iter("material"))) == expected["xml_counts"]["material"]
    assert len(list(actuator) if actuator is not None else []) == expected["xml_counts"]["actuator"]
    assert len(list(equality) if equality is not None else []) == expected["xml_counts"]["equality"]

    compiler = root.find("compiler")
    assert compiler is not None
    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_root = out_path.parent / meshdir
    for mesh in root.iter("mesh"):
        mesh_file = mesh.attrib.get("file")
        assert mesh_file is not None
        assert (mesh_root / mesh_file).exists(), f"Missing mesh asset: {mesh_file}"

    model = mujoco.MjModel.from_xml_path(str(out_path))
    assert model.nbody == expected["model_counts"]["nbody"]
    assert model.njnt == expected["model_counts"]["njnt"]
    assert model.ngeom == expected["model_counts"]["ngeom"]
    assert model.nmesh == expected["model_counts"]["nmesh"]


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
    assert_example_signature(out_path, "agilex-piper")


def test_convert_rm65_with_metadata(tmp_dir: Path) -> None:
    robot_dir = EXAMPLES_DIR / "realman-rm65"
    urdf_path = robot_dir / "rm65b_eg24c2_description.urdf"
    metadata_path = robot_dir / "metadata" / "metadata.json"
    appendix_path = robot_dir / "metadata" / "appendix.xml"

    out_path = tmp_dir / "rm65.xml"

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
    assert_example_signature(out_path, "realman-rm65")


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

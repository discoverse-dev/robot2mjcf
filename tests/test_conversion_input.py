"""Tests for conversion input loading helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from robot2mjcf.conversion.input import load_conversion_inputs


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_load_conversion_inputs_loads_robot_metadata_and_materials(tmp_path) -> None:
    urdf_path = write_text(
        tmp_path / "robot.urdf",
        """
        <robot name="demo">
          <material name="blue"><color rgba="0 0 1 1" /></material>
          <link name="base">
            <visual>
              <geometry><box size="1 1 1" /></geometry>
              <material name="blue" />
            </visual>
          </link>
        </robot>
        """.strip(),
    )
    metadata_path = write_text(tmp_path / "metadata.json", "{}")

    inputs = load_conversion_inputs(urdf_path, tmp_path / "out" / "robot.xml", metadata_path, collision_only=False)

    assert inputs.urdf_path == urdf_path
    assert inputs.urdf_dir == tmp_path
    assert inputs.mjcf_path == tmp_path / "out" / "robot.xml"
    assert inputs.robot.attrib["name"] == "demo"
    assert inputs.metadata.height_offset == 0.0
    assert inputs.materials["blue"] == "0 0 1 1"


def test_load_conversion_inputs_returns_output_warning_for_same_directory(tmp_path) -> None:
    urdf_path = write_text(tmp_path / "robot.urdf", "<robot name='demo'><link name='base' /></robot>")

    inputs = load_conversion_inputs(urdf_path, tmp_path / "same_dir.xml", None, collision_only=False)

    assert inputs.output_warning is not None
    assert inputs.mjcf_path == tmp_path / "output_mjcf" / "robot.xml"


def test_load_conversion_inputs_rejects_missing_or_invalid_urdf(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_conversion_inputs(tmp_path / "missing.urdf", None, None, collision_only=False)

    invalid_path = write_text(tmp_path / "invalid.urdf", "")
    with pytest.raises(ET.ParseError):
        load_conversion_inputs(invalid_path, None, None, collision_only=False)

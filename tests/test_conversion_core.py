"""Tests for shared conversion context helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from robot2mjcf.conversion.pipeline import (
    build_conversion_context,
    create_empty_actuator_metadata,
    resolve_root_link_name,
)
from robot2mjcf.core.model import ActuatorMetadata, ConversionMetadata, DefaultJointMetadata, dActuator, dJoint


def test_create_empty_actuator_metadata_creates_motor_entries() -> None:
    robot = ET.fromstring(
        """
        <robot>
          <joint name="joint1" />
          <joint name="joint2" />
          <joint />
        </robot>
        """
    )

    metadata = create_empty_actuator_metadata(robot)

    assert list(metadata) == ["joint1", "joint2"]
    assert metadata["joint1"].actuator_type == "motor"


def test_resolve_root_link_name_returns_only_root() -> None:
    link_map = {"base": ET.Element("link"), "arm": ET.Element("link")}
    child_joints = {"arm": ET.Element("joint")}

    assert resolve_root_link_name(link_map, child_joints) == "base"


def test_resolve_root_link_name_raises_when_missing_root() -> None:
    with pytest.raises(ValueError, match="No root link found"):
        resolve_root_link_name({"base": ET.Element("link")}, {"base": ET.Element("joint")})


def test_build_conversion_context_creates_base_tree_and_resolves_metadata() -> None:
    robot = ET.fromstring(
        """
        <robot name="demo">
          <link name="base" />
          <link name="arm" />
          <joint name="joint1" type="revolute">
            <parent link="base" />
            <child link="arm" />
            <mimic joint="joint0" multiplier="2" offset="0.5" />
          </joint>
        </robot>
        """
    )
    default_metadata = {
        "arm": DefaultJointMetadata(
            joint=dJoint(damping=1.0),
            actuator=dActuator(actuator_type="motor"),
        )
    }

    context = build_conversion_context(
        robot,
        metadata=ConversionMetadata(),
        default_metadata=default_metadata,
        actuator_metadata=None,
        collision_only=False,
    )

    assert context.mjcf_root.attrib["model"] == "demo"
    assert context.worldbody.tag == "worldbody"
    assert context.root_link_name == "base"
    assert context.actuator_metadata["joint1"].actuator_type == "motor"
    assert context.mimic_constraints == [("joint0", "joint1", 2.0, 0.5)]
    assert context.mjcf_root.find("compiler") is not None
    assert context.mjcf_root.find("visual") is not None
    assert context.mjcf_root.find("default") is not None


def test_build_conversion_context_respects_provided_actuator_metadata() -> None:
    robot = ET.fromstring(
        """
        <robot>
          <link name="base" />
          <link name="arm" />
          <joint name="joint1">
            <parent link="base" />
            <child link="arm" />
          </joint>
        </robot>
        """
    )
    actuator_metadata = {"joint1": ActuatorMetadata(actuator_type="position", kp=10.0)}

    context = build_conversion_context(
        robot,
        metadata=ConversionMetadata(),
        default_metadata=None,
        actuator_metadata=actuator_metadata,
        collision_only=True,
    )

    assert context.actuator_metadata is actuator_metadata
    assert context.mjcf_root.find(".//default[@class='visual']") is None

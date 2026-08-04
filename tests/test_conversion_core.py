"""Tests for shared conversion context helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from urdf_to_mjcf.conversion.pipeline import (
    build_conversion_context,
    resolve_joint_data,
    resolve_root_link_name,
)
from urdf_to_mjcf.core.model import (
    ActuatorConfig,
    ConversionMetadata,
    ExtraJoint,
    ExtraJointGroup,
    JointData,
    JointMetadata,
)


def test_resolve_joint_data_creates_default_motors_for_movable_urdf_joints() -> None:
    robot = ET.fromstring(
        """
        <robot>
          <joint name="joint1" type="revolute" />
          <joint name="joint2" type="prismatic" />
          <joint name="fixed" type="fixed" />
          <joint type="continuous" />
        </robot>
        """
    )

    joint_data = resolve_joint_data(robot, None)

    assert list(joint_data.joints) == ["joint1", "joint2"]
    assert joint_data.joints["joint1"].actuator is not None
    assert joint_data.joints["joint1"].actuator.actuator_type == "motor"


def test_resolve_joint_data_preserves_explicit_empty_configuration() -> None:
    robot = ET.fromstring('<robot><joint name="joint1" type="revolute" /></robot>')
    joint_data = JointData()

    assert resolve_joint_data(robot, joint_data) is joint_data


def test_resolve_joint_data_rejects_unknown_and_duplicate_joints() -> None:
    robot = ET.fromstring(
        """
        <robot>
          <link name="base" />
          <link name="arm" />
          <joint name="arm_joint" type="revolute" />
        </robot>
        """
    )

    with pytest.raises(ValueError, match="unknown or fixed joints"):
        resolve_joint_data(robot, JointData(joints={"typo": JointMetadata()}))

    duplicate = ExtraJoint(name="base_x", type="slide", axis="x")
    with pytest.raises(ValueError, match="Duplicate MJCF-only joint names"):
        resolve_joint_data(
            robot,
            JointData(
                extra_joints=[
                    ExtraJointGroup(body="base", joints=[duplicate]),
                    ExtraJointGroup(body="base", joints=[duplicate]),
                ]
            ),
        )


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
    context = build_conversion_context(
        robot,
        metadata=ConversionMetadata(),
        collision_only=False,
    )

    assert context.mjcf_root.attrib["model"] == "demo"
    assert context.worldbody.tag == "worldbody"
    assert context.root_link_name == "base"
    actuator = context.joint_data.joints["joint1"].actuator
    assert actuator is not None
    assert actuator.actuator_type == "motor"
    assert context.mimic_constraints == [("joint0", "joint1", 2.0, 0.5)]
    assert context.mjcf_root.find("compiler") is not None
    assert context.mjcf_root.find("visual") is not None
    assert context.mjcf_root.find("default") is not None


def test_build_conversion_context_preserves_explicit_joint_data() -> None:
    robot = ET.fromstring(
        """
        <robot name="demo">
          <link name="base" />
          <link name="arm" />
          <joint name="joint1" type="revolute">
            <parent link="base" />
            <child link="arm" />
          </joint>
          <joint name="joint2" type="revolute">
            <parent link="arm" />
            <child link="finger" />
          </joint>
          <link name="finger" />
        </robot>
        """
    )
    joint_metadata = {
        "joint1": JointMetadata(
            damping=0.5,
            actuator=ActuatorConfig(
                actuator_type="position",
                ctrllimited=True,
                kp=10.0,
                forcelimited=True,
            ),
        ),
        "joint2": JointMetadata(actuator=ActuatorConfig()),
    }
    joint_data = JointData(joints=joint_metadata)

    context = build_conversion_context(
        robot,
        metadata=ConversionMetadata(),
        collision_only=False,
        joint_data=joint_data,
    )

    assert context.joint_data is joint_data
    actuator = context.joint_data.joints["joint1"].actuator
    assert actuator is not None
    assert actuator.actuator_type == "position"
    assert actuator.ctrllimited is True
    assert actuator.kp == 10.0
    assert actuator.forcelimited is True


def test_build_conversion_context_skips_visual_defaults_for_collision_only() -> None:
    robot = ET.fromstring(
        """
        <robot>
          <link name="base" />
          <link name="arm" />
          <joint name="joint1" type="revolute">
            <parent link="base" />
            <child link="arm" />
          </joint>
        </robot>
        """
    )
    context = build_conversion_context(
        robot,
        metadata=ConversionMetadata(),
        collision_only=True,
    )

    assert context.mjcf_root.find(".//default[@class='visual']") is None

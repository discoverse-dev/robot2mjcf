"""Tests for MJCF assembly helpers extracted from convert.py."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from robot2mjcf.conversion_mjcf_assembly import add_actuators, add_mimic_equality_constraints
from robot2mjcf.geometry import ParsedJointParams
from robot2mjcf.model import ActuatorMetadata


def test_add_actuators_uses_metadata_and_preserves_metadata_order() -> None:
    root = ET.fromstring("<mujoco />")
    actuator_joints = [
        ParsedJointParams(name="joint_b", type="hinge"),
        ParsedJointParams(name="joint_missing", type="hinge"),
        ParsedJointParams(name="joint_a", type="hinge"),
    ]
    actuator_metadata = {
        "joint_a": ActuatorMetadata(
            actuator_type="position",
            joint_class="arm",
            kp=50.0,
            kv=5.0,
            ctrlrange=[-1.0, 1.0],
            forcerange=[-2.0, 2.0],
            gear=3.0,
        ),
        "joint_b": ActuatorMetadata(actuator_type="motor"),
    }

    add_actuators(root, actuator_joints, actuator_metadata)

    actuator = root.find("actuator")
    children = list(actuator) if actuator is not None else []

    assert [child.attrib["joint"] for child in children] == ["joint_a", "joint_b"]
    assert children[0].tag == "position"
    assert children[0].attrib["class"] == "arm"
    assert children[0].attrib["kp"] == "50.0"
    assert children[0].attrib["kv"] == "5.0"
    assert children[0].attrib["ctrlrange"] == "-1.0 1.0"
    assert children[0].attrib["forcerange"] == "-2.0 2.0"
    assert children[0].attrib["gear"] == "3.0"
    assert children[1].tag == "motor"


def test_add_mimic_equality_constraints_adds_polycoef_constraints() -> None:
    root = ET.fromstring("<mujoco />")

    add_mimic_equality_constraints(
        root,
        [
            ("joint1", "joint2", 2.0, 0.5),
            ("joint3", "joint4", -1.0, 0.0),
        ],
    )

    equality = root.find("equality")
    joints = list(equality) if equality is not None else []

    assert len(joints) == 2
    assert joints[0].attrib == {
        "joint1": "joint1",
        "joint2": "joint2",
        "polycoef": "0.5 2.0 0 0 0",
        "solimp": "0.95 0.99 0.001",
        "solref": "0.005 1",
    }
    assert joints[1].attrib["polycoef"] == "0.0 -1.0 0 0 0"


def test_add_mimic_equality_constraints_skips_empty_input() -> None:
    root = ET.fromstring("<mujoco />")

    add_mimic_equality_constraints(root, [])

    assert root.find("equality") is None

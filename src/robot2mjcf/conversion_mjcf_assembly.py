"""Helpers for assembling MJCF actuator and equality blocks."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

from robot2mjcf.geometry import ParsedJointParams
from robot2mjcf.model import ActuatorMetadata

logger = logging.getLogger(__name__)

MimicConstraint = tuple[str, str, float, float]


def add_actuators(
    root: ET.Element,
    actuator_joints: Sequence[ParsedJointParams],
    actuator_metadata: Mapping[str, ActuatorMetadata],
) -> None:
    """Add ordered actuator elements to the MJCF root."""
    actuator_elem = ET.SubElement(root, "actuator")
    actuator_order = list(actuator_metadata)

    for actuator_joint in actuator_joints:
        metadata = actuator_metadata.get(actuator_joint.name)
        if metadata is None:
            logger.info("Actuator %s not found in actuator_metadata", actuator_joint.name)
            continue

        attrib: dict[str, str] = {"joint": actuator_joint.name}
        actuator_type = metadata.actuator_type or "motor"
        logger.info("Joint %s assigned to class: %s", actuator_joint.name, actuator_type)

        if metadata.joint_class is not None:
            attrib["class"] = str(metadata.joint_class)
            logger.info("Joint %s assigned to class: %s", actuator_joint.name, metadata.joint_class)
        if metadata.kp is not None:
            attrib["kp"] = str(metadata.kp)
        if metadata.kv is not None:
            attrib["kv"] = str(metadata.kv)
        if metadata.ctrlrange is not None:
            attrib["ctrlrange"] = f"{metadata.ctrlrange[0]} {metadata.ctrlrange[1]}"
        if metadata.forcerange is not None:
            attrib["forcerange"] = f"{metadata.forcerange[0]} {metadata.forcerange[1]}"
        if metadata.gear is not None:
            attrib["gear"] = str(metadata.gear)

        logger.info("Creating actuator %s with class: %s", actuator_joint.name, actuator_type)
        ET.SubElement(actuator_elem, actuator_type, attrib={"name": actuator_joint.name, **attrib})

    actuator_children = [child for child in list(actuator_elem) if child.attrib["joint"] in actuator_metadata]
    actuator_children.sort(key=lambda elem: actuator_order.index(elem.attrib["joint"]))

    for child in actuator_children:
        actuator_elem.remove(child)
    for child in actuator_children:
        actuator_elem.append(child)


def add_mimic_equality_constraints(root: ET.Element, mimic_constraints: Sequence[MimicConstraint]) -> None:
    """Add equality constraints for mimic joints."""
    if not mimic_constraints:
        return

    equality_elem = ET.SubElement(root, "equality")
    for mimicked_joint, mimicking_joint, multiplier, offset in mimic_constraints:
        ET.SubElement(
            equality_elem,
            "joint",
            attrib={
                "joint1": mimicked_joint,
                "joint2": mimicking_joint,
                "polycoef": f"{offset} {multiplier} 0 0 0",
                "solimp": "0.95 0.99 0.001",
                "solref": "0.005 1",
            },
        )
        logger.info("Added equality constraint: %s = %s + %s * %s", mimicking_joint, offset, multiplier, mimicked_joint)

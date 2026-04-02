"""Core helpers for preparing conversion context before body/asset assembly."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass

from robot2mjcf.conversion_helpers import build_joint_maps, collect_mimic_constraints
from robot2mjcf.mjcf_builders import add_compiler, add_default, add_visual
from robot2mjcf.model import ActuatorMetadata, ConversionMetadata, DefaultJointMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversionContext:
    """Prepared conversion state shared by the main conversion pipeline."""

    mjcf_root: ET.Element
    worldbody: ET.Element
    link_map: dict[str, ET.Element]
    parent_map: dict[str, list[tuple[str, ET.Element]]]
    root_link_name: str
    actuator_metadata: dict[str, ActuatorMetadata]
    mimic_constraints: list[tuple[str, str, float, float]]


def create_empty_actuator_metadata(robot_elem: ET.Element) -> dict[str, ActuatorMetadata]:
    """Create placeholder metadata when actuator metadata is omitted."""
    actuator_meta: dict[str, ActuatorMetadata] = {}
    for joint in robot_elem.findall("joint"):
        name = joint.attrib.get("name")
        if name:
            actuator_meta[name] = ActuatorMetadata(actuator_type="motor")
    return actuator_meta


def resolve_root_link_name(link_map: Mapping[str, ET.Element], child_joints: Mapping[str, ET.Element]) -> str:
    """Resolve the single URDF root link name from the joint graph."""
    root_links = list(set(link_map) - set(child_joints))
    if not root_links:
        raise ValueError("No root link found in URDF.")
    return root_links[0]


def build_conversion_context(
    robot: ET.Element,
    *,
    metadata: ConversionMetadata,
    default_metadata: Mapping[str, DefaultJointMetadata] | None,
    actuator_metadata: dict[str, ActuatorMetadata] | None,
    collision_only: bool,
) -> ConversionContext:
    """Build the shared conversion context used by convert_urdf_to_mjcf."""
    resolved_actuator_metadata = actuator_metadata
    if resolved_actuator_metadata is None:
        logger.warning("Missing joint metadata, falling back to single empty 'motor' class.")
        resolved_actuator_metadata = create_empty_actuator_metadata(robot)

    mjcf_root = ET.Element("mujoco", attrib={"model": robot.attrib.get("name", "converted_robot")})
    add_compiler(mjcf_root)
    add_visual(mjcf_root)
    add_default(mjcf_root, metadata, default_metadata, collision_only)
    worldbody = ET.SubElement(mjcf_root, "worldbody")

    link_map, parent_map, child_joints = build_joint_maps(robot)
    root_link_name = resolve_root_link_name(link_map, child_joints)

    mimic_constraints = collect_mimic_constraints(robot)
    for mimicked_joint, joint_name, multiplier, offset in mimic_constraints:
        logger.info(
            "Found mimic constraint: %s mimics %s with multiplier=%s, offset=%s",
            joint_name,
            mimicked_joint,
            multiplier,
            offset,
        )

    return ConversionContext(
        mjcf_root=mjcf_root,
        worldbody=worldbody,
        link_map=link_map,
        parent_map=parent_map,
        root_link_name=root_link_name,
        actuator_metadata=resolved_actuator_metadata,
        mimic_constraints=mimic_constraints,
    )

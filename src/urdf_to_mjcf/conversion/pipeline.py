"""Core conversion context and scene assembly pipeline.

Merged from: conversion_core.py + conversion_scene.py
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from urdf_to_mjcf.conversion.assets import (
    add_mesh_assets_to_xml,
    collect_single_obj_materials,
    copy_mesh_assets,
    resolve_workspace_search_paths,
)
from urdf_to_mjcf.conversion.body_builder import apply_joint_metadata, build_robot_body_tree
from urdf_to_mjcf.conversion.input import build_joint_maps, collect_mimic_constraints
from urdf_to_mjcf.conversion.mjcf_assembly import (
    ROBOT_CLASS,
    add_actuators,
    add_assets,
    add_compiler,
    add_default,
    add_joint_sensors,
    add_mimic_equality_constraints,
    add_visual,
)
from urdf_to_mjcf.core.geometry import ParsedJointParams
from urdf_to_mjcf.core.model import (
    ActuatorMetadata,
    ConversionMetadata,
    DefaultJointMetadata,
    ExtraJointGroup,
    JointData,
    JointMetadata,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion context (from conversion_core.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionContext:
    """Prepared conversion state shared by the main conversion pipeline."""

    mjcf_root: ET.Element
    worldbody: ET.Element
    link_map: dict[str, ET.Element]
    parent_map: dict[str, list[tuple[str, ET.Element]]]
    root_link_name: str
    actuator_metadata: dict[str, ActuatorMetadata]
    joint_metadata: dict[str, JointMetadata]
    extra_joints: list[ExtraJointGroup]
    mimic_constraints: list[tuple[str, str, float, float]]
    metadata: ConversionMetadata


def create_empty_actuator_metadata(robot_elem: ET.Element) -> dict[str, ActuatorMetadata]:
    """Create placeholder metadata when actuator metadata is omitted."""
    actuator_meta: dict[str, ActuatorMetadata] = {}
    for joint in robot_elem.findall("joint"):
        name = joint.attrib.get("name")
        if name:
            actuator_meta[name] = ActuatorMetadata(actuator_type="motor")
    return actuator_meta


def create_actuator_metadata_from_joint_metadata(
    joint_metadata: Mapping[str, JointMetadata],
) -> dict[str, ActuatorMetadata]:
    """Extract explicit actuators from per-joint metadata."""
    actuator_metadata: dict[str, ActuatorMetadata] = {}
    for name, metadata in joint_metadata.items():
        actuator = metadata.actuator
        if actuator is None or actuator.actuator_type is None:
            continue
        actuator_metadata[name] = ActuatorMetadata(**dict(actuator))
    return actuator_metadata


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
    joint_data: JointData | None = None,
) -> ConversionContext:
    """Build the shared conversion context used by convert_urdf_to_mjcf."""
    resolved_joint_data = joint_data or JointData()
    resolved_joint_metadata = resolved_joint_data.joints
    if actuator_metadata is not None:
        resolved_actuator_metadata = actuator_metadata
    elif joint_data is not None:
        resolved_actuator_metadata = create_actuator_metadata_from_joint_metadata(resolved_joint_metadata)
    else:
        logger.warning("Missing joint metadata, falling back to single empty 'motor' class.")
        resolved_actuator_metadata = create_empty_actuator_metadata(robot)

    mjcf_root = ET.Element("mujoco", attrib={"model": robot.attrib.get("name", "converted_robot")})
    add_compiler(mjcf_root)
    add_visual(mjcf_root)
    add_default(mjcf_root, metadata, None if joint_data is not None else default_metadata, collision_only)
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
        joint_metadata=resolved_joint_metadata,
        extra_joints=resolved_joint_data.extra_joints,
        mimic_constraints=mimic_constraints,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Scene assembly (from conversion_scene.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneAssemblyResult:
    """Artifacts produced by robot scene assembly."""

    robot_body: ET.Element
    actuator_joints: list[ParsedJointParams]
    mesh_file_paths: dict[str, Path]


def add_extra_joints(
    robot_body: ET.Element,
    actuator_joints: list[ParsedJointParams],
    extra_joints: list[ExtraJointGroup],
    joint_metadata: Mapping[str, JointMetadata] | None = None,
) -> None:
    """Inject MJCF-only joints into generated bodies."""
    insert_indices: dict[str, int] = {}
    bodies = {body.attrib["name"]: body for body in robot_body.iter("body") if "name" in body.attrib}
    touched_bodies: dict[str, ET.Element] = {}

    for group in extra_joints:
        body = bodies.get(group.body)
        if body is None:
            raise ValueError(f"Extra joint body not found: {group.body}")

        for joint in group.joints:
            attrib = {
                "name": joint.name,
                "type": joint.type,
                "axis": " ".join(str(value) for value in joint.axis_values()),
            }
            lower = upper = None
            if joint.range is not None:
                lower, upper = joint.range
                attrib["range"] = f"{lower} {upper}"
            metadata = joint_metadata.get(joint.name) if joint_metadata is not None else None
            apply_joint_metadata(attrib, metadata)
            insert_at = insert_indices.get(group.body, 0)
            body.insert(insert_at, ET.Element("joint", attrib=attrib))
            insert_indices[group.body] = insert_at + 1
            touched_bodies[group.body] = body
            actuator_joints.append(ParsedJointParams(name=joint.name, type=joint.type, lower=lower, upper=upper))

    for body_name, body in touched_bodies.items():
        if body.find("inertial") is None:
            insert_at = insert_indices[body_name]
            body.insert(insert_at, minimal_inertial())


def minimal_inertial() -> ET.Element:
    return ET.Element(
        "inertial",
        attrib={
            "pos": "0 0 0",
            "mass": "0.001",
            "diaginertia": "1e-06 1e-06 1e-06",
        },
    )


def assemble_robot_scene(
    context: ConversionContext,
    *,
    urdf_path: Path,
    urdf_dir: Path,
    mjcf_path: Path,
    collision_only: bool,
    materials: dict[str, Any],
) -> SceneAssemblyResult:
    """Build the robot body tree, assets, and mesh resources into the MJCF root."""
    mesh_assets: dict[str, str] = {}

    target_mesh_dir = (mjcf_path.parent / "meshes").resolve()
    target_mesh_dir.mkdir(parents=True, exist_ok=True)
    workspace_search_paths = resolve_workspace_search_paths(urdf_path)

    robot_body, actuator_joints = build_robot_body_tree(
        context.root_link_name,
        link_map=context.link_map,
        parent_map=context.parent_map,
        actuator_metadata=context.actuator_metadata,
        collision_only=collision_only,
        materials=materials,
        mesh_assets=mesh_assets,
        workspace_search_paths=workspace_search_paths,
        urdf_dir=urdf_dir,
        joint_metadata=context.joint_metadata,
    )
    robot_body.attrib["childclass"] = ROBOT_CLASS
    add_extra_joints(robot_body, actuator_joints, context.extra_joints, context.joint_metadata)
    context.worldbody.append(robot_body)

    obj_materials = collect_single_obj_materials(
        mesh_assets,
        urdf_dir=urdf_dir,
        workspace_search_paths=workspace_search_paths,
    )
    add_assets(context.mjcf_root, materials, obj_materials)
    add_actuators(context.mjcf_root, actuator_joints, context.actuator_metadata)
    add_joint_sensors(context.mjcf_root, context.joint_metadata, actuator_joints)
    add_mimic_equality_constraints(context.mjcf_root, context.mimic_constraints)

    mesh_copy_result = copy_mesh_assets(
        context.mjcf_root,
        mesh_assets,
        urdf_dir=urdf_dir,
        target_mesh_dir=target_mesh_dir,
        workspace_search_paths=workspace_search_paths,
    )
    add_mesh_assets_to_xml(context.mjcf_root, mesh_copy_result.mesh_assets, urdf_dir=urdf_dir)

    return SceneAssemblyResult(
        robot_body=robot_body,
        actuator_joints=actuator_joints,
        mesh_file_paths=mesh_copy_result.mesh_file_paths,
    )

"""Helpers for assembling the main robot scene into the MJCF tree."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot2mjcf.conversion_assets import (
    add_mesh_assets_to_xml,
    collect_single_obj_materials,
    copy_mesh_assets,
    resolve_workspace_search_paths,
)
from robot2mjcf.conversion_body_builder import build_robot_body_tree
from robot2mjcf.conversion_core import ConversionContext
from robot2mjcf.conversion_mjcf_assembly import add_actuators, add_mimic_equality_constraints
from robot2mjcf.geometry import ParsedJointParams
from robot2mjcf.mjcf_builders import ROBOT_CLASS, add_assets


@dataclass(frozen=True)
class SceneAssemblyResult:
    """Artifacts produced by robot scene assembly."""

    robot_body: ET.Element
    actuator_joints: list[ParsedJointParams]
    mesh_file_paths: dict[str, Path]


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
    )
    robot_body.attrib["childclass"] = ROBOT_CLASS
    context.worldbody.append(robot_body)

    obj_materials = collect_single_obj_materials(
        mesh_assets,
        urdf_dir=urdf_dir,
        workspace_search_paths=workspace_search_paths,
    )
    add_assets(context.mjcf_root, materials, obj_materials)
    add_actuators(context.mjcf_root, actuator_joints, context.actuator_metadata)
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

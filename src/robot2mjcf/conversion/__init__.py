"""Conversion pipeline subpackage."""

from robot2mjcf.conversion.assets import (
    MeshCopyResult,
    add_mesh_assets_to_xml,
    collect_single_obj_materials,
    copy_mesh_assets,
    resolve_mesh_source_path,
    resolve_workspace_search_paths,
)
from robot2mjcf.conversion.body_builder import build_robot_body_tree
from robot2mjcf.conversion.input import ConversionInputs, load_conversion_inputs
from robot2mjcf.conversion.mjcf_assembly import (
    ROBOT_CLASS,
    MimicConstraint,
    add_actuators,
    add_assets,
    add_compiler,
    add_contact,
    add_default,
    add_mimic_equality_constraints,
    add_option,
    add_visual,
    add_weld_constraints,
)
from robot2mjcf.conversion.output import (
    adjust_robot_body_height,
    build_postprocess_options,
    save_initial_mjcf_and_apply_postprocess,
)
from robot2mjcf.conversion.pipeline import (
    ConversionContext,
    SceneAssemblyResult,
    assemble_robot_scene,
    build_conversion_context,
    create_empty_actuator_metadata,
    resolve_root_link_name,
)

__all__ = [
    # assets
    "MeshCopyResult",
    "add_mesh_assets_to_xml",
    "collect_single_obj_materials",
    "copy_mesh_assets",
    "resolve_mesh_source_path",
    "resolve_workspace_search_paths",
    # body_builder
    "build_robot_body_tree",
    # input
    "ConversionInputs",
    "load_conversion_inputs",
    # mjcf_assembly
    "ROBOT_CLASS",
    "MimicConstraint",
    "add_actuators",
    "add_assets",
    "add_compiler",
    "add_contact",
    "add_default",
    "add_mimic_equality_constraints",
    "add_option",
    "add_visual",
    "add_weld_constraints",
    # output
    "adjust_robot_body_height",
    "build_postprocess_options",
    "save_initial_mjcf_and_apply_postprocess",
    # pipeline
    "ConversionContext",
    "SceneAssemblyResult",
    "assemble_robot_scene",
    "build_conversion_context",
    "create_empty_actuator_metadata",
    "resolve_root_link_name",
]

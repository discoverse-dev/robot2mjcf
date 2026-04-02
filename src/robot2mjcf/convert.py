"""Converts URDF files to MJCF files."""

import argparse
import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from robot2mjcf.conversion_assets import (
    add_mesh_assets_to_xml,
    collect_single_obj_materials,
    copy_mesh_assets,
    resolve_workspace_search_paths,
)
from robot2mjcf.conversion_body_builder import build_robot_body_tree
from robot2mjcf.conversion_cli import (
    load_actuator_metadata_files,
    load_default_metadata_files,
    normalize_appendix_files,
)
from robot2mjcf.conversion_helpers import (
    build_joint_maps,
    collect_mimic_constraints,
    collect_urdf_materials,
    load_conversion_metadata,
    resolve_output_path,
)
from robot2mjcf.conversion_mjcf_assembly import add_actuators, add_mimic_equality_constraints
from robot2mjcf.conversion_output import adjust_robot_body_height, save_initial_mjcf_and_apply_postprocess
from robot2mjcf.conversion_postprocess import PostprocessOptions
from robot2mjcf.geometry import ParsedJointParams
from robot2mjcf.mjcf_builders import (
    ROBOT_CLASS,
    add_assets,
    add_compiler,
    add_default,
    add_visual,
    add_weld_constraints,
)
from robot2mjcf.model import ActuatorMetadata, DefaultJointMetadata

logger = logging.getLogger(__name__)


def _get_empty_actuator_metadata(
    robot_elem: ET.Element,
) -> dict[str, ActuatorMetadata]:
    """Create placeholder metadata for joints and actuators if none are provided.

    Each joint is simply assigned a "motor" actuator type, which has no other parameters.
    """
    actuator_meta: dict[str, ActuatorMetadata] = {}
    for joint in robot_elem.findall("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        actuator_meta[name] = ActuatorMetadata(
            actuator_type="motor",
        )

    return actuator_meta


def convert_urdf_to_mjcf(
    urdf_path: str | Path,
    mjcf_path: str | Path | None = None,
    metadata_file: str | Path | None = None,
    *,
    default_metadata: Mapping[str, DefaultJointMetadata] | None = None,
    actuator_metadata: dict[str, ActuatorMetadata] | None = None,
    appendix_files: list[Path] | None = None,
    max_vertices: int = 1000000,
    collision_only: bool = False,
    collision_type: str | None = None,
    capture_images: bool = False,
    run_mesh_postprocess: bool = True,
) -> None:
    """Converts a URDF file to an MJCF file.

    Args:
        urdf_path: The path to the URDF file.
        mjcf_path: The desired output MJCF file path.
        metadata_file: Optional path to metadata file.
        default_metadata: Optional default metadata.
        actuator_metadata: Optional actuator metadata.
        appendix_files: Optional list of appendix files.
        max_vertices: Maximum number of vertices in the mesh.
        collision_only: If true, use simplified collision geometry without visual appearance for visual representation.
        collision_type: The type of collision geometry to use.
        capture_images: If true, capture rendered preview images after conversion.
        run_mesh_postprocess: If false, skip the heavy mesh post-processing pipeline.
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    urdf_dir = urdf_path.parent.resolve()
    mjcf_path, output_warning = resolve_output_path(urdf_path, mjcf_path)
    if output_warning is not None:
        print(f"\033[33m{output_warning}\033[0m")

    mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    urdf_tree = ET.parse(urdf_path)
    robot = urdf_tree.getroot()
    if robot is None:
        raise ValueError("URDF file has no root element")

    metadata = load_conversion_metadata(metadata_file)

    if actuator_metadata is None:
        missing = []
        if actuator_metadata is None:
            missing.append("joint")
        logger.warning("Missing %s metadata, falling back to single empty 'motor' class.", " and ".join(missing))
        actuator_metadata = _get_empty_actuator_metadata(robot)
    assert actuator_metadata is not None

    materials = collect_urdf_materials(robot, collision_only)

    # Create a new MJCF tree root element.
    mjcf_root: ET.Element = ET.Element("mujoco", attrib={"model": robot.attrib.get("name", "converted_robot")})

    # Add compiler, option, visual, and assets
    add_compiler(mjcf_root)
    # add_option(mjcf_root)
    add_visual(mjcf_root)
    add_default(mjcf_root, metadata, default_metadata, collision_only)

    # Creates the worldbody element.
    worldbody = ET.SubElement(mjcf_root, "worldbody")

    link_map, parent_map, child_joints = build_joint_maps(robot)

    all_links = set(link_map.keys())
    child_links = set(child_joints.keys())
    root_links: list[str] = list(all_links - child_links)
    if not root_links:
        raise ValueError("No root link found in URDF.")
    root_link_name: str = root_links[0]

    # These dictionaries are used to collect mesh assets and actuator joints.
    mesh_assets: dict[str, str] = {}
    actuator_joints: list[ParsedJointParams] = []
    mimic_constraints = collect_mimic_constraints(robot)
    for mimicked_joint, joint_name, multiplier, offset in mimic_constraints:
        logger.info(
            f"Found mimic constraint: {joint_name} mimics {mimicked_joint} with multiplier={multiplier}, offset={offset}"
        )

    # Prepare paths for mesh processing
    target_mesh_dir: Path = (mjcf_path.parent / "meshes").resolve()
    target_mesh_dir.mkdir(parents=True, exist_ok=True)

    workspace_search_paths = resolve_workspace_search_paths(urdf_path)

    robot_body, actuator_joints = build_robot_body_tree(
        root_link_name,
        link_map=link_map,
        parent_map=parent_map,
        actuator_metadata=actuator_metadata,
        collision_only=collision_only,
        materials=materials,
        mesh_assets=mesh_assets,
        workspace_search_paths=workspace_search_paths,
        urdf_dir=urdf_dir,
    )

    robot_body.attrib["childclass"] = ROBOT_CLASS
    worldbody.append(robot_body)

    obj_materials = collect_single_obj_materials(
        mesh_assets,
        urdf_dir=urdf_dir,
        workspace_search_paths=workspace_search_paths,
    )

    # Add assets
    add_assets(mjcf_root, materials, obj_materials)

    add_actuators(mjcf_root, actuator_joints, actuator_metadata)
    add_mimic_equality_constraints(mjcf_root, mimic_constraints)

    # add_contact(mjcf_root, robot)

    # Add weld constraints if specified in metadata
    add_weld_constraints(mjcf_root, metadata)

    mesh_copy_result = copy_mesh_assets(
        mjcf_root,
        mesh_assets,
        urdf_dir=urdf_dir,
        target_mesh_dir=target_mesh_dir,
        workspace_search_paths=workspace_search_paths,
    )
    mesh_assets = mesh_copy_result.mesh_assets
    mesh_file_paths = mesh_copy_result.mesh_file_paths
    add_mesh_assets_to_xml(mjcf_root, mesh_assets, urdf_dir=urdf_dir)

    adjust_robot_body_height(
        robot_body,
        mesh_file_paths=mesh_file_paths,
        height_offset=metadata.height_offset,
    )
    save_initial_mjcf_and_apply_postprocess(
        mjcf_root,
        mjcf_path=mjcf_path,
        options=PostprocessOptions(
            metadata=metadata,
            collision_only=collision_only,
            collision_type=collision_type,
            max_vertices=max_vertices,
            appendix_files=appendix_files,
            capture_images=capture_images,
            run_mesh_postprocess=run_mesh_postprocess,
        ),
    )


def main() -> None:
    """Parse command-line arguments and execute the URDF to MJCF conversion."""
    parser = argparse.ArgumentParser(description="Convert a URDF file to an MJCF file.")

    parser.add_argument(
        "urdf_path",
        type=str,
        help="The path to the URDF file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the output MJCF file.",
    )
    parser.add_argument(
        "--collision-only",
        action="store_true",
        help="If true, use collision geometry without visual appearance for visual representation.",
    )
    parser.add_argument(
        "-cp",
        "--collision-type",
        type=str,
        # 保持原样mesh，进行凸分解，进行凸包络
        choices=["mesh", "decomposition", "convex_hull"],
        help="The type of collision geometry to use.",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default=None,
        help="A JSON file containing conversion metadata (joint params and sensors).",
    )
    parser.add_argument(
        "-dm",
        "--default-metadata",
        nargs="*",
        default=None,
        help="JSON files containing default metadata. Multiple files will be merged, with later files overriding earlier ones.",
    )
    parser.add_argument(
        "-am",
        "--actuator-metadata",
        nargs="*",
        default=None,
        help="JSON files containing actuator metadata. Multiple files will be merged, with later files overriding earlier ones.",
    )
    parser.add_argument(
        "-a",
        "--appendix",
        nargs="*",
        default=None,
        help="XML files containing appendix. Multiple files will be applied in order.",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=logging.INFO,
        help="The log level to use.",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=200000,
        help="Maximum number of vertices in the mesh.",
    )
    parser.add_argument(
        "--capture-images",
        action="store_true",
        help="Capture rendered preview images after conversion.",
    )
    parser.add_argument(
        "--skip-mesh-postprocess",
        action="store_true",
        help="Skip heavy mesh post-processing and only keep lightweight XML-side postprocess steps.",
    )
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    convert_urdf_to_mjcf(
        urdf_path=args.urdf_path,
        mjcf_path=args.output,
        metadata_file=args.metadata,
        default_metadata=load_default_metadata_files(args.default_metadata),
        actuator_metadata=load_actuator_metadata_files(args.actuator_metadata),
        appendix_files=normalize_appendix_files(args.appendix),
        max_vertices=args.max_vertices,
        collision_only=args.collision_only,
        collision_type=args.collision_type,
        capture_images=args.capture_images,
        run_mesh_postprocess=not args.skip_mesh_postprocess,
    )


if __name__ == "__main__":
    main()

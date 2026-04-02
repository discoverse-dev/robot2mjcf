"""Converts URDF files to MJCF files."""

import argparse
import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from robot2mjcf.conversion_cli import (
    load_actuator_metadata_files,
    load_default_metadata_files,
    normalize_appendix_files,
)
from robot2mjcf.conversion_core import build_conversion_context
from robot2mjcf.conversion_helpers import (
    collect_urdf_materials,
    load_conversion_metadata,
    resolve_output_path,
)
from robot2mjcf.conversion_output import adjust_robot_body_height, save_initial_mjcf_and_apply_postprocess
from robot2mjcf.conversion_postprocess import PostprocessOptions
from robot2mjcf.conversion_scene import assemble_robot_scene
from robot2mjcf.mjcf_builders import add_weld_constraints
from robot2mjcf.model import ActuatorMetadata, DefaultJointMetadata

logger = logging.getLogger(__name__)


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
    materials = collect_urdf_materials(robot, collision_only)
    context = build_conversion_context(
        robot,
        metadata=metadata,
        default_metadata=default_metadata,
        actuator_metadata=actuator_metadata,
        collision_only=collision_only,
    )
    scene = assemble_robot_scene(
        context,
        urdf_path=urdf_path,
        urdf_dir=urdf_dir,
        mjcf_path=mjcf_path,
        collision_only=collision_only,
        materials=materials,
    )

    # add_contact(mjcf_root, robot)

    # Add weld constraints if specified in metadata
    add_weld_constraints(context.mjcf_root, metadata)

    adjust_robot_body_height(
        scene.robot_body,
        mesh_file_paths=scene.mesh_file_paths,
        height_offset=metadata.height_offset,
    )
    save_initial_mjcf_and_apply_postprocess(
        context.mjcf_root,
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

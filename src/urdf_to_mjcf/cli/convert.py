"""Converts URDF files to MJCF files.

Merged from: convert.py + conversion_cli.py
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from urdf_to_mjcf.conversion.input import load_conversion_inputs
from urdf_to_mjcf.conversion.mjcf_assembly import add_weld_constraints
from urdf_to_mjcf.conversion.output import (
    adjust_robot_body_height,
    build_postprocess_options,
    save_initial_mjcf_and_apply_postprocess,
)
from urdf_to_mjcf.conversion.pipeline import assemble_robot_scene, build_conversion_context
from urdf_to_mjcf.core.model import ActuatorMetadata, DefaultJointMetadata, JointData

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# CLI helpers (from conversion_cli.py)
# ---------------------------------------------------------------------------


def _load_json_file(path: str, *, label: str, parser: Callable[[dict], T]) -> T:
    """Load and parse one CLI JSON input with consistent error reporting."""
    try:
        with open(path, "r") as file:
            value = parser(json.load(file))
    except Exception as exc:
        logger.warning("Failed to load %s from %s: %s", label, path, exc)
        traceback.print_exc()
        raise SystemExit(1) from exc
    logger.info("Loaded %s from %s", label, path)
    return value


def _load_metadata_files(
    metadata_files: Sequence[str] | None,
    *,
    label: str,
    parser: Callable[[dict], T],
) -> dict[str, T] | None:
    """Load keyed metadata from one or more JSON files."""
    if not metadata_files:
        return None

    loaded: dict[str, T] = {}
    for metadata_file in metadata_files:
        file_metadata = _load_json_file(
            metadata_file,
            label=f"{label} metadata",
            parser=lambda data: {key: parser(value) for key, value in data.items()},
        )
        loaded.update(file_metadata)

    return loaded or None


def load_default_metadata_files(metadata_files: Sequence[str] | None) -> dict[str, DefaultJointMetadata] | None:
    """Load default metadata files from CLI arguments."""
    return _load_metadata_files(
        metadata_files,
        label="default",
        parser=DefaultJointMetadata.from_dict,
    )


def load_actuator_metadata_files(metadata_files: Sequence[str] | None) -> dict[str, ActuatorMetadata] | None:
    """Load actuator metadata files from CLI arguments."""
    return _load_metadata_files(
        metadata_files,
        label="actuator",
        parser=ActuatorMetadata.from_dict,
    )


def load_joint_data_files(joint_data_files: Sequence[str] | None) -> JointData | None:
    """Load and merge joint data files from CLI arguments."""
    if not joint_data_files:
        return None

    extra_joints = []
    joints = {}
    for joint_data_file in joint_data_files:
        joint_data = _load_json_file(joint_data_file, label="joint data", parser=JointData.from_dict)
        extra_joints.extend(joint_data.extra_joints)
        joints.update(joint_data.joints)
    return JointData(extra_joints=extra_joints, joints=joints)


def normalize_appendix_files(appendix_files: Sequence[str] | None) -> list[Path] | None:
    """Convert appendix file CLI arguments to Path objects."""
    if not appendix_files:
        return None
    return [Path(appendix_file) for appendix_file in appendix_files]


# ---------------------------------------------------------------------------
# Public API and CLI entry point (from convert.py)
# ---------------------------------------------------------------------------


def convert_urdf_to_mjcf(
    urdf_path: str | Path,
    mjcf_path: str | Path | None = None,
    metadata_file: str | Path | None = None,
    *,
    default_metadata: Mapping[str, DefaultJointMetadata] | None = None,
    actuator_metadata: dict[str, ActuatorMetadata] | None = None,
    joint_data: JointData | None = None,
    appendix_files: list[Path] | None = None,
    max_vertices: int = 1000000,
    collision_only: bool = False,
    collision_type: str | None = None,
    freejoint: bool | None = None,
    add_floor: bool | None = None,
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
        joint_data: Optional extra joints, per-joint dynamics, and actuator metadata.
        appendix_files: Optional list of appendix files.
        max_vertices: Maximum number of vertices in the mesh.
        collision_only: If true, use simplified collision geometry without visual appearance for visual representation.
        collision_type: The type of collision geometry to use.
        freejoint: Optional CLI override for base freejoint generation.
        add_floor: Optional CLI override for floor generation.
        capture_images: If true, capture rendered preview images after conversion.
        run_mesh_postprocess: If false, skip the heavy mesh post-processing pipeline.
    """
    inputs = load_conversion_inputs(
        urdf_path,
        mjcf_path,
        metadata_file,
        collision_only=collision_only,
    )
    if inputs.output_warning is not None:
        print(f"\033[33m{inputs.output_warning}\033[0m")
    metadata = inputs.metadata
    if freejoint is not None:
        metadata.freejoint = freejoint
    if add_floor is not None:
        metadata.add_floor = add_floor
    context = build_conversion_context(
        inputs.robot,
        metadata=metadata,
        default_metadata=default_metadata,
        actuator_metadata=actuator_metadata,
        joint_data=joint_data,
        collision_only=collision_only,
    )
    scene = assemble_robot_scene(
        context,
        urdf_path=inputs.urdf_path,
        urdf_dir=inputs.urdf_dir,
        mjcf_path=inputs.mjcf_path,
        collision_only=collision_only,
        materials=inputs.materials,
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
        mjcf_path=inputs.mjcf_path,
        options=build_postprocess_options(
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
        "-ct",
        "--collision-type",
        type=str,
        # 保持原样mesh，进行凸分解，进行凸包络
        choices=["mesh", "decomposition", "convex_hull"],
        help="The collision mesh processing mode to use.",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default=None,
        help="Advanced conversion metadata JSON. Common add_floor/freejoint options are available as CLI flags.",
    )
    parser.add_argument(
        "--freejoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add a MuJoCo freejoint to the root body. Use --no-freejoint for fixed or planar bases.",
    )
    parser.add_argument(
        "--add-floor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add the default floor geom. Use --no-add-floor to disable it.",
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
        "-jd",
        "--joint-data",
        nargs="*",
        default=None,
        help="JSON files containing extra joints, per-joint dynamics, and actuator metadata. Multiple files will be merged in order.",
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
        joint_data=load_joint_data_files(args.joint_data),
        appendix_files=normalize_appendix_files(args.appendix),
        max_vertices=args.max_vertices,
        collision_only=args.collision_only,
        collision_type=args.collision_type,
        freejoint=args.freejoint,
        add_floor=args.add_floor,
        capture_images=args.capture_images,
        run_mesh_postprocess=not args.skip_mesh_postprocess,
    )


if __name__ == "__main__":
    main()

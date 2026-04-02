"""Helpers for final MJCF output adjustment and persistence."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from robot2mjcf.conversion_postprocess import PostprocessOptions, apply_postprocess_pipeline
from robot2mjcf.geometry import compute_min_z, format_value
from robot2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def adjust_robot_body_height(
    robot_body: ET.Element,
    *,
    mesh_file_paths: dict[str, Path],
    height_offset: float,
) -> float:
    """Adjust the robot body height based on the minimum geometry z value."""
    print("Computing minimum z coordinate from geometries...")
    min_z = compute_min_z(robot_body, mesh_file_paths=mesh_file_paths)
    computed_offset = -min_z + height_offset
    logger.info("Auto-detected base offset: %s (min z = %s)", computed_offset, min_z)

    body_pos = [float(x) for x in robot_body.attrib.get("pos", "0 0 0").split()]
    body_pos[2] += computed_offset
    robot_body.attrib["pos"] = " ".join(format_value(x) for x in body_pos)
    return computed_offset


def save_initial_mjcf_and_apply_postprocess(
    mjcf_root: ET.Element,
    *,
    mjcf_path: Path,
    options: PostprocessOptions,
) -> None:
    """Persist the initial MJCF and then run the configured postprocess pipeline."""
    print(f"Saving initial MJCF file to {mjcf_path}")
    save_xml(mjcf_path, ET.ElementTree(mjcf_root))
    apply_postprocess_pipeline(mjcf_path, options=options)

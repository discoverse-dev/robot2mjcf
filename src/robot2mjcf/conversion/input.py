"""Helpers for loading conversion inputs, resolving output paths, and URDF parsing.

Merged from: conversion_input.py + conversion_helpers.py
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from robot2mjcf.core.model import ConversionMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (from conversion_helpers.py)
# ---------------------------------------------------------------------------


def resolve_output_path(urdf_path: str | Path, mjcf_path: str | Path | None) -> tuple[Path, str | None]:
    """Resolve the output MJCF path and return an optional warning message."""
    urdf_path = Path(urdf_path)
    urdf_dir = urdf_path.parent.resolve()
    default_mjcf_path = urdf_dir / "output_mjcf" / "robot.xml"

    if mjcf_path is None:
        return default_mjcf_path, None

    output_path = Path(mjcf_path)
    if output_path.parent.resolve() == urdf_dir:
        warning = (
            f"Warning: output file cannot be in the same directory as the URDF file ({urdf_dir}). "
            f"Using default output path: {default_mjcf_path}"
        )
        return default_mjcf_path, warning

    return output_path, None


def load_conversion_metadata(metadata_file: str | Path | None) -> ConversionMetadata:
    """Load conversion metadata, falling back to defaults on parse failure."""
    if metadata_file is None:
        return ConversionMetadata()

    try:
        text = Path(metadata_file).read_text()
        if hasattr(ConversionMetadata, "model_validate_json"):
            return ConversionMetadata.model_validate_json(text)
        return ConversionMetadata.parse_raw(text)
    except Exception as exc:
        logger.warning("Failed to load metadata from %s: %s", metadata_file, exc)
        return ConversionMetadata()


def collect_urdf_materials(robot: ET.Element, collision_only: bool) -> dict[str, str]:
    """Collect named URDF material colors from root and visual elements."""
    materials: dict[str, str] = {}

    for material in robot.findall("material"):
        name = material.attrib.get("name")
        if name is None:
            continue
        if name == "":
            logger.warning("Material name is empty, using default_material")
            material.attrib["name"] = "default_material"
            name = "default_material"

        color = material.find("color")
        if color is not None and (rgba := color.attrib.get("rgba")) is not None:
            materials[name] = rgba

    if collision_only:
        return materials

    for link in robot.findall("link"):
        for visual in link.findall("visual"):
            visual_material = visual.find("material")
            if visual_material is None:
                continue
            if visual_material.attrib.get("name") == "":
                visual_material.attrib["name"] = "default_material"

            name = visual_material.attrib.get("name")
            if name is None:
                continue
            color = visual_material.find("color")
            if color is not None and (rgba := color.attrib.get("rgba")) is not None:
                materials[name] = rgba

    return materials


def build_joint_maps(
    robot: ET.Element,
) -> tuple[dict[str, ET.Element], dict[str, list[tuple[str, ET.Element]]], dict[str, ET.Element]]:
    """Build link and joint lookup tables from a URDF robot element."""
    link_map: dict[str, ET.Element] = {link.attrib["name"]: link for link in robot.findall("link")}
    parent_map: dict[str, list[tuple[str, ET.Element]]] = {}
    child_joints: dict[str, ET.Element] = {}

    for joint in robot.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            logger.warning("Joint missing parent or child element")
            continue

        parent_name = parent_elem.attrib.get("link", "")
        child_name = child_elem.attrib.get("link", "")
        if not parent_name or not child_name:
            logger.warning("Joint missing parent or child link name")
            continue

        parent_map.setdefault(parent_name, []).append((child_name, joint))
        child_joints[child_name] = joint

    return link_map, parent_map, child_joints


def collect_mimic_constraints(robot: ET.Element) -> list[tuple[str, str, float, float]]:
    """Collect mimic joint constraints from a URDF robot element."""
    mimic_constraints: list[tuple[str, str, float, float]] = []

    for joint in robot.findall("joint"):
        mimic_elem = joint.find("mimic")
        if mimic_elem is None:
            continue

        joint_name = joint.attrib.get("name")
        mimicked_joint = mimic_elem.attrib.get("joint")
        multiplier = float(mimic_elem.attrib.get("multiplier", "1.0"))
        offset = float(mimic_elem.attrib.get("offset", "0.0"))

        if joint_name and mimicked_joint:
            mimic_constraints.append((mimicked_joint, joint_name, multiplier, offset))

    return mimic_constraints


# ---------------------------------------------------------------------------
# ConversionInputs (from conversion_input.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversionInputs:
    """Loaded inputs for the conversion pipeline."""

    urdf_path: Path
    urdf_dir: Path
    mjcf_path: Path
    output_warning: str | None
    robot: ET.Element
    metadata: ConversionMetadata
    materials: dict[str, str]


def load_conversion_inputs(
    urdf_path: str | Path,
    mjcf_path: str | Path | None,
    metadata_file: str | Path | None,
    *,
    collision_only: bool,
) -> ConversionInputs:
    """Load the URDF, metadata, material map, and resolved output path."""
    resolved_urdf_path = Path(urdf_path)
    if not resolved_urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {resolved_urdf_path}")

    urdf_dir = resolved_urdf_path.parent.resolve()
    resolved_mjcf_path, output_warning = resolve_output_path(resolved_urdf_path, mjcf_path)
    resolved_mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    robot = ET.parse(resolved_urdf_path).getroot()
    if robot is None:
        raise ValueError("URDF file has no root element")

    metadata = load_conversion_metadata(metadata_file)
    materials = collect_urdf_materials(robot, collision_only)

    return ConversionInputs(
        urdf_path=resolved_urdf_path,
        urdf_dir=urdf_dir,
        mjcf_path=resolved_mjcf_path,
        output_warning=output_warning,
        robot=robot,
        metadata=metadata,
        materials=materials,
    )

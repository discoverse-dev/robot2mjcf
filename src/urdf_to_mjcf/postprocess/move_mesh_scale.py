"""Move scale attributes from geom elements to mesh asset definitions.

This post-processing script handles the case where geom elements have scale
attributes, which MuJoCo doesn't support directly. Positive scales are moved to
the mesh asset definition. Negative scales are baked into generated mesh files so
mirrored meshes keep correct face winding and lighting.
"""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import trimesh

from urdf_to_mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


def _parse_scale(scale_str: str) -> tuple[float, float, float] | None:
    try:
        values = tuple(float(value) for value in scale_str.split())
    except ValueError:
        return None

    if len(values) == 1:
        value = values[0]
        return (value, value, value)
    if len(values) == 3:
        return values
    return None


def _format_number(value: float) -> str:
    return f"{value:g}"


def _format_scale(scale: tuple[float, float, float]) -> str:
    return " ".join(_format_number(value) for value in scale)


def _scale_token(scale: tuple[float, float, float]) -> str:
    parts = []
    for value in scale:
        token = _format_number(value).replace("-", "m").replace(".", "p")
        parts.append(token)
    return "_".join(parts)


def _is_identity_scale(scale: tuple[float, float, float]) -> bool:
    return all(abs(value - 1.0) < 1e-12 for value in scale)


def _has_negative_scale(scale: tuple[float, float, float]) -> bool:
    return any(value < 0.0 for value in scale)


def _mesh_root(mjcf_path: Path, root: ET.Element) -> Path:
    compiler = root.find("compiler")
    meshdir_ref = "." if compiler is None else compiler.attrib.get("meshdir", ".")
    return (mjcf_path.parent / meshdir_ref).resolve()


def _baked_mesh_relative_path(mesh_file: str, scale: tuple[float, float, float], mesh_root: Path) -> str | None:
    mesh_path = (mesh_root / mesh_file).resolve()
    if not mesh_path.exists():
        logger.warning("Cannot bake scaled mesh; file does not exist: %s", mesh_path)
        return None

    baked_path = mesh_path.with_name(f"{mesh_path.stem}_scaled_{_scale_token(scale)}{mesh_path.suffix}")
    if not baked_path.exists() or baked_path.stat().st_mtime < mesh_path.stat().st_mtime:
        try:
            loaded = trimesh.load(mesh_path, force="scene", process=False)
            if isinstance(loaded, trimesh.Scene):
                meshes = loaded.dump(concatenate=False)
            else:
                meshes = [loaded]

            scaled_meshes = []
            scale_matrix = [
                [scale[0], 0.0, 0.0, 0.0],
                [0.0, scale[1], 0.0, 0.0],
                [0.0, 0.0, scale[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            for mesh in meshes:
                if not isinstance(mesh, trimesh.Trimesh):
                    continue
                scaled = mesh.copy()
                scaled.apply_transform(scale_matrix)
                scaled_meshes.append(scaled)

            if not scaled_meshes:
                logger.warning("Cannot bake scaled mesh; no triangle geometry found in %s", mesh_path)
                return None

            if len(scaled_meshes) == 1:
                scaled_meshes[0].export(baked_path)
            else:
                trimesh.util.concatenate(scaled_meshes).export(baked_path)
        except Exception as exc:
            logger.warning("Failed to bake scaled mesh %s with scale %s: %s", mesh_path, _format_scale(scale), exc)
            return None

    try:
        return baked_path.relative_to(mesh_root).as_posix()
    except ValueError:
        return baked_path.as_posix()


def move_mesh_scale(mjcf_path: str | Path) -> None:
    """Move scale attributes from geom to mesh assets.

    Args:
        mjcf_path: Path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Find asset section
    asset = root.find("asset")
    if asset is None:
        logger.warning("No <asset> section found in MJCF file")
        return

    mesh_root = _mesh_root(mjcf_path, root)

    # Build a mapping of mesh name -> mesh element and file path
    mesh_map: Dict[str, Tuple[ET.Element, str]] = {}
    for mesh_elem in asset.findall("mesh"):
        mesh_name = mesh_elem.attrib.get("name")
        mesh_file = mesh_elem.attrib.get("file")
        if mesh_name and mesh_file:
            mesh_map[mesh_name] = (mesh_elem, mesh_file)

    # Track mesh file + scale combinations to create unique mesh names
    # Key: (original_mesh_name, normalized_scale_str), Value: new_mesh_name
    scale_mesh_map: Dict[Tuple[str, str], str] = {}

    # Counter for generating unique mesh names
    mesh_counters: Dict[str, int] = {}

    # Find all geoms with scale attributes in worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        logger.warning("No <worldbody> section found in MJCF file")
        return

    # Process all geom elements recursively
    def process_geoms(element: ET.Element) -> None:
        """Recursively process geom elements in the tree."""
        for geom in element.findall(".//geom"):
            geom_type = geom.attrib.get("type")
            mesh_name = geom.attrib.get("mesh")
            scale_str = geom.attrib.get("scale")

            # Only process mesh geoms with scale attribute
            if geom_type != "mesh" or not mesh_name or not scale_str:
                continue

            # Check if this mesh exists in assets
            if mesh_name not in mesh_map:
                logger.warning(f"Geom references non-existent mesh: {mesh_name}")
                continue

            parsed_scale = _parse_scale(scale_str)
            if parsed_scale is None:
                logger.warning("Invalid mesh scale '%s' on geom '%s'", scale_str, geom.attrib.get("name", "unnamed"))
                continue
            normalized_scale = _format_scale(parsed_scale)

            original_mesh_elem, mesh_file = mesh_map[mesh_name]

            # Create a key for this mesh+scale combination
            key = (mesh_name, normalized_scale)

            # Check if we already created a mesh for this combination
            if key in scale_mesh_map:
                # Reuse existing mesh name
                new_mesh_name = scale_mesh_map[key]
            else:
                # Check if the original mesh already has this scale
                original_scale = original_mesh_elem.attrib.get("scale")
                if original_scale == normalized_scale:
                    # The mesh already has this scale, just remove from geom
                    new_mesh_name = mesh_name
                elif _is_identity_scale(parsed_scale) and original_scale is None:
                    new_mesh_name = mesh_name
                else:
                    # Need to create a new mesh entry
                    # Generate unique name
                    base_name = mesh_name.rsplit(".", 1)[0]  # Remove extension if present

                    # Initialize counter if not exists
                    if base_name not in mesh_counters:
                        mesh_counters[base_name] = 1

                    # Find a unique name
                    counter = mesh_counters[base_name]
                    while True:
                        new_mesh_name = f"{base_name}_{counter}"
                        # Check if this name already exists in mesh_map or scale_mesh_map
                        if new_mesh_name not in mesh_map and new_mesh_name not in [v for v in scale_mesh_map.values()]:
                            break
                        counter += 1

                    mesh_counters[base_name] = counter + 1

                    mesh_attrib = {"name": new_mesh_name, "file": mesh_file}
                    if _has_negative_scale(parsed_scale):
                        baked_file = _baked_mesh_relative_path(mesh_file, parsed_scale, mesh_root)
                        if baked_file is not None:
                            mesh_attrib["file"] = baked_file
                        else:
                            mesh_attrib["scale"] = normalized_scale
                    else:
                        mesh_attrib["scale"] = normalized_scale

                    # Create new mesh element in asset section
                    new_mesh_elem = ET.Element("mesh", attrib=mesh_attrib)

                    # Insert after the original mesh
                    mesh_index = list(asset).index(original_mesh_elem)
                    asset.insert(mesh_index + 1, new_mesh_elem)

                    # Update tracking
                    mesh_map[new_mesh_name] = (new_mesh_elem, mesh_attrib["file"])
                    scale_mesh_map[key] = new_mesh_name

                    logger.info(
                        "Created new mesh '%s' with scale '%s' for file '%s'",
                        new_mesh_name,
                        normalized_scale,
                        mesh_file,
                    )

            # Update geom to reference the new mesh and remove scale
            geom.attrib["mesh"] = new_mesh_name
            del geom.attrib["scale"]

    # Process all geoms in worldbody
    process_geoms(worldbody)

    # Also handle geoms that might have no scale (to normalize naming)
    # Now handle geoms without scale that reference meshes with extensions in name
    for geom in worldbody.findall(".//geom"):
        geom_type = geom.attrib.get("type")
        mesh_name = geom.attrib.get("mesh")

        if geom_type == "mesh" and mesh_name and "scale" not in geom.attrib:
            # Check if mesh name still has extension
            if mesh_name in mesh_map:
                original_mesh_elem, _mesh_file = mesh_map[mesh_name]

                # If the original mesh doesn't have a scale, normalize its name (remove extension)
                if "scale" not in original_mesh_elem.attrib:
                    base_name = mesh_name.rsplit(".", 1)[0]
                    if base_name != mesh_name and base_name not in mesh_map:
                        # Rename the mesh to remove extension
                        original_mesh_elem.attrib["name"] = base_name
                        mesh_map[base_name] = mesh_map.pop(mesh_name)
                        geom.attrib["mesh"] = base_name
                        logger.info(f"Normalized mesh name from '{mesh_name}' to '{base_name}'")

    # Save the modified MJCF file
    save_xml(mjcf_path, tree)
    logger.info(f"Processed mesh scales in {mjcf_path}")


def main() -> None:
    """Command-line interface for move_mesh_scale."""
    parser = argparse.ArgumentParser(description="Move scale attributes from geom elements to mesh assets")
    parser.add_argument(
        "mjcf_path",
        type=Path,
        help="Path to the MJCF file to process",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    move_mesh_scale(args.mjcf_path)


if __name__ == "__main__":
    main()

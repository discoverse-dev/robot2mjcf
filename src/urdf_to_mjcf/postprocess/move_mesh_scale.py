"""Move scale attributes from geom elements to mesh asset definitions.

This post-processing script handles the case where geom elements have scale
attributes, which MuJoCo doesn't support directly. Scales are moved to the mesh
asset definition, where MuJoCo applies them to the loaded vertex data.
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
    return "_".join(_format_number(value).replace("-", "m").replace(".", "p") for value in scale)


def _is_identity_scale(scale: tuple[float, float, float]) -> bool:
    return all(abs(value - 1.0) < 1e-12 for value in scale)


def _is_reflection_scale(scale: tuple[float, float, float]) -> bool:
    return scale[0] * scale[1] * scale[2] < 0.0


def _combine_scale(
    base_scale: tuple[float, float, float],
    geom_scale: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        base_scale[0] * geom_scale[0],
        base_scale[1] * geom_scale[1],
        base_scale[2] * geom_scale[2],
    )


def _mesh_root(mjcf_path: Path, root: ET.Element) -> Path:
    compiler = root.find("compiler")
    meshdir_ref = "." if compiler is None else compiler.attrib.get("meshdir", ".")
    return (mjcf_path.parent / meshdir_ref).resolve()


def _strip_obj_material_directives(obj_path: Path) -> None:
    lines = obj_path.read_text().splitlines(keepends=True)
    stripped_lines = [
        line for line in lines if not line.lstrip().startswith(("mtllib ", "usemtl "))
    ]
    if len(stripped_lines) != len(lines):
        obj_path.write_text("".join(stripped_lines))


def _remove_generated_obj_sidecars(obj_path: Path) -> None:
    for sidecar_path in obj_path.parent.glob("*.mtl"):
        if sidecar_path.is_file():
            sidecar_path.unlink()


def _baked_mesh_relative_path(mesh_file: str, scale: tuple[float, float, float], mesh_root: Path) -> str:
    mesh_path = (mesh_root / mesh_file).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Cannot bake reflected visual mesh; file does not exist: {mesh_path}")

    baked_dir = mesh_path.parent / "_generated"
    baked_dir.mkdir(parents=True, exist_ok=True)
    baked_path = baked_dir / f"{mesh_path.stem}_scaled_{_scale_token(scale)}.obj"

    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = loaded.dump(concatenate=False)
    else:
        meshes = [loaded]

    scaled_meshes: list[trimesh.Trimesh] = []
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
        raise ValueError(f"Cannot bake reflected visual mesh; no triangle geometry found in {mesh_path}")

    baked_mesh = scaled_meshes[0] if len(scaled_meshes) == 1 else trimesh.util.concatenate(scaled_meshes)
    baked_mesh.export(baked_path, file_type="obj")
    _strip_obj_material_directives(baked_path)
    _remove_generated_obj_sidecars(baked_path)

    return baked_path.relative_to(mesh_root).as_posix()


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
    # Key: (original_mesh_name, normalized_scale_str, strategy), Value: new_mesh_name
    scale_mesh_map: Dict[Tuple[str, str, str], str] = {}

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

            geom_scale = _parse_scale(scale_str)
            if geom_scale is None:
                logger.warning("Invalid mesh scale '%s' on geom '%s'", scale_str, geom.attrib.get("name", "unnamed"))
                continue

            original_mesh_elem, mesh_file = mesh_map[mesh_name]
            base_scale = _parse_scale(original_mesh_elem.attrib.get("scale", "1 1 1"))
            if base_scale is None:
                logger.warning("Invalid mesh asset scale '%s' on mesh '%s'", original_mesh_elem.attrib["scale"], mesh_name)
                continue
            effective_scale = _combine_scale(base_scale, geom_scale)
            normalized_scale = _format_scale(effective_scale)
            bake_visual_reflection = geom.attrib.get("class") != "collision" and _is_reflection_scale(effective_scale)
            scale_strategy = "baked_visual_reflection" if bake_visual_reflection else "mesh_scale"

            # Create a key for this mesh+scale combination
            key = (mesh_name, normalized_scale, scale_strategy)

            # Check if we already created a mesh for this combination
            if key in scale_mesh_map:
                # Reuse existing mesh name
                new_mesh_name = scale_mesh_map[key]
            else:
                # Check if the original mesh already has this scale
                original_scale = original_mesh_elem.attrib.get("scale")
                if not bake_visual_reflection and original_scale == normalized_scale:
                    # The mesh already has this scale, just remove from geom
                    new_mesh_name = mesh_name
                elif not bake_visual_reflection and _is_identity_scale(effective_scale) and original_scale is None:
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
                    if bake_visual_reflection:
                        mesh_attrib["file"] = _baked_mesh_relative_path(mesh_file, effective_scale, mesh_root)
                    elif not _is_identity_scale(effective_scale):
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

"""Move scale attributes from geom elements to mesh asset definitions.

This post-processing script handles the case where geom elements have scale
attributes, which MuJoCo doesn't support directly. The scale is moved to the
mesh asset definition in the <asset> section. When multiple geoms reference
the same mesh file with different scales, new mesh entries are created.

Example transformation:
    Before:
        <geom type="mesh" mesh="gripper.stl" scale="1 -1 1"/>
        <geom type="mesh" mesh="gripper.stl"/>

    After:
        <geom type="mesh" mesh="gripper"/>
        <geom type="mesh" mesh="gripper_1"/>

        <mesh name="gripper" file="meshes/collision/gripper.stl"/>
        <mesh name="gripper_1" file="meshes/collision/gripper.stl" scale="1 -1 1"/>
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

from robot2mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


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

    # Build a mapping of mesh name -> mesh element and file path
    mesh_map: Dict[str, Tuple[ET.Element, str]] = {}
    for mesh_elem in asset.findall("mesh"):
        mesh_name = mesh_elem.attrib.get("name")
        mesh_file = mesh_elem.attrib.get("file")
        if mesh_name and mesh_file:
            mesh_map[mesh_name] = (mesh_elem, mesh_file)

    # Track mesh file + scale combinations to create unique mesh names
    # Key: (original_mesh_name, scale_str), Value: new_mesh_name
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

            original_mesh_elem, mesh_file = mesh_map[mesh_name]

            # Create a key for this mesh+scale combination
            key = (mesh_name, scale_str)

            # Check if we already created a mesh for this combination
            if key in scale_mesh_map:
                # Reuse existing mesh name
                new_mesh_name = scale_mesh_map[key]
            else:
                # Check if the original mesh already has this scale
                original_scale = original_mesh_elem.attrib.get("scale")
                if original_scale == scale_str:
                    # The mesh already has this scale, just remove from geom
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

                    # Create new mesh element in asset section
                    new_mesh_elem = ET.Element(
                        "mesh", attrib={"name": new_mesh_name, "file": mesh_file, "scale": scale_str}
                    )

                    # Insert after the original mesh
                    mesh_index = list(asset).index(original_mesh_elem)
                    asset.insert(mesh_index + 1, new_mesh_elem)

                    # Update tracking
                    mesh_map[new_mesh_name] = (new_mesh_elem, mesh_file)
                    scale_mesh_map[key] = new_mesh_name

                    logger.info(f"Created new mesh '{new_mesh_name}' with scale '{scale_str}' for file '{mesh_file}'")

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
                original_mesh_elem, mesh_file = mesh_map[mesh_name]

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

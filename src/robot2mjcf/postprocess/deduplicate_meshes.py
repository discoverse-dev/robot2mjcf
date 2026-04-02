"""Remove duplicate mesh definitions from MJCF asset section.

This post-processing script handles cases where the same mesh name appears
multiple times in the asset section. If the mesh files are identical, the
duplicates are removed. If they differ, an error is logged and the first
occurrence is kept.
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

from robot2mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


def deduplicate_meshes(mjcf_path: str | Path) -> None:
    """Remove duplicate mesh definitions from MJCF asset section.

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

    # Track mesh names and their file paths
    mesh_registry: Dict[str, str] = {}  # name -> file path
    mesh_elements: Dict[str, ET.Element] = {}  # name -> first element
    meshes_to_remove: List[ET.Element] = []

    duplicate_count = 0
    conflict_count = 0

    # Process all mesh elements
    for mesh_elem in asset.findall("mesh"):
        mesh_name = mesh_elem.attrib.get("name")
        mesh_file = mesh_elem.attrib.get("file")

        if not mesh_name:
            logger.warning("Found mesh element without name attribute, skipping")
            continue

        if not mesh_file:
            logger.warning(f"Mesh '{mesh_name}' has no file attribute, skipping")
            continue

        # Normalize file path for comparison (resolve relative paths)
        mesh_file_normalized = str(Path(mesh_file).as_posix())

        if mesh_name in mesh_registry:
            # Duplicate mesh name found
            existing_file = mesh_registry[mesh_name]

            if mesh_file_normalized == existing_file:
                # Same file path - safe to remove duplicate
                logger.info(f"Removing duplicate mesh '{mesh_name}' with identical file path: {mesh_file}")
                meshes_to_remove.append(mesh_elem)
                duplicate_count += 1
            else:
                # Different file paths - this is a conflict
                logger.error(
                    f"Mesh name conflict detected: '{mesh_name}'\n"
                    f"  First occurrence:  {existing_file}\n"
                    f"  Duplicate found:   {mesh_file}\n"
                    f"  Using first occurrence and removing duplicate."
                )
                meshes_to_remove.append(mesh_elem)
                conflict_count += 1
        else:
            # First occurrence of this mesh name
            mesh_registry[mesh_name] = mesh_file_normalized
            mesh_elements[mesh_name] = mesh_elem

    # Remove duplicate mesh elements
    for mesh_elem in meshes_to_remove:
        asset.remove(mesh_elem)

    # Log summary
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate mesh(es) with identical file paths")

    if conflict_count > 0:
        logger.warning(f"Resolved {conflict_count} mesh name conflict(s) - check logs for details")

    if duplicate_count == 0 and conflict_count == 0:
        logger.info("No duplicate meshes found")

    # Save the modified MJCF file
    save_xml(mjcf_path, tree)
    logger.info(f"Processed mesh deduplication in {mjcf_path}")


def main() -> None:
    """Command-line interface for deduplicate_meshes."""
    parser = argparse.ArgumentParser(description="Remove duplicate mesh definitions from MJCF asset section")
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

    deduplicate_meshes(args.mjcf_path)


if __name__ == "__main__":
    main()

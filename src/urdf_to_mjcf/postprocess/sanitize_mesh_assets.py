"""Sanitize final MJCF mesh asset references."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf_to_mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


def _mesh_path(mjcf_path: Path, meshdir: str, mesh_file: str) -> Path:
    path = Path(mesh_file)
    if path.is_absolute():
        return path
    return mjcf_path.parent / meshdir / path


def sanitize_mesh_assets(mjcf_path: str | Path) -> None:
    """Remove mesh assets and geoms that reference missing mesh files."""
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        return

    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", ".") if compiler is not None else "."

    valid_mesh_names: set[str] = set()
    missing_mesh_elems: list[ET.Element] = []
    for mesh in asset.findall("mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")
        if not mesh_name or not mesh_file:
            continue
        if _mesh_path(mjcf_path, meshdir, mesh_file).exists():
            valid_mesh_names.add(mesh_name)
        else:
            missing_mesh_elems.append(mesh)

    if not missing_mesh_elems:
        return

    missing_mesh_names = {
        mesh.attrib["name"] for mesh in missing_mesh_elems if mesh.attrib["name"] not in valid_mesh_names
    }
    parent_map = {child: parent for parent in root.iter() for child in parent}
    removed_geoms = 0
    for geom in list(root.iter("geom")):
        if geom.attrib.get("mesh") not in missing_mesh_names:
            continue
        parent = parent_map.get(geom)
        if parent is not None:
            parent.remove(geom)
            removed_geoms += 1

    for mesh in missing_mesh_elems:
        asset.remove(mesh)

    remaining_missing = []
    for mesh in asset.findall("mesh"):
        mesh_file = mesh.attrib.get("file")
        if mesh_file and not _mesh_path(mjcf_path, meshdir, mesh_file).exists():
            remaining_missing.append(mesh_file)
    if remaining_missing:
        raise FileNotFoundError(f"Missing MJCF mesh assets after sanitization: {sorted(remaining_missing)}")

    logger.warning(
        "Removed %d missing mesh asset(s) and %d geom(s) from %s",
        len(missing_mesh_elems),
        removed_geoms,
        mjcf_path,
    )
    save_xml(mjcf_path, tree)

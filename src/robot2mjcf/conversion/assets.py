"""Helpers for mesh asset resolution and copying."""

from __future__ import annotations

import logging
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from robot2mjcf.core.materials import Material, copy_obj_with_mtl, get_obj_material_info, parse_mtl_name
from robot2mjcf.core.package_resolver import find_workspace_from_path, resolve_package_path

logger = logging.getLogger(__name__)


@dataclass
class MeshCopyResult:
    mesh_assets: dict[str, str]
    mesh_file_paths: dict[str, Path]


def resolve_workspace_search_paths(urdf_path: Path) -> list[Path]:
    """Resolve candidate search roots for package:// resources."""
    workspace_search_paths: list[Path] = []

    workspace_from_urdf = find_workspace_from_path(urdf_path)
    if workspace_from_urdf:
        workspace_search_paths.append(workspace_from_urdf)
        logger.debug("Found ROS workspace from URDF location: %s", workspace_from_urdf)

    from robot2mjcf.core.package_resolver import _default_resolver

    package_root = _default_resolver._find_package_root_from_urdf_path(urdf_path)
    if package_root and package_root not in workspace_search_paths:
        workspace_search_paths.append(package_root)
        logger.debug("Found package root from URDF location: %s", package_root)

    return workspace_search_paths


def resolve_mesh_source_path(
    filename: str, *, urdf_dir: Path, workspace_search_paths: list[Path]
) -> tuple[Path | None, str]:
    """Resolve a mesh source path and its target subpath."""
    if "package://" in filename:
        package_path = filename[len("package://") :]
        package_name = package_path.split("/")[0]
        sub_path = "/".join(package_path.split("/")[1:])
        try:
            pkg_root = resolve_package_path(package_name, workspace_search_paths)
            source_path = pkg_root / sub_path if pkg_root else None
        except Exception:
            source_path = None
        return source_path, f"{package_name}/{sub_path}"

    if filename.startswith("/"):
        return Path(filename), os.path.relpath(filename, urdf_dir)

    return (urdf_dir / filename).resolve(), filename


def collect_single_obj_materials(
    mesh_assets: dict[str, str], *, urdf_dir: Path, workspace_search_paths: list[Path]
) -> dict[str, Material]:
    """Collect materials from single-material OBJ files."""
    obj_materials: dict[str, Material] = {}

    for filename in mesh_assets.values():
        if not filename.lower().endswith(".obj"):
            continue

        obj_file_path, _ = resolve_mesh_source_path(
            filename,
            urdf_dir=urdf_dir,
            workspace_search_paths=workspace_search_paths,
        )
        if obj_file_path is None:
            continue

        has_single_material, material_name = get_obj_material_info(obj_file_path)
        if not has_single_material or not material_name:
            continue

        try:
            mtl_name = parse_mtl_name(obj_file_path.open("r").readlines())
            if not mtl_name:
                continue
            mtl_file = obj_file_path.parent / mtl_name
            if not mtl_file.exists():
                continue

            material_lines: list[str] = []
            in_material = False
            for line in mtl_file.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("newmtl ") and stripped.split()[1] == material_name:
                    in_material = True
                    material_lines = [stripped]
                elif stripped.startswith("newmtl ") and in_material:
                    break
                elif in_material:
                    material_lines.append(stripped)

            if material_lines:
                material = Material.from_string(material_lines)
                material.name = f"{obj_file_path.stem}_{material_name}"
                obj_materials[material.name] = material
                logger.info("Added single OBJ material: %s", material.name)
        except Exception as exc:
            logger.warning("Failed to parse single-material OBJ %s: %s", obj_file_path, exc)

    return obj_materials


def copy_mesh_assets(
    mjcf_root: ET.Element,
    mesh_assets: dict[str, str],
    *,
    urdf_dir: Path,
    target_mesh_dir: Path,
    workspace_search_paths: list[Path],
) -> MeshCopyResult:
    """Copy mesh assets into the output tree and prune missing references."""
    processed_targets: set[str] = set()
    missing_meshes: set[str] = set()
    mesh_file_paths: dict[str, Path] = {}

    for mesh_name, filename in mesh_assets.items():
        source_path, sub_path = resolve_mesh_source_path(
            filename,
            urdf_dir=urdf_dir,
            workspace_search_paths=workspace_search_paths,
        )

        if source_path is not None and not source_path.exists():
            logger.error("Mesh file not found: %s", filename)
            missing_meshes.add(mesh_name)
            continue

        target_path = target_mesh_dir / sub_path
        if source_path is None or not source_path.exists():
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if str(target_path) not in processed_targets and source_path != target_path:
            try:
                if source_path.suffix.lower() == ".obj":
                    copy_obj_with_mtl(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)
                processed_targets.add(str(target_path))
                logger.debug("Copied mesh file: %s -> %s", source_path, target_path)
            except Exception as exc:
                logger.warning("Failed to copy mesh file %s to %s: %s", source_path, target_path, exc)

        mesh_file_paths[mesh_name] = target_path

    cleaned_mesh_assets = dict(mesh_assets)
    if missing_meshes:
        print(f"Remove non-existent mesh files from mesh_assets: {len(missing_meshes)}")
        print(missing_meshes)
        for mesh_name in missing_meshes:
            cleaned_mesh_assets.pop(mesh_name, None)

        geoms_to_remove = []
        for geom in mjcf_root.iter("geom"):
            geom_mesh_name = geom.attrib.get("mesh")
            if geom_mesh_name and geom_mesh_name in missing_meshes:
                geoms_to_remove.append(geom)

        parent_map = {child: parent for parent in mjcf_root.iter() for child in parent}
        for geom in geoms_to_remove:
            parent = parent_map.get(geom)
            if parent is not None:
                parent.remove(geom)

    return MeshCopyResult(mesh_assets=cleaned_mesh_assets, mesh_file_paths=mesh_file_paths)


def add_mesh_assets_to_xml(mjcf_root: ET.Element, mesh_assets: dict[str, str], *, urdf_dir: Path) -> None:
    """Add mesh asset XML entries under the MJCF asset section."""
    asset_elem = mjcf_root.find("asset")
    if asset_elem is None:
        asset_elem = ET.SubElement(mjcf_root, "asset")

    for mesh_name, filename in mesh_assets.items():
        if "package://" in filename:
            package_path = filename[len("package://") :]
            normalized_filename = f"meshes/{package_path}"
        elif filename.startswith("/"):
            normalized_filename = f"meshes/{os.path.relpath(filename, urdf_dir)}"
        else:
            normalized_filename = f"meshes/{filename}"

        ET.SubElement(asset_elem, "mesh", attrib={"name": mesh_name, "file": normalized_filename})

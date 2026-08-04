"""Helpers for building MJCF body trees from URDF links."""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from urdf_to_mjcf.conversion.assets import resolve_mesh_source_path
from urdf_to_mjcf.core.geometry import GeomElement, ParsedJointParams, rpy_to_quat
from urdf_to_mjcf.core.materials import get_obj_material_info, make_mjcf_material_name
from urdf_to_mjcf.core.model import JointMetadata

logger = logging.getLogger(__name__)

JOINT_METADATA_ATTRS = ("stiffness", "actuatorfrcrange", "margin", "armature", "damping", "frictionloss")


def apply_joint_metadata(attrib: dict[str, str], metadata: JointMetadata | None) -> None:
    if metadata is None:
        return
    for name in JOINT_METADATA_ATTRS:
        value = getattr(metadata, name)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            attrib[name] = " ".join(str(item) for item in value)
        else:
            attrib[name] = str(value)


def build_robot_body_tree(
    root_link_name: str,
    *,
    link_map: Mapping[str, ET.Element],
    parent_map: Mapping[str, list[tuple[str, ET.Element]]],
    collision_only: bool,
    materials: Mapping[str, object],
    mesh_assets: dict[str, str],
    workspace_search_paths: list[Path],
    urdf_dir: Path,
    joint_metadata: Mapping[str, JointMetadata] | None = None,
) -> tuple[ET.Element, list[ParsedJointParams]]:
    """Build the MJCF body hierarchy for a URDF robot."""

    movable_joints: list[ParsedJointParams] = []

    mesh_key_by_name: dict[str, str] = {}
    mesh_name_by_key: dict[str, str] = {}

    def mesh_scale_key(scale: str | None) -> str:
        if scale is None:
            return ""
        try:
            values = tuple(float(value) for value in scale.split())
        except ValueError:
            return " ".join(scale.split())
        if len(values) == 1:
            values = (values[0], values[0], values[0])
        if len(values) == 3:
            return " ".join(f"{value:g}" for value in values)
        return " ".join(scale.split())

    def mesh_asset_key(filename: str, *, mode: str, scale: str | None) -> str:
        source_path, sub_path = resolve_mesh_source_path(
            filename,
            urdf_dir=urdf_dir,
            workspace_search_paths=workspace_search_paths,
        )
        if source_path is not None:
            file_key = f"path:{source_path.expanduser().resolve(strict=False).as_posix()}"
        else:
            file_key = f"unresolved:{sub_path}"
        return f"{file_key}|mode:{mode}|scale:{mesh_scale_key(scale)}"

    def clean_mesh_name_part(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
        return cleaned or "mesh"

    def preferred_mesh_name(filename: str, prefix: str = "", link_prefix: str = "") -> str:
        name_parts = []
        if link_prefix:
            name_parts.append(clean_mesh_name_part(link_prefix))
        if prefix:
            name_parts.append(clean_mesh_name_part(prefix))
        stem = clean_mesh_name_part(Path(filename).stem)
        name_parts.append(stem)
        return "_".join(name_parts)

    def register_mesh_asset(filename: str, prefix: str = "", link_prefix: str = "", scale: str | None = None) -> str:
        mode = prefix or "visual"
        key = mesh_asset_key(filename, mode=mode, scale=scale)
        existing_name = mesh_name_by_key.get(key)
        if existing_name is not None:
            return existing_name

        base_name = preferred_mesh_name(filename, prefix=prefix, link_prefix=link_prefix)
        mesh_name = base_name
        existing_key = mesh_key_by_name.get(mesh_name)
        if mesh_name in mesh_assets and existing_key != key:
            suffix = hashlib.sha1(key.encode()).hexdigest()[:8]
            mesh_name = f"{base_name}_{suffix}"
            counter = 1
            while mesh_name in mesh_assets and mesh_key_by_name.get(mesh_name) != key:
                mesh_name = f"{base_name}_{suffix}_{counter}"
                counter += 1

        mesh_assets[mesh_name] = filename
        mesh_key_by_name[mesh_name] = key
        mesh_name_by_key[key] = mesh_name
        return mesh_name

    for existing_name, existing_filename in mesh_assets.items():
        existing_key = mesh_asset_key(existing_filename, mode="existing", scale=None)
        mesh_key_by_name[existing_name] = existing_key
        mesh_name_by_key.setdefault(existing_key, existing_name)

    def handle_geom_element(
        geom_elem: ET.Element | None, default_size: str, prefix: str = "", link_prefix: str = ""
    ) -> GeomElement:
        if geom_elem is None:
            return GeomElement(type="box", size=default_size, scale=None, mesh=None)

        box_elem = geom_elem.find("box")
        if box_elem is not None:
            size_str = box_elem.attrib.get("size", default_size)
            return GeomElement(
                type="box",
                size=" ".join(str(float(s) / 2) for s in size_str.split()),
            )

        cyl_elem = geom_elem.find("cylinder")
        if cyl_elem is not None:
            radius = cyl_elem.attrib.get("radius", "0.1")
            length = cyl_elem.attrib.get("length", "1")
            return GeomElement(
                type="cylinder",
                size=f"{radius} {float(length) / 2}",
            )

        sph_elem = geom_elem.find("sphere")
        if sph_elem is not None:
            radius = sph_elem.attrib.get("radius", "0.1")
            return GeomElement(
                type="sphere",
                size=radius,
            )

        mesh_elem = geom_elem.find("mesh")
        if mesh_elem is not None:
            filename = mesh_elem.attrib.get("filename")
            if filename is not None:
                scale = mesh_elem.attrib.get("scale")
                mesh_name = register_mesh_asset(filename, prefix=prefix, link_prefix=link_prefix, scale=scale)

                return GeomElement(
                    type="mesh",
                    size=None,
                    scale=scale,
                    mesh=mesh_name,
                )

        return GeomElement(
            type="box",
            size=default_size,
        )

    def build_body(link_name: str, joint: ET.Element | None = None) -> ET.Element:
        link = link_map[link_name]

        if joint is not None:
            origin_elem = joint.find("origin")
            if origin_elem is not None:
                pos = origin_elem.attrib.get("xyz", "0 0 0")
                quat = rpy_to_quat(origin_elem.attrib.get("rpy", "0 0 0"))
            else:
                pos = "0 0 0"
                quat = "1 0 0 0"
        else:
            pos = "0 0 0"
            quat = "1 0 0 0"

        body_attrib = {"name": link_name}
        if not np.allclose(np.array(list(map(float, pos.split()))), [0.0, 0.0, 0.0]):
            body_attrib["pos"] = pos
        if not np.allclose(list(map(float, quat.split())), [1.0, 0.0, 0.0, 0.0]):
            body_attrib["quat"] = quat
        body = ET.Element("body", attrib=body_attrib)

        if joint is not None:
            joint_type = joint.attrib.get("type", "fixed")
            if joint_type in ("revolute", "continuous", "prismatic"):
                joint_name = joint.attrib.get("name", f"{link_name}_joint")
                joint_attrib: dict[str, str] = {"name": joint_name}

                if joint_type in ("revolute", "continuous"):
                    joint_attrib["type"] = "hinge"
                else:
                    joint_attrib["type"] = "slide"

                limit = joint.find("limit")
                lower_num: float | None
                upper_num: float | None
                if limit is not None:
                    lower_val = limit.attrib.get("lower")
                    upper_val = limit.attrib.get("upper")
                    if lower_val is not None and upper_val is not None:
                        joint_attrib["range"] = f"{lower_val} {upper_val}"
                        lower_num = float(lower_val)
                        upper_num = float(upper_val)
                    else:
                        lower_num = upper_num = None
                else:
                    lower_num = upper_num = None

                axis_elem = joint.find("axis")
                if axis_elem is not None:
                    joint_attrib["axis"] = axis_elem.attrib.get("xyz", "0 0 1")
                metadata = joint_metadata.get(joint_name) if joint_metadata is not None else None
                apply_joint_metadata(joint_attrib, metadata)
                ET.SubElement(body, "joint", attrib=joint_attrib)

                movable_joints.append(
                    ParsedJointParams(
                        name=joint_name,
                        type=joint_attrib["type"],
                        lower=lower_num,
                        upper=upper_num,
                    )
                )

        inertial = link.find("inertial")
        if inertial is not None:
            inertial_elem = ET.Element("inertial")
            origin_inertial = inertial.find("origin")
            if origin_inertial is not None:
                inertial_elem.attrib["pos"] = origin_inertial.attrib.get("xyz", "0 0 0")
                rpy = origin_inertial.attrib.get("rpy", "0 0 0")
                if rpy != "0 0 0":
                    inertial_elem.attrib["quat"] = rpy_to_quat(rpy)
            else:
                inertial_elem.attrib["pos"] = "0 0 0"
                inertial_elem.attrib["quat"] = "1 0 0 0"
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                mass = mass_elem.attrib.get("value", "0")
                inertial_elem.attrib["mass"] = str(max(float(mass), 1e-6))
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                ixx = float(inertia_elem.attrib.get("ixx", "0"))
                ixy = float(inertia_elem.attrib.get("ixy", "0"))
                ixz = float(inertia_elem.attrib.get("ixz", "0"))
                iyy = float(inertia_elem.attrib.get("iyy", "0"))
                iyz = float(inertia_elem.attrib.get("iyz", "0"))
                izz = float(inertia_elem.attrib.get("izz", "0"))
                if abs(ixy) > 1e-6 or abs(ixz) > 1e-6 or abs(iyz) > 1e-6:
                    logger.info(
                        "Warning: off-diagonal inertia terms for link '%s' are nonzero and will be ignored.",
                        link_name,
                    )
                inertial_elem.attrib["diaginertia"] = f"{max(ixx, 1e-9)} {max(iyy, 1e-9)} {max(izz, 1e-9)}"
            body.append(inertial_elem)

        collisions = link.findall("collision")
        for idx, collision in enumerate(collisions):
            origin_collision = collision.find("origin")
            if origin_collision is not None:
                pos_geom = origin_collision.attrib.get("xyz", "0 0 0")
                quat_geom = rpy_to_quat(origin_collision.attrib.get("rpy", "0 0 0"))
            else:
                pos_geom = "0 0 0"
                quat_geom = "1 0 0 0"
            name = f"{link_name}_collision" if len(collisions) == 1 else f"{link_name}_collision_{idx}"

            collision_geom_attrib: dict[str, str] = {"name": name}
            if not np.allclose(np.array(list(map(float, pos_geom.split()))), [0.0, 0.0, 0.0]):
                collision_geom_attrib["pos"] = pos_geom
            if not np.allclose(list(map(float, quat_geom.split())), [1.0, 0.0, 0.0, 0.0]):
                collision_geom_attrib["quat"] = quat_geom

            collision_geom_elem = collision.find("geometry")
            if collision_geom_elem is not None:
                geom = handle_geom_element(collision_geom_elem, "1 1 1", prefix="collision", link_prefix=link_name)
                collision_geom_attrib["type"] = geom.type
                if geom.type == "mesh":
                    if geom.mesh is not None:
                        collision_geom_attrib["mesh"] = geom.mesh
                elif geom.size is not None:
                    collision_geom_attrib["size"] = geom.size
                if geom.scale is not None:
                    collision_geom_attrib["scale"] = geom.scale
            collision_geom_attrib["class"] = "collision"
            ET.SubElement(body, "geom", attrib=collision_geom_attrib)

        if not collision_only:
            visuals = link.findall("visual")
            for idx, visual in enumerate(visuals):
                origin_elem = visual.find("origin")
                if origin_elem is not None:
                    pos_geom = origin_elem.attrib.get("xyz", "0 0 0")
                    quat_geom = rpy_to_quat(origin_elem.attrib.get("rpy", "0 0 0"))
                else:
                    pos_geom = "0 0 0"
                    quat_geom = "1 0 0 0"

                visual_geom_elem = visual.find("geometry")
                if visual_geom_elem is not None:
                    geom = handle_geom_element(visual_geom_elem, "1 1 1", link_prefix=link_name)
                    name = f"{link_name}_visual" if len(visuals) == 1 else f"{link_name}_visual_{idx}"
                    visual_geom_attrib: dict[str, str] = {"name": name}
                    if not np.allclose(np.array(list(map(float, pos_geom.split()))), [0.0, 0.0, 0.0]):
                        visual_geom_attrib["pos"] = pos_geom
                    if not np.allclose(list(map(float, quat_geom.split())), [1.0, 0.0, 0.0, 0.0]):
                        visual_geom_attrib["quat"] = quat_geom
                    visual_geom_attrib["type"] = geom.type
                    if geom.type == "mesh" and geom.mesh is not None:
                        visual_geom_attrib["mesh"] = geom.mesh
                    elif geom.size is not None:
                        visual_geom_attrib["size"] = geom.size
                    if geom.scale is not None:
                        visual_geom_attrib["scale"] = geom.scale
                else:
                    logger.warning("No geometry element link_name=%s, use default attribute.", link_name)
                    geom = GeomElement(type="box", size="1 1 1")
                    name = f"{link_name}_visual" if len(visuals) == 1 else f"{link_name}_visual_{idx}"
                    visual_geom_attrib = {
                        "name": name,
                        "pos": pos_geom,
                        "quat": quat_geom,
                        "type": "box",
                        "size": "1 1 1",
                    }

                assigned_material = "default_material"
                material_elem = visual.find("material")
                if material_elem is not None:
                    material_name = material_elem.attrib.get("name")
                    if material_name and material_name in materials:
                        assigned_material = material_name

                if geom.type == "mesh" and geom.mesh is not None and assigned_material == "default_material":
                    obj_filename = mesh_assets.get(geom.mesh)
                    if obj_filename and obj_filename.lower().endswith(".obj"):
                        obj_file_path, material_source = resolve_mesh_source_path(
                            obj_filename,
                            urdf_dir=urdf_dir,
                            workspace_search_paths=workspace_search_paths,
                        )
                        if obj_file_path is not None:
                            has_single_material, material_name = get_obj_material_info(obj_file_path)
                            if has_single_material and material_name:
                                assigned_material = make_mjcf_material_name(material_source, material_name)
                                logger.info(
                                    "Assigned single OBJ material %s to geom %s",
                                    assigned_material,
                                    visual_geom_attrib["name"],
                                )

                visual_geom_attrib["material"] = assigned_material
                visual_geom_attrib["class"] = "visual"
                ET.SubElement(body, "geom", attrib=visual_geom_attrib)

        if link_name in parent_map:
            for child_name, child_joint in parent_map[link_name]:
                body.append(build_body(child_name, child_joint))

        return body

    return build_body(root_link_name), movable_joints

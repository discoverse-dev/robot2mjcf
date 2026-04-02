"""Helpers for building MJCF body trees from URDF links."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from robot2mjcf.core.geometry import GeomElement, ParsedJointParams, rpy_to_quat
from robot2mjcf.core.materials import get_obj_material_info
from robot2mjcf.core.model import ActuatorMetadata
from robot2mjcf.core.package_resolver import resolve_package_path

logger = logging.getLogger(__name__)


def build_robot_body_tree(
    root_link_name: str,
    *,
    link_map: Mapping[str, ET.Element],
    parent_map: Mapping[str, list[tuple[str, ET.Element]]],
    actuator_metadata: Mapping[str, ActuatorMetadata],
    collision_only: bool,
    materials: Mapping[str, object],
    mesh_assets: dict[str, str],
    workspace_search_paths: list[Path],
    urdf_dir: Path,
) -> tuple[ET.Element, list[ParsedJointParams]]:
    """Build the MJCF body hierarchy for a URDF robot."""

    actuator_joints: list[ParsedJointParams] = []

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
                mesh_name = Path(filename).stem
                if prefix:
                    mesh_name = f"{prefix}_{mesh_name}"
                if link_prefix:
                    mesh_name = f"{link_prefix}_{mesh_name}"
                if mesh_name not in mesh_assets:
                    mesh_assets[mesh_name] = filename

                return GeomElement(
                    type="mesh",
                    size=None,
                    scale=mesh_elem.attrib.get("scale"),
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

                if joint_name in actuator_metadata and actuator_metadata[joint_name].joint_class is not None:
                    joint_class_value = actuator_metadata[joint_name].joint_class
                    joint_attrib["class"] = str(joint_class_value)
                    logger.info("Joint %s assigned to class: %s", joint_name, joint_class_value)

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
                ET.SubElement(body, "joint", attrib=joint_attrib)

                actuator_joints.append(
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
                geom = handle_geom_element(collision_geom_elem, "1 1 1", prefix="collision")
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
                        if "package://" in obj_filename:
                            package_path = obj_filename[len("package://") :]
                            package_name = package_path.split("/")[0]
                            sub_path = "/".join(package_path.split("/")[1:])
                            try:
                                pkg_root = resolve_package_path(package_name, workspace_search_paths)
                                obj_file_path = pkg_root / sub_path if pkg_root else None
                            except Exception:
                                obj_file_path = None
                        elif obj_filename.startswith("/"):
                            obj_file_path = Path(obj_filename)
                        else:
                            obj_file_path = urdf_dir / obj_filename

                        if obj_file_path is not None:
                            has_single_material, material_name = get_obj_material_info(obj_file_path)
                            if has_single_material and material_name:
                                assigned_material = f"{obj_file_path.stem}_{material_name}"
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

    return build_body(root_link_name), actuator_joints

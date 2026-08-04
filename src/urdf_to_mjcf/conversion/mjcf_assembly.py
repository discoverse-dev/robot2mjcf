"""MJCF XML element builders, actuator assembly, and equality constraints.

Merged from: mjcf_builders.py + conversion_mjcf_assembly.py
"""

import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

from urdf_to_mjcf.core.geometry import ParsedJointParams
from urdf_to_mjcf.core.materials import Material
from urdf_to_mjcf.core.model import ActuatorConfig, ConversionMetadata, JointMetadata

logger = logging.getLogger(__name__)

ROBOT_CLASS = "robot"

MimicConstraint = tuple[str, str, float, float]


def _actuator_attributes(metadata: ActuatorConfig) -> dict[str, str]:
    """Serialize a joint's actuator configuration."""
    attributes = {name: str(value) for name in ("kp", "kv", "gear") if (value := getattr(metadata, name)) is not None}
    for name in ("ctrllimited", "forcelimited"):
        value = getattr(metadata, name)
        if value is not None:
            attributes[name] = "true" if value else "false"
    for name in ("ctrlrange", "forcerange"):
        value = getattr(metadata, name)
        if value is not None and len(value) == 2:
            attributes[name] = f"{value[0]} {value[1]}"
    return attributes


# ---------------------------------------------------------------------------
# MJCF root-level builders (from mjcf_builders.py)
# ---------------------------------------------------------------------------


def add_compiler(root: ET.Element) -> None:
    """Add a compiler element to the MJCF root.

    Args:
        root: The MJCF root element.
    """
    attrib = {
        "angle": "radian",
        "meshdir": ".",
        "balanceinertia": "true",
        # "eulerseq": "zyx",
        # "autolimits": "true",
    }

    element = ET.Element("compiler", attrib=attrib)
    existing_element = root.find("compiler")
    if isinstance(existing_element, ET.Element):
        root.remove(existing_element)
    root.insert(0, element)


def add_default(
    root: ET.Element,
    metadata: ConversionMetadata,
    collision_only: bool = False,
) -> None:
    """Add default settings with hierarchical structure for robot components."""
    default = ET.Element("default")

    robot_default = ET.SubElement(default, "default", attrib={"class": ROBOT_CLASS})

    # Visual geometry class
    if not collision_only:
        visual_default = ET.SubElement(
            robot_default,
            "default",
            attrib={"class": "visual"},
        )
        ET.SubElement(
            visual_default,
            "geom",
            attrib={
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )

    # Collision geometry class
    collision_default = ET.SubElement(
        robot_default,
        "default",
        attrib={"class": "collision"},
    )
    ET.SubElement(
        collision_default,
        "geom",
        attrib={
            "contype": str(metadata.collision_params.contype),
            "conaffinity": str(metadata.collision_params.conaffinity),
            "group": "3" if not collision_only else "2",
        },
    )

    # Add maxhullvert for efficient collising handling.
    if metadata.maxhullvert is not None:
        ET.SubElement(default, "mesh", attrib={"maxhullvert": str(metadata.maxhullvert)})

    # Replace existing default element if present
    existing_element = root.find("default")
    if isinstance(existing_element, ET.Element):
        root.remove(existing_element)
    root.insert(0, default)


def add_contact(root: ET.Element, robot: ET.Element) -> None:
    """Add a contact element to the MJCF root.

    For each pair of adjacent links that each have collision elements, we need
    to add an exclude tag to the contact element to make sure the links do not
    collide with each other.

    Args:
        root: The MJCF root element.
        robot: The URDF robot element.
    """
    links_with_collision: dict[str, ET.Element] = {}
    for link in robot.findall("link"):
        if link.find("collision") is not None and (name := link.attrib.get("name")) is not None:
            links_with_collision[name] = link

    contact: ET.Element | None = None
    for joint in robot.findall("joint"):
        parent_link = joint.find("parent")
        child_link = joint.find("child")
        if (
            parent_link is None
            or child_link is None
            or (parent_name := parent_link.attrib.get("link")) is None
            or (child_name := child_link.attrib.get("link")) is None
        ):
            continue

        if parent_name in links_with_collision and child_name in links_with_collision:
            if contact is None:
                contact = ET.SubElement(root, "contact")

            ET.SubElement(
                contact,
                "exclude",
                attrib={
                    "body1": parent_name,
                    "body2": child_name,
                },
            )


def add_weld_constraints(root: ET.Element, metadata: ConversionMetadata) -> None:
    """Add weld constraints to the MJCF root.

    Args:
        root: The MJCF root element.
        metadata: The conversion metadata containing weld constraints.
    """
    if not metadata.weld_constraints:
        return

    equality = ET.SubElement(root, "equality")
    for weld in metadata.weld_constraints:
        ET.SubElement(
            equality,
            "weld",
            attrib={
                "body1": weld.body1,
                "body2": weld.body2,
                "solimp": " ".join(f"{x:.6g}" for x in weld.solimp),
                "solref": " ".join(f"{x:.6g}" for x in weld.solref),
            },
        )


def add_option(root: ET.Element) -> None:
    """Add an option element to the MJCF root.

    Args:
        root: The MJCF root element.
    """
    ET.SubElement(
        root,
        "option",
        attrib={
            # "timestep": "0.001",
            # "gravity": "0 0 -9.81",
            # "density": "0",
            # "impratio": "20",
            # "viscosity": "0.00002",
            "integrator": "implicitfast",
            # "cone": "elliptic",
            # "jacobian": "auto",
            # "solver": "Newton",
            # "iterations": "100",
            # "tolerance": "1e-8",
        },
    )


def add_visual(root: ET.Element) -> None:
    """Add a visual element to the MJCF root.

    Args:
        root: The MJCF root element.
    """
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        attrib={"offwidth": "3840", "offheight": "2160"},
    )


def add_assets(root: ET.Element, materials: dict[str, str], mtl_materials: dict[str, Material] | None = None) -> None:
    """Add texture and material assets to the MJCF root.

    Args:
        root: The MJCF root element.
        materials: Dictionary mapping material names to RGBA color strings.
        mtl_materials: Dictionary mapping material names to MTL Material objects.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add MTL materials first (they take priority)
    if mtl_materials:
        for material in mtl_materials.values():
            material_attrib = {
                "name": material.name,
                # "specular": material.mjcf_specular(),
                # "shininess": material.mjcf_shininess(),
            }

            if material.map_Kd is not None:
                # Create texture asset for diffuse map
                texture_name = Path(material.map_Kd).stem
                ET.SubElement(
                    asset,
                    "texture",
                    attrib={
                        "type": "2d",
                        "name": texture_name,
                        "file": material.map_Kd,
                    },
                )
                # Reference the texture in the material
                material_attrib["texture"] = texture_name
            else:
                # Use RGBA if no texture
                material_attrib["rgba"] = material.mjcf_rgba()

            ET.SubElement(asset, "material", attrib=material_attrib)
            logger.info(f"Added MTL material: {material.name}")

    # Add materials from URDF (skip if already added from MTL)
    for name, rgba in materials.items():
        if name == "default_material":
            continue
        if mtl_materials and name in mtl_materials:
            continue  # Skip if already added from MTL
        ET.SubElement(
            asset,
            "material",
            attrib={
                "name": name,
                "rgba": rgba,
            },
        )

    # Add default material for visual elements without materials
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "default_material",
            "rgba": "0.7 0.7 0.7 1",
        },
    )


# ---------------------------------------------------------------------------
# Actuator and equality builders (from conversion_mjcf_assembly.py)
# ---------------------------------------------------------------------------


def add_actuators(
    root: ET.Element,
    movable_joints: Sequence[ParsedJointParams],
    joint_metadata: Mapping[str, JointMetadata],
) -> None:
    """Add actuators declared by joint metadata, preserving joint-data order."""
    actuator_elem = ET.SubElement(root, "actuator")
    available_joints = {joint.name for joint in movable_joints}

    for joint_name, metadata in joint_metadata.items():
        actuator = metadata.actuator
        if actuator is None or actuator.actuator_type is None:
            continue
        if joint_name not in available_joints:
            logger.info("Joint %s not found in converted joints", joint_name)
            continue

        attrib = {"joint": joint_name, **_actuator_attributes(actuator)}

        logger.info("Creating %s actuator for joint %s", actuator.actuator_type, joint_name)
        ET.SubElement(actuator_elem, actuator.actuator_type, attrib={"name": joint_name, **attrib})


def add_joint_sensors(
    root: ET.Element,
    joint_metadata: Mapping[str, JointMetadata],
    available_joints: Sequence[ParsedJointParams],
) -> None:
    """Add joint sensors described by joint metadata."""
    available_joint_names = {joint.name for joint in available_joints}
    sensor_joints = [
        joint_name
        for joint_name, metadata in joint_metadata.items()
        if metadata.sensors is not None and metadata.sensors.jointvel and joint_name in available_joint_names
    ]
    if not sensor_joints:
        return

    sensor_elem = root.find("sensor")
    if sensor_elem is None:
        sensor_elem = ET.SubElement(root, "sensor")
    for joint_name in sensor_joints:
        ET.SubElement(sensor_elem, "jointvel", attrib={"name": f"vel_{joint_name}", "joint": joint_name})


def add_mimic_equality_constraints(root: ET.Element, mimic_constraints: Sequence[MimicConstraint]) -> None:
    """Add equality constraints for mimic joints."""
    if not mimic_constraints:
        return

    equality_elem = ET.SubElement(root, "equality")
    for mimicked_joint, mimicking_joint, multiplier, offset in mimic_constraints:
        ET.SubElement(
            equality_elem,
            "joint",
            attrib={
                "joint1": mimicked_joint,
                "joint2": mimicking_joint,
                "polycoef": f"{offset} {multiplier} 0 0 0",
                "solimp": "0.95 0.99 0.001",
                "solref": "0.005 1",
            },
        )
        logger.info("Added equality constraint: %s = %s + %s * %s", mimicking_joint, offset, multiplier, mimicked_joint)

"""Core data models and foundational utilities."""

from robot2mjcf.core.geometry import (
    GeomElement,
    ParsedJointParams,
    build_transform,
    compute_min_z,
    format_value,
    mat_mult,
    parse_vector,
    quat_from_str,
    quat_to_rot,
    rpy_to_quat,
)
from robot2mjcf.core.materials import Material, copy_obj_with_mtl, get_obj_material_info, parse_mtl_name
from robot2mjcf.core.model import (
    ActuatorMetadata,
    Angle,
    CameraSensor,
    CollisionGeometry,
    CollisionParams,
    CollisionType,
    ConversionMetadata,
    DefaultJointMetadata,
    ExplicitFloorContacts,
    ForceSensor,
    ImuSensor,
    SiteMetadata,
    SiteType,
    TouchSensor,
    WeldConstraint,
    dActuator,
    dJoint,
)
from robot2mjcf.core.package_resolver import (
    PackageResolver,
    find_workspace_from_path,
    resolve_package_path,
    resolve_package_resource,
)
from robot2mjcf.core.utils import save_xml, sort_body_elements

__all__ = [
    # geometry
    "GeomElement",
    "ParsedJointParams",
    "build_transform",
    "compute_min_z",
    "format_value",
    "mat_mult",
    "parse_vector",
    "quat_from_str",
    "quat_to_rot",
    "rpy_to_quat",
    # materials
    "Material",
    "copy_obj_with_mtl",
    "get_obj_material_info",
    "parse_mtl_name",
    # model
    "ActuatorMetadata",
    "Angle",
    "CameraSensor",
    "CollisionGeometry",
    "CollisionParams",
    "CollisionType",
    "ConversionMetadata",
    "DefaultJointMetadata",
    "ExplicitFloorContacts",
    "ForceSensor",
    "ImuSensor",
    "SiteMetadata",
    "SiteType",
    "TouchSensor",
    "WeldConstraint",
    "dActuator",
    "dJoint",
    # package_resolver
    "PackageResolver",
    "find_workspace_from_path",
    "resolve_package_path",
    "resolve_package_resource",
    # utils
    "save_xml",
    "sort_body_elements",
]

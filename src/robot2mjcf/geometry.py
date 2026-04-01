"""Geometry processing and mathematical transformation utilities."""

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedJointParams:
    """Parsed joint parameters from URDF.

    Attributes:
        name: Joint name.
        type: Joint type (hinge, slide, etc.).
        lower: Lower joint limit, if any.
        upper: Upper joint limit, if any.
    """

    name: str
    type: str
    lower: float | None = None
    upper: float | None = None


@dataclass
class GeomElement:
    type: str
    size: str | None = None
    scale: str | None = None
    mesh: str | None = None


# 将数值格式化为最多保留4位小数，去除末尾的多余0和小数点，且将 -0 统一为 0
def format_value(val: float) -> str:
    """格式化数值为字符串。

    行为：
    - 使用四位小数精度进行四舍五入。
    - 删除末尾多余的零。
    - 若结果以小数点结尾则删除小数点。
    - 将 "-0" 或 "-0.0000" 规范为 "0"。

    Args:
        val: 要格式化的浮点数。

    Returns:
        处理后的字符串表示。
    """
    # 先用四位小数进行格式化（确保四舍五入）
    formatted = f"{val:.4f}"
    # 去除末尾的零
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    # 规范 -0 -> 0
    if formatted in ("-0", "-0.0", "-0.00", "-0.000", "-0.0000", ""):
        return "0"
    return formatted


def parse_vector(s: str) -> list[float]:
    """Convert a string of space-separated numbers to a list of floats.

    Args:
        s: Space-separated string of numbers (e.g., "1 2 3").

    Returns:
        List of parsed float values.
    """
    return list(map(float, s.split()))


def quat_from_str(s: str) -> list[float]:
    """Convert a quaternion string to a list of floats.

    Args:
        s: Space-separated string of quaternion values (w x y z).

    Returns:
        List of parsed quaternion values [w, x, y, z].
    """
    return list(map(float, s.split()))


def quat_to_rot(q: list[float]) -> list[list[float]]:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]


def build_transform(pos_str: str, quat_str: str) -> list[list[float]]:
    """Build a 4x4 homogeneous transformation matrix from position and quaternion strings.

    Args:
        pos_str: Space-separated string of position values (x y z).
        quat_str: Space-separated string of quaternion values (w x y z).

    Returns:
        A 4x4 homogeneous transformation matrix.
    """
    pos = parse_vector(pos_str)
    q = quat_from_str(quat_str)
    r_mat = quat_to_rot(q)
    transform = [
        [r_mat[0][0], r_mat[0][1], r_mat[0][2], pos[0]],
        [r_mat[1][0], r_mat[1][1], r_mat[1][2], pos[1]],
        [r_mat[2][0], r_mat[2][1], r_mat[2][2], pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return transform


def mat_mult(mat_a: list[list[float]], mat_b: list[list[float]]) -> list[list[float]]:
    """Multiply two 4x4 matrices A and B."""
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            result[i][j] = sum(mat_a[i][k] * mat_b[k][j] for k in range(4))
    return result


def compute_min_z(
    body: ET.Element,
    parent_transform: list[list[float]] | None = None,
    mesh_file_paths: dict[str, Path] | None = None,
    mesh_cache: dict[str, float] | None = None,
) -> float:
    """Recursively computes the minimum Z value in the world frame.

    This is used to compute the starting height of the robot.

    Args:
        body: The current body element.
        parent_transform: The transform of the parent body.
        mesh_file_paths: Dictionary mapping mesh names to their actual file paths.
        mesh_cache: Cache for mesh min_z values to avoid reloading.

    Returns:
        The minimum Z value in the world frame.
    """
    if parent_transform is None:
        parent_transform = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    if mesh_cache is None:
        mesh_cache = {}

    pos_str: str = body.attrib.get("pos", "0 0 0")
    quat_str: str = body.attrib.get("quat", "1 0 0 0")
    body_tf: list[list[float]] = mat_mult(parent_transform, build_transform(pos_str, quat_str))
    local_min_z: float = float("inf")

    for child in body:
        if child.tag == "geom":
            gpos_str: str = child.attrib.get("pos", "0 0 0")
            gquat_str: str = child.attrib.get("quat", "1 0 0 0")
            geom_tf: list[list[float]] = build_transform(gpos_str, gquat_str)
            total_tf: list[list[float]] = mat_mult(body_tf, geom_tf)

            # The translation part of T_total is in column 3.
            z: float = total_tf[2][3]
            geom_type: str = child.attrib.get("type", "")
            if geom_type == "box":
                size_vals: list[float] = list(map(float, child.attrib.get("size", "0 0 0").split()))
                half_height: float = size_vals[2] if len(size_vals) >= 3 else 0.0
                candidate: float = z - half_height
            elif geom_type == "cylinder":
                size_vals = list(map(float, child.attrib.get("size", "0 0").split()))
                half_length: float = size_vals[1] if len(size_vals) >= 2 else 0.0
                candidate = z - half_length
            elif geom_type == "sphere":
                r = float(child.attrib.get("size", "0"))
                candidate = z - r
            elif geom_type == "mesh":
                # Load mesh and compute actual min_z
                mesh_name = child.attrib.get("mesh")
                if mesh_name and mesh_file_paths and mesh_name in mesh_file_paths:
                    if mesh_name not in mesh_cache:
                        mesh_file_path = mesh_file_paths[mesh_name]
                        mesh_min_z = _compute_mesh_min_z(mesh_file_path, child.attrib.get("scale"))
                        mesh_cache[mesh_name] = mesh_min_z
                    else:
                        mesh_min_z = mesh_cache[mesh_name]

                    candidate = z + mesh_min_z
                else:
                    # Fallback to conservative estimate if mesh not available
                    candidate = z - 0.2
                    if mesh_name:
                        logger.warning(f"Mesh {mesh_name} not found in mesh_file_paths, using fallback estimate")
            else:
                candidate = z

            local_min_z = min(candidate, local_min_z)

        elif child.tag == "body":
            child_min: float = compute_min_z(child, body_tf, mesh_file_paths, mesh_cache)
            local_min_z = min(child_min, local_min_z)

    return local_min_z


def _compute_mesh_min_z(mesh_file_path: Path, scale_str: str | None = None) -> float:
    """Compute the minimum Z value from a mesh file.

    Args:
        mesh_file_path: Full path to the mesh file.
        scale_str: Optional scale string (e.g., "1 1 1").

    Returns:
        The minimum Z value in the mesh's local frame.
    """
    try:
        import trimesh
    except ImportError:
        logger.warning("trimesh not available, using fallback for mesh min_z computation")
        return -0.2

    if not mesh_file_path.exists():
        logger.warning(f"compute mesh z min: Mesh file not found: {mesh_file_path}")
        return 0.0

    logger.info(f"Loading mesh file: {mesh_file_path}")

    try:
        # Load the mesh
        mesh = trimesh.load(str(mesh_file_path), force="mesh")

        # Handle scale if provided
        if scale_str:
            scale_vals = list(map(float, scale_str.split()))
            if len(scale_vals) == 3:
                scale_matrix = [
                    [scale_vals[0], 0, 0, 0],
                    [0, scale_vals[1], 0, 0],
                    [0, 0, scale_vals[2], 0],
                    [0, 0, 0, 1],
                ]
                mesh.apply_transform(scale_matrix)
            elif len(scale_vals) == 1:
                mesh.apply_scale(scale_vals[0])

        # Get minimum Z coordinate from vertices
        if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            min_z = float(mesh.vertices[:, 2].min())
            logger.info(f"Computed min_z for mesh '{mesh_file_path.name}': {min_z}")
            return min_z
        else:
            logger.warning(f"Mesh '{mesh_file_path.name}' has no vertices")
            return 0.0

    except Exception as e:
        logger.warning(f"Failed to load mesh '{mesh_file_path.name}': {e}")
        return 0.0


def rpy_to_quat(rpy_str: str) -> str:
    """Convert roll, pitch, yaw angles (in radians) to a quaternion (w, x, y, z)."""
    try:
        r, p, y = map(float, rpy_str.split())
    except Exception:
        r, p, y = 0.0, 0.0, 0.0
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return f"{format_value(qw)} {format_value(qx)} {format_value(qy)} {format_value(qz)}"

#!/usr/bin/env python3
"""OBJ 网格位姿变换工具。

支持：
  - 平移 (--translate x y z)
  - 绕指定点按欧拉角旋转 (--euler rx ry rz --euler-order XYZ --pivot px py pz)
  - 绕指定点按四元数旋转 (--quat w x y z --pivot px py pz)
  - 旋转与平移可同时指定，旋转先于平移执行

保留 OBJ 文件中的 UV、法线、材质、顶点色等信息（文本级变换）。
当输出目录与输入目录不同时，自动将关联的 MTL 文件及其引用的贴图复制到输出目录。

用法示例：
  # 欧拉角旋转（默认 XYZ 顺序，角度单位 °）
  python transform_mesh.py input.obj -o output.obj --euler 0 0 90

  # 绕指定轴心点旋转
  python transform_mesh.py input.obj -o output.obj --euler 30 0 0 --pivot 1.0 2.0 0.5

  # 四元数旋转
  python transform_mesh.py input.obj -o output.obj --quat 0.7071 0 0 0.7071

  # 平移
  python transform_mesh.py input.obj -o output.obj --translate 1.0 0.0 -0.5

  # 旋转 + 平移（先旋转后平移）
  python transform_mesh.py input.obj -o output.obj --euler 0 90 0 --translate 0 0 1
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 旋转矩阵构建
# ---------------------------------------------------------------------------

def _rot_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=float)


def _rot_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


_AXIS_ROT = {"X": _rot_x, "Y": _rot_y, "Z": _rot_z}


def euler_to_matrix(rx: float, ry: float, rz: float, order: str = "XYZ") -> np.ndarray:
    """欧拉角（角度，单位°）→ 3×3 旋转矩阵。

    order: 旋转轴顺序字符串，如 'XYZ'、'ZYX'、'ZXZ' 等（最多 3 个轴，均为大写）。
    变换按从左到右的顺序依次右乘（外旋/固定轴约定）。
    """
    order = order.upper()
    if len(order) not in (2, 3) or not all(c in "XYZ" for c in order):
        raise ValueError(f"无效的欧拉角顺序 '{order}'，示例：'XYZ'、'ZYX'")

    angles = [rx, ry, rz]
    if len(order) == 2:
        angles = angles[:2]
    elif len(order) == 3:
        pass

    # 映射轴名称→角度
    axis_angles = {"X": math.radians(rx), "Y": math.radians(ry), "Z": math.radians(rz)}

    R = np.eye(3)
    for axis in order:
        R = R @ _AXIS_ROT[axis](axis_angles[axis])
    return R


def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """单位四元数 (w, x, y, z) → 3×3 旋转矩阵。"""
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("四元数模长为零")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)


# ---------------------------------------------------------------------------
# 4×4 齐次变换矩阵
# ---------------------------------------------------------------------------

def build_transform(
    R: np.ndarray | None,
    t: np.ndarray | None,
    pivot: np.ndarray | None,
) -> np.ndarray:
    """构造 4×4 齐次变换矩阵。

    执行顺序：
      1. 平移到原点（减去 pivot）
      2. 旋转 R
      3. 平移回 pivot
      4. 再加上平移向量 t
    """
    T = np.eye(4)

    if R is not None:
        p = pivot if pivot is not None else np.zeros(3)
        # T = translate(p) @ rotate(R) @ translate(-p)
        T[:3, :3] = R
        T[:3, 3] = p - R @ p  # = -R@(-p) + p 简化

    if t is not None:
        T[:3, 3] += t

    return T


# ---------------------------------------------------------------------------
# OBJ 文本级变换（保留 UV/法线/材质等）
# ---------------------------------------------------------------------------

_VERTEX_RE = re.compile(
    r"^(v)\s+([-+]?\S+)\s+([-+]?\S+)\s+([-+]?\S+)(.*)"
)
_NORMAL_RE = re.compile(
    r"^(vn)\s+([-+]?\S+)\s+([-+]?\S+)\s+([-+]?\S+)(.*)"
)


def _fmt(v: float) -> str:
    """格式化浮点数，去除多余零。"""
    return f"{v:.10g}"


def transform_mesh_text(lines: list[str], T: np.ndarray) -> list[str]:
    """对 OBJ 文本行进行变换，仅修改 v/vn 行，其余原样保留。"""
    R3 = T[:3, :3]
    t3 = T[:3, 3]

    # 法线变换使用逆转置（对于纯旋转即 R 本身）
    R_normal = np.linalg.inv(R3).T

    out: list[str] = []
    for line in lines:
        stripped = line.rstrip("\n\r")

        m = _VERTEX_RE.match(stripped)
        if m:
            x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
            nx, ny, nz = R3 @ np.array([x, y, z]) + t3
            tail = m.group(5)  # 可能含顶点色 r g b
            # 顶点色不参与变换，原样保留
            out.append(f"v {_fmt(nx)} {_fmt(ny)} {_fmt(nz)}{tail}\n")
            continue

        m = _NORMAL_RE.match(stripped)
        if m:
            x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
            nx, ny, nz = R_normal @ np.array([x, y, z])
            # 重新归一化法线
            length = math.sqrt(nx*nx + ny*ny + nz*nz)
            if length > 1e-12:
                nx, ny, nz = nx / length, ny / length, nz / length
            tail = m.group(5)
            out.append(f"vn {_fmt(nx)} {_fmt(ny)} {_fmt(nz)}{tail}\n")
            continue

        out.append(line if line.endswith("\n") else line + "\n")

    return out


# ---------------------------------------------------------------------------
# 材质迁移（MTL + 贴图）
# ---------------------------------------------------------------------------

# MTL 文件中引用贴图的指令前缀（支持带选项的写法，如 map_Kd -s 1 1 tex.png）
_MTL_MAP_DIRECTIVES = re.compile(
    r"^\s*(map_Ka|map_Kd|map_Ks|map_Ns|map_d|map_bump|bump|disp|norm|refl"
    r"|map_Ke|map_Pr|map_Pm|map_Pc|map_Pcr|map_aniso|map_anisor)\s+(.*)",
    re.IGNORECASE,
)


def _collect_mtllib_names(obj_lines: list[str]) -> list[str]:
    """从 OBJ 行中提取所有 mtllib 引用的文件名列表。"""
    names: list[str] = []
    for line in obj_lines:
        stripped = line.strip()
        if stripped.lower().startswith("mtllib"):
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                names.append(parts[1].strip())
    return names


def _collect_texture_paths_from_mtl(mtl_path: Path) -> list[str]:
    """解析 MTL 文件，返回其中引用的贴图相对路径列表。"""
    textures: list[str] = []
    if not mtl_path.exists():
        return textures
    with open(mtl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _MTL_MAP_DIRECTIVES.match(line)
            if m:
                rest = m.group(2).strip()
                # 选项以 '-' 开头，贴图文件名是最后一个 token
                tokens = rest.split()
                if tokens:
                    textures.append(tokens[-1])
    return textures


def transfer_materials(
    obj_lines: list[str],
    input_dir: Path,
    output_dir: Path,
    output_stem: str,
) -> str | None:
    """将 OBJ 引用的所有 MTL 文件合并为 <output_stem>.mtl 写入 output_dir，
    并将其中引用的贴图复制到 output_dir（保持相对目录结构）。

    返回新 MTL 文件名（如 "mesh_new.mtl"），若无 MTL 引用则返回 None。
    input_dir 与 output_dir 相同时同样生效（会生成新名称的 MTL）。
    """
    mtl_names = _collect_mtllib_names(obj_lines)
    if not mtl_names:
        return None

    new_mtl_name = output_stem + ".mtl"
    dst_mtl = output_dir / new_mtl_name
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_lines: list[str] = []
    for mtl_name in mtl_names:
        src_mtl = input_dir / mtl_name
        if not src_mtl.exists():
            print(f"  [材质] 警告：找不到 MTL 文件 '{src_mtl}'，跳过。")
            continue

        with open(src_mtl, "r", encoding="utf-8", errors="replace") as f:
            mtl_file_lines = f.readlines()

        # 多文件合并时确保段落间有空行
        if merged_lines and not merged_lines[-1].endswith("\n"):
            merged_lines.append("\n")
        merged_lines.extend(mtl_file_lines)

        # 收集并复制贴图
        for tex_rel in _collect_texture_paths_from_mtl(src_mtl):
            src_tex = (src_mtl.parent / tex_rel).resolve()
            try:
                rel = src_tex.relative_to(input_dir.resolve())
                dst_tex = output_dir / rel
            except ValueError:
                dst_tex = output_dir / src_tex.name

            if not src_tex.exists():
                print(f"  [材质] 警告：找不到贴图 '{src_tex}'，跳过。")
                continue
            if dst_tex.resolve() == src_tex.resolve():
                continue

            dst_tex.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_tex, dst_tex)
            print(f"  [材质] 已复制贴图：{src_tex.name} → {dst_tex}")

    if not merged_lines:
        return None

    with open(dst_mtl, "w", encoding="utf-8") as f:
        f.writelines(merged_lines)
    print(f"  [材质] 已写入 MTL：{dst_mtl}（合并自 {mtl_names}）")

    return new_mtl_name


def rewrite_mtllib_lines(out_lines: list[str], new_mtl_name: str) -> list[str]:
    """将 OBJ 输出行中所有 mtllib 指令替换为 new_mtl_name，去重只保留首次出现。"""
    result: list[str] = []
    inserted = False
    for line in out_lines:
        if line.strip().lower().startswith("mtllib"):
            if not inserted:
                result.append(f"mtllib {new_mtl_name}\n")
                inserted = True
            # 丢弃后续重复的 mtllib 行（多 MTL 已合并）
        else:
            result.append(line)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OBJ 网格位姿变换（旋转 + 平移）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="输入 OBJ 文件路径")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="输出 OBJ 文件路径（默认在原文件名后追加 _new）")

    # 旋转参数组
    rot_group = parser.add_mutually_exclusive_group()
    rot_group.add_argument(
        "--euler", nargs=3, type=float, metavar=("RX", "RY", "RZ"),
        help="欧拉角旋转，单位°（配合 --euler-order 指定顺序，默认 XYZ）",
    )
    rot_group.add_argument(
        "--quat", nargs=4, type=float, metavar=("W", "X", "Y", "Z"),
        help="四元数旋转 (w x y z)，自动归一化",
    )

    parser.add_argument(
        "--euler-order", default="XYZ", metavar="ORDER",
        help="欧拉角旋转顺序，如 XYZ / ZYX / ZXZ（默认 XYZ）",
    )
    parser.add_argument(
        "--pivot", nargs=3, type=float, metavar=("PX", "PY", "PZ"),
        default=None,
        help="旋转轴心点坐标（默认坐标原点 0 0 0）",
    )

    # 平移参数
    parser.add_argument(
        "--translate", nargs=3, type=float, metavar=("TX", "TY", "TZ"),
        default=None,
        help="平移向量 (tx ty tz)，单位：m",
    )

    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印变换矩阵，不写入文件",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- 构建旋转矩阵 ---
    R: np.ndarray | None = None
    if args.euler is not None:
        rx, ry, rz = args.euler
        R = euler_to_matrix(rx, ry, rz, args.euler_order)
        print(f"[旋转] 欧拉角 ({rx}°, {ry}°, {rz}°)，顺序 {args.euler_order.upper()}")
    elif args.quat is not None:
        w, x, y, z = args.quat
        R = quat_to_matrix(w, x, y, z)
        print(f"[旋转] 四元数 (w={w}, x={x}, y={y}, z={z})")

    # --- 构建平移向量 ---
    t: np.ndarray | None = None
    if args.translate is not None:
        t = np.array(args.translate, dtype=float)
        print(f"[平移] ({t[0]}, {t[1]}, {t[2]})")

    if R is None and t is None:
        print("警告：未指定任何变换参数（--euler / --quat / --translate），输出与输入相同。")

    # --- 轴心点 ---
    pivot: np.ndarray | None = None
    if args.pivot is not None:
        pivot = np.array(args.pivot, dtype=float)
        print(f"[轴心] ({pivot[0]}, {pivot[1]}, {pivot[2]})")

    # --- 组合 4×4 变换矩阵 ---
    T = build_transform(R, t, pivot)
    print(f"\n[变换矩阵 4×4]\n{np.array2string(T, precision=6, suppress_small=True)}\n")

    if args.dry_run:
        print("--dry-run 模式，跳过文件写入。")
        return

    # --- 读取 OBJ ---
    input_path: Path = args.input
    if not input_path.exists():
        print(f"错误：找不到文件 '{input_path}'", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # --- 变换 ---
    out_lines = transform_mesh_text(lines, T)

    # --- 确定输出路径 ---
    if args.output is not None:
        output_path: Path = args.output
    else:
        output_path = input_path.with_stem(input_path.stem + "_new")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 材质迁移（MTL 合并 + 贴图复制）并更新 mtllib 引用 ---
    new_mtl_name = transfer_materials(
        lines, input_path.parent, output_path.parent, output_path.stem
    )
    if new_mtl_name:
        out_lines = rewrite_mtllib_lines(out_lines, new_mtl_name)

    # --- 写出 OBJ ---
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"已写入：{output_path}")


if __name__ == "__main__":
    main()

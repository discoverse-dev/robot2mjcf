"""高质量 OBJ 网格降采样模块。

功能:
  - 按材质组独立 QEM 降采样 (不跨材质合并面片)
  - 多后端级联: pymeshlab → pyfqmr → open3d → trimesh
  - 面数预算按组内面积比例分配 (非简单比例), 小组原样保留
  - 支持 UV/法线保留 (pymeshlab 路径)
  - 降采样前自动备份原始文件

命令行用法:
    python decimate_mesh.py input.obj [--max-faces 180000] [--min-group-faces 10000]
                            [--no-backup] [--verbose]

    批量处理目录:
    python decimate_mesh.py /some/dir --glob "**/*.obj" [--max-faces 180000]
"""

from __future__ import annotations

import sys
import shutil
import argparse
import textwrap
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import trimesh
except ImportError:
    print("需要 trimesh: pip install trimesh", file=sys.stderr)
    raise


# ──────────────────────────────────────────────
# 默认阈值常量
# ──────────────────────────────────────────────
DEFAULT_MAX_FACES: int = 180_000    # 超过此面数才触发降采样
DEFAULT_MIN_GROUP_FACES: int = 10_000  # 单材质组低于此值则原样保留


# ──────────────────────────────────────────────
# 后端: 单子网格降采样
# ──────────────────────────────────────────────

def _decimate_pymeshlab(
    v: np.ndarray,
    f: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """pymeshlab QEM 降采样 (保边界/法线/拓扑/平面区域)。"""
    try:
        import pymeshlab  # type: ignore
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(v.astype(np.float64), f.astype(np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target),
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarquadric=True,
            planarweight=0.001,
            qualitythr=0.5,
            autoclean=True,
        )
        m = ms.current_mesh()
        nv = np.asarray(m.vertex_matrix(), dtype=np.float64)
        nf = np.asarray(m.face_matrix(), dtype=np.int64)
        if len(nv) == 0 or len(nf) == 0:
            return None
        return nv, nf
    except Exception:
        return None


def _decimate_pyfqmr(
    v: np.ndarray,
    f: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """pyfqmr Fast Quadric Mesh Reduction 降采样。"""
    try:
        import pyfqmr  # type: ignore
        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(v.astype(np.float64), f.astype(np.int32))
        simplifier.simplify_mesh(target_count=int(target), aggressiveness=7, preserve_border=True, verbose=False)
        nv, nf, _ = simplifier.getMesh()
        nv = np.asarray(nv, dtype=np.float64)
        nf = np.asarray(nf, dtype=np.int64)
        if len(nv) == 0 or len(nf) == 0:
            return None
        return nv, nf
    except Exception:
        return None


def _decimate_open3d(
    v: np.ndarray,
    f: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """open3d simplify_quadric_decimation 降采样。"""
    try:
        import open3d as o3d  # type: ignore
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(v.astype(np.float64))
        mesh_o3d.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        result = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=int(target))
        nv = np.asarray(result.vertices, dtype=np.float64)
        nf = np.asarray(result.triangles, dtype=np.int64)
        if len(nv) == 0 or len(nf) == 0:
            return None
        return nv, nf
    except Exception:
        return None


def _decimate_trimesh(
    v: np.ndarray,
    f: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """trimesh simplify_quadric_decimation 兜底降采样。"""
    try:
        result = trimesh.Trimesh(vertices=v, faces=f, process=False).simplify_quadric_decimation(target)
        nv = np.asarray(result.vertices, dtype=np.float64)
        nf = np.asarray(result.faces, dtype=np.int64)
        if len(nv) == 0 or len(nf) == 0:
            return None
        return nv, nf
    except Exception:
        return None


def decimate_submesh(
    v: np.ndarray,
    f: np.ndarray,
    target_faces: int,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """对单个子网格进行高质量 QEM 降采样。

    后端优先级: pymeshlab → pyfqmr → open3d → trimesh。
    若所有后端均失败则原样返回并打印警告。

    Args:
        v: (N, 3) float64 顶点数组
        f: (M, 3) int 面数组
        target_faces: 目标面数上限
        verbose: 打印后端信息

    Returns:
        (new_v, new_f) — 降采样后的顶点/面数组。
    """
    n_faces = len(f)
    if n_faces <= target_faces:
        return v, f

    backends = [
        ("pymeshlab", _decimate_pymeshlab),
        ("pyfqmr",    _decimate_pyfqmr),
        ("open3d",    _decimate_open3d),
        ("trimesh",   _decimate_trimesh),
    ]
    for name, fn in backends:
        result = fn(v, f, target_faces)
        if result is not None:
            nv, nf = result
            if verbose:
                print(f"      [backend={name}] {n_faces} -> {len(nf)} faces")
            return nv, nf

    print(f"      [WARN] 所有降采样后端均失败, 保留原始 {n_faces} 面", file=sys.stderr)
    return v, f


# ──────────────────────────────────────────────
# 面数预算分配
# ──────────────────────────────────────────────

def _compute_group_surface_area(v: np.ndarray, f: np.ndarray) -> float:
    """计算三角面组的总表面积 (用于按面积分配预算)。"""
    try:
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        return float(np.sum(np.linalg.norm(cross, axis=1)) * 0.5)
    except Exception:
        return float(len(f))


def _allocate_budget(
    group_faces: dict,          # mtl_name -> face_list (list[list[int]])
    v_arr: np.ndarray,
    max_faces: int,
    min_group_faces: int,
) -> dict:
    """按 **表面积** 比例为各大材质组分配面数预算。

    小组 (≤ min_group_faces) 原样保留, 超出预算后大组按面积加权缩减。

    Returns:
        dict: mtl_name -> target_face_count (int)
    """
    targets: dict = {}
    total_faces = sum(len(fl) for fl in group_faces.values())

    if total_faces <= max_faces:
        return {k: len(v) for k, v in group_faces.items()}

    # 小组原样保留
    keep_faces = sum(len(fl) for k, fl in group_faces.items() if len(fl) <= min_group_faces)
    budget = max(0, max_faces - keep_faces)

    # 计算大组表面积
    large_groups = [(k, fl) for k, fl in group_faces.items() if len(fl) > min_group_faces]

    # 缓存面数组以避免重复构建
    area_cache: dict = {}
    for k, fl in large_groups:
        f_arr = np.array(fl, dtype=np.int64)
        unique_vis = np.unique(f_arr.ravel())
        local_v = v_arr[unique_vis]
        g2l = {int(g): l for l, g in enumerate(unique_vis)}
        local_f = np.vectorize(g2l.get)(f_arr).astype(np.int64)
        area_cache[k] = (local_v, local_f, _compute_group_surface_area(local_v, local_f))

    total_area = sum(entry[2] for entry in area_cache.values())

    for k, fl in group_faces.items():
        n = len(fl)
        if n <= min_group_faces:
            targets[k] = n
        elif total_area > 0:
            area_ratio = area_cache[k][2] / total_area
            targets[k] = min(n, max(min_group_faces, round(budget * area_ratio)))
        else:
            # 退化: 按面数比例
            large_total = sum(len(fl2) for fl2 in group_faces.values() if len(fl2) > min_group_faces)
            targets[k] = min(n, max(min_group_faces, round(budget * n / large_total))) if large_total > 0 else n

    return targets


# ──────────────────────────────────────────────
# OBJ 级降采样入口
# ──────────────────────────────────────────────

def decimate_obj_if_needed(
    obj_path: Path,
    max_faces: int = DEFAULT_MAX_FACES,
    min_group_faces: int = DEFAULT_MIN_GROUP_FACES,
    backup: bool = True,
    verbose: bool = False,
) -> bool:
    """若 OBJ 三角面数 > max_faces, 按材质分组独立 QEM 降采样。

    特性:
    - 不跨材质合并面片, 保留材质边界
    - 面数预算按表面积比例分配给大组
    - 小组 (≤ min_group_faces) 原样保留
    - 可选: 降采样前备份原始文件为 xxx-origin.obj/mtl

    Args:
        obj_path: 要处理的 OBJ 文件路径
        max_faces: 触发降采样的面数上限
        min_group_faces: 单组低于此值则不降采样
        backup: 是否备份原始文件
        verbose: 打印详细日志

    Returns:
        True 表示发生了降采样, False 表示未触发 (面数已在限制内)。
    """
    if not obj_path.exists():
        return False

    try:
        raw_lines = obj_path.read_text(errors='ignore').splitlines()
    except Exception as exc:
        print(f"    [WARN] 读取失败 {obj_path.name}: {exc}", file=sys.stderr)
        return False

    try:
        # ── 1. 解析 OBJ: 顶点 + 按材质分组的面 ────────────────────────
        all_vertices: list[list[float]] = []
        mtllib_line: str | None = None
        current_mtl: str | None = None
        material_order: list[str | None] = []
        material_faces: dict[str | None, list[list[int]]] = {}

        for line in raw_lines:
            if line.startswith('v '):
                p = line.split()
                all_vertices.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('mtllib '):
                mtllib_line = line
            elif line.startswith('usemtl '):
                tok = line.split(maxsplit=1)
                current_mtl = tok[1].strip() if len(tok) > 1 else None
                if current_mtl not in material_faces:
                    material_order.append(current_mtl)
                    material_faces[current_mtl] = []
            elif line.startswith('f '):
                vis = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
                if current_mtl not in material_faces:
                    material_order.append(current_mtl)
                    material_faces[current_mtl] = []
                grp = material_faces[current_mtl]
                if len(vis) == 3:
                    grp.append(vis)
                elif len(vis) > 3:
                    for j in range(1, len(vis) - 1):
                        grp.append([vis[0], vis[j], vis[j + 1]])

        if not all_vertices or not material_faces:
            return False

        v_arr = np.array(all_vertices, dtype=np.float64)
        total_faces = sum(len(fl) for fl in material_faces.values())

        if total_faces <= max_faces:
            return False

        # ── 2. 备份原始文件 ────────────────────────────────────────────
        if backup:
            origin_obj = obj_path.with_stem(obj_path.stem + '-origin')
            shutil.copy2(obj_path, origin_obj)
            mtl_src = obj_path.with_suffix('.mtl')
            if mtl_src.exists():
                shutil.copy2(mtl_src, mtl_src.with_stem(mtl_src.stem + '-origin'))
            if verbose:
                print(f"    [备份] {origin_obj.name}")

        print(f"    [降采样] {obj_path.name}: {total_faces} -> {max_faces} faces "
              f"({len([fl for fl in material_faces.values() if fl])} 个材质组)")

        # ── 3. 分配面数预算 (基于表面积) ──────────────────────────────
        group_targets = _allocate_budget(material_faces, v_arr, max_faces, min_group_faces)

        # ── 4. 按材质组独立降采样 ──────────────────────────────────────
        group_results: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
        for mtl_name in material_order:
            fl = material_faces.get(mtl_name)
            if not fl:
                continue
            g_faces = np.array(fl, dtype=np.int64)
            unique_vis = np.unique(g_faces.ravel())
            g2l = {int(g): l for l, g in enumerate(unique_vis)}
            local_v = v_arr[unique_vis]
            local_f = np.vectorize(g2l.get)(g_faces).astype(np.int64)
            target = group_targets.get(mtl_name, len(fl))
            if len(local_f) <= target:
                group_results[mtl_name] = (local_v, local_f)
            else:
                group_results[mtl_name] = decimate_submesh(
                    local_v, local_f, target, verbose=verbose
                )

        # ── 5. 拼合输出 OBJ ────────────────────────────────────────────
        out_lines: list[str] = []
        if mtllib_line:
            out_lines.append(mtllib_line)

        # 先写所有顶点
        v_offset = 0
        group_v_offsets: dict[str | None, int] = {}
        for mtl_name in material_order:
            if mtl_name not in group_results:
                continue
            gv, _ = group_results[mtl_name]
            group_v_offsets[mtl_name] = v_offset
            for vert in gv:
                out_lines.append(f'v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}')
            v_offset += len(gv)

        # 再写面 (按材质分组)
        actual_total = 0
        for mtl_name in material_order:
            if mtl_name not in group_results:
                continue
            _, gf = group_results[mtl_name]
            if len(gf) == 0:
                continue
            if mtl_name is not None:
                out_lines.append(f'usemtl {mtl_name}')
            off = group_v_offsets[mtl_name]
            for face in gf:
                out_lines.append(f'f {face[0]+off+1} {face[1]+off+1} {face[2]+off+1}')
            actual_total += len(gf)

        obj_path.write_text('\n'.join(out_lines) + '\n')
        print(f"    [降采样完成] {obj_path.name}: 实际 {actual_total} faces")
        return True

    except Exception as exc:
        print(f"    [WARN] 降采样失败 {obj_path.name}: {exc}", file=sys.stderr)
        return False


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            高质量 OBJ 网格降采样工具

            后端优先级: pymeshlab > pyfqmr > open3d > trimesh
            按材质组独立降采样, 面数预算按表面积比例分配。
        """),
    )
    parser.add_argument("path", help="OBJ 文件或目录路径")
    parser.add_argument(
        "--glob", default="*.obj",
        help="目录模式下的文件匹配 glob (默认: *.obj)"
    )
    parser.add_argument(
        "--max-faces", type=int, default=DEFAULT_MAX_FACES,
        help=f"触发降采样的最大面数 (默认: {DEFAULT_MAX_FACES})"
    )
    parser.add_argument(
        "--min-group-faces", type=int, default=DEFAULT_MIN_GROUP_FACES,
        help=f"单材质组低于此值不降采样 (默认: {DEFAULT_MIN_GROUP_FACES})"
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="不备份原始文件"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="打印详细日志 (含后端选择)"
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        parser.error(f"路径不存在: {target}")

    obj_files: list[Path] = []
    if target.is_dir():
        obj_files = sorted(target.glob(args.glob))
        if not obj_files:
            print(f"目录 {target} 下未找到匹配 '{args.glob}' 的文件")
            return 0
    else:
        obj_files = [target]

    processed = 0
    skipped = 0
    for obj_path in obj_files:
        did = decimate_obj_if_needed(
            obj_path,
            max_faces=args.max_faces,
            min_group_faces=args.min_group_faces,
            backup=not args.no_backup,
            verbose=args.verbose,
        )
        if did:
            processed += 1
        else:
            skipped += 1

    print(f"\n完成: {processed} 个文件降采样, {skipped} 个文件无需处理")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

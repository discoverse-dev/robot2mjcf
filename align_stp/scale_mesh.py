"""高精度等比例网格缩放模块。

功能:
  - 等比例缩放 (X/Y/Z 相同比例), 不引入形变
  - 三种缩放模式:
      factor   -- 直接指定缩放系数 (如 0.001 = mm→m)
      fit      -- 缩放到目标包围盒最长边尺寸
      volume   -- 缩放到目标体积 (取体积比的立方根)
  - 三种缩放中心 (pivot):
      origin   -- 世界原点 (0,0,0), 不偏移顶点分布
      centroid -- 几何质心为中心 (缩放后质心保持不变)
      bbox     -- 包围盒中心为中心
  - OBJ 文本级处理: 仅修改 v 行, UV/法线/面索引/材质完整保留
  - 非 OBJ 格式通过 trimesh 加载后导出
  - 支持单文件或整目录批量处理
  - 缩放前自动备份原始文件 (可关闭)

命令行用法:
    # mm → m 单位转换
    python scale_mesh.py model.obj --factor 0.001

    # 缩放到最长边 = 0.5 m
    python scale_mesh.py model.obj --fit 0.5

    # 批量缩放目录
    python scale_mesh.py ./meshes --factor 0.001 --glob "**/*.obj"

    # 以包围盒中心为缩放原点, 不备份
    python scale_mesh.py model.obj --factor 2.0 --pivot bbox --no-backup
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
# 内部工具
# ──────────────────────────────────────────────

def _parse_vertices_from_obj(lines: list[str]) -> np.ndarray:
    """从 OBJ 文本行列表中提取所有顶点, 返回 (N, 3) float64。"""
    verts: list[list[float]] = []
    for line in lines:
        if line.startswith('v '):
            p = line.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts, dtype=np.float64) if verts else np.empty((0, 3), dtype=np.float64)


def _compute_pivot(verts: np.ndarray, pivot: str) -> np.ndarray:
    """计算缩放中心点。

    Args:
        verts: (N, 3) 顶点数组
        pivot: 'origin' | 'centroid' | 'bbox'

    Returns:
        (3,) 缩放中心坐标
    """
    if pivot == 'origin' or len(verts) == 0:
        return np.zeros(3, dtype=np.float64)
    if pivot == 'centroid':
        return verts.mean(axis=0)
    if pivot == 'bbox':
        return 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    return np.zeros(3, dtype=np.float64)


def _resolve_factor(
    verts: np.ndarray,
    mode: str,
    factor: float | None,
    fit_size: float | None,
    target_volume: float | None,
) -> float:
    """根据模式解算最终缩放系数。"""
    if mode == 'factor':
        assert factor is not None
        return float(factor)

    if mode == 'fit':
        assert fit_size is not None
        if len(verts) == 0:
            return 1.0
        extents = verts.max(axis=0) - verts.min(axis=0)
        longest = float(extents.max())
        if longest < 1e-12:
            return 1.0
        return float(fit_size) / longest

    if mode == 'volume':
        assert target_volume is not None
        if len(verts) == 0:
            return 1.0
        extents = verts.max(axis=0) - verts.min(axis=0)
        current_vol = float(extents[0] * extents[1] * extents[2])
        if current_vol < 1e-30:
            return 1.0
        return (float(target_volume) / current_vol) ** (1.0 / 3.0)

    raise ValueError(f"未知缩放模式: {mode}")


# ──────────────────────────────────────────────
# OBJ 文本级缩放 (高精度核心路径)
# ──────────────────────────────────────────────

def _scale_obj_text(
    lines: list[str],
    scale: float,
    pivot: np.ndarray,
    decimal_places: int = 8,
) -> list[str]:
    """对 OBJ 文本行列表做等比例缩放, 仅修改 v 行。

    UV (vt)、法线 (vn)、面索引 (f)、材质引用 (usemtl/mtllib) 完整保留。
    等比例缩放时法线方向不变, 无需重新归一化。

    Args:
        lines: OBJ 文件按行分割的列表
        scale: 缩放系数
        pivot: (3,) 缩放中心点
        decimal_places: 顶点坐标输出精度 (默认 8 位, 远高于 float32 精度)

    Returns:
        缩放后的文本行列表
    """
    fmt = f'{{:.{decimal_places}f}}'
    px, py, pz = pivot

    out: list[str] = []
    for line in lines:
        if line.startswith('v ') and not line.startswith('vt ') and not line.startswith('vn '):
            p = line.split()
            x = (float(p[1]) - px) * scale + px
            y = (float(p[2]) - py) * scale + py
            z = (float(p[3]) - pz) * scale + pz
            suffix = '  ' + ' '.join(p[4:]) if len(p) > 4 else ''  # 保留颜色/扩展属性
            out.append(f'v {fmt.format(x)} {fmt.format(y)} {fmt.format(z)}{suffix}')
        else:
            out.append(line)
    return out


# ──────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────

def scale_obj(
    obj_path: Path,
    *,
    mode: str = 'factor',
    factor: float | None = None,
    fit_size: float | None = None,
    target_volume: float | None = None,
    pivot: str = 'origin',
    decimal_places: int = 8,
    backup: bool = True,
    verbose: bool = False,
) -> float:
    """对单个 OBJ 文件做高精度等比例缩放 (文本级, 原地修改)。

    Args:
        obj_path: OBJ 文件路径
        mode: 'factor' | 'fit' | 'volume'
        factor: mode='factor' 时的缩放系数
        fit_size: mode='fit' 时的目标最长边尺寸
        target_volume: mode='volume' 时的目标体积
        pivot: 缩放中心 'origin' | 'centroid' | 'bbox'
        decimal_places: 顶点坐标小数位数 (默认 8)
        backup: 是否备份原始文件为 xxx-origin.obj
        verbose: 打印详细信息

    Returns:
        实际使用的缩放系数。
    """
    if not obj_path.exists():
        raise FileNotFoundError(str(obj_path))

    lines = obj_path.read_text(errors='ignore').splitlines()
    verts = _parse_vertices_from_obj(lines)

    scale = _resolve_factor(verts, mode, factor, fit_size, target_volume)

    if abs(scale - 1.0) < 1e-12:
        if verbose:
            print(f"  [跳过] {obj_path.name}: scale≈1.0, 无需缩放")
        return scale

    pivot_pt = _compute_pivot(verts, pivot)

    if backup:
        origin = obj_path.with_stem(obj_path.stem + '-origin')
        shutil.copy2(obj_path, origin)
        mtl = obj_path.with_suffix('.mtl')
        if mtl.exists():
            shutil.copy2(mtl, mtl.with_stem(mtl.stem + '-origin'))

    scaled_lines = _scale_obj_text(lines, scale, pivot_pt, decimal_places)
    obj_path.write_text('\n'.join(scaled_lines) + '\n')

    if verbose or True:  # 始终打印缩放信息
        bbox_before = verts.max(axis=0) - verts.min(axis=0) if len(verts) else np.zeros(3)
        bbox_after = bbox_before * scale
        print(
            f"  [缩放] {obj_path.name}: ×{scale:.8g} | pivot={pivot} "
            f"| bbox {bbox_before} → {bbox_after}"
        )

    return scale


def scale_mesh_trimesh(
    in_path: Path,
    out_path: Path,
    *,
    mode: str = 'factor',
    factor: float | None = None,
    fit_size: float | None = None,
    target_volume: float | None = None,
    pivot: str = 'origin',
    backup: bool = True,
    verbose: bool = False,
) -> float:
    """通过 trimesh 对非 OBJ 格式 (STL/PLY/GLB 等) 做等比例缩放。

    Args:
        in_path: 输入文件路径
        out_path: 输出文件路径 (可与 in_path 相同则原地覆盖)
        其余参数同 scale_obj。

    Returns:
        实际使用的缩放系数。
    """
    mesh = trimesh.load(str(in_path), force='mesh', skip_materials=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    scale = _resolve_factor(verts, mode, factor, fit_size, target_volume)

    if abs(scale - 1.0) < 1e-12:
        if verbose:
            print(f"  [跳过] {in_path.name}: scale≈1.0")
        if out_path != in_path:
            shutil.copy2(in_path, out_path)
        return scale

    pivot_pt = _compute_pivot(verts, pivot)

    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pivot_pt * (1.0 - scale)
    T[0, 0] = T[1, 1] = T[2, 2] = scale
    mesh.apply_transform(T)

    if backup and out_path == in_path:
        origin = in_path.with_stem(in_path.stem + '-origin')
        shutil.copy2(in_path, origin)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    if verbose:
        print(f"  [缩放] {in_path.name}: ×{scale:.8g} | pivot={pivot} → {out_path}")

    return scale


def scale_file(
    path: Path,
    out_path: Path | None = None,
    *,
    mode: str = 'factor',
    factor: float | None = None,
    fit_size: float | None = None,
    target_volume: float | None = None,
    pivot: str = 'origin',
    decimal_places: int = 8,
    backup: bool = True,
    verbose: bool = False,
) -> float:
    """统一入口: 根据文件后缀自动选择文本级 (OBJ) 或 trimesh 路径。

    Args:
        path: 输入文件
        out_path: 输出路径 (None = 原地修改)
        其余参数同 scale_obj / scale_mesh_trimesh。

    Returns:
        实际使用的缩放系数。
    """
    if out_path is None:
        out_path = path

    suffix = path.suffix.lower()
    if suffix == '.obj':
        if out_path != path:
            shutil.copy2(path, out_path)
        return scale_obj(
            out_path,
            mode=mode, factor=factor, fit_size=fit_size, target_volume=target_volume,
            pivot=pivot, decimal_places=decimal_places,
            backup=backup and (out_path == path),
            verbose=verbose,
        )
    else:
        return scale_mesh_trimesh(
            path, out_path,
            mode=mode, factor=factor, fit_size=fit_size, target_volume=target_volume,
            pivot=pivot,
            backup=backup,
            verbose=verbose,
        )


# ──────────────────────────────────────────────
# 批量处理
# ──────────────────────────────────────────────

def scale_directory(
    dir_path: Path,
    glob: str = '*.obj',
    out_dir: Path | None = None,
    *,
    mode: str = 'factor',
    factor: float | None = None,
    fit_size: float | None = None,
    target_volume: float | None = None,
    pivot: str = 'origin',
    decimal_places: int = 8,
    backup: bool = True,
    verbose: bool = False,
) -> dict[str, float]:
    """批量缩放目录下所有匹配文件。

    Args:
        dir_path: 目录路径
        glob: 文件匹配模式 (默认 '*.obj')
        out_dir: 输出目录 (None = 原地修改)
        其余参数同 scale_file。

    Returns:
        {文件名: 缩放系数} 字典。
    """
    files = sorted(dir_path.glob(glob))
    if not files:
        print(f"目录 {dir_path} 下未找到 '{glob}' 匹配文件")
        return {}

    results: dict[str, float] = {}
    for p in files:
        if out_dir is not None:
            rel = p.relative_to(dir_path)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst = None
        try:
            s = scale_file(
                p, dst,
                mode=mode, factor=factor, fit_size=fit_size, target_volume=target_volume,
                pivot=pivot, decimal_places=decimal_places,
                backup=backup, verbose=verbose,
            )
            results[p.name] = s
        except Exception as exc:
            print(f"  [WARN] {p.name} 缩放失败: {exc}", file=sys.stderr)

    return results


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            高精度等比例网格缩放工具

            缩放模式 (三选一):
              --factor F      直接指定系数 (如 --factor 0.001 做 mm→m 转换)
              --fit S         缩放至包围盒最长边 = S (单位与坐标一致)
              --volume V      缩放至目标体积 = V (取立方根系数)

            缩放中心 (--pivot):
              origin          世界原点 [默认], 顶点分布整体平移
              centroid        几何质心不变
              bbox            包围盒中心不变

            示例:
              # mm → m
              python scale_mesh.py robot.obj --factor 0.001

              # 最长边缩放到 1.0
              python scale_mesh.py model.obj --fit 1.0 --pivot bbox

              # 批量处理目录
              python scale_mesh.py ./meshes --factor 0.001 --glob "**/*.obj"

              # 输出到新目录 (不修改原文件)
              python scale_mesh.py ./meshes --factor 0.001 --out ./meshes_scaled --no-backup
        """),
    )
    parser.add_argument("path", help="OBJ/STL/PLY 文件或目录路径")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--factor", "-f", type=float, metavar="F",
                            help="等比例缩放系数")
    mode_group.add_argument("--fit", type=float, metavar="S",
                            help="目标包围盒最长边尺寸")
    mode_group.add_argument("--volume", type=float, metavar="V",
                            help="目标体积")

    parser.add_argument("--pivot", choices=["origin", "centroid", "bbox"], default="origin",
                        help="缩放中心 (默认: origin)")
    parser.add_argument("--decimals", type=int, default=8,
                        help="OBJ 顶点坐标小数位数 (默认: 8)")
    parser.add_argument("--glob", default="*.obj",
                        help="目录模式下文件匹配 glob (默认: *.obj)")
    parser.add_argument("--out", metavar="DIR_OR_FILE",
                        help="输出路径 (目录 or 文件); 默认原地修改")
    parser.add_argument("--no-backup", action="store_true",
                        help="不备份原始文件")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细日志")
    args = parser.parse_args(argv)

    target = Path(args.path)
    if not target.exists():
        parser.error(f"路径不存在: {target}")

    if args.factor is not None:
        mode, factor, fit_size, target_volume = 'factor', args.factor, None, None
    elif args.fit is not None:
        mode, factor, fit_size, target_volume = 'fit', None, args.fit, None
    else:
        mode, factor, fit_size, target_volume = 'volume', None, None, args.volume

    common = dict(
        mode=mode, factor=factor, fit_size=fit_size, target_volume=target_volume,
        pivot=args.pivot, decimal_places=args.decimals,
        backup=not args.no_backup, verbose=args.verbose,
    )

    if target.is_dir():
        out_dir = Path(args.out) if args.out else None
        results = scale_directory(target, glob=args.glob, out_dir=out_dir, **common)
        print(f"\n完成: {len(results)} 个文件缩放处理")
    else:
        out_path = Path(args.out) if args.out else None
        s = scale_file(target, out_path, **common)
        print(f"\n完成: ×{s:.8g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""将高精度 OBJ 对齐到低精度 STL/OBJ 参考系。

Pipeline:
  网格采样→点云 → FPFH+RANSAC 粗配准 → 多尺度 ICP 精配准
  → 尺度微调 → GICP 终极精修
  → 文本级 OBJ 变换 (保留 UV/材质/法线/顶点色)

依赖: pip install open3d trimesh numpy

用法:
  python align_meshes.py --ref ref.STL --src high.obj --out aligned.obj
  python align_meshes.py --ref-dir lo/ --src-dir hi/ --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("需要 open3d: pip install open3d")

try:
    import trimesh
except ImportError:
    sys.exit("需要 trimesh: pip install trimesh")


# ── OBJ 解析正则（模块级预编译，所有函数共享）────────────────────────────────
_NUM    = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
# v x y z [optional tail: vertex color etc.]
_V_RE   = re.compile(r'^v ('  + _NUM + r') (' + _NUM + r') (' + _NUM + r')([ \t][^\n]*)?',
                     re.MULTILINE)
# vn x y z
_VN_RE  = re.compile(r'^vn (' + _NUM + r') (' + _NUM + r') (' + _NUM + r')',
                     re.MULTILINE)
# v x y z (无 tail 组，用于纯顶点提取)
_V_XYZ_RE = re.compile(r'^v (' + _NUM + r') (' + _NUM + r') (' + _NUM + r')',
                        re.MULTILINE)
_MTL_RE = re.compile(r'^(mtllib ).*$', re.MULTILINE)


# ── 配置 ──────────────────────────────────────────────────────────────────────

@dataclass
class AlignConfig:
    """对齐算法的全部超参数。"""
    voxel_ratio:        float = 0.010       # voxel_size = 对角线 × 此值
    icp_dist_ratio:     float = 0.030       # max_icp_dist = 对角线 × 此值
    icp_scales:         tuple  = (5.0, 2.0, 1.0, 0.5, 0.2)  # 多尺度粗→细
    icp_max_iter:       int   = 200
    ultra_icp_max_iter: int   = 200
    ultra_icp_stages:   int   = 6
    ransac_iters:       int   = 200_000
    ransac_confidence:  float = 0.999
    snap_factors:       tuple  = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    snap_tolerance:     float = 0.30
    n_points:           int   = 200_000    # FPFH 采样点数（OBJ 用快速顶点采样）
    target_rmse:        float = 0.0001     # 单位: 米 (0.1 mm)
    do_global:          bool  = True
    no_scale:           bool  = False
    voxel_size:         float = 0.0        # 0 → 自动
    max_icp_dist:       float = 0.0        # 0 → 自动
    max_ultra_pts:      int   = 150_000    # ultra_fine_icp 最大点数，0=不限


CFG = AlignConfig()   # 全局默认配置 (可被 CLI 覆盖)


# ── 网格 I/O ──────────────────────────────────────────────────────────────────

def load_trimesh(path: Path) -> trimesh.Trimesh:
    """加载网格文件，Scene 自动合并为单一 Trimesh。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    tm = trimesh.load(str(path), force="mesh")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(
            [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return tm


def to_o3d_mesh(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """trimesh → Open3D TriangleMesh（含顶点法线）。"""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces)),
    )
    mesh.compute_vertex_normals()
    return mesh


def sample_pcd(path: Path, n_points: int) -> o3d.geometry.PointCloud:
    """均匀采样点云（通用，非 OBJ 文件使用）。"""
    return to_o3d_mesh(load_trimesh(path)).sample_points_uniformly(n_points)


def fast_sample_pcd(path: Path, n_points: int) -> o3d.geometry.PointCloud:
    """OBJ 顶点随机采样点云，跳过 trimesh 加载，速度快 5-10x。

    非 OBJ 文件回退到 trimesh 均匀采样。法线由 ensure_normals 按需估计。
    """
    if path.suffix.lower() != ".obj":
        return sample_pcd(path, n_points)
    text = path.read_bytes().decode("utf-8", errors="replace")
    raw = _V_XYZ_RE.findall(text)
    if not raw:
        return sample_pcd(path, n_points)
    verts = np.array(raw, dtype=np.float64)
    if len(verts) > n_points:
        idx = np.random.default_rng().choice(len(verts), n_points, replace=False)
        verts = verts[idx]
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))


def vertex_pcd(path: Path, scale: float = 1.0) -> o3d.geometry.PointCloud:
    """全量顶点点云（通用，非 OBJ 文件使用）。"""
    tm = load_trimesh(path)
    verts = np.asarray(tm.vertices, dtype=np.float64) * scale
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    o3d_mesh = to_o3d_mesh(tm)
    if scale != 1.0:
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.compute_vertex_normals()
    pcd.normals = o3d_mesh.vertex_normals
    return pcd


def fast_vertex_pcd(path: Path, scale: float = 1.0,
                    max_pts: int = 0) -> o3d.geometry.PointCloud:
    """OBJ 全量（或限额）顶点点云，跳过 trimesh，速度快 10-20x。

    非 OBJ 文件回退到 trimesh 路径。法线由 ensure_normals 按需估计。
    """
    if path.suffix.lower() != ".obj":
        return vertex_pcd(path, scale)
    text = path.read_bytes().decode("utf-8", errors="replace")
    raw = _V_XYZ_RE.findall(text)
    if not raw:
        return vertex_pcd(path, scale)
    verts = np.array(raw, dtype=np.float64) * scale
    if max_pts and len(verts) > max_pts:
        step = max(1, len(verts) // max_pts)
        verts = verts[::step][:max_pts]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    return pcd


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def diagonal(pcd: o3d.geometry.PointCloud) -> float:
    bb = pcd.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))


def ensure_normals(pcd: o3d.geometry.PointCloud, radius: float):
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )


# ── 尺度估计 ──────────────────────────────────────────────────────────────────

def estimate_scale(pcd_ref: o3d.geometry.PointCloud,
                   pcd_src: o3d.geometry.PointCloud,
                   cfg: AlignConfig = CFG) -> tuple[float, float]:
    """估算 src→ref 尺度比，snap 到常见 10^n 整数倍。返回 (scale, raw_ratio)。"""
    src_diag = diagonal(pcd_src)
    if src_diag < 1e-12:
        return 1.0, 1.0

    raw = diagonal(pcd_ref) / src_diag
    best_f = min(cfg.snap_factors, key=lambda f: abs(raw - f) / f)
    err = abs(raw - best_f) / best_f
    return (best_f if err < cfg.snap_tolerance else raw), raw


def refine_scale(pcd_ref: o3d.geometry.PointCloud,
                 src_pts_raw: np.ndarray,
                 coarse_scale: float,
                 raw_ratio: float,
                 T: np.ndarray,
                 max_dist: float,
                 cfg: AlignConfig = CFG) -> float:
    """在 coarse_scale 附近网格搜索最优尺度（snap 误差 > 0.1% 时触发）。"""
    if abs(raw_ratio - coarse_scale) / coarse_scale < 0.001:
        return coarse_scale

    reg = o3d.pipelines.registration
    icp_criteria = reg.ICPConvergenceCriteria(
        relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=50
    )
    best_scale, best_rmse, baseline_fitness = coarse_scale, float("inf"), 0.0

    for d in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02):
        s = coarse_scale * (1.0 + d)
        trial = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(src_pts_raw * s)
        )
        trial.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist * 2, max_nn=30)
        )
        r = reg.registration_icp(
            source=trial, target=pcd_ref,
            max_correspondence_distance=max_dist, init=T,
            estimation_method=reg.TransformationEstimationPointToPlane(),
            criteria=icp_criteria,
        )
        if d == 0.0:
            baseline_fitness, best_rmse = r.fitness, r.inlier_rmse
        if r.inlier_rmse < best_rmse and r.fitness >= baseline_fitness * 0.95:
            best_rmse, best_scale = r.inlier_rmse, s

    return best_scale


# ── 配准算法 ──────────────────────────────────────────────────────────────────

def preprocess(pcd: o3d.geometry.PointCloud,
               voxel_size: float
               ) -> tuple[o3d.geometry.PointCloud,
                          o3d.pipelines.registration.Feature]:
    """降采样 + 法线估计 + FPFH 特征。"""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return down, fpfh


def global_ransac(src_down, ref_down, src_fpfh, ref_fpfh,
                  voxel_size: float,
                  cfg: AlignConfig = CFG) -> o3d.pipelines.registration.RegistrationResult:
    """FPFH + RANSAC 全局配准。"""
    reg = o3d.pipelines.registration
    dist = voxel_size * 1.5
    return reg.registration_ransac_based_on_feature_matching(
        source=src_down, target=ref_down,
        source_feature=src_fpfh, target_feature=ref_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=reg.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=reg.RANSACConvergenceCriteria(cfg.ransac_iters, cfg.ransac_confidence),
    )


def icp(src, ref, init_T: np.ndarray, max_dist: float,
        max_iter: int = CFG.icp_max_iter) -> o3d.pipelines.registration.RegistrationResult:
    """Point-to-Plane ICP。"""
    reg = o3d.pipelines.registration
    ensure_normals(src, max_dist * 2)
    ensure_normals(ref, max_dist * 2)
    return reg.registration_icp(
        source=src, target=ref,
        max_correspondence_distance=max_dist, init=init_T,
        estimation_method=reg.TransformationEstimationPointToPlane(),
        criteria=reg.ICPConvergenceCriteria(
            relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=max_iter,
        ),
    )


def gicp(src, ref, init_T: np.ndarray, max_dist: float,
         max_iter: int = CFG.icp_max_iter) -> o3d.pipelines.registration.RegistrationResult:
    """Generalized ICP（利用局部协方差，精度高于 Point-to-Plane）。"""
    reg = o3d.pipelines.registration
    ensure_normals(src, max_dist * 2)
    ensure_normals(ref, max_dist * 2)
    return reg.registration_generalized_icp(
        source=src, target=ref,
        max_correspondence_distance=max_dist, init=init_T,
        estimation_method=reg.TransformationEstimationForGeneralizedICP(),
        criteria=reg.ICPConvergenceCriteria(
            relative_fitness=1e-12, relative_rmse=1e-12, max_iteration=max_iter,
        ),
    )


def multiscale_icp(src, ref, init_T: np.ndarray, base_dist: float,
                   scales: tuple = CFG.icp_scales,
                   max_iter: int = CFG.icp_max_iter) -> tuple[np.ndarray,
                                                               o3d.pipelines.registration.RegistrationResult]:
    """多尺度 Point-to-Plane ICP，粗→细逐步收敛。"""
    T = init_T
    for s in scales:
        r = icp(src, ref, T, base_dist * s, max_iter)
        T = r.transformation
    return T, r


def ultra_fine_icp(ref_path: Path, src_path: Path,
                   init_T: np.ndarray, scale: float,
                   cfg: AlignConfig = CFG) -> tuple[np.ndarray, float, float]:
    """全量顶点超精细配准，达到亚毫米精度。

    用递减搜索距离的多阶段 ICP + GICP 组合，逐步逼近目标精度。
    返回: (T_4x4, fitness, rmse)
    """
    t_load = time.perf_counter()
    print("  加载全量顶点 ...")
    # fast_vertex_pcd 直接解析 OBJ 文本，跳过 trimesh，比原方案快 10-20x
    pcd_ref = fast_vertex_pcd(ref_path, max_pts=cfg.max_ultra_pts)
    pcd_src = fast_vertex_pcd(src_path, scale, max_pts=cfg.max_ultra_pts)
    print(f"    ref={len(pcd_ref.points):,}  src={len(pcd_src.points):,} 顶点"
          + (f"  (上限 {cfg.max_ultra_pts:,})" if cfg.max_ultra_pts else "")
          + f"  [{time.perf_counter()-t_load:.1f}s]")

    diag = diagonal(pcd_ref)
    start_dist = max(diag * 0.01, cfg.target_rmse * 50)
    end_dist   = max(cfg.target_rmse * 5, 1e-7)
    dists = np.geomspace(start_dist, end_dist, cfg.ultra_icp_stages)

    T = init_T.copy()
    print(f"  超精细 ICP ({cfg.ultra_icp_stages} 阶段, "
          f"{start_dist*1000:.2f}mm → {end_dist*1000:.2f}mm) ...")

    t_ultra = time.perf_counter()
    r = None
    for i, d in enumerate(dists):
        t_stage = time.perf_counter()
        r = icp(pcd_src, pcd_ref, T, d, cfg.ultra_icp_max_iter)
        T = r.transformation
        # GICP 精修
        used_gicp = False
        try:
            g = gicp(pcd_src, pcd_ref, T, d, cfg.ultra_icp_max_iter)
            if g.fitness >= r.fitness * 0.95:
                T, r = g.transformation, g
                used_gicp = True
        except Exception:
            pass

        reached = r.inlier_rmse <= cfg.target_rmse
        tag = "✓" if reached else "·"
        algo = "ICP+GICP" if used_gicp else "ICP"
        print(f"    [{i+1}/{cfg.ultra_icp_stages}] dist={d*1000:.3f}mm  "
              f"fitness={r.fitness:.4f}  RMSE={r.inlier_rmse*1000:.4f}mm  "
              f"{tag}  {algo}  [{time.perf_counter()-t_stage:.1f}s]")
        if reached:
            print(f"    已达目标精度 {cfg.target_rmse*1000:.2f}mm，提前终止")
            break

    print(f"  超精细 ICP 合计: [{time.perf_counter()-t_ultra:.1f}s]")
    return T, r.fitness, r.inlier_rmse


# ── OBJ 文本级变换 ────────────────────────────────────────────────────────────

def _parse_coords(matches: list) -> np.ndarray:
    """从 regex finditer 结果批量提取 (N, 3) float64 坐标（比逐行 float() 快 3-5x）。"""
    if not matches:
        return np.zeros((0, 3), dtype=np.float64)
    raw = np.array([[m.group(1), m.group(2), m.group(3)] for m in matches])
    return raw.astype(np.float64)


def _flush_subnormals(arr: np.ndarray) -> int:
    """将次正规（denormal）浮点数归零，返回被归零的元素数。

    open3d BLAS 操作可能修改 FPU 异常标志，使得次正规数在后续 matmul 中触发
    RuntimeWarning。归零不影响精度（次正规值远小于模型坐标量级）。
    """
    mask = (arr != 0.0) & (np.abs(arr) < np.finfo(np.float64).tiny)
    n = int(mask.sum())
    if n:
        arr[mask] = 0.0
    return n


def transform_obj(src: Path, dst: Path, T: np.ndarray, scale: float = 1.0):
    """regex + numpy 单遍扫描变换 OBJ 顶点/法线，保留 UV/材质/顶点色等所有属性。

    性能优化：
    - 用预编译正则一次性提取所有 v/vn 坐标，避免 Python 行循环
    - numpy 批量字符串→float64 转换（比逐行 float() 快 3-5x）
    - 用 arr @ R.T 替代 (R @ arr.T).T，保持 C-contiguous 内存布局，
      避免 open3d 污染 FPU 状态后 Fortran-order 转置触发 RuntimeWarning
    """
    if not np.isfinite(T).all():
        print(f"  [ERROR] 变换矩阵含 NaN/Inf，跳过写出:\n{T}", file=sys.stderr)
        return
    if not (np.isfinite(scale) and scale > 0):
        print(f"  [ERROR] scale={scale!r} 非法，重置为 1.0", file=sys.stderr)
        scale = 1.0

    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)

    text = src.read_text(errors="ignore")

    # 重命名 mtllib 引用
    mtl_name = dst.with_suffix(".mtl").name
    text = _MTL_RE.sub(lambda m: m.group(1) + mtl_name, text)

    # ── 提取所有顶点/法线匹配（保留位置信息用于重建）──
    v_ms  = list(_V_RE.finditer(text))
    vn_ms = list(_VN_RE.finditer(text))

    # ── 批量变换顶点 ──
    # np.errstate：屏蔽 numpy 2.x + Apple Accelerate BLAS 对大矩阵 matmul 误报的
    # FPU 异常警告（divide/overflow/invalid）。这是该 BLAS 组合的已知 false-positive，
    # 计算结果本身正确；之后通过 np.isfinite 显式检查真实数值异常。
    verts: np.ndarray | None = None
    if v_ms:
        arr = _parse_coords(v_ms) * scale
        n_sub = _flush_subnormals(arr)
        if n_sub:
            print(f"  [INFO] 归零 {n_sub} 个次正规顶点坐标", file=sys.stderr)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            verts = arr @ R.T + t
        bad = ~np.isfinite(verts).all(axis=1)
        if bad.any():
            print(f"  [WARN] 变换后 {bad.sum()} 个顶点含 NaN/Inf，回退原始值", file=sys.stderr)
            verts[bad] = arr[bad] / scale

    # ── 批量变换法线 ──
    norms: np.ndarray | None = None
    if vn_ms:
        arr_n = _parse_coords(vn_ms)
        _flush_subnormals(arr_n)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            raw_n = arr_n @ R.T
        lengths = np.linalg.norm(raw_n, axis=1, keepdims=True).clip(1e-12)
        norms = raw_n / lengths

    # ── 单遍重建文本（v 和 vn 按位置顺序混合处理）──
    all_ms: list = []
    if verts is not None:
        all_ms += [(m, "v", i) for i, m in enumerate(v_ms)]
    if norms is not None:
        all_ms += [(m, "vn", i) for i, m in enumerate(vn_ms)]
    all_ms.sort(key=lambda x: x[0].start())

    parts: list[str] = []
    prev = 0
    for m, kind, idx in all_ms:
        parts.append(text[prev:m.start()])
        if kind == "v":
            v = verts[idx]
            tail = m.group(4) or ""
            parts.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}{tail}")
        else:
            n = norms[idx]
            parts.append(f"vn {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}")
        prev = m.end()
    parts.append(text[prev:])

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(parts))
    _copy_materials(src, dst)


def _safe_copy(src: Path, dst: Path):
    try:
        if dst.exists():
            dst.chmod(0o644)
        shutil.copyfile(str(src), str(dst))
    except (PermissionError, OSError):
        try:
            dst.write_bytes(src.read_bytes())
        except (PermissionError, OSError) as e:
            print(f"  [WARN] 复制失败 {src.name}: {e}", file=sys.stderr)


def _copy_materials(src_obj: Path, dst_obj: Path):
    mtl = src_obj.with_suffix(".mtl")
    if not mtl.exists():
        return
    _safe_copy(mtl, dst_obj.with_suffix(".mtl"))
    for m in re.finditer(r"^\s*map_\w+\s+(.+)$",
                         mtl.read_text(errors="ignore"), re.MULTILINE):
        tex = src_obj.parent / m.group(1).strip()
        if tex.exists():
            _safe_copy(tex, dst_obj.parent / tex.name)


# ── 单对对齐（主流程）────────────────────────────────────────────────────────

def align_pair(ref_path: Path, src_path: Path, out_path: Path,
               cfg: AlignConfig = CFG) -> dict:
    """对齐单对模型，返回结果字典。"""
    t_total = time.perf_counter()

    # ── 采样点云 ──
    t0 = time.perf_counter()
    print(f"  加载 ref: {ref_path.name}")
    pcd_ref = fast_sample_pcd(ref_path, cfg.n_points)
    print(f"  加载 src: {src_path.name}")
    pcd_src = fast_sample_pcd(src_path, cfg.n_points)
    print(f"  对角线  ref={diagonal(pcd_ref):.4f}  src={diagonal(pcd_src):.4f}"
          f"  [{time.perf_counter()-t0:.1f}s]")
    print(f"  目标精度: RMSE ≤ {cfg.target_rmse*1000:.2f}mm")

    # ── 尺度补偿 ──
    src_pts_raw = np.asarray(pcd_src.points).copy()
    scale, raw_ratio = 1.0, 1.0

    if not cfg.no_scale:
        scale, raw_ratio = estimate_scale(pcd_ref, pcd_src, cfg)
        if abs(scale - 1.0) > 0.01:
            print(f"  尺度补偿: raw={raw_ratio:.4f} → scale={scale}")
            pcd_src.points = o3d.utility.Vector3dVector(src_pts_raw * scale)
        else:
            scale, raw_ratio = 1.0, 1.0

    # ── 自动参数 ──
    diag = max(diagonal(pcd_ref), diagonal(pcd_src), 1e-6)
    voxel = cfg.voxel_size or diag * cfg.voxel_ratio
    icp_d = cfg.max_icp_dist or diag * cfg.icp_dist_ratio
    print(f"  自动参数: voxel={voxel:.4f}  icp_dist={icp_d:.4f}")

    # ── FPFH + RANSAC 全局配准 ──
    T = np.eye(4)
    if cfg.do_global:
        t0 = time.perf_counter()
        print("  FPFH + RANSAC ...")
        sd, sf = preprocess(pcd_src, voxel)
        rd, rf = preprocess(pcd_ref, voxel)
        g = global_ransac(sd, rd, sf, rf, voxel, cfg)
        T = g.transformation
        print(f"    粗配准 fitness={g.fitness:.4f}  RMSE={g.inlier_rmse:.6f}"
              f"  [{time.perf_counter()-t0:.1f}s]")

    # ── 多尺度 ICP ──
    t0 = time.perf_counter()
    print("  多尺度 ICP ...")
    T, result = multiscale_icp(pcd_src, pcd_ref, T, icp_d, cfg.icp_scales, cfg.icp_max_iter)
    print(f"    P2Plane fitness={result.fitness:.4f}  RMSE={result.inlier_rmse*1000:.4f}mm"
          f"  [{time.perf_counter()-t0:.1f}s]")

    # ── 尺度微调 ──
    if scale != 1.0:
        t0 = time.perf_counter()
        refined = refine_scale(pcd_ref, src_pts_raw, scale, raw_ratio, T, icp_d, cfg)
        if refined != scale:
            print(f"  尺度微调: {scale} → {refined:.6f}  [{time.perf_counter()-t0:.1f}s]")
            scale = refined
            pcd_src.points = o3d.utility.Vector3dVector(src_pts_raw * scale)
            t0 = time.perf_counter()
            T, result = multiscale_icp(
                pcd_src, pcd_ref, T, icp_d, cfg.icp_scales[-2:], cfg.icp_max_iter
            )
            print(f"    微调后 RMSE={result.inlier_rmse*1000:.4f}mm"
                  f"  [{time.perf_counter()-t0:.1f}s]")

    # ── 全量顶点超精细配准 ──
    t0 = time.perf_counter()
    T, fitness, rmse = ultra_fine_icp(ref_path, src_path, T, scale, cfg)
    # ultra_fine_icp 内部已打印各子阶段耗时，此处仅记录总计
    _ = t0  # 总时间已在函数内打印

    # ── 精度报告 ──
    pct = rmse / max(diagonal(pcd_ref), 1e-6) * 100
    tag = "✓" if rmse <= cfg.target_rmse else "⚠"
    print(f"  {tag} 最终精度: RMSE={rmse*1000:.4f}mm ({pct:.4f}%)  目标≤{cfg.target_rmse*1000:.2f}mm")

    # ── 写出 ──
    t0 = time.perf_counter()
    print(f"  输出: {out_path}")
    if src_path.suffix.lower() == ".obj":
        transform_obj(src_path, out_path, T, scale)
    else:
        tm = load_trimesh(src_path)
        if scale != 1.0:
            tm.vertices *= scale
        tm.apply_transform(T)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tm.export(str(out_path))
    print(f"    写出完成  [{time.perf_counter()-t0:.1f}s]")

    t_elapsed = time.perf_counter() - t_total
    print(f"  总耗时: {t_elapsed:.1f}s  ({t_elapsed/60:.1f}min)")

    return dict(
        ref=str(ref_path), src=str(src_path), out=str(out_path),
        transform=T.tolist(), scale=scale,
        fitness=fitness, rmse=rmse, rmse_mm=rmse * 1000, precision_pct=pct,
        elapsed_s=round(t_elapsed, 2),
    )


# ── 批量对齐 ──────────────────────────────────────────────────────────────────

def batch_align(ref_dir: Path, src_dir: Path, out_dir: Path,
                ref_ext: str = ".STL", src_ext: str = ".obj",
                cfg: AlignConfig = CFG) -> list[dict]:
    """批量对齐目录下所有同名模型对，结果写入 alignment_record.json。"""
    results, ok, fail = [], 0, 0
    t_batch = time.perf_counter()

    for rf in sorted(ref_dir.glob(f"*{ref_ext}")):
        # 查找同名 src（大小写不敏感）
        sf = src_dir / f"{rf.stem}{src_ext}"
        if not sf.exists():
            cands = [f for f in src_dir.iterdir()
                     if f.stem.lower() == rf.stem.lower()
                     and f.suffix.lower() == src_ext.lower()]
            sf = cands[0] if cands else None

        if sf is None:
            print(f"  [跳过] {rf.stem}: 无对应 {src_ext}")
            continue

        sep = "=" * 60
        print(f"\n{sep}\n对齐: {rf.stem}\n{sep}")
        try:
            r = align_pair(rf, sf, out_dir / f"{rf.stem}.obj", cfg)
            results.append(r)
            ok += 1
            print(f"  ✓ fitness={r['fitness']:.4f}  RMSE={r['rmse_mm']:.4f}mm"
                  f"  ({r['precision_pct']:.4f}%)  耗时={r['elapsed_s']:.1f}s")
        except Exception as e:
            print(f"  ✗ {e}", file=sys.stderr)
            fail += 1
            results.append(dict(ref=str(rf), src=str(sf), error=str(e)))

    t_batch_elapsed = time.perf_counter() - t_batch
    sep = "=" * 60
    print(f"\n{sep}\n完成: {ok}/{ok+fail} 成功，{fail} 失败  "
          f"批量总耗时: {t_batch_elapsed:.1f}s ({t_batch_elapsed/60:.1f}min)\n{sep}")
    record = out_dir / "alignment_record.json"
    record.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"记录: {record}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="高精度模型 → 低精度参考系对齐 (FPFH + RANSAC + ICP + GICP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g1 = p.add_argument_group("单对模式")
    g1.add_argument("--ref",  type=Path, help="参考网格 (STL/OBJ)")
    g1.add_argument("--src",  type=Path, help="待对齐网格 (OBJ)")
    g1.add_argument("--out",  type=Path, help="输出路径")

    g2 = p.add_argument_group("批量模式")
    g2.add_argument("--ref-dir", type=Path)
    g2.add_argument("--src-dir", type=Path)
    g2.add_argument("--out-dir", type=Path)
    g2.add_argument("--ref-ext", default=".STL")
    g2.add_argument("--src-ext", default=".obj")

    g3 = p.add_argument_group("算法参数")
    g3.add_argument("--voxel-size",   type=float, default=0,
                    help="体素尺寸，0=自动")
    g3.add_argument("--max-icp-dist", type=float, default=0,
                    help="ICP 最大对应距离，0=自动")
    g3.add_argument("--num-points",   type=int,   default=CFG.n_points,
                    help="采样点数")
    g3.add_argument("--no-global",    action="store_true",
                    help="跳过全局配准")
    g3.add_argument("--no-scale",     action="store_true",
                    help="禁用自动尺度补偿")
    g3.add_argument("--target-rmse",  type=float, default=CFG.target_rmse,
                    help=f"目标 RMSE（米），默认 {CFG.target_rmse}={CFG.target_rmse*1000}mm")
    g3.add_argument("--max-ultra-pts", type=int, default=CFG.max_ultra_pts,
                    help=f"ultra_fine_icp 最大顶点数，0=不限，默认 {CFG.max_ultra_pts}")
    return p


def main():
    a = build_parser().parse_args()

    cfg = AlignConfig(
        voxel_size=a.voxel_size,
        max_icp_dist=a.max_icp_dist,
        n_points=a.num_points,
        do_global=not a.no_global,
        no_scale=a.no_scale,
        target_rmse=a.target_rmse,
        max_ultra_pts=a.max_ultra_pts,
    )

    if a.ref_dir and a.src_dir:
        out = a.out_dir or a.src_dir.parent / "meshes_output"
        out.mkdir(parents=True, exist_ok=True)
        batch_align(a.ref_dir, a.src_dir, out, a.ref_ext, a.src_ext, cfg)

    elif a.ref and a.src:
        out = a.out or a.src.with_name(f"{a.src.stem}_aligned.obj")
        r = align_pair(a.ref, a.src, out, cfg)
        print(f"\n完成! fitness={r['fitness']:.4f}  "
              f"RMSE={r['rmse_mm']:.4f}mm  精度={r['precision_pct']:.4f}%")
        print(f"变换矩阵:\n{np.array(r['transform'])}")

    else:
        build_parser().error("请指定 --ref/--src 或 --ref-dir/--src-dir")


if __name__ == "__main__":
    main()
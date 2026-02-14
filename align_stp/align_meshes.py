#!/usr/bin/env python3
"""将高精度 OBJ 对齐到低精度 STL/OBJ 参考系。

Pipeline: 网格采样→点云 → FPFH+RANSAC 粗配准 → 多尺度 ICP 精配准
          → 尺度微调 → GICP 终极精修
          → 文本级 OBJ 变换 (保留 UV/材质/法线/顶点色)

依赖:  pip install open3d trimesh numpy

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
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("需要 open3d: pip install open3d")

try:
    import trimesh
except ImportError:
    sys.exit("需要 trimesh: pip install trimesh")

# ── 常量 ──────────────────────────────────────────────────────────────────────

VOXEL_RATIO = 0.010                # voxel_size = 对角线 × 此值
ICP_DIST_RATIO = 0.03              # max_icp_dist = 对角线 × 此值
ICP_SCALES = (5.0, 2.0, 1.0, 0.5, 0.2)  # 多尺度 ICP (粗→亚体素)
ICP_MAX_ITER = 300                 # 常规 ICP 迭代上限
ULTRA_ICP_MAX_ITER = 500           # 超精细 ICP 迭代上限
RANSAC_ITERS = 500_000
RANSAC_CONFIDENCE = 0.9999
SNAP_FACTORS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SNAP_TOLERANCE = 0.30
DEFAULT_POINTS = 500_000           # 默认采样密度
DEFAULT_TARGET_RMSE = 0.0001       # 默认目标 RMSE (0.1mm)

# ── 网格 → 点云 ──────────────────────────────────────────────────────────────


def _load_trimesh(path: str | Path) -> trimesh.Trimesh:
    """加载网格文件为 trimesh 对象。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"网格文件不存在: {path}")
    tm = trimesh.load(str(path), force="mesh")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(
            [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return tm


def _trimesh_to_o3d(tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """trimesh → Open3D TriangleMesh。"""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces)),
    )
    mesh.compute_vertex_normals()
    return mesh


def mesh_to_pcd(path: str | Path, n_points: int = DEFAULT_POINTS) -> o3d.geometry.PointCloud:
    """加载网格并均匀采样为点云。"""
    return _trimesh_to_o3d(_load_trimesh(path)).sample_points_uniformly(
        number_of_points=n_points
    )


def mesh_to_full_pcd(path: str | Path, scale: float = 1.0) -> o3d.geometry.PointCloud:
    """加载网格的全量顶点作为点云 (最大密度, 用于超精细配准)。"""
    tm = _load_trimesh(path)
    verts = np.asarray(tm.vertices, dtype=np.float64) * scale
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(verts))
    # 从面片计算精确顶点法线
    o3d_mesh = _trimesh_to_o3d(tm)
    if scale != 1.0:
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.compute_vertex_normals()
    pcd.normals = o3d_mesh.vertex_normals
    return pcd


# ── 尺度估计 ─────────────────────────────────────────────────────────────────


def _pcd_diagonal(pcd: o3d.geometry.PointCloud) -> float:
    bb = pcd.get_axis_aligned_bounding_box()
    return float(np.linalg.norm(bb.get_max_bound() - bb.get_min_bound()))


def estimate_scale(pcd_ref: o3d.geometry.PointCloud,
                   pcd_src: o3d.geometry.PointCloud) -> tuple[float, float]:
    """估算 src→ref 尺度比, snap 到常见 10^n 整数倍。

    Returns:  (scale, raw_ratio)
    """
    src_diag = _pcd_diagonal(pcd_src)
    if src_diag < 1e-12:
        return 1.0, 1.0

    raw = _pcd_diagonal(pcd_ref) / src_diag

    best_f, best_err = raw, float("inf")
    for f in SNAP_FACTORS:
        err = abs(raw - f) / f
        if err < best_err:
            best_f, best_err = f, err

    return (best_f if best_err < SNAP_TOLERANCE else raw), raw


def _refine_scale(pcd_ref: o3d.geometry.PointCloud,
                  pcd_src_raw: np.ndarray,
                  coarse_scale: float,
                  raw_ratio: float,
                  T: np.ndarray,
                  max_dist: float) -> float:
    """在 coarse_scale 附近网格搜索最优尺度。

    仅当 snap 误差 > 0.1% 时才触发。
    用 RMSE (越低越好) 选最优, 同时要求 fitness 不低于基线的 95%。
    """
    # snap 完全准确时无需微调
    if abs(raw_ratio - coarse_scale) / coarse_scale < 0.001:
        return coarse_scale

    deltas = (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)
    best_scale, best_rmse, baseline_fitness = coarse_scale, float("inf"), 0.0

    for d in deltas:
        s = coarse_scale * (1.0 + d)
        pts = pcd_src_raw * s
        trial = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        trial.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist * 2, max_nn=30)
        )
        r = o3d.pipelines.registration.registration_icp(
            source=trial, target=pcd_ref,
            max_correspondence_distance=max_dist, init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=50,
            ),
        )
        # 记录基线 (delta=0)
        if d == 0.0:
            baseline_fitness = r.fitness
            best_rmse = r.inlier_rmse
            best_scale = s

        # RMSE 更低 且 fitness 不低于基线 95% 才采纳
        if r.inlier_rmse < best_rmse and r.fitness >= baseline_fitness * 0.95:
            best_rmse = r.inlier_rmse
            best_scale = s

    return best_scale


def auto_params(pcd_ref: o3d.geometry.PointCloud,
                pcd_src: o3d.geometry.PointCloud) -> tuple[float, float]:
    """自动计算 (voxel_size, max_icp_dist), 应在尺度补偿后调用。"""
    diag = max(_pcd_diagonal(pcd_ref), _pcd_diagonal(pcd_src), 1e-6)
    return diag * VOXEL_RATIO, diag * ICP_DIST_RATIO


# ── FPFH 特征 ────────────────────────────────────────────────────────────────


def preprocess_pcd(
    pcd: o3d.geometry.PointCloud, voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """降采样 + 法线 + FPFH。"""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return down, fpfh


# ── RANSAC 全局配准 ───────────────────────────────────────────────────────────


def global_registration(src_down, ref_down, src_fpfh, ref_fpfh, voxel_size):
    dist_thr = voxel_size * 1.5
    reg = o3d.pipelines.registration
    return reg.registration_ransac_based_on_feature_matching(
        source=src_down, target=ref_down,
        source_feature=src_fpfh, target_feature=ref_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thr,
        estimation_method=reg.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(dist_thr),
        ],
        criteria=reg.RANSACConvergenceCriteria(RANSAC_ITERS, RANSAC_CONFIDENCE),
    )


# ── ICP ──────────────────────────────────────────────────────────────────────


def _ensure_normals(pcd: o3d.geometry.PointCloud, radius: float):
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )


def refine_icp(src_pcd, ref_pcd, init_T, max_dist):
    """Point-to-Plane ICP。"""
    _ensure_normals(src_pcd, max_dist * 2)
    _ensure_normals(ref_pcd, max_dist * 2)
    return o3d.pipelines.registration.registration_icp(
        source=src_pcd, target=ref_pcd,
        max_correspondence_distance=max_dist, init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=ICP_MAX_ITER,
        ),
    )


def refine_gicp(src_pcd, ref_pcd, init_T, max_dist, max_iter=ICP_MAX_ITER):
    """Generalized ICP (利用局部协方差, 比 Point-to-Plane 更精确)。"""
    _ensure_normals(src_pcd, max_dist * 2)
    _ensure_normals(ref_pcd, max_dist * 2)
    return o3d.pipelines.registration.registration_generalized_icp(
        source=src_pcd, target=ref_pcd,
        max_correspondence_distance=max_dist, init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-12, relative_rmse=1e-12, max_iteration=max_iter,
        ),
    )


def ultra_fine_icp(
    ref_path: Path, src_path: Path,
    init_T: np.ndarray, scale: float,
    target_rmse: float = DEFAULT_TARGET_RMSE,
) -> tuple[np.ndarray, float, float]:
    """全量顶点超精细 ICP — 达到亚毫米精度。

    用两个网格的全量顶点 (而非采样) 做多轮递减距离的 ICP + GICP,
    搜索距离一路降到 target_rmse 附近。

    Returns:  (T_4x4, fitness, rmse)
    """
    print("  加载全量顶点 ...")
    pcd_ref = mesh_to_full_pcd(ref_path, scale=1.0)
    pcd_src = mesh_to_full_pcd(src_path, scale=scale)

    n_ref = len(pcd_ref.points)
    n_src = len(pcd_src.points)
    print(f"    ref 顶点={n_ref:,}  src 顶点={n_src:,}")

    diag = _pcd_diagonal(pcd_ref)
    # 从当前 RMSE 估计起始距离, 逐步递减到目标
    start_dist = max(diag * 0.01, target_rmse * 50)   # ~1% 对角线
    end_dist = max(target_rmse * 5, 1e-7)              # 目标 × 5 (留余量)

    # 生成递减距离序列 (对数等分)
    n_stages = 8
    dists = np.geomspace(start_dist, end_dist, n_stages)

    T = init_T.copy()
    print(f"  超精细 ICP ({n_stages} 阶段, {start_dist*1000:.2f}mm → {end_dist*1000:.2f}mm) ...")

    for i, d in enumerate(dists):
        # 先 Point-to-Plane
        r = refine_icp(pcd_src, pcd_ref, T, d)
        T = r.transformation

        # 再 GICP 精修
        try:
            g = refine_gicp(pcd_src, pcd_ref, T, d, max_iter=ULTRA_ICP_MAX_ITER)
            if g.fitness >= r.fitness * 0.95:
                T = g.transformation
                r = g
        except Exception:
            pass

        tag = "✓" if r.inlier_rmse <= target_rmse else "·"
        print(f"    [{i+1}/{n_stages}] dist={d*1000:.3f}mm  "
              f"fitness={r.fitness:.4f}  RMSE={r.inlier_rmse*1000:.4f}mm  {tag}")

        if r.inlier_rmse <= target_rmse:
            print(f"    已达目标精度 {target_rmse*1000:.2f}mm, 提前终止")
            return T, r.fitness, r.inlier_rmse

    return T, r.fitness, r.inlier_rmse


# ── OBJ 文本级变换 ───────────────────────────────────────────────────────────


def apply_transform_to_obj(
    src: Path, dst: Path, T: np.ndarray, scale: float = 1.0,
):
    """文本级 OBJ 变换 — 向量化批处理, 保留 UV/材质/法线/顶点色。"""
    R, t = T[:3, :3], T[:3, 3]
    lines = src.read_text(errors="ignore").splitlines()
    dst_mtl_name = dst.with_suffix(".mtl").name

    v_idx, v_data, v_extra = [], [], []
    vn_idx, vn_data = [], []

    for i, line in enumerate(lines):
        if line.startswith("v "):
            parts = line.split()
            v_idx.append(i)
            v_data.append([float(parts[1]), float(parts[2]), float(parts[3])])
            v_extra.append(" ".join(parts[4:]))
        elif line.startswith("vn "):
            parts = line.split()
            vn_idx.append(i)
            vn_data.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("mtllib "):
            lines[i] = f"mtllib {dst_mtl_name}"

    if v_data:
        verts = (R @ (np.array(v_data) * scale).T).T + t
        for j, idx in enumerate(v_idx):
            extra = f" {v_extra[j]}" if v_extra[j] else ""
            lines[idx] = f"v {verts[j, 0]:.8f} {verts[j, 1]:.8f} {verts[j, 2]:.8f}{extra}"

    if vn_data:
        normals = (R @ np.array(vn_data).T).T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
        for j, idx in enumerate(vn_idx):
            lines[idx] = f"vn {normals[j, 0]:.8f} {normals[j, 1]:.8f} {normals[j, 2]:.8f}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")
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
    src_mtl = src_obj.with_suffix(".mtl")
    if not src_mtl.exists():
        return
    _safe_copy(src_mtl, dst_obj.with_suffix(".mtl"))
    for m in re.finditer(r"^\s*map_\w+\s+(.+)$",
                         src_mtl.read_text(errors="ignore"), re.MULTILINE):
        tex_src = src_obj.parent / m.group(1).strip()
        if tex_src.exists():
            _safe_copy(tex_src, dst_obj.parent / tex_src.name)


# ── 单对对齐 ─────────────────────────────────────────────────────────────────


def align_one_pair(
    ref_path: Path, src_path: Path, out_path: Path, *,
    voxel_size: float = 0, max_icp_dist: float = 0,
    n_points: int = DEFAULT_POINTS, do_global: bool = True,
    no_scale: bool = False, target_rmse: float = DEFAULT_TARGET_RMSE,
) -> dict:
    """对齐单对模型, 返回 {transform, scale, fitness, rmse, …}。"""
    print(f"  加载 ref: {ref_path.name}")
    pcd_ref = mesh_to_pcd(ref_path, n_points)
    print(f"  加载 src: {src_path.name}")
    pcd_src = mesh_to_pcd(src_path, n_points)

    ref_diag = _pcd_diagonal(pcd_ref)
    src_diag = _pcd_diagonal(pcd_src)
    print(f"  对角线  ref={ref_diag:.4f}  src={src_diag:.4f}")
    print(f"  目标精度: RMSE ≤ {target_rmse*1000:.2f}mm")

    # ─ 尺度补偿 ─
    scale, raw_ratio = 1.0, 1.0
    src_pts_raw = np.asarray(pcd_src.points).copy()

    if not no_scale:
        scale, raw_ratio = estimate_scale(pcd_ref, pcd_src)
        if abs(scale - 1.0) > 0.01:
            print(f"  尺度补偿: raw={raw_ratio:.4f} → scale={scale}")
            pcd_src.points = o3d.utility.Vector3dVector(src_pts_raw * scale)
        else:
            scale, raw_ratio = 1.0, 1.0

    # ─ 自动参数 (尺度补偿后) ─
    if voxel_size <= 0 or max_icp_dist <= 0:
        vs_a, icp_a = auto_params(pcd_ref, pcd_src)
        voxel_size = voxel_size or vs_a
        max_icp_dist = max_icp_dist or icp_a
        print(f"  自动参数: voxel={voxel_size:.4f}  icp_dist={max_icp_dist:.4f}")

    # ─ RANSAC 全局 ─
    T = np.eye(4)
    if do_global:
        print("  FPFH + RANSAC ...")
        sd, sf = preprocess_pcd(pcd_src, voxel_size)
        rd, rf = preprocess_pcd(pcd_ref, voxel_size)
        g = global_registration(sd, rd, sf, rf, voxel_size)
        T = g.transformation
        print(f"    粗配准 fitness={g.fitness:.4f}  RMSE={g.inlier_rmse:.6f}")

    # ─ 多尺度 Point-to-Plane ICP (采样点云) ─
    print("  多尺度 ICP ...")
    for factor in ICP_SCALES:
        result = refine_icp(pcd_src, pcd_ref, T, max_icp_dist * factor)
        T = result.transformation
    print(f"    P2Plane fitness={result.fitness:.4f}  RMSE={result.inlier_rmse*1000:.4f}mm")

    # ─ 尺度微调 (仅当 snap 不精确时触发) ─
    if scale != 1.0:
        refined = _refine_scale(pcd_ref, src_pts_raw, scale, raw_ratio, T, max_icp_dist)
        if refined != scale:
            print(f"  尺度微调: {scale} → {refined:.6f}")
            scale = refined
            pcd_src.points = o3d.utility.Vector3dVector(src_pts_raw * scale)
            for factor in ICP_SCALES[-2:]:
                result = refine_icp(pcd_src, pcd_ref, T, max_icp_dist * factor)
                T = result.transformation
            print(f"    微调后 RMSE={result.inlier_rmse*1000:.4f}mm")

    # ─ 全量顶点超精细 ICP (核心精度阶段) ─
    T, fitness, rmse = ultra_fine_icp(ref_path, src_path, T, scale, target_rmse)

    # ─ 精度评估 ─
    diag = max(ref_diag, 1e-6)
    rmse_mm = rmse * 1000
    pct = rmse / diag * 100
    tag = "✓" if rmse <= target_rmse else "⚠"
    print(f"  {tag} 最终精度: RMSE={rmse_mm:.4f}mm  ({pct:.4f}%)  目标≤{target_rmse*1000:.2f}mm")

    # ─ 写出 ─
    print(f"  输出: {out_path}")
    if src_path.suffix.lower() == ".obj":
        apply_transform_to_obj(src_path, out_path, T, scale)
    else:
        tm = _load_trimesh(src_path)
        if scale != 1.0:
            tm.vertices *= scale
        tm.apply_transform(T)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tm.export(str(out_path))

    return dict(
        ref=str(ref_path), src=str(src_path), out=str(out_path),
        transform=T.tolist(), scale=scale,
        fitness=fitness, rmse=rmse, rmse_mm=rmse_mm,
        precision_pct=pct,
    )


# ── 批量对齐 ─────────────────────────────────────────────────────────────────


def batch_align(
    ref_dir: Path, src_dir: Path, out_dir: Path,
    ref_ext: str = ".STL", src_ext: str = ".obj", **kw,
) -> list[dict]:
    ref_files = sorted(ref_dir.glob(f"*{ref_ext}"))
    results, ok, fail = [], 0, 0

    for rf in ref_files:
        name = rf.stem
        sf = src_dir / f"{name}{src_ext}"
        if not sf.exists():
            cands = [f for f in src_dir.iterdir()
                     if f.stem.lower() == name.lower()
                     and f.suffix.lower() == src_ext.lower()]
            sf = cands[0] if cands else None
        if sf is None:
            print(f"  [跳过] {name}: 无对应 {src_ext}")
            continue

        print(f"\n{'=' * 60}\n对齐: {name}\n{'=' * 60}")
        try:
            r = align_one_pair(rf, sf, out_dir / f"{name}.obj", **kw)
            results.append(r)
            ok += 1
            print(f"  ✓ fitness={r['fitness']:.4f}  RMSE={r['rmse_mm']:.4f}mm  ({r['precision_pct']:.4f}%)")
        except Exception as e:
            print(f"  ✗ {e}", file=sys.stderr)
            fail += 1
            results.append(dict(ref=str(rf), src=str(sf), error=str(e)))

    print(f"\n{'=' * 60}\n完成: {ok}/{ok + fail} 成功, {fail} 失败\n{'=' * 60}")

    record = out_dir / "alignment_record.json"
    record.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"记录: {record}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="高精度模型 → 低精度参考系对齐 (FPFH + RANSAC + ICP + GICP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g1 = p.add_argument_group("单对模式")
    g1.add_argument("--ref", type=Path, help="参考网格 (STL/OBJ)")
    g1.add_argument("--src", type=Path, help="待对齐网格 (OBJ)")
    g1.add_argument("--out", type=Path, help="输出路径")

    g2 = p.add_argument_group("批量模式")
    g2.add_argument("--ref-dir", type=Path)
    g2.add_argument("--src-dir", type=Path)
    g2.add_argument("--out-dir", type=Path)
    g2.add_argument("--ref-ext", default=".STL")
    g2.add_argument("--src-ext", default=".obj")

    g3 = p.add_argument_group("算法参数")
    g3.add_argument("--voxel-size", type=float, default=0, help="0=自动")
    g3.add_argument("--max-icp-dist", type=float, default=0, help="0=自动")
    g3.add_argument("--num-points", type=int, default=DEFAULT_POINTS)
    g3.add_argument("--no-global", action="store_true", help="跳过全局配准")
    g3.add_argument("--no-scale", action="store_true", help="禁用自动尺度补偿")
    g3.add_argument("--target-rmse", type=float, default=DEFAULT_TARGET_RMSE,
                    help=f"目标 RMSE 精度, 单位米 (默认 {DEFAULT_TARGET_RMSE}={DEFAULT_TARGET_RMSE*1000}mm)")

    a = p.parse_args()
    kw = dict(
        voxel_size=a.voxel_size, max_icp_dist=a.max_icp_dist,
        n_points=a.num_points, do_global=not a.no_global, no_scale=a.no_scale,
        target_rmse=a.target_rmse,
    )

    if a.ref_dir and a.src_dir:
        out = a.out_dir or a.src_dir.parent / "meshes_output"
        out.mkdir(parents=True, exist_ok=True)
        batch_align(a.ref_dir, a.src_dir, out, a.ref_ext, a.src_ext, **kw)
    elif a.ref and a.src:
        out = a.out or a.src.with_name(f"{a.src.stem}_aligned.obj")
        r = align_one_pair(a.ref, a.src, out, **kw)
        print(f"\n完成! fitness={r['fitness']:.4f}  RMSE={r['rmse_mm']:.4f}mm  精度={r['precision_pct']:.4f}%")
        print(f"变换矩阵:\n{np.array(r['transform'])}")
    else:
        p.error("请指定 --ref/--src 或 --ref-dir/--src-dir")


if __name__ == "__main__":
    main()

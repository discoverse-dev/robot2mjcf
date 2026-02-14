"""根据 mapping.json 判断一个独立 mesh 属于 OBJ 模型的哪个 part 并做坐标转换保存。

使用流程示例:
    单个 mesh:
        python single_mesh.obj \\
            -g path/to/merged_model/visual.obj \\
            -m path/to/mapping.json \\
            -o output_dir

    多子部件 OBJ (自动拆分各 geometry):
        python multi_parts.obj \\
            -g path/to/merged_model/visual.obj \\
            -m path/to/mapping.json \\
            -o output_dir

主要步骤:
1. 读取输入 mesh (若为多子部件 OBJ 则展开)，对每个子部件计算其 AABB。
2. 读取 mapping.json 中每个 part 的 AABB, 先做 AABB "包含" 粗筛 (要求 mesh AABB 被 part AABB 完全包裹, 允许 eps 松弛)。
3. 对候选 part, 使用 KD-Tree 最近点距离做精细匹配。
4. 选出最佳 part.
5. 将每个匹配到的子部件 mesh 根据其 part 的 transform_4x4 变换到局部坐标系后保存: out_dir / <part_name> / <子部件名>.obj；保留材质 (若存在)。
6. 输出匹配结果统计。

注意: mapping.json 中 transform_4x4 视为 part->world 变换 (列/行主假设为常见行主 4x4)。
"""

from __future__ import annotations

import sys
import json
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import multiprocessing as mp

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("需要 scipy 依赖: pip install scipy", file=sys.stderr)
    raise

try:
    import trimesh
    from trimesh.exchange import obj as obj_io
    from trimesh.visual.material import SimpleMaterial  # type: ignore
except Exception as exc:
    print("需要 trimesh 依赖: pip install trimesh", file=sys.stderr)
    raise

MAX_FACE_COUNT = 180_000  # 超过 18 万面时自动降采样


@dataclass
class PartInfo:
    name: str
    aabb_min: np.ndarray  # (3,)
    aabb_max: np.ndarray  # (3,)
    transform: np.ndarray  # (4,4) part->world

    def contains_aabb(self, other_min: np.ndarray, other_max: np.ndarray, eps: float = 0.0) -> bool:
        """判断 other AABB 是否被当前 part AABB 完整包含 (带 eps 宽松)。"""
        return bool(
            np.all(self.aabb_min - eps <= other_min) and np.all(other_max <= self.aabb_max + eps)
        )

@dataclass
class PreParsedOBJ:
    """Holds pre-parsed data from an OBJ file for text-level operations."""
    all_v: list[str]                                      # "v x y z" raw strings
    all_vt: list[str]                                     # "vt u v" raw strings
    all_vn: list[str]                                     # "vn nx ny nz" raw strings
    object_faces: dict[str, list[tuple[str, str | None]]] # name → [(face_line, material)]
    object_meshes: dict[str, trimesh.Trimesh]              # name → trimesh (v+f, for matching)
    object_materials: dict[str, str]                       # name → dominant material


def load_mapping(path: Path) -> list[PartInfo]:
    data = json.loads(path.read_text())
    parts: list[PartInfo] = []
    for p in data.get("parts", []):
        try:
            name = p["name"]
            aabb = p["aabb"]
            t = p["transform_4x4"]
        except KeyError:
            continue
        aabb_min = np.array(aabb["min"], dtype=float)
        aabb_max = np.array(aabb["max"], dtype=float)
        transform = np.array(t, dtype=float).reshape(4, 4)
        parts.append(PartInfo(name=name, aabb_min=aabb_min, aabb_max=aabb_max, transform=transform))
    return parts

def compute_aabb(mesh: "trimesh.Trimesh") -> tuple[np.ndarray, np.ndarray]:
    return mesh.bounds[0].copy(), mesh.bounds[1].copy()

def load_mesh_any(path: Path) -> "trimesh.Trimesh":
    loaded = trimesh.load(str(path), force="mesh", skip_materials=True)
    if isinstance(loaded, trimesh.Scene):  # 退化
        if not loaded.geometry:  # pragma: no cover
            raise ValueError(f"空场景: {path}")
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):  # pragma: no cover
        raise TypeError("无法加载为 mesh")
    return loaded


def _preparse_obj_full(path: Path) -> PreParsedOBJ:
    """Parse an OBJ file at the text level, preserving ALL data per object.

    This is the **primary** OBJ parser. It:
      - Preserves original object names (unlike trimesh which may silently rename)
      - Preserves per-face material assignments from ``usemtl`` directives
      - Retains raw v/vt/vn text lines for lossless text-level export
      - Builds lightweight trimesh objects (v+f only) for AABB / KD-Tree matching
    """
    text = path.read_text(errors='ignore').splitlines()

    all_v: list[str] = []
    all_vt: list[str] = []
    all_vn: list[str] = []
    object_faces: dict[str, list[tuple[str, str | None]]] = {}
    current_obj: str | None = None
    current_mtl: str | None = None

    for line in text:
        if line.startswith('v '):
            all_v.append(line)
        elif line.startswith('vt '):
            all_vt.append(line)
        elif line.startswith('vn '):
            all_vn.append(line)
        elif line.startswith('o ') or line.startswith('g '):
            name = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else f'unnamed_{len(object_faces)}'
            current_obj = name
            object_faces.setdefault(current_obj, [])
        elif line.startswith('usemtl '):
            current_mtl = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else None
        elif line.startswith('f ') and current_obj is not None:
            object_faces[current_obj].append((line, current_mtl))

    # Build per-object lightweight trimesh (v+f) for matching
    v_arr = np.zeros((len(all_v), 3), dtype=np.float64)
    for i, vline in enumerate(all_v):
        parts = vline.split()
        v_arr[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

    object_meshes: dict[str, trimesh.Trimesh] = {}
    object_materials: dict[str, str] = {}

    for obj_name, faces_data in object_faces.items():
        if not faces_data:
            continue
        # Determine dominant material
        mat_counts: dict[str, int] = {}
        for _, mtl in faces_data:
            if mtl:
                mat_counts[mtl] = mat_counts.get(mtl, 0) + 1
        if mat_counts:
            object_materials[obj_name] = max(mat_counts, key=mat_counts.get)  # type: ignore

        # Parse faces for trimesh (v index only)
        global_vi_set: set[int] = set()
        raw_faces: list[list[int]] = []
        for face_line, _ in faces_data:
            tokens = face_line.split()[1:]
            vis = []
            for tok in tokens:
                vi = int(tok.split('/')[0]) - 1  # 0-based
                vis.append(vi)
                global_vi_set.add(vi)
            if len(vis) == 3:
                raw_faces.append(vis)
            elif len(vis) > 3:
                for j in range(1, len(vis) - 1):
                    raw_faces.append([vis[0], vis[j], vis[j + 1]])

        if not raw_faces:
            continue
        sorted_vis = sorted(global_vi_set)
        g2l = {g: l for l, g in enumerate(sorted_vis)}
        local_verts = v_arr[sorted_vis]
        local_faces = [[g2l[vi] for vi in f] for f in raw_faces]
        mesh = trimesh.Trimesh(
            vertices=local_verts,
            faces=np.array(local_faces, dtype=np.int64),
            process=False,
        )
        mesh.metadata['name'] = obj_name
        object_meshes[obj_name] = mesh

    return PreParsedOBJ(
        all_v=all_v,
        all_vt=all_vt,
        all_vn=all_vn,
        object_faces=object_faces,
        object_meshes=object_meshes,
        object_materials=object_materials,
    )


def _raw_load_obj_separate(path: Path) -> dict:
    """使用底层 load_obj 强制拆分 (object/group) 且不按材质合并, 返回 name->Trimesh.

    与默认 trimesh.load 不同: 这里设置 split_object=True, split_group=True, group_material=False,
    这样即使材质相同也不会被合并。"""
    with path.open('r', errors='ignore') as f:
        data = obj_io.load_obj(
            f,
            split_object=True,
            split_group=True,
            group_material=False,
            skip_materials=False,
            merge_vertices=False,
        )
    geoms = data.get('geometry', {})
    out = {}
    iterable = geoms.items() if isinstance(geoms, dict) else enumerate(geoms)
    for key, entry in iterable:
        name = str(key)
        g = entry
        if isinstance(g, trimesh.Trimesh):
            out[name] = g
            continue
        if isinstance(g, dict):
            verts = g.get('vertices')
            faces = g.get('faces')
            if verts is None or faces is None:
                continue
            try:
                visual = g.get('visual')
                tm = trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)
                tm.metadata['name'] = str(name)
                out[name] = tm
            except Exception:
                continue
    return out


def load_obj_parts(obj_path: Path):
    """加载 OBJ 按 object/group 拆分, 确保不因同材质合并。"""
    try:
        preparsed = _preparse_obj_full(obj_path)
        if preparsed.object_meshes:
            return preparsed.object_meshes
    except Exception:
        pass
    try:
        mapping = _raw_load_obj_separate(obj_path)
        if mapping:
            return mapping
    except Exception:
        pass
    scene = trimesh.load(str(obj_path), force='scene', skip_materials=True)
    if isinstance(scene, trimesh.Trimesh):
        return {scene.metadata.get('name', 'mesh'): scene}
    mapping = {}
    for name, geom in scene.geometry.items():
        clean = str(name)
        if clean in mapping:
            mapping[clean] = trimesh.util.concatenate([mapping[clean], geom])
        else:
            mapping[clean] = geom
    return mapping


def _text_export_object(
    preparsed: PreParsedOBJ,
    object_name: str,
    out_path: Path,
    inv_transform: np.ndarray,
    original_mtl_map: dict,
):
    """Export a single object from pre-parsed OBJ to a component OBJ file.

    Applies the inverse transform to vertices and normals at the TEXT level,
    perfectly preserving UVs, materials, and all OBJ structure.
    """
    faces_data = preparsed.object_faces.get(object_name)
    if not faces_data:
        return False

    rot = inv_transform[:3, :3]
    trans = inv_transform[:3, 3]

    # Collect all unique v/vt/vn indices referenced by this object's faces (1-based)
    v_set: set[int] = set()
    vt_set: set[int] = set()
    vn_set: set[int] = set()
    for face_line, _ in faces_data:
        for tok in face_line.split()[1:]:
            parts = tok.split('/')
            v_set.add(int(parts[0]))
            if len(parts) > 1 and parts[1]:
                vt_set.add(int(parts[1]))
            if len(parts) > 2 and parts[2]:
                vn_set.add(int(parts[2]))

    # Sorted indices → remap to local 1-based
    v_sorted = sorted(v_set)
    vt_sorted = sorted(vt_set)
    vn_sorted = sorted(vn_set)
    v_map = {old: new + 1 for new, old in enumerate(v_sorted)}
    vt_map = {old: new + 1 for new, old in enumerate(vt_sorted)}
    vn_map = {old: new + 1 for new, old in enumerate(vn_sorted)}

    out_lines = [f'mtllib {out_path.stem}.mtl', f'o {object_name}']

    # Write transformed vertices
    for old_idx in v_sorted:
        vline = preparsed.all_v[old_idx - 1]
        parts = vline.split()
        pt = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        pt_local = rot @ pt + trans
        out_lines.append(f'v {pt_local[0]:.6f} {pt_local[1]:.6f} {pt_local[2]:.6f}')

    # Write texture coords (unchanged)
    for old_idx in vt_sorted:
        out_lines.append(preparsed.all_vt[old_idx - 1])

    # Write transformed normals (rotation only)
    for old_idx in vn_sorted:
        nline = preparsed.all_vn[old_idx - 1]
        parts = nline.split()
        n = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        n_local = rot @ n
        length = np.linalg.norm(n_local)
        if length > 1e-12:
            n_local /= length
        out_lines.append(f'vn {n_local[0]:.6f} {n_local[1]:.6f} {n_local[2]:.6f}')

    # Write faces with usemtl (remapped indices)
    current_written_mtl = None
    for face_line, mtl in faces_data:
        if mtl and mtl != current_written_mtl:
            out_lines.append(f'usemtl {mtl}')
            current_written_mtl = mtl
        tokens = face_line.split()
        new_tokens = ['f']
        for tok in tokens[1:]:
            parts = tok.split('/')
            new_v = str(v_map[int(parts[0])])
            new_vt = ''
            new_vn = ''
            if len(parts) > 1 and parts[1]:
                new_vt = str(vt_map[int(parts[1])])
            if len(parts) > 2 and parts[2]:
                new_vn = str(vn_map[int(parts[2])])
            if new_vn:
                new_tokens.append(f'{new_v}/{new_vt}/{new_vn}')
            elif new_vt:
                new_tokens.append(f'{new_v}/{new_vt}')
            else:
                new_tokens.append(new_v)
        out_lines.append(' '.join(new_tokens))

    # Write OBJ
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out_lines) + '\n')

    # Write MTL with correct material blocks from original
    mtl_path = out_path.with_suffix('.mtl')
    used_mats: list[str] = []
    seen: set[str] = set()
    for _, mtl in faces_data:
        if mtl and mtl not in seen:
            seen.add(mtl)
            used_mats.append(mtl)
    if original_mtl_map and used_mats:
        mtl_blocks: list[str] = []
        for m in used_mats:
            block = original_mtl_map.get(m)
            if block:
                mtl_blocks.extend([ln for ln in block if not ln.strip().lower().startswith('map_')])
                if mtl_blocks and not mtl_blocks[-1].startswith('newmtl'):
                    mtl_blocks.append('')
        if mtl_blocks:
            mtl_path.write_text('\n'.join(mtl_blocks).rstrip() + '\n')

    return True


def build_part_kdtrees(obj_parts: dict) -> dict[str, cKDTree]:
    """为每个 part 几何体预构建 KD-Tree (用于线程安全的最近点查询)。"""
    trees: dict[str, cKDTree] = {}
    for name, geom in obj_parts.items():
        if geom is None or len(geom.vertices) == 0:
            continue
        trees[name] = cKDTree(geom.vertices)
    return trees


def _resolve_part_geom_name(part_name: str, obj_parts: dict) -> str | None:
    """精确匹配或大小写不敏感匹配 part 名称到 obj_parts 的 key。"""
    if part_name in obj_parts:
        return part_name
    norm = part_name.strip().lower()
    for n in obj_parts:
        if n.strip().lower() == norm:
            return n
    return None


def pick_best_part(
    candidates: list[PartInfo],
    obj_parts: dict,
    input_mesh: "trimesh.Trimesh",
    sample: int,
    kdtrees: dict | None = None,
) -> PartInfo:
    """从候选 parts 中选出与 input_mesh 最匹配的 part。

    使用 KD-Tree 最近点距离 (若提供 kdtrees) 或回退到 AABB 质心距离。
    """
    if len(candidates) == 1:
        return candidates[0]

    verts = input_mesh.vertices
    if len(verts) > sample:
        pts = verts[np.random.default_rng(0).choice(len(verts), size=sample, replace=False)]
    else:
        pts = verts

    centroid = verts.mean(axis=0)

    best = None
    best_score = float('inf')

    for part in candidates:
        geom_key = _resolve_part_geom_name(part.name, obj_parts)
        if geom_key is None:
            continue

        if kdtrees and geom_key in kdtrees:
            dists, _ = kdtrees[geom_key].query(pts, k=1, workers=1)
            score = float(np.median(dists))
        else:
            geom = obj_parts.get(geom_key)
            if geom is None:
                continue
            gmin, gmax = geom.bounds
            center_g = 0.5 * (gmin + gmax)
            score = float(np.linalg.norm(center_g - centroid))

        if score < best_score:
            best_score = score
            best = part

    if best is None:
        raise RuntimeError("未能匹配到任何候选 part (细化阶段全部失败)")
    return best


def transform_to_local(mesh: "trimesh.Trimesh", transform_part_to_world: np.ndarray) -> "trimesh.Trimesh":
    """将 mesh 从世界坐标变换到 part 局部坐标。"""
    T_inv = np.linalg.inv(transform_part_to_world)
    verts_h = np.hstack([mesh.vertices, np.ones((len(mesh.vertices), 1))])
    verts_local = (T_inv @ verts_h.T).T[:, :3]
    local_mesh = mesh.copy()
    local_mesh.vertices[:] = verts_local
    return local_mesh



def _sanitize_obj_mtl(obj_path: Path):
    """确保 OBJ 的 mtllib 引用 <obj_stem>.mtl, 必要时重命名 MTL 文件并写回 OBJ。"""
    if not obj_path.exists():
        return
    try:
        text = obj_path.read_text(errors='ignore').splitlines()
    except Exception:
        return

    desired_mtl = obj_path.stem + '.mtl'
    desired_line = f'mtllib {desired_mtl}'
    mtl_path = None
    modified = False

    # 查找已有 mtllib 行
    for i, line in enumerate(text):
        if line.startswith('mtllib '):
            parts_line = line.strip().split(maxsplit=1)
            if len(parts_line) == 2:
                mtl_path = obj_path.parent / parts_line[1].strip()
            if line.strip() != desired_line:
                text[i] = desired_line
                modified = True
            break
    else:
        # 无 mtllib 行 — 查找孤立 mtl 文件
        for cand in [obj_path.with_suffix('.mtl'), obj_path.parent / 'material.mtl']:
            if cand.exists():
                mtl_path = cand
                text.insert(0, desired_line)
                modified = True
                break

    # 重命名 mtl 文件以匹配 obj 基名
    if mtl_path and mtl_path.exists() and mtl_path.name != desired_mtl:
        target = obj_path.parent / desired_mtl
        try:
            if target.exists():
                target.unlink()
            mtl_path.rename(target)
        except Exception:
            pass

    if modified:
        try:
            obj_path.write_text('\n'.join(text) + '\n')
        except Exception:
            pass


MIN_GROUP_FACES = 10000  # 面数低于此值的材质组不降采样


def _decimate_submesh(v_arr: np.ndarray, f_arr: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """对单个子网格做 QEM 降采样, 返回 (new_vertices, new_faces)。

    优先 pymeshlab (保边界/法线/拓扑), 回退到 trimesh。
    """
    n = len(f_arr)
    if n <= target_faces:
        return v_arr, f_arr

    # pymeshlab: MeshLab Quadric Edge Collapse — 保边界/法线/拓扑, 平面区域优先简化
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(v_arr.astype(np.float64), f_arr.astype(np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True, preservenormal=True, preservetopology=True,
            optimalplacement=True, planarquadric=True, planarweight=0.002,
            qualitythr=0.5,
        )
        m = ms.current_mesh()
        return m.vertex_matrix(), m.face_matrix()
    except Exception:
        pass

    # 兜底: trimesh
    result = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False
                             ).simplify_quadric_decimation(target_faces)
    return np.asarray(result.vertices), np.asarray(result.faces)


def _decimate_obj_if_needed(obj_path: Path, max_faces: int = MAX_FACE_COUNT):
    """若 OBJ 三角面数 > max_faces, 按材质分组独立 QEM 降采样 (不跨材质合并面片)。

    小组 (≤ MIN_GROUP_FACES) 原样保留, 缩减预算集中分配给大组。
    降采样前备份原始文件为 xxx-origin.obj/mtl。
    """
    if not obj_path.exists():
        return
    try:
        raw_lines = obj_path.read_text(errors='ignore').splitlines()
    except Exception:
        return

    try:
        # ---- 1. 解析 OBJ: 顶点 + 按材质分组的面 ----
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
                current_mtl = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else None
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
                elif len(vis) > 3:  # fan 三角化
                    for j in range(1, len(vis) - 1):
                        grp.append([vis[0], vis[j], vis[j + 1]])

        if not all_vertices or not material_faces:
            return

        v_arr = np.array(all_vertices, dtype=np.float64)
        total_faces = sum(len(fl) for fl in material_faces.values())
        if total_faces <= max_faces:
            return

        # 备份原始文件
        origin_obj = obj_path.with_stem(obj_path.stem + '-origin')
        shutil.copy2(obj_path, origin_obj)
        mtl_src = obj_path.with_suffix('.mtl')
        if mtl_src.exists():
            shutil.copy2(mtl_src, mtl_src.with_stem(mtl_src.stem + '-origin'))

        print(f"    [降采样] {obj_path.name}: {total_faces} -> {max_faces} faces "
              f"(原始备份: {origin_obj.name})")

        # ---- 2. 分配面数预算: 小组保留, 大组按比例缩减 ----
        keep_faces = sum(len(fl) for fl in material_faces.values() if len(fl) <= MIN_GROUP_FACES)
        large_total = total_faces - keep_faces
        budget = max(0, max_faces - keep_faces)

        group_targets: dict[str | None, int] = {}
        for name, fl in material_faces.items():
            ng = len(fl)
            if ng <= MIN_GROUP_FACES:
                group_targets[name] = ng
            elif large_total > 0:
                group_targets[name] = min(ng, max(MIN_GROUP_FACES, round(budget * ng / large_total)))
            else:
                group_targets[name] = ng

        # ---- 3. 按材质组独立降采样 ----
        group_results: dict[str | None, tuple[np.ndarray, np.ndarray]] = {}
        for mtl_name in material_order:
            fl = material_faces[mtl_name]
            if not fl:
                continue
            g_faces = np.array(fl, dtype=np.int64)
            unique_vis = np.unique(g_faces.ravel())
            g2l = {int(g): l for l, g in enumerate(unique_vis)}
            local_v = v_arr[unique_vis]
            local_f = np.vectorize(g2l.get)(g_faces).astype(np.int64)
            target = group_targets[mtl_name]
            if len(local_f) <= target:
                group_results[mtl_name] = (local_v, local_f)
            else:
                group_results[mtl_name] = _decimate_submesh(local_v, local_f, target)

        # ---- 4. 拼合输出 OBJ ----
        out_lines: list[str] = []
        if mtllib_line:
            out_lines.append(mtllib_line)

        v_offset = 0
        group_v_offsets: dict[str | None, int] = {}
        for mtl_name in material_order:
            if mtl_name not in group_results:
                continue
            gv, _ = group_results[mtl_name]
            group_v_offsets[mtl_name] = v_offset
            for v in gv:
                out_lines.append(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}')
            v_offset += len(gv)

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
            for f in gf:
                out_lines.append(f'f {f[0]+off+1} {f[1]+off+1} {f[2]+off+1}')
            actual_total += len(gf)

        obj_path.write_text('\n'.join(out_lines) + '\n')
        print(f"    [降采样完成] {obj_path.name}: {actual_total} faces "
              f"({len(group_results)} 个材质组)")

    except Exception as exc:
        print(f"    [WARN] 降采样失败 {obj_path.name}: {exc}", file=sys.stderr)


def _parse_original_mtls(obj_path: Path) -> dict:
    """解析输入 OBJ 引用的 mtl 文件, 返回 {material_name: [lines]} 去除贴图行(map_*)。
    若找不到 mtllib 则返回空。"""
    base_dir = obj_path.parent
    try:
        lines = obj_path.read_text(errors='ignore').splitlines()
    except Exception:
        return {}
    mtl_files = []
    for l in lines:
        if l.startswith('mtllib '):
            name = l.split(maxsplit=1)[1].strip()
            if name:
                for part in name.split():  # 支持多个
                    mtl_files.append(base_dir / part)
    materials = {}
    for mtl in mtl_files:
        if not mtl.exists():
            continue
        try:
            mlines = mtl.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        current = None
        buffer = []
        def _commit():
            if current and buffer:
                materials[current] = list(buffer)
        for raw in mlines:
            s = raw.strip()
            if s.lower().startswith('map_'):
                continue
            if s.startswith('newmtl '):
                _commit()
                current = s.split(maxsplit=1)[1].strip()
                buffer = [raw]
            else:
                buffer.append(raw)
        _commit()
    return materials


def _parse_obj_material_map(path: Path) -> dict:
    """解析 OBJ 中 o/g/usemtl -> material 统计映射 (返回 name->dominant_material)。"""
    try:
        lines = path.read_text(errors='ignore').splitlines()
    except Exception:
        return {}
    current_obj = None
    current_group = None
    current_mtl = None
    counts = {}
    def add_face():
        if current_mtl is None:
            return
        keys = []
        if current_obj and current_group:
            keys.append(current_group)
            keys.append(current_obj)
        elif current_obj:
            keys.append(current_obj)
        elif current_group:
            keys.append(current_group)
        for k in keys:
            d = counts.setdefault(k, {})
            d[current_mtl] = d.get(current_mtl, 0) + 1
    for line in lines:
        if not line or line.startswith('#'):
            continue
        if line.startswith('o '):
            current_obj = line.split(maxsplit=1)[1].strip()
        elif line.startswith('g '):
            current_group = line.split(maxsplit=1)[1].strip()
        elif line.startswith('usemtl '):
            current_mtl = line.split(maxsplit=1)[1].strip()
        elif line.startswith('f '):
            add_face()
    mapping = {}
    for k, d in counts.items():
        if not d:
            continue
        mat = max(d.items(), key=lambda kv: kv[1])[0]
        mapping[k] = mat
    return mapping


def _rebuild_mtl_from_original(obj_path: Path, original_materials: dict):
    """根据 obj 中实际使用的 usemtl 顺序，重建对应 mtl 文件内容 (移除贴图)。"""
    if not obj_path.exists() or not original_materials:
        return
    try:
        lines = obj_path.read_text(errors='ignore').splitlines()
    except Exception:
        return
    used = []
    seen = set()
    for l in lines:
        if l.startswith('usemtl '):
            nm = l.split(maxsplit=1)[1].strip()
            if nm and nm not in seen:
                seen.add(nm)
                used.append(nm)
    if not used:
        return
    mtl_path = obj_path.with_suffix('.mtl')
    blocks = []
    for nm in used:
        block = original_materials.get(nm)
        if not block:
            continue
        for raw in block:
            if raw.strip().lower().startswith('map_'):
                continue
            blocks.append(raw)
        if blocks and not blocks[-1].startswith('newmtl'):
            blocks.append('')
    if blocks:
        try:
            mtl_path.write_text('\n'.join(blocks).rstrip() + '\n')
        except Exception:
            pass


def _offset_idx(idx_str: str, offset: int) -> str:
    """OBJ 索引偏移: 空串/0 原样返回, 否则 int(idx_str)+offset。"""
    if not idx_str or idx_str == '0':
        return idx_str
    try:
        return str(int(idx_str) + offset)
    except ValueError:
        return idx_str


def _adjust_face_indices(face_line: str, v_offset: int, vt_offset: int, vn_offset: int) -> str:
    """调整 f 行内的 v/vt/vn 索引, 加上各自的偏移量。"""
    try:
        parts = face_line.strip().split()
        if len(parts) < 4 or parts[0] != 'f':
            return face_line
        out_tokens = ['f']
        for token in parts[1:]:
            if '/' not in token:
                out_tokens.append(str(int(token) + v_offset))
                continue
            a = token.split('/')
            v_str = a[0]
            vt_str = a[1] if len(a) >= 2 else ''
            vn_str = a[2] if len(a) >= 3 else ''
            v_new = _offset_idx(v_str, v_offset)
            vt_new = _offset_idx(vt_str, vt_offset)
            vn_new = _offset_idx(vn_str, vn_offset)
            if vn_str or len(a) == 3:
                out_tokens.append(f"{v_new}/{vt_new}/{vn_new}")
            elif vt_str:
                out_tokens.append(f"{v_new}/{vt_new}")
            else:
                out_tokens.append(v_new)
        return ' '.join(out_tokens)
    except Exception:
        return face_line


def _manual_merge_component_objs(part_name: str, component_paths: list[Path], out_dir: Path, original_materials: dict) -> Path | None:
    """合并多个子 OBJ 为一个, 保留 v/vt/vn/usemtl 结构并重建 MTL (去贴图)。"""
    if not component_paths:
        return None
    merged_obj = out_dir / f"{part_name}.obj"
    merged_mtl = out_dir / f"{part_name}.mtl"
    out_dir.mkdir(parents=True, exist_ok=True)

    v_offset = 0
    vt_offset = 0
    vn_offset = 0
    out_lines = [f"mtllib {merged_mtl.name}"]
    used_materials: list[str] = []
    used_set = set()
    for comp_idx, comp_path in enumerate(component_paths):
        path_obj = Path(comp_path)
        if not path_obj.exists():
            continue
        try:
            lines = path_obj.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        v_count = sum(1 for l in lines if l.startswith('v '))
        vt_count = sum(1 for l in lines if l.startswith('vt '))
        vn_count = sum(1 for l in lines if l.startswith('vn '))
        out_lines.append(f"# component {comp_idx}: {path_obj.name}")
        if not any(l.startswith('o ') for l in lines[:20]):
            out_lines.append(f"o {path_obj.stem}")
        # 第一遍: v/vt/vn (OBJ 要求顶点在面之前)
        for l in lines:
            if l.startswith('mtllib '):
                continue
            if l.startswith('v ') or l.startswith('vt ') or l.startswith('vn '):
                out_lines.append(l)
        # 第二遍: f/usemtl/o/g
        for l in lines:
            if l.startswith('mtllib '):
                continue
            if l.startswith('usemtl '):
                current_material = l.split(maxsplit=1)[1].strip()
                out_lines.append(l)
                if current_material and current_material not in used_set:
                    used_set.add(current_material)
                    used_materials.append(current_material)
                continue
            if l.startswith('o ') or l.startswith('g '):
                out_lines.append(l)
                continue
            if l.startswith('f '):
                adjusted = _adjust_face_indices(l, v_offset, vt_offset, vn_offset)
                out_lines.append(adjusted)
        v_offset += v_count
        vt_offset += vt_count
        vn_offset += vn_count
    try:
        merged_obj.write_text('\n'.join(out_lines).rstrip() + '\n')
    except Exception:
        return None
    if original_materials and used_materials:
        mtl_blocks = []
        for m in used_materials:
            block = original_materials.get(m)
            if not block:
                continue
            for raw in block:
                if raw.strip().lower().startswith('map_'):
                    continue
                mtl_blocks.append(raw)
        if mtl_blocks:
            try:
                merged_mtl.write_text('\n'.join(mtl_blocks).rstrip() + '\n')
            except Exception:
                pass
    return merged_obj if merged_obj.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="基于 mapping.json 将独立 mesh 或多子部件 OBJ 分配到对应 part 并输出局部坐标")
    parser.add_argument("mesh", help="输入 mesh (obj/stl 等); 若为 multi-geometry OBJ 将逐子部件处理")
    parser.add_argument("-g", "--groups", required=True, help="包含多个 part 的多组 OBJ (整体) 用于精细匹配")
    parser.add_argument("-m", "--mapping", required=True, help="mapping.json 路径")
    parser.add_argument("-o", "--outdir", required=False, default=None, help="输出目录 (默认: ./meshes_aligned)")
    parser.add_argument("--epsilon", type=float, default=5e-3, help="AABB 包含判断松弛 eps")
    parser.add_argument("--sample", type=int, default=500, help="精细检测采样点上限")
    parser.add_argument("-nw", "--num-workers", type=int, default=0, help="并行工作线程数 (1=单线程)")
    args = parser.parse_args(argv)

    mesh_path = Path(args.mesh)
    group_obj_path = Path(args.groups)
    mapping_path = Path(args.mapping)
    out_dir = Path(args.outdir) if args.outdir else (mesh_path.parent / "meshes_aligned")

    if not mesh_path.exists():
        parser.error(f"mesh 不存在: {mesh_path}")
    if not group_obj_path.exists():
        parser.error(f"group obj 不存在: {group_obj_path}")
    if not mapping_path.exists():
        parser.error(f"mapping 不存在: {mapping_path}")

    print("[1/7] 读取 mapping ...")
    parts = load_mapping(mapping_path)
    if not parts:
        parser.error("mapping.json 未包含 parts")
    print(f"    共 {len(parts)} 个 part")

    print("[2/7] 解析输入 mesh/scene ...")
    submeshes: list[tuple[str, trimesh.Trimesh]] = []
    multi = False
    original_mtl_map = {}
    name_to_material = {}
    preparsed_obj: PreParsedOBJ | None = None

    if mesh_path.suffix.lower() == '.obj':
        original_mtl_map = _parse_original_mtls(mesh_path)
        try:
            preparsed_obj = _preparse_obj_full(mesh_path)
            name_to_material = preparsed_obj.object_materials
            if len(preparsed_obj.object_meshes) > 1:
                multi = True
                for name, geom in preparsed_obj.object_meshes.items():
                    clean_name = str(name).replace('/', '_')
                    submeshes.append((clean_name, geom))
            elif len(preparsed_obj.object_meshes) == 1:
                only_name, only_mesh = next(iter(preparsed_obj.object_meshes.items()))
                submeshes.append((only_name, only_mesh))
            else:
                m = load_mesh_any(mesh_path)
                submeshes.append((mesh_path.stem, m))
        except Exception as exc:
            print(f"    [WARN] 文本级解析失败 ({exc}), 回退到 trimesh ...", file=sys.stderr)
            preparsed_obj = None
            name_to_material = _parse_obj_material_map(mesh_path)
            try:
                split_geoms = _raw_load_obj_separate(mesh_path)
            except Exception:
                split_geoms = {}
            if len(split_geoms) > 1:
                multi = True
                for name, geom in split_geoms.items():
                    clean_name = str(name).replace('/', '_')
                    submeshes.append((clean_name, geom))
            elif len(split_geoms) == 1:
                only_mesh = next(iter(split_geoms.values()))
                submeshes.append((mesh_path.stem, only_mesh))
            else:
                m = load_mesh_any(mesh_path)
                submeshes.append((mesh_path.stem, m))
    else:
        m = load_mesh_any(mesh_path)
        submeshes.append((mesh_path.stem, m))
    print(f"    子部件数量: {len(submeshes)} (multi={multi})")

    print("[3/7] 加载组 OBJ 子几何 + 构建 KD-Tree ...")
    obj_parts = load_obj_parts(group_obj_path)
    kdtrees = build_part_kdtrees(obj_parts)
    print(f"    组 OBJ 几何数量: {len(obj_parts)}, KD-Tree: {len(kdtrees)}")

    if args.num_workers == 0:
        args.num_workers = max(1, mp.cpu_count() // 2)

    print("[4/7] 遍历子部件并做 AABB 包含粗筛 + KD-Tree 精细匹配 ... (workers={})".format(args.num_workers))

    visual_parts_dedup: list[PartInfo] = []
    seen_names: set[str] = set()
    for p in parts:
        if p.name not in seen_names:
            seen_names.add(p.name)
            visual_parts_dedup.append(p)

    def _process_one(sub_name: str, sub_mesh: "trimesh.Trimesh"):
        in_min, in_max = compute_aabb(sub_mesh)
        candidates = [p for p in parts if p.contains_aabb(in_min, in_max, eps=args.epsilon)]
        if not candidates:
            try:
                best_part = pick_best_part(visual_parts_dedup, obj_parts, sub_mesh,
                                           sample=args.sample, kdtrees=kdtrees)
            except Exception as exc:
                return None, {
                    'submesh': sub_name,
                    'reason': f'no_part_contains_aabb_and_fallback_failed: {exc}',
                    'aabb': {'min': in_min.tolist(), 'max': in_max.tolist()},
                }
        else:
            try:
                best_part = pick_best_part(candidates, obj_parts, sub_mesh,
                                           sample=args.sample, kdtrees=kdtrees)
            except Exception as exc:
                return None, {'submesh': sub_name, 'reason': f'match_error: {exc}'}

        part_dir = out_dir / best_part.name
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / f"{sub_name}.obj"

        inv_transform = np.linalg.inv(best_part.transform)

        if preparsed_obj is not None and sub_name in preparsed_obj.object_faces:
            try:
                ok = _text_export_object(preparsed_obj, sub_name, out_path,
                                         inv_transform, original_mtl_map)
                if not ok:
                    return None, {'submesh': sub_name, 'reason': 'text_export_empty'}
            except Exception as exc:
                return None, {'submesh': sub_name, 'reason': f'text_export_failed: {exc}'}
        else:
            local_mesh = transform_to_local(sub_mesh, best_part.transform)
            correct_material = name_to_material.get(sub_name)
            if correct_material:
                try:
                    local_mesh.visual.material = SimpleMaterial(name=correct_material)
                except Exception:
                    pass
            try:
                local_mesh.export(str(out_path))
            except Exception:
                try:
                    local_mesh.visual = None  # type: ignore
                    local_mesh.export(str(out_path))
                except Exception as exc2:
                    return None, {'submesh': sub_name, 'reason': f'export_failed: {exc2}'}
            _sanitize_obj_mtl(out_path)

        rec = {
            'submesh': sub_name,
            'matched_part': best_part.name,
            'output': str(out_path),
            'aabb_input': {'min': in_min.tolist(), 'max': in_max.tolist()}
        }
        return rec, None

    assignments = []
    failures = []

    _iter_wrap = (lambda it, **kw: tqdm(it, **kw)) if tqdm else (lambda it, **kw: it)

    if args.num_workers == 1 or len(submeshes) == 1:
        for sub_name, sub_mesh in _iter_wrap(submeshes, desc="匹配子部件", unit="mesh"):
            rec, fail = _process_one(sub_name, sub_mesh)
            if rec:
                assignments.append(rec)
            if fail:
                failures.append(fail)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {ex.submit(_process_one, n, m): n for n, m in submeshes}
            for fut in _iter_wrap(as_completed(futs), total=len(futs), desc="匹配子部件", unit="mesh"):
                try:
                    rec, fail = fut.result()
                    if rec:
                        assignments.append(rec)
                    if fail:
                        failures.append(fail)
                except Exception as exc:  # pragma: no cover
                    failures.append({'submesh': futs[fut], 'reason': f'worker_exception: {exc}'})

    matched_count = len(assignments)
    failed_count = len(failures)

    if multi and assignments:
        print("[5/7] 合并同 part 子部件 -> 生成整体 OBJ+MTL ...")
        from collections import defaultdict
        part_groups = defaultdict(list)
        for rec in assignments:
            part_groups[rec['matched_part']].append(rec['output'])
        for part_name, paths in part_groups.items():
            try:
                merged_path = _manual_merge_component_objs(part_name, [Path(p) for p in paths], out_dir, original_mtl_map)
                if merged_path:
                    _decimate_obj_if_needed(merged_path)
                else:
                    print(f"    [WARN] part {part_name} 合并失败 (结果为空)", file=sys.stderr)
            except Exception as exc:
                print(f"    [WARN] 处理 part {part_name} 合并异常: {exc}", file=sys.stderr)
            part_dir = out_dir / part_name
            if part_dir.exists() and part_dir.is_dir():
                try:
                    shutil.rmtree(part_dir)
                except Exception as exc:
                    print(f"    [WARN] 删除 part 文件夹失败: {part_dir} ({exc})", file=sys.stderr)
    else:
        print("[5/7] 跳过合并 (非多子部件或无成功匹配)")
        for rec in assignments:
            _decimate_obj_if_needed(Path(rec['output']))

    # 汇总未匹配子部件成一个 OBJ
    if failures:
        fail_names = {f['submesh'] for f in failures if 'submesh' in f}
        scene_unmatched = trimesh.Scene()
        for sub_name, sub_mesh in submeshes:
            if sub_name not in fail_names:
                continue
            try:
                mat = getattr(sub_mesh.visual, 'material', None)
                mat_name = getattr(mat, 'name', None) or 'material_0'
                diffuse = getattr(mat, 'diffuse', None)
                sub_mesh.visual.material = SimpleMaterial(name=mat_name, diffuse=diffuse)
                scene_unmatched.add_geometry(sub_mesh, node_name=sub_name)
            except Exception:
                pass
        if scene_unmatched.geometry:
            unmatched_path = out_dir / 'unmatched.obj'
            try:
                scene_unmatched.export(str(unmatched_path))
                _sanitize_obj_mtl(unmatched_path)
                _rebuild_mtl_from_original(unmatched_path, original_mtl_map)
                print(f"[额外] 未匹配聚合输出: {unmatched_path}")
            except Exception as exc:
                print(f"[WARN] 未匹配聚合导出失败: {exc}", file=sys.stderr)

    print(f"[6/7] 统计: 成功 {matched_count}/{len(submeshes)}")
    if failures:
        print(f"[7/7] 有未匹配子部件: {failed_count}", file=sys.stderr)
        if matched_count == 0:
            return 3
    else:
        print("[7/7] 全部子部件匹配成功")

    print(f"保存路径: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

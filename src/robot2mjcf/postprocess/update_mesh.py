"""Defines a post-processing function that updates the mesh of the Mujoco model.

This script updates the mesh of the MJCF file.
"""

import argparse
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from robot2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def simplify_mesh_assets(mjcf_path: str | Path, max_vertices: int) -> None:
    """Update the mesh assets of the MJCF file.

    Args:
        root: The root element of the MJCF file.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    import pymeshlab

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.attrib["meshdir"] = "."

    dir_path = mjcf_path.parent / compiler.attrib["meshdir"]

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    for mesh in asset.findall("mesh"):
        mesh_file = dir_path / mesh.attrib["file"]
        if not mesh_file.exists():
            logger.warning(f"<update mesh> Mesh file {mesh_file} does not exist.")
            continue

        # 检查文件扩展名
        if mesh_file.suffix.lower() not in [".obj", ".stl"]:
            logger.warning(f"Unsupported mesh format: {mesh_file.suffix}")
            continue

        if mesh.attrib["file"].startswith("./"):
            mesh.attrib["file"] = str(mesh_file.relative_to(dir_path))

        try:
            # 使用PyMeshLab加载网格获取顶点数
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_file))

            max_simple_times = 5
            vertices = ms.current_mesh().vertex_matrix()
            if len(vertices) >= max_vertices:
                while len(vertices) >= max_vertices:
                    max_simple_times -= 1
                    if max_simple_times <= 0:
                        break
                    nm = min(max_vertices, len(vertices))
                    percentage = nm / max_vertices * 0.999
                    ms.meshing_decimation_clustering(threshold=pymeshlab.PercentageValue(percentage))
                    vertices = ms.current_mesh().vertex_matrix()
                ms.save_current_mesh(str(mesh_file))
                if os.path.exists(mesh_file.with_suffix(".mtl")):
                    os.remove(mesh_file.with_suffix(".mtl"))
                elif os.path.exists(str(mesh_file) + ".mtl"):
                    os.remove(str(mesh_file) + ".mtl")

        except Exception as e:
            logger.error(f"处理网格文件 {mesh_file} 时出错: {e}")
            continue

    save_xml(mjcf_path, root)


def remove_unused_mesh(mjcf_path: str | Path) -> None:
    """Remove the unused mesh of the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # 获取meshdir路径
    compiler = root.find("compiler")
    if compiler is None:
        logger.warning("No compiler element found in MJCF file")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir

    # 查找所有使用的mesh和material
    used_meshes = set()
    used_materials = set()

    # 1. 遍历所有geom元素，收集使用的mesh
    for geom in root.iter("geom"):
        mesh_name = geom.attrib.get("mesh")
        if mesh_name:
            used_meshes.add(mesh_name)

        material_name = geom.attrib.get("material")
        if material_name:
            used_materials.add(material_name)

    # 2. 遍历所有site元素，收集使用的material
    for site in root.iter("site"):
        material_name = site.attrib.get("material")
        if material_name:
            used_materials.add(material_name)

    # 3. 遍历所有body元素，收集使用的material
    for body in root.iter("body"):
        material_name = body.attrib.get("material")
        if material_name:
            used_materials.add(material_name)

    logger.info(f"使用中的mesh: {used_meshes}")
    logger.info(f"使用中的material: {used_materials}")

    # 获取asset元素
    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    # 收集要删除的mesh和对应文件
    meshes_to_remove = []
    mesh_files_to_remove = []

    for mesh in asset.findall("mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")

        if mesh_name not in used_meshes:
            logger.info(f"发现未使用的mesh: {mesh_name} (文件: {mesh_file})")
            meshes_to_remove.append(mesh)

            if mesh_file:
                # 添加mesh文件到删除列表
                full_mesh_path = mesh_dir_path / mesh_file
                if full_mesh_path.exists():
                    mesh_files_to_remove.append(full_mesh_path)

                # 如果是obj文件，检查对应的mtl文件
                if mesh_file.lower().endswith(".obj"):
                    mtl_file = full_mesh_path.with_suffix(".mtl")
                    if mtl_file.exists():
                        mesh_files_to_remove.append(mtl_file)

    # 收集要删除的material
    materials_to_remove = []

    for material in asset.findall("material"):
        material_name = material.attrib.get("name")

        if material_name not in used_materials:
            logger.info(f"发现未使用的material: {material_name}")
            materials_to_remove.append(material)

    # 删除未使用的mesh元素
    for mesh in meshes_to_remove:
        asset.remove(mesh)
        logger.info(f"已删除mesh元素: {mesh.attrib.get('name')}")

    # 删除未使用的material元素
    for material in materials_to_remove:
        asset.remove(material)
        logger.info(f"已删除material元素: {material.attrib.get('name')}")

    # 检查mesh目录中的所有mesh文件，删除未被引用的文件
    if mesh_dir_path.exists():
        # 收集所有被引用的文件名
        referenced_files = set()
        for mesh in asset.findall("mesh"):
            mesh_file = mesh.attrib.get("file")
            if mesh_file:
                referenced_files.add(mesh_file)
                # 如果是obj文件，也包含对应的mtl文件
                # if mesh_file.lower().endswith('.obj'):
                #     mtl_file = Path(mesh_file).with_suffix('.mtl').name
                #     referenced_files.add(str(Path(mesh_file).parent / mtl_file))

        # 查找所有mesh文件
        mesh_extensions = [".obj", ".stl", ".dae", ".mtl"]
        for mesh_file_path in mesh_dir_path.rglob("*"):
            if mesh_file_path.is_file() and mesh_file_path.suffix.lower() in mesh_extensions:
                # 计算相对于mesh目录的路径
                relative_path = mesh_file_path.relative_to(mesh_dir_path)
                relative_path_str = str(relative_path).replace("\\", "/")  # 统一使用正斜杠

                if relative_path_str not in referenced_files:
                    logger.info(f"发现未被引用的文件: {relative_path_str}")
                    mesh_files_to_remove.append(mesh_file_path)

    # 删除文件
    deleted_files = []
    for file_path in mesh_files_to_remove:
        try:
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))
                logger.info(f"已删除文件: {file_path}")
        except Exception as e:
            logger.error(f"删除文件 {file_path} 时出错: {e}")

    # 保存修改后的MJCF文件
    if meshes_to_remove or materials_to_remove:
        save_xml(mjcf_path, tree)
        logger.info(f"已更新MJCF文件: {mjcf_path}")

    # 总结
    logger.info("清理完成:")
    logger.info(f"  删除的mesh元素: {len(meshes_to_remove)}")
    logger.info(f"  删除的material元素: {len(materials_to_remove)}")
    logger.info(f"  删除的文件: {len(deleted_files)}")

    if deleted_files:
        logger.info("删除的文件列表:")
        for file_path in deleted_files:
            logger.info(f"  - {file_path}")


def merge_materials(mjcf_path: str | Path) -> None:
    """Merge materials with identical attributes in the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # 获取asset元素
    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    # 合并属性相同的材质
    # 1. 收集所有材质的属性签名和名称映射
    material_signature_map = {}  # 签名 -> 第一个材质名称
    material_name_to_signature = {}  # 材质名称 -> 签名

    logger.info("开始收集材质信息...")
    for material in asset.findall("material"):
        material_name = material.attrib.get("name")
        if not material_name:
            continue

        # 创建材质属性签名(排除name属性)
        attrib_items = sorted([(k, v) for k, v in material.attrib.items() if k != "name"])
        signature = str(attrib_items)

        logger.debug(f"材质 '{material_name}' 的签名: {signature}")

        material_name_to_signature[material_name] = signature

        if signature not in material_signature_map:
            material_signature_map[signature] = material_name
            logger.debug("  -> 首次遇到此签名，设为规范名称")
        else:
            # 发现重复的材质
            canonical_name = material_signature_map[signature]
            if material_name != canonical_name:
                logger.info(f"⚠️  发现重复材质: '{material_name}' 与 '{canonical_name}' 属性相同，将合并")

    # 2. 更新所有引用了重复材质的元素
    material_rename_map = {}  # 旧名称 -> 新名称
    for mat_name, signature in material_name_to_signature.items():
        canonical_name = material_signature_map[signature]
        if mat_name != canonical_name:
            material_rename_map[mat_name] = canonical_name

    logger.info(
        f"材质合并结果: 共扫描 {len(material_name_to_signature)} 个材质，发现 {len(material_rename_map)} 个重复材质"
    )

    if material_rename_map:
        logger.info(f"⚠️  材质重命名映射 (共 {len(material_rename_map)} 项):")
        for old_name, new_name in material_rename_map.items():
            logger.info(f"    {old_name} -> {new_name}")

        # 更新所有geom元素中的material引用
        for geom in root.iter("geom"):
            material_name = geom.attrib.get("material")
            if material_name and material_name in material_rename_map:
                new_name = material_rename_map[material_name]
                geom.attrib["material"] = new_name
                logger.info(f"更新geom的material引用: {material_name} -> {new_name}")

        # 更新所有site元素中的material引用
        for site in root.iter("site"):
            material_name = site.attrib.get("material")
            if material_name and material_name in material_rename_map:
                new_name = material_rename_map[material_name]
                site.attrib["material"] = new_name
                logger.info(f"更新site的material引用: {material_name} -> {new_name}")

        # 更新所有body元素中的material引用
        for body in root.iter("body"):
            material_name = body.attrib.get("material")
            if material_name and material_name in material_rename_map:
                new_name = material_rename_map[material_name]
                body.attrib["material"] = new_name
                logger.info(f"更新body的material引用: {material_name} -> {new_name}")

        # 从asset中删除重复的材质定义
        for material in list(asset.findall("material")):
            material_name = material.attrib.get("name")
            if material_name and material_name in material_rename_map:
                asset.remove(material)
                logger.info(f"已删除重复的material元素: {material_name}")

        # 保存修改后的MJCF文件
        save_xml(mjcf_path, tree)
        logger.info(f"✅ 材质合并完成，已更新MJCF文件: {mjcf_path}")


def remove_empty_or_invalid_meshes(mjcf_path: str | Path) -> None:
    """检测并移除顶点数为0的空mesh，同时删除worldbody中的引用。

    - 遍历`asset/mesh`并读取对应文件
    - 若顶点数为0：
      1) 记录warning日志
      2) 从`asset`中删除该`mesh`
      3) 从`worldbody`（实际为各级`body`下的`geom`）中删除引用该mesh的`geom`
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        logger.warning("No compiler element found in MJCF file")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir

    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    import pymeshlab

    # 收集空mesh及其名称
    empty_mesh_elements: list[ET.Element] = []
    empty_mesh_names: set[str] = set()

    for mesh in list(asset.findall("mesh")):
        mesh_name = mesh.attrib.get("name")
        mesh_file_attr = mesh.attrib.get("file")
        if not mesh_file_attr:
            continue

        mesh_file_path = mesh_dir_path / mesh_file_attr
        if not mesh_file_path.exists():
            continue

        # 仅处理常见的网格格式
        if mesh_file_path.suffix.lower() not in [".obj", ".stl"]:
            continue

        try:
            ws_path = os.getcwd()
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(mesh_file_path))
            vertices = ms.current_mesh().vertex_matrix()
            if len(vertices) == 0:
                # 记录warning并标记删除
                logger.warning(
                    f"检测到空mesh: name={mesh_name}, file={mesh_file_attr}. 将从asset和worldbody中删除引用。"
                )
                empty_mesh_elements.append(mesh)
                if mesh_name:
                    empty_mesh_names.add(mesh_name)
        except Exception as e:
            # 读取失败不等同于空mesh，这里只记录错误
            logger.error(f"读取mesh文件失败 {mesh_file_path}: {e}")
            os.chdir(ws_path)
            logger.warning(f"检测到非法mesh: name={mesh_name}, file={mesh_file_attr}. 将从asset和worldbody中删除引用。")
            empty_mesh_elements.append(mesh)
            if mesh_name:
                empty_mesh_names.add(mesh_name)

            continue

    if not empty_mesh_elements and not empty_mesh_names:
        return

    # 从worldbody（各级body）中删除引用这些mesh的geom
    # 需要父节点来执行remove
    if empty_mesh_names:
        for parent in root.iter():
            # 遍历直接子元素，避免跨层误删
            for child in list(parent):
                if child.tag == "geom":
                    ref_name = child.attrib.get("mesh")
                    if ref_name and ref_name in empty_mesh_names:
                        parent.remove(child)
                        logger.info(
                            f"已从{parent.tag}中删除引用空mesh '{ref_name}' 的geom: name={child.attrib.get('name')}"
                        )

    # 从asset中删除空mesh定义
    for mesh in empty_mesh_elements:
        try:
            mesh_name = mesh.attrib.get("name")
            asset.remove(mesh)
            logger.info(f"已从asset中删除空mesh定义: {mesh_name}")
        except Exception as e:
            logger.error(f"从asset删除mesh失败: {e}")

    # 保存修改
    save_xml(mjcf_path, tree)


def remove_empty_mesh_dirs(mjcf_path: str | Path) -> None:
    """Recursively remove empty directories under the meshdir declared in MJCF.

    This will walk the meshdir bottom-up and remove any empty subdirectories.
    The meshdir root itself will not be removed.
    """
    mjcf_path = Path(mjcf_path)
    try:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
    except Exception:
        logger.debug("无法解析 MJCF 文件以获取 meshdir，跳过空文件夹清理")
        return

    compiler = root.find("compiler")
    if compiler is None:
        logger.debug("No compiler element found in MJCF file; skipping empty directory cleanup")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir
    if not mesh_dir_path.exists():
        logger.debug(f"Mesh dir {mesh_dir_path} does not exist; skipping cleanup")
        return

    removed = []
    # Walk bottom-up so subdirectories are processed before their parents
    for dirpath, dirnames, filenames in os.walk(mesh_dir_path, topdown=False):
        p = Path(dirpath)
        # Don't remove the meshdir root itself
        if p == mesh_dir_path:
            continue
        try:
            # If directory is empty (no files and no subdirectories), remove it
            if not any(p.iterdir()):
                p.rmdir()
                removed.append(str(p))
        except Exception as e:
            logger.debug(f"Failed to remove directory {p}: {e}")

    if removed:
        logger.info(f"Removed {len(removed)} empty directories under {mesh_dir_path}")
        for d in removed:
            logger.info(f"  - {d}")


def merge_geoms_by_material(mjcf_path: str | Path) -> None:
    """合并同一body内具有相同材质属性的geom及其mesh文件.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        logger.warning("No compiler element found in MJCF file")
        return

    meshdir = compiler.attrib.get("meshdir", ".")
    mesh_dir_path = mjcf_path.parent / meshdir

    asset = root.find("asset")
    if asset is None:
        logger.warning("No asset element found in MJCF file")
        return

    import numpy as np
    import pymeshlab

    # 构建mesh名称到文件路径的映射
    mesh_name_to_file = {}
    for mesh in asset.findall("mesh"):
        mesh_name = mesh.attrib.get("name")
        mesh_file = mesh.attrib.get("file")
        if mesh_name and mesh_file:
            # 如果mesh_file已经包含meshdir，则直接使用；否则添加meshdir
            if mesh_file.startswith(meshdir):
                mesh_name_to_file[mesh_name] = mjcf_path.parent / mesh_file
            else:
                mesh_name_to_file[mesh_name] = mesh_dir_path / mesh_file

    # 遍历所有body
    merged_count = 0
    for body in root.iter("body"):
        body_name = body.attrib.get("name", "unnamed")

        # 收集该body下所有的geom，按材质属性分组
        material_to_geoms = {}  # 材质签名 -> [(geom_element, mesh_name, pose)]

        for geom in list(body.findall("geom")):
            if geom.attrib.get("type") != "mesh":
                continue

            mesh_name = geom.attrib.get("mesh")
            mesh_class = geom.attrib.get("class", "")
            if not mesh_name or mesh_class == "collision":
                continue

            # 创建材质属性签名
            material_attribs = []
            for key in ["material", "rgba", "class"]:
                if key in geom.attrib:
                    material_attribs.append((key, geom.attrib[key]))

            material_signature = str(sorted(material_attribs))

            # 获取geom的位姿
            pos = geom.attrib.get("pos", "0 0 0")
            quat = geom.attrib.get("quat", "1 0 0 0")
            euler = geom.attrib.get("euler", None)

            if material_signature not in material_to_geoms:
                material_to_geoms[material_signature] = []

            material_to_geoms[material_signature].append(
                {
                    "element": geom,
                    "mesh_name": mesh_name,
                    "pos": np.array([float(x) for x in pos.split()]),
                    "quat": np.array([float(x) for x in quat.split()]) if euler is None else None,
                    "euler": np.array([float(x) for x in euler.split()]) if euler is not None else None,
                    "material_attribs": material_attribs,
                }
            )

        # 对每个材质组，如果有多个geom，则合并
        for material_sig, geom_list in material_to_geoms.items():
            if len(geom_list) <= 1:
                continue  # 只有一个geom，无需合并

            logger.info(f"Body '{body_name}': 发现 {len(geom_list)} 个相同材质的geom，准备合并")

            try:
                # 创建合并后的mesh - 先加载所有mesh并应用变换
                all_transformed_meshes = []

                for i, geom_info in enumerate(geom_list):
                    mesh_name = geom_info["mesh_name"]
                    mesh_file = mesh_name_to_file.get(mesh_name)

                    if not mesh_file or not mesh_file.exists():
                        logger.warning(f"  Mesh文件不存在: {mesh_name}")
                        continue

                    # 加载mesh
                    temp_ms = pymeshlab.MeshSet()
                    temp_ms.load_new_mesh(str(mesh_file))

                    # 应用变换
                    pos = geom_info["pos"]

                    # 构建旋转矩阵
                    if geom_info["euler"] is not None:
                        # 从欧拉角构建旋转矩阵 (XYZ顺序)
                        euler = geom_info["euler"]
                        # MuJoCo使用的是extrinsic XYZ (相当于intrinsic ZYX)
                        cx, cy, cz = np.cos(euler)
                        sx, sy, sz = np.sin(euler)

                        # 旋转矩阵: Rz * Ry * Rx
                        rot_matrix = np.array(
                            [
                                [cy * cz, -cy * sz, sy],
                                [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
                                [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy],
                            ]
                        )
                    elif geom_info["quat"] is not None:
                        # 从四元数构建旋转矩阵 (w, x, y, z)
                        quat = geom_info["quat"]
                        w, x, y, z = quat

                        rot_matrix = np.array(
                            [
                                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                            ]
                        )
                    else:
                        rot_matrix = np.eye(3)

                    # 构建4x4变换矩阵
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = rot_matrix
                    transform_matrix[:3, 3] = pos

                    # 手动应用变换到顶点
                    vertices = temp_ms.current_mesh().vertex_matrix()
                    faces = temp_ms.current_mesh().face_matrix()

                    vertices_homo = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
                    transformed_vertices = (transform_matrix @ vertices_homo.T).T[:, :3]

                    # 保存变换后的顶点和面
                    all_transformed_meshes.append({"vertices": transformed_vertices, "faces": faces})

                    logger.debug(f"  Mesh {i}: {len(transformed_vertices)} 顶点, {len(faces)} 面")

                if len(all_transformed_meshes) == 0:
                    logger.warning(f"  Body '{body_name}': 无法加载任何mesh，跳过合并")
                    continue

                # 合并所有mesh的顶点和面
                vertex_offset = 0
                all_vertices = []
                all_faces = []

                for mesh_data in all_transformed_meshes:
                    all_vertices.append(mesh_data["vertices"])
                    # 更新面索引（添加顶点偏移）
                    adjusted_faces = mesh_data["faces"] + vertex_offset
                    all_faces.append(adjusted_faces)
                    vertex_offset += len(mesh_data["vertices"])

                combined_vertices = np.vstack(all_vertices)
                combined_faces = np.vstack(all_faces)

                logger.info(f"  合并后: {len(combined_vertices)} 顶点, {len(combined_faces)} 面")

                # 创建合并后的mesh
                merged_ms = pymeshlab.MeshSet()
                merged_mesh = pymeshlab.Mesh(combined_vertices, combined_faces)
                merged_ms.add_mesh(merged_mesh)

                # 生成新的mesh文件名 - 保存在第一个mesh文件所在的文件夹
                first_geom = geom_list[0]
                original_mesh_name = first_geom["mesh_name"]
                original_mesh_file = mesh_name_to_file.get(original_mesh_name)

                # 获取原mesh文件所在的文件夹
                if original_mesh_file and original_mesh_file.exists():
                    target_dir = original_mesh_file.parent
                else:
                    # 如果找不到原文件，则使用meshdir
                    target_dir = mesh_dir_path

                merged_mesh_name = f"{body_name}_merged_{original_mesh_name.split('_')[0]}"
                merged_mesh_file = target_dir / f"{merged_mesh_name}.obj"

                # 确保文件名唯一
                counter = 1
                while merged_mesh_file.exists():
                    merged_mesh_name = f"{body_name}_merged_{original_mesh_name.split('_')[0]}_{counter}"
                    merged_mesh_file = target_dir / f"{merged_mesh_name}.obj"
                    counter += 1

                # 保存合并后的mesh
                merged_ms.save_current_mesh(str(merged_mesh_file))
                logger.info(f"  保存合并后的mesh: {merged_mesh_file}")

                # 计算相对于MJCF文件的路径
                mesh_file_relative = str(merged_mesh_file.relative_to(mjcf_path.parent))

                # 在asset中添加新的mesh定义
                new_mesh_element = ET.Element("mesh", name=merged_mesh_name, file=mesh_file_relative)
                asset.append(new_mesh_element)

                # 更新第一个geom，删除其他geom
                first_geom_element = geom_list[0]["element"]
                first_geom_element.attrib["mesh"] = merged_mesh_name

                # 确保材质属性被保留（从material_attribs中恢复）
                material_attribs = geom_list[0]["material_attribs"]
                if material_attribs:
                    logger.info("  保留材质属性:")
                    for key, value in material_attribs:
                        first_geom_element.attrib[key] = value
                        logger.info(f"    {key} = {value}")

                # 重置位姿为原点（因为已经应用到mesh上了）
                first_geom_element.attrib["pos"] = "0 0 0"
                if "quat" in first_geom_element.attrib:
                    first_geom_element.attrib["quat"] = "1 0 0 0"
                if "euler" in first_geom_element.attrib:
                    del first_geom_element.attrib["euler"]

                # 删除其他geom
                for i in range(1, len(geom_list)):
                    body.remove(geom_list[i]["element"])

                merged_count += 1
                logger.info(f"  成功合并 {len(geom_list)} 个geom为 1 个")

            except Exception as e:
                logger.error(f"  合并body '{body_name}' 的geom时出错: {e}")
                import traceback

                traceback.print_exc()
                continue

    if merged_count > 0:
        save_xml(mjcf_path, tree)
        logger.info(f"✅ 合并完成: 共处理 {merged_count} 组geom")
    else:
        logger.info("未找到需要合并的geom")


def update_mesh(mjcf_path: str | Path, max_vertices: int = 1000000) -> None:
    """Update the mesh of the MJCF file.

    Args:
        mjcf_path: The path to the MJCF file to process.
    """
    remove_empty_or_invalid_meshes(mjcf_path)
    simplify_mesh_assets(mjcf_path, max_vertices)
    merge_materials(mjcf_path)
    merge_geoms_by_material(mjcf_path)
    remove_unused_mesh(mjcf_path)
    # 最后清理 meshdir 下的空文件夹（递归），但不删除 meshdir 根目录本身
    try:
        remove_empty_mesh_dirs(mjcf_path)
    except Exception as e:
        logger.warning(f"清理 meshdir 空文件夹时发生错误: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Updates the mesh of the MJCF file.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    parser.add_argument("--max-vertices", type=int, default=200000, help="Maximum number of vertices in the mesh.")
    args = parser.parse_args()
    update_mesh(args.mjcf_path, args.max_vertices)


if __name__ == "__main__":
    main()

"""为 Mujoco 模型更新网格的后处理脚本。

此模块将 MJCF 文件中的碰撞网格（class="collision"）转换为 STL 格式并更新对应的 asset 条目与 geom 引用。
仅修改文本和 asset 引用，不改变原有逻辑。
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from robot2mjcf.core.utils import save_xml

logger = logging.getLogger(__name__)


def collision_to_stl(mjcf_path: str | Path) -> None:
    """将 MJCF 文件中的碰撞网格转换为 STL 并更新 asset。

    参数:
        mjcf_path: 要处理的 MJCF 文件路径（Path 或字符串）。
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("MJCF file does not contain a compiler element")

    mesh_dir_path = mjcf_path.parent / compiler.attrib["meshdir"]
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MJCF file does not contain an asset element")

    asset_to_add: list[tuple[str, str]] = []
    for geom in root.iter("geom"):
        if geom.attrib.get("type") == "mesh":
            mesh_name = geom.attrib.get("mesh")
            class_name = geom.attrib.get("class")
            if class_name == "collision" and mesh_name is not None:
                for mesh in asset.findall("mesh"):
                    if mesh.attrib.get("name") == mesh_name:
                        mesh_file = mesh_dir_path / mesh.attrib["file"]
                        if not mesh_file.exists():
                            logger.error(f"网格文件 {mesh_file} 不存在。")
                            raise FileNotFoundError(f"网格文件 {mesh_file} 不存在。")

                        if Path(mesh_file).suffix.lower() != ".stl":
                            logger.info(f"将碰撞网格 {mesh_name} 转换为 STL 格式。")
                            # 尝试转换为 STL（如果已存在则跳过转换）
                            stl_file = mesh_file.with_suffix(".stl")
                            if not stl_file.exists():
                                try:
                                    import pymeshlab

                                    ms = pymeshlab.MeshSet()
                                    ms.load_new_mesh(str(mesh_file))
                                    ms.save_current_mesh(str(stl_file))
                                    logger.info(f"已将 {mesh_file} 转换为 {stl_file}")
                                except Exception as e:
                                    logger.error(f"将 {mesh_file} 转换为 STL 时出错: {e}")
                            # 更新 geom 的 mesh 属性为新的 stl 名称
                            geom.attrib["mesh"] = Path(mesh_name).with_suffix(".stl").name
                            logger.info(f"已更新 geom {geom.attrib.get('name')} 使用网格 {stl_file.name}")
                            # 将新的 asset 条目记录到待添加列表（稍后追加到 asset）
                            stl_rel_path = str(stl_file.relative_to(mesh_dir_path))
                            asset_to_add.append((geom.attrib["mesh"], stl_rel_path))

                            logger.info(f"已记录 asset 更新: {mesh.attrib.get('name')} -> {mesh.attrib.get('file')}")
                        break

    for mesh_name, mesh_file_attr in asset_to_add:
        asset.append(ET.Element("mesh", name=mesh_name, file=mesh_file_attr))

    save_xml(mjcf_path, tree)


def main() -> None:
    parser = argparse.ArgumentParser(description="Updates the mesh of the MJCF file.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file.")
    args = parser.parse_args()
    collision_to_stl(args.mjcf_path)


if __name__ == "__main__":
    main()

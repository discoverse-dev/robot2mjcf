"""Split OBJ files by materials and create separate geoms for each material."""

import argparse
import logging
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh

from robot2mjcf.materials import Material, parse_mtl_name
from robot2mjcf.postprocess.mesh_converter import dae2obj
from robot2mjcf.utils import save_xml

logger = logging.getLogger(__name__)


def process_obj_materials(obj_file: Path, files_to_delete: list = None) -> dict[str, Material]:
    """Process MTL materials from OBJ file and split by materials.

    Args:
        obj_file: Path to the OBJ file
        files_to_delete: List to collect files that should be deleted later

    Returns:
        Dictionary mapping material names to Material objects
    """
    if files_to_delete is None:
        files_to_delete = []

    materials = {}

    if not obj_file.exists():
        return materials

    # Check if the OBJ file references an MTL file
    try:
        with obj_file.open("r") as f:
            mtl_name = parse_mtl_name(f.readlines())
    except Exception as e:
        logger.warning(f"Failed to read OBJ file {obj_file}: {e}")
        return materials

    if mtl_name is None:
        return materials

    # Make sure the MTL file exists
    mtl_file = obj_file.parent / mtl_name
    if not mtl_file.exists():
        logger.warning(f"MTL file {mtl_file} referenced in {obj_file} does not exist")
        return materials

    logger.info(f"Found MTL file: {mtl_file}")

    try:
        # Parse the MTL file into separate materials
        with open(mtl_file, "r") as f:
            lines = f.readlines()

        # Remove comments and empty lines
        lines = [line for line in lines if not line.startswith("#")]
        lines = [line for line in lines if line.strip()]
        lines = [line.strip() for line in lines]

        # Split at each new material definition
        sub_mtls = []
        for line in lines:
            if line.startswith("newmtl"):
                sub_mtls.append([])
            if sub_mtls:  # Only append if we have started a material
                sub_mtls[-1].append(line)

        # Skip processing if there's only one material
        if len(sub_mtls) <= 1:
            logger.info(f"OBJ file {obj_file.name} has only one material, skipping split processing")
            mesh = trimesh.load_scene(obj_file)
            mesh.export(obj_file.as_posix(), mtl_name=mtl_file.name)
            # Don't delete MTL here - it might be used by other OBJs
            return materials

        # Process each material
        for sub_mtl in sub_mtls:
            if sub_mtl:  # Make sure the material has content
                material = Material.from_string(sub_mtl)
                material.name = f"{obj_file.stem}_{material.name}"
                materials[material.name] = material
                logger.info(f"Found material: {material.name}")

        # Now process the OBJ file to split by materials
        try:
            # Load the OBJ file with material grouping
            mesh = trimesh.load_scene(obj_file)

            if isinstance(mesh, trimesh.base.Trimesh):
                logger.warning(
                    f"OBJ file {obj_file.name} is a single mesh, but has more than one material, skipping split processing, please check the mtl file"
                )
                mesh.export(obj_file.as_posix())
            else:
                # Create target directory in the same directory as the original OBJ file
                # This maintains the original directory structure
                obj_stem = obj_file.stem
                obj_target_dir = obj_file.parent / obj_stem
                obj_target_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Splitting OBJ file {obj_file.name} by materials in: {obj_target_dir}")
                # Multiple submeshes, save each one separately
                logger.info(f"Splitting OBJ into {len(mesh.geometry)} submeshes by material")
                for i, (material_name, geom) in enumerate(mesh.geometry.items()):
                    submesh_name = obj_target_dir / f"{obj_stem}_{i}.obj"
                    if type(geom.visual) is trimesh.visual.texture.TextureVisuals:
                        geom.visual.material.name = material_name
                    else:
                        logger.warning(
                            f"Submesh: {submesh_name.name}, Geometry {material_name} does not have texture visuals"
                        )
                    mtl_file = f"{obj_stem}_{i}_{material_name}.mtl"
                    geom.export(submesh_name.as_posix(), include_texture=True, header=None, mtl_name=mtl_file)
                    # Mark files for deletion instead of deleting immediately
                    files_to_delete.append(obj_target_dir / mtl_file)
                    logger.info(f"Saved submesh: {submesh_name.name} (material: {material_name})")
                # Mark original files for deletion
                files_to_delete.append(obj_file)

        except ImportError:
            logger.warning("trimesh not available, cannot split OBJ by materials")
        except Exception as e:
            logger.warning(f"Failed to split OBJ file {obj_file} by materials: {e}")
            traceback.print_exc()

    except Exception as e:
        logger.error(f"Failed to process MTL file {mtl_file}: {e}")

    return materials


def split_obj_by_materials(mjcf_path: str | Path) -> None:
    """Split OBJ files by materials and update MJCF to use separate geoms.

    Args:
        mjcf_path: Path to the MJCF file
    """
    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # Track files to delete at the end
    files_to_delete = []

    # Get mesh directory
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")

    # Use existing meshdir setting or default to "."
    meshdir_ref = compiler.attrib.get("meshdir", ".")
    mesh_dir = mjcf_path.parent / meshdir_ref

    logger.info(f"Using meshdir: {meshdir_ref}, mesh_dir: {mesh_dir}")

    # Get asset section
    asset = root.find("asset")
    if asset is None:
        logger.info("No asset section found, skipping OBJ material splitting")
        return

    # First, convert DAE files to OBJ files
    dae_meshes = {}
    for mesh_elem in asset.findall("mesh"):
        mesh_name = mesh_elem.get("name", "")
        mesh_file = mesh_elem.get("file", "")
        if mesh_file.lower().endswith(".dae"):
            dae_meshes[mesh_name] = mesh_file

    if dae_meshes:
        # Convert DAE files to OBJ files
        for mesh_name, mesh_file in dae_meshes.items():
            dae_file_path = mesh_dir / mesh_file
            if not dae_file_path.exists():
                logger.warning(f"DAE file {dae_file_path} does not exist, skipping")
                continue

            # Generate OBJ file path
            obj_file_path = dae_file_path.with_suffix(".obj")

            try:
                logger.info(f"Converting DAE to OBJ: {dae_file_path} -> {obj_file_path}")
                dae2obj(dae_file_path, obj_file_path)

                # Verify the OBJ file was created
                if not obj_file_path.exists():
                    logger.error(f"OBJ file was not created: {obj_file_path}")
                    continue

                # Update mesh element in asset to reference OBJ file
                # Calculate relative path from mesh_dir to maintain directory structure
                obj_relative_path = obj_file_path.relative_to(mesh_dir)
                for mesh_elem in asset.findall("mesh"):
                    if mesh_elem.get("name") == mesh_name:
                        mesh_elem.attrib["file"] = str(obj_relative_path)
                        logger.info(f"Updated asset reference: {mesh_name} -> {obj_relative_path}")
                        break

                # Mark original DAE file for deletion
                files_to_delete.append(dae_file_path)
                logger.info(f"Marked for deletion: {dae_file_path}")

            except Exception as e:
                logger.error(f"Failed to convert DAE file {dae_file_path}: {e}")
                continue

        # 直接读取mjcf文件，不使用tree，将所有的".dae"替换为".obj"
        with open(mjcf_path, "r") as f:
            mjcf_content = f.read()
        mjcf_content = mjcf_content.replace(".dae", ".obj")
        with open(mjcf_path, "w") as f:
            f.write(mjcf_content)
        # reload the mjcf file
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        asset = root.find("asset")
        if asset is None:
            logger.info("No asset section found, skipping OBJ material splitting")
            return

    # Collect all OBJ mesh assets (including converted ones)
    obj_meshes = {}
    for mesh_elem in asset.findall("mesh"):
        mesh_name = mesh_elem.get("name", "")
        mesh_file = mesh_elem.get("file", "")
        if mesh_file.lower().endswith(".obj"):
            obj_meshes[mesh_name] = mesh_file

    logger.info(f"Found {len(obj_meshes)} OBJ meshes: {list(obj_meshes.keys())}")
    logger.info(f"Mesh directory: {mesh_dir}")

    if not obj_meshes:
        logger.info("No OBJ meshes found, skipping material splitting")
        return

    # Process each OBJ file
    all_mtl_materials = {}
    mesh_splits = {}  # mesh_name -> [(submesh_name, submesh_file), ...]
    mesh_single_materials = {}  # mesh_name -> material_name (for single mesh with multiple materials)
    processed_obj_files = {}  # obj_file_path -> (obj_materials, split_info, single_material)

    for mesh_name, mesh_file in obj_meshes.items():
        obj_file_path = mesh_dir / mesh_file
        logger.info(f"Processing OBJ: {mesh_name} -> {mesh_file} -> {obj_file_path}")
        if not obj_file_path.exists():
            logger.warning(f"OBJ file {obj_file_path} does not exist, skipping")
            continue

        # Check if we've already processed this OBJ file
        if obj_file_path in processed_obj_files:
            logger.info(f"OBJ file {obj_file_path} already processed, reusing results")
            obj_materials, split_info, single_material = processed_obj_files[obj_file_path]
            all_mtl_materials.update(obj_materials)
            if split_info is not None:
                mesh_splits[mesh_name] = split_info
            if single_material is not None:
                mesh_single_materials[mesh_name] = single_material
            continue

        # Process this OBJ file for the first time
        obj_materials = process_obj_materials(obj_file_path, files_to_delete)
        all_mtl_materials.update(obj_materials)

        # Check for split meshes in the same directory as the original OBJ file
        obj_stem = obj_file_path.stem
        split_dir = obj_file_path.parent / obj_stem

        split_info = None
        single_material = None

        if split_dir.exists():
            submeshes = list(split_dir.glob(f"{obj_stem}_*.obj"))
            if submeshes:
                # Found split meshes
                submesh_info = []
                # Extract link prefix from mesh_name if it exists
                # mesh_name format: linkname_meshstem or just meshstem
                link_prefix = ""
                if mesh_name != obj_stem:
                    # mesh_name has a prefix (link name)
                    # Extract everything before the last occurrence of obj_stem
                    if obj_stem in mesh_name:
                        prefix_part = mesh_name[: mesh_name.rfind(obj_stem)]
                        if prefix_part.endswith("_"):
                            link_prefix = prefix_part[:-1]  # Remove trailing underscore

                for i, submesh_path in enumerate(sorted(submeshes)):
                    submesh_stem = submesh_path.stem
                    # If mesh_name has a link prefix, add it to submesh name
                    if link_prefix:
                        submesh_name_with_prefix = f"{link_prefix}_{submesh_stem}"
                    else:
                        submesh_name_with_prefix = submesh_stem
                    # Get relative path from mesh_dir
                    submesh_rel_path = submesh_path.relative_to(mesh_dir)
                    submesh_info.append((submesh_name_with_prefix, str(submesh_rel_path)))

                mesh_splits[mesh_name] = submesh_info
                split_info = submesh_info
                logger.info(f"Found {len(submesh_info)} split meshes for {mesh_name}")

        # Handle single mesh with multiple materials case
        # If obj_materials has content but no split meshes found, read actual material used in OBJ
        if obj_materials and mesh_name not in mesh_splits:
            # This is a single mesh with multiple materials that wasn't split
            # Read the OBJ file to find which material is actually used
            actual_material = None
            try:
                with open(obj_file_path, "r") as f:
                    for line in f:
                        if line.startswith("usemtl "):
                            mtl_name_raw = line.split()[1].strip()
                            # Construct the expected material name with obj stem prefix
                            expected_material_name = f"{obj_file_path.stem}_{mtl_name_raw}"
                            if expected_material_name in obj_materials:
                                actual_material = expected_material_name
                                break
            except Exception as e:
                logger.warning(f"Failed to read material from OBJ {obj_file_path}: {e}")

            # Fallback to first material if we couldn't find the actual one
            if not actual_material:
                actual_material = next(iter(obj_materials.keys()))
                logger.warning(
                    f"Could not determine actual material used in {mesh_name}, using first material: {actual_material}"
                )

            mesh_single_materials[mesh_name] = actual_material
            single_material = actual_material
            logger.info(f"Single mesh {mesh_name} with multiple materials will use material: {actual_material}")

        # Cache the results for this OBJ file
        processed_obj_files[obj_file_path] = (obj_materials, split_info, single_material)

    # Update asset section - add new mesh assets for submeshes
    for mesh_name, submesh_info in mesh_splits.items():
        for submesh_name, submesh_file in submesh_info:
            # Add new mesh asset
            new_mesh = ET.SubElement(asset, "mesh")
            new_mesh.attrib["name"] = submesh_name
            new_mesh.attrib["file"] = submesh_file

    # Add MTL materials to asset section
    for material in all_mtl_materials.values():
        material_attrib = {
            "name": material.name,
            # "specular": material.mjcf_specular(),
            # "shininess": material.mjcf_shininess(),
            "rgba": material.mjcf_rgba(),
        }
        ET.SubElement(asset, "material", attrib=material_attrib)
        logger.info(f"Added MTL material: {material.name}")

    # Update geom elements to use split meshes or assign materials
    for body in root.findall(".//body"):
        geoms_to_update = []
        geoms_to_assign_material = []
        for geom in body.findall("geom"):
            if geom.get("class") == "visual" and geom.get("type") == "mesh":
                mesh_ref = geom.get("mesh")
                if mesh_ref in mesh_splits:
                    geoms_to_update.append((geom, mesh_ref))
                elif mesh_ref in mesh_single_materials:
                    geoms_to_assign_material.append((geom, mesh_ref))

        # First, assign materials to single meshes that have multiple materials but weren't split
        for geom, mesh_ref in geoms_to_assign_material:
            material_name = mesh_single_materials[mesh_ref]
            geom.attrib["material"] = material_name
            logger.info(f"Assigned material {material_name} to geom {geom.get('name', 'unnamed')} for mesh {mesh_ref}")

        # Replace each geom with multiple geoms for submeshes
        for geom, mesh_ref in geoms_to_update:
            submesh_info = mesh_splits[mesh_ref]
            geom_name_base = geom.get("name", "visual")

            # Remove original geom
            body.remove(geom)

            # Add new geoms for each submesh
            for i, (submesh_name, submesh_file) in enumerate(submesh_info):
                new_geom = ET.SubElement(body, "geom")

                # Copy all attributes from original geom
                for attr_name, attr_value in geom.attrib.items():
                    if attr_name not in ["name", "mesh", "material"]:
                        new_geom.attrib[attr_name] = attr_value

                # Set new attributes
                new_geom.attrib["name"] = f"{geom_name_base}_{i}"
                new_geom.attrib["mesh"] = submesh_name

                # Try to find corresponding MTL material
                assigned_material = "default_material"

                # Read the submesh to find material reference
                submesh_path = mesh_dir / submesh_file
                if submesh_path.exists():
                    try:
                        with open(submesh_path, "r") as f:
                            submesh_lines = f.readlines()
                        for line in submesh_lines:
                            if line.startswith("usemtl "):
                                mtl_name_raw = line.split()[1].strip()
                                # Look for matching material with correct prefix
                                # The material name should be: {actual_obj_stem}_{mtl_name_raw}
                                # Need to extract the actual obj stem from mesh_ref (which may have link prefix)

                                # Get the actual OBJ file stem by looking at the submesh_file path
                                # submesh_file format: path/to/obj_stem/obj_stem_N.obj
                                submesh_parts = Path(submesh_file).parts
                                if len(submesh_parts) >= 2:
                                    # The parent directory name is the actual obj_stem
                                    actual_obj_stem = submesh_parts[-2]
                                else:
                                    # Fallback: try to extract from mesh_ref
                                    if mesh_ref.endswith(".obj"):
                                        _ = mesh_ref[:-4]
                                    else:
                                        _ = mesh_ref
                                    # Remove link prefix if present
                                    # Find the obj_stem in the mesh_ref
                                    actual_obj_stem = submesh_path.parent.name

                                expected_material_name = f"{actual_obj_stem}_{mtl_name_raw}"
                                if expected_material_name in all_mtl_materials:
                                    assigned_material = expected_material_name
                                    break
                                break
                    except Exception as e:
                        logger.warning(f"Could not read submesh {submesh_path}: {e}")

                new_geom.attrib["material"] = assigned_material
                logger.info(f"Created geom {new_geom.attrib['name']} with material {assigned_material}")

    # Reorganize asset section: materials first, then meshes
    # And remove unused original OBJ meshes that were split

    # Collect all used mesh names from geoms
    used_meshes = set()
    collision_meshes = set()
    for body in root.findall(".//body"):
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh":
                mesh_name = geom.get("mesh")
                if mesh_name:
                    used_meshes.add(mesh_name)
                if geom.get("class") == "collision":
                    collision_meshes.add(mesh_name)

    # Remove unused OBJ meshes that were split
    split_original_meshes = set(mesh_splits.keys())

    # Collect all existing elements in asset section
    existing_materials = []
    existing_meshes = []
    other_elements = []

    for child in list(asset):
        if child.tag == "material":
            existing_materials.append(child)
        elif child.tag == "mesh":
            mesh_name = child.get("name", "")
            # Only keep meshes that are used and not the original split meshes
            if mesh_name in used_meshes:
                if mesh_name not in split_original_meshes:
                    existing_meshes.append(child)
                elif mesh_name in collision_meshes:
                    existing_meshes.append(child)
                else:
                    logger.info(f"Removing unused original OBJ mesh: {mesh_name}")
            else:
                logger.info(f"Removing unused mesh: {mesh_name}")
        else:
            other_elements.append(child)
        asset.remove(child)

    # Re-add elements in the desired order: materials first, then meshes, then others
    for material in existing_materials:
        asset.append(material)

    for mesh in existing_meshes:
        asset.append(mesh)

    for other in other_elements:
        asset.append(other)

    logger.info(f"Reorganized asset section: {len(existing_materials)} materials, {len(existing_meshes)} meshes")

    # Save the updated MJCF file
    save_xml(mjcf_path, tree)
    logger.info(f"Updated MJCF file with split OBJ materials: {mjcf_path}")

    # Now safely delete all marked files
    deleted_count = 0
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"Deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

    logger.info(f"Cleanup complete: deleted {deleted_count} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split OBJ files by materials and update MJCF to use separate geoms.")
    parser.add_argument("mjcf_path", type=Path, help="Path to the MJCF file")
    args = parser.parse_args()
    split_obj_by_materials(args.mjcf_path)


if __name__ == "__main__":
    main()

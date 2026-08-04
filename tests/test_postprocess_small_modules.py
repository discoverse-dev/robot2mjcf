"""Tests for smaller MJCF postprocess utilities."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh

from urdf_to_mjcf.postprocess.base_joint import fix_base_joint
from urdf_to_mjcf.postprocess.explicit_floor_contacts import add_explicit_floor_contacts
from urdf_to_mjcf.postprocess.make_degrees import (
    convert_radians_to_degrees,
    make_degrees,
    update_compiler_angle,
    update_default_joint_limits,
    update_default_motor_limits,
    update_joint_axes,
    update_joint_limits,
    update_rpy_attributes,
)
from urdf_to_mjcf.postprocess.move_mesh_scale import move_mesh_scale
from urdf_to_mjcf.postprocess.remove_redundancies import remove_redundancies
from urdf_to_mjcf.postprocess.sanitize_mesh_assets import sanitize_mesh_assets
from urdf_to_mjcf.postprocess.split_obj_materials import (
    build_submesh_info,
    process_obj_materials,
    remove_stale_generated_submeshes,
    split_obj_by_materials,
)
from urdf_to_mjcf.postprocess.update_mesh import merge_materials, update_mesh


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_convert_radians_to_degrees_handles_valid_and_invalid_values() -> None:
    assert convert_radians_to_degrees(f"0 {math.pi}") == "0 180"
    assert convert_radians_to_degrees("not-a-number") == "not-a-number"


def test_make_degrees_updates_expected_angle_fields(tmp_path) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <compiler angle="radian" />
          <default>
            <joint range="0 1.57079632679" />
            <motor ctrlrange="-1.57079632679 1.57079632679" />
          </default>
          <worldbody>
            <body name="base" rpy="0 0 3.14159265359">
              <joint name="hinge" axis="1 0 0" range="-3.14159265359 0" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    make_degrees(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    compiler = root.find("compiler")
    default_joint = root.find(".//default/joint")
    default_motor = root.find(".//default/motor")
    body = root.find(".//body")
    joint = root.find(".//body/joint")

    assert compiler is not None
    assert compiler.attrib["angle"] == "degree"
    assert default_joint is not None
    assert default_joint.attrib["range"] == "0 90"
    assert default_motor is not None
    assert default_motor.attrib["ctrlrange"] == "-90 90"
    assert body is not None
    assert body.attrib["rpy"] == "0 0 180"
    assert joint is not None
    assert joint.attrib["range"] == "-180 0"
    assert joint.attrib["axis"] == "1 0 0"


def test_angle_update_helpers_skip_missing_elements() -> None:
    root = ET.fromstring("<mujoco><worldbody><body /></worldbody></mujoco>")

    update_compiler_angle(root)
    update_joint_limits(root)
    update_default_joint_limits(root)
    update_default_motor_limits(root)
    update_rpy_attributes(root)
    update_joint_axes(root)

    assert root.find("compiler") is None


def test_fix_base_joint_wraps_existing_joint_under_new_root(tmp_path) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <worldbody>
            <body name="robot" pos="1 2 3" quat="0 0 0 1">
              <joint name="base_joint" type="hinge" />
              <inertial mass="1" pos="0 0 0" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    fix_base_joint(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    new_root = root.find("./worldbody/body[@name='root']")
    robot_body = root.find("./worldbody/body[@name='root']/body[@name='robot']")

    assert new_root is not None
    assert new_root.attrib["pos"] == "1 2 3"
    assert new_root.attrib["quat"] == "0 0 0 1"
    assert new_root.find("freejoint") is not None
    assert robot_body is not None
    assert robot_body.attrib["pos"] == "0 0 0"
    assert robot_body.attrib["quat"] == "1 0 0 0"
    assert robot_body.find("inertial") is None


def test_fix_base_joint_adds_freejoint_when_root_body_has_no_joint(tmp_path) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <worldbody>
            <body name="robot">
              <geom type="box" size="1 1 1" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    fix_base_joint(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    robot_body = root.find("./worldbody/body[@name='robot']")

    assert robot_body is not None
    assert robot_body.find("freejoint") is not None


def test_fix_base_joint_handles_missing_worldbody_or_body(tmp_path) -> None:
    no_worldbody = write_text(tmp_path / "no_worldbody.xml", "<mujoco />")
    no_body = write_text(tmp_path / "no_body.xml", "<mujoco><worldbody /></mujoco>")

    fix_base_joint(no_worldbody)
    fix_base_joint(no_body, add_freejoint=False)

    assert ET.parse(no_worldbody).getroot().find("worldbody") is None
    assert ET.parse(no_body).getroot().find("worldbody/body") is None


def test_add_explicit_floor_contacts_creates_pairs_for_named_box_geoms(tmp_path) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <worldbody>
            <body name="arm">
              <geom name="arm_box" class="collision" type="box" />
              <geom name="arm_other" class="visual" type="box" />
            </body>
            <body name="leg">
              <geom name="leg_box_a" class="collision" type="box" />
              <geom name="leg_box_b" class="collision" type="box" />
            </body>
            <body name="head">
              <geom class="collision" type="box" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    add_explicit_floor_contacts(mjcf_path, ["arm", "leg", "head", "missing"], floor_name="ground")

    root = ET.parse(mjcf_path).getroot()
    pairs = root.findall("./contact/pair")

    assert [(pair.attrib["geom1"], pair.attrib["geom2"]) for pair in pairs] == [
        ("arm_box", "ground"),
        ("leg_box_a", "ground"),
    ]


def test_process_obj_materials_registers_single_material_obj(tmp_path) -> None:
    obj_path = write_text(
        tmp_path / "link1.obj",
        "\n".join(
            [
                "mtllib link1.mtl",
                "usemtl metal_black",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
    )
    write_text(obj_path.with_suffix(".mtl"), "newmtl metal_black\nKd 0.01 0.01 0.01\nd 1.0\n")

    materials = process_obj_materials(obj_path)

    assert list(materials) == ["mtl_link1_metal_black"]
    assert materials["mtl_link1_metal_black"].mjcf_rgba() == "0.01 0.01 0.01 1.0"


def test_mesh_postprocess_preserves_obj_texture_materials(tmp_path) -> None:
    obj_path = write_text(
        tmp_path / "meshes" / "torso" / "torso.obj",
        "\n".join(
            [
                "mtllib torso.mtl",
                "usemtl material_0",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
    )
    write_text(
        obj_path.with_suffix(".mtl"),
        "\n".join(
            [
                "newmtl material_0",
                "Kd 0.4 0.4 0.4",
                "map_Kd material_0.png",
            ]
        ),
    )
    write_text(obj_path.parent / "material_0.png", "png")
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <compiler meshdir="." />
          <asset>
            <material name="chassis_material_0" rgba="0.4 0.4 0.4 1.0" />
            <mesh name="torso_base_link_torso" file="meshes/torso/torso.obj" />
          </asset>
          <worldbody>
            <body name="torso_base_link">
              <geom name="torso_base_link_visual" type="mesh" mesh="torso_base_link_torso" material="default_material" class="visual" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    split_obj_by_materials(mjcf_path)
    update_mesh(mjcf_path, max_vertices=1000000)

    root = ET.parse(mjcf_path).getroot()
    geom = root.find(".//geom[@name='torso_base_link_visual']")
    texture = root.find("./asset/texture[@name='mtl_meshes_torso_torso_material_0_texture']")
    material = root.find("./asset/material[@name='mtl_meshes_torso_torso_material_0']")

    assert geom is not None
    assert geom.attrib["material"] == "mtl_meshes_torso_torso_material_0"
    assert texture is not None
    assert texture.attrib["file"] == "meshes/torso/material_0.png"
    assert material is not None
    assert material.attrib["texture"] == "mtl_meshes_torso_torso_material_0_texture"
    assert not obj_path.with_suffix(".mtl").exists()
    assert "mtllib" not in obj_path.read_text()
    assert "usemtl" not in obj_path.read_text()


def test_split_obj_by_materials_scopes_materials_by_source_path(tmp_path) -> None:
    arm_obj = write_text(
        tmp_path / "meshes" / "visual" / "arm" / "link3.obj",
        "\n".join(
            [
                "mtllib link3.mtl",
                "usemtl material_0",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
    )
    gripper_obj = write_text(
        tmp_path / "meshes" / "visual" / "gripper" / "link3.obj",
        "\n".join(
            [
                "mtllib link3.mtl",
                "usemtl material_0",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
    )
    write_text(arm_obj.with_suffix(".mtl"), "newmtl material_0\nKd 0.1 0.2 0.3\n")
    write_text(gripper_obj.with_suffix(".mtl"), "newmtl material_0\nKd 0.7 0.8 0.9\n")
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <compiler meshdir='.' />
          <asset>
            <mesh name='arm_link3' file='meshes/visual/arm/link3.obj' />
            <mesh name='gripper_link3' file='meshes/visual/gripper/link3.obj' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='arm_visual' class='visual' type='mesh' mesh='arm_link3' />
              <geom name='gripper_visual' class='visual' type='mesh' mesh='gripper_link3' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    split_obj_by_materials(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    arm_material = "mtl_meshes_visual_arm_link3_material_0"
    gripper_material = "mtl_meshes_visual_gripper_link3_material_0"
    arm_material_elem = root.find(f"./asset/material[@name='{arm_material}']")
    gripper_material_elem = root.find(f"./asset/material[@name='{gripper_material}']")

    assert arm_material_elem is not None
    assert gripper_material_elem is not None
    assert arm_material_elem.attrib["rgba"] == "0.1 0.2 0.3 1.0"
    assert gripper_material_elem.attrib["rgba"] == "0.7 0.8 0.9 1.0"
    assert root.find(f".//geom[@name='arm_visual'][@material='{arm_material}']") is not None
    assert root.find(f".//geom[@name='gripper_visual'][@material='{gripper_material}']") is not None


def test_split_obj_by_materials_uses_split_time_submesh_material_mapping(tmp_path) -> None:
    obj_path = write_text(
        tmp_path / "meshes" / "part.obj",
        "\n".join(
            [
                "mtllib part.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "usemtl red",
                "f 1 2 3",
                "usemtl blue",
                "f 1 3 4",
            ]
        ),
    )
    write_text(
        obj_path.with_suffix(".mtl"),
        "\n".join(
            [
                "newmtl red",
                "Kd 1 0 0",
                "newmtl blue",
                "Kd 0 0 1",
            ]
        ),
    )
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <compiler meshdir='.' />
          <asset>
            <mesh name='part' file='meshes/part.obj' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='part_visual' class='visual' type='mesh' mesh='part' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    split_obj_by_materials(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    geom_materials = {geom.attrib["material"] for geom in root.findall(".//geom")}
    assert geom_materials == {"mtl_meshes_part_red", "mtl_meshes_part_blue"}
    assert root.find("./asset/material[@name='mtl_meshes_part_red']") is not None
    assert root.find("./asset/material[@name='mtl_meshes_part_blue']") is not None


def test_material_compactors_preserve_source_scoped_mtl_materials(tmp_path) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <asset>
            <material name='mtl_meshes_arm_link3_material_0' rgba='0.1 0.2 0.3 1' />
            <material name='mtl_meshes_gripper_link3_material_0' rgba='0.1 0.2 0.3 1' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='arm_visual' material='mtl_meshes_arm_link3_material_0' />
              <geom name='gripper_visual' material='mtl_meshes_gripper_link3_material_0' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    merge_materials(mjcf_path)
    remove_redundancies(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    arm_geom = root.find(".//geom[@name='arm_visual']")
    gripper_geom = root.find(".//geom[@name='gripper_visual']")

    assert root.find("./asset/material[@name='mtl_meshes_arm_link3_material_0']") is not None
    assert root.find("./asset/material[@name='mtl_meshes_gripper_link3_material_0']") is not None
    assert arm_geom is not None
    assert gripper_geom is not None
    assert arm_geom.attrib["material"] == "mtl_meshes_arm_link3_material_0"
    assert gripper_geom.attrib["material"] == "mtl_meshes_gripper_link3_material_0"


def test_move_mesh_scale_bakes_reflected_visual_mesh_and_scales_collision_mesh(tmp_path) -> None:
    mesh_path = write_text(
        tmp_path / "meshes" / "triangle.obj",
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
    )
    mjcf_path = write_text(
        tmp_path / "model.xml",
        f"""
        <mujoco>
          <compiler meshdir='.' />
          <asset>
            <mesh name='triangle_visual' file='{mesh_path.relative_to(tmp_path).as_posix()}' />
            <mesh name='triangle_collision' file='{mesh_path.relative_to(tmp_path).as_posix()}' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='mirrored_visual' class='visual' type='mesh' mesh='triangle_visual' scale='1 -1 1' />
              <geom name='mirrored_collision' class='collision' type='mesh' mesh='triangle_collision' scale='1 -1 1' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    move_mesh_scale(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    visual_geom = root.find(".//geom[@name='mirrored_visual']")
    collision_geom = root.find(".//geom[@name='mirrored_collision']")
    assert visual_geom is not None
    assert collision_geom is not None
    assert "scale" not in visual_geom.attrib
    assert "scale" not in collision_geom.attrib

    visual_mesh = root.find(f"./asset/mesh[@name='{visual_geom.attrib['mesh']}']")
    collision_mesh = root.find(f"./asset/mesh[@name='{collision_geom.attrib['mesh']}']")
    assert visual_mesh is not None
    assert collision_mesh is not None
    assert "scale" not in visual_mesh.attrib
    assert visual_mesh.attrib["file"] == "meshes/_generated/triangle_scaled_1_m1_1.obj"
    assert collision_mesh.attrib["file"] == "meshes/triangle.obj"
    assert collision_mesh.attrib["scale"] == "1 -1 1"

    baked_mesh_path = tmp_path / visual_mesh.attrib["file"]
    baked_mesh = trimesh.load(baked_mesh_path, force="mesh", process=False)
    assert isinstance(baked_mesh, trimesh.Trimesh)
    assert baked_mesh.vertices.tolist() == [[0.0, -0.0, 0.0], [1.0, -0.0, 0.0], [0.0, -1.0, 0.0]]
    assert baked_mesh.face_normals[0].tolist() == [0.0, 0.0, 1.0]
    assert list(baked_mesh_path.parent.glob("*.mtl")) == []


def test_move_mesh_scale_renames_all_shared_extension_mesh_references(tmp_path) -> None:
    mesh_path = write_text(
        tmp_path / "meshes" / "triangle.obj",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    )
    mjcf_path = write_text(
        tmp_path / "model.xml",
        f"""
        <mujoco>
          <compiler meshdir='.' />
          <asset>
            <mesh name='triangle.obj' file='{mesh_path.relative_to(tmp_path).as_posix()}' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='first' type='mesh' mesh='triangle.obj' />
              <geom name='second' type='mesh' mesh='triangle.obj' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    move_mesh_scale(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    assert root.find("./asset/mesh[@name='triangle']") is not None
    assert [geom.attrib["mesh"] for geom in root.findall(".//geom")] == ["triangle", "triangle"]


def test_sanitize_mesh_assets_removes_missing_mesh_assets_and_geoms(tmp_path) -> None:
    write_text(tmp_path / "meshes" / "existing.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <compiler meshdir='meshes' />
          <asset>
            <mesh name='existing' file='existing.obj' />
            <mesh name='missing' file='missing.obj' />
            <mesh name='unused_missing' file='unused_missing.obj' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='existing_geom' type='mesh' mesh='existing' />
              <geom name='missing_geom' type='mesh' mesh='missing' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    sanitize_mesh_assets(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    assert root.find("./asset/mesh[@name='existing']") is not None
    assert root.find("./asset/mesh[@name='missing']") is None
    assert root.find("./asset/mesh[@name='unused_missing']") is None
    assert root.find(".//geom[@name='existing_geom']") is not None
    assert root.find(".//geom[@name='missing_geom']") is None


def test_split_obj_by_materials_rebuilds_submesh_names_for_reused_obj(tmp_path) -> None:
    mesh_path = write_text(tmp_path / "meshes" / "part.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    split_dir = mesh_path.parent / "part"
    write_text(split_dir / "part_0.obj", "usemtl silver\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    write_text(split_dir / "part_1.obj", "usemtl black\nv 0 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n")
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <compiler meshdir='.' />
          <asset>
            <mesh name='left_part' file='meshes/part.obj' />
            <mesh name='right_part' file='meshes/part.obj' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='left_visual' class='visual' type='mesh' mesh='left_part' />
              <geom name='right_visual' class='visual' type='mesh' mesh='right_part' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    split_obj_by_materials(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    mesh_names = {mesh.attrib["name"] for mesh in root.findall("./asset/mesh")}
    assert {"left_part_0", "left_part_1", "right_part_0", "right_part_1"} <= mesh_names
    assert root.find(".//geom[@name='left_visual_0'][@mesh='left_part_0']") is not None
    assert root.find(".//geom[@name='right_visual_0'][@mesh='right_part_0']") is not None
    assert "usemtl" not in (split_dir / "part_0.obj").read_text()
    assert "usemtl" not in (split_dir / "part_1.obj").read_text()


def test_generated_submesh_cleanup_removes_stale_scaled_outputs(tmp_path) -> None:
    mesh_dir = tmp_path / "meshes"
    obj_file = write_text(mesh_dir / "part.obj", "v 0 0 0\n")
    split_dir = mesh_dir / "part"
    current_a = write_text(split_dir / "part_0.obj", "v 0 0 0\n")
    current_b = write_text(split_dir / "part_1.obj", "v 1 0 0\n")
    stale_submesh = write_text(
        split_dir / "part_2_scaled_1_m1_1.obj",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl red\nf 1 2 3\n",
    )

    assert len(build_submesh_info("part", obj_file, tmp_path) or []) == 3

    remove_stale_generated_submeshes(split_dir, "part")
    current_a.write_text("v 0 0 0\n")
    current_b.write_text("v 1 0 0\n")

    submeshes = build_submesh_info("part", obj_file, tmp_path)
    assert submeshes == [
        ("part_0", "meshes/part/part_0.obj"),
        ("part_1", "meshes/part/part_1.obj"),
    ]
    assert not stale_submesh.exists()

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
from urdf_to_mjcf.postprocess.split_obj_materials import process_obj_materials, split_obj_by_materials


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

    assert list(materials) == ["link1_metal_black"]
    assert materials["link1_metal_black"].mjcf_rgba() == "0.01 0.01 0.01 1.0"


def test_move_mesh_scale_bakes_negative_scale_mesh(tmp_path) -> None:
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
            <mesh name='triangle' file='{mesh_path.relative_to(tmp_path).as_posix()}' />
          </asset>
          <worldbody>
            <body name='body'>
              <geom name='mirrored' type='mesh' mesh='triangle' scale='1 -1 1' />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    move_mesh_scale(mjcf_path)

    root = ET.parse(mjcf_path).getroot()
    geom = root.find(".//geom[@name='mirrored']")
    assert geom is not None
    assert "scale" not in geom.attrib

    mesh = root.find(f"./asset/mesh[@name='{geom.attrib['mesh']}']")
    assert mesh is not None
    assert "scale" not in mesh.attrib
    assert mesh.attrib["file"] == "meshes/triangle_scaled_1_m1_1.obj"

    baked_mesh = trimesh.load(tmp_path / mesh.attrib["file"], force="mesh", process=False)
    assert isinstance(baked_mesh, trimesh.Trimesh)
    assert baked_mesh.vertices.tolist() == [[0.0, -0.0, 0.0], [1.0, -0.0, 0.0], [0.0, -1.0, 0.0]]
    assert baked_mesh.face_normals[0].tolist() == [0.0, 0.0, 1.0]


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

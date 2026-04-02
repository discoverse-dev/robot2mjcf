"""Focused tests for asset resolution and MJCF builder helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from robot2mjcf.conversion.assets import (
    add_mesh_assets_to_xml,
    collect_single_obj_materials,
    copy_mesh_assets,
    resolve_mesh_source_path,
    resolve_workspace_search_paths,
)
from robot2mjcf.conversion.mjcf_assembly import (
    add_assets,
    add_compiler,
    add_contact,
    add_default,
    add_weld_constraints,
)
from robot2mjcf.core.materials import Material
from robot2mjcf.core.model import (
    CollisionParams,
    ConversionMetadata,
    DefaultJointMetadata,
    WeldConstraint,
    dActuator,
    dJoint,
)


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_resolve_workspace_search_paths_collects_unique_roots(tmp_path, monkeypatch) -> None:
    urdf_path = tmp_path / "pkg" / "robot.urdf"
    workspace = tmp_path / "ws"

    monkeypatch.setattr("robot2mjcf.conversion.assets.find_workspace_from_path", lambda path: workspace)
    monkeypatch.setattr(
        "robot2mjcf.core.package_resolver._default_resolver._find_package_root_from_urdf_path",
        lambda path: workspace,
    )

    assert resolve_workspace_search_paths(urdf_path) == [workspace]


def test_resolve_mesh_source_path_handles_package_absolute_and_relative(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "workspace" / "demo_pkg"
    urdf_dir = tmp_path / "robot"
    absolute_mesh = tmp_path / "shared" / "mesh.stl"

    monkeypatch.setattr("robot2mjcf.conversion.assets.resolve_package_path", lambda name, roots: package_root)

    package_source, package_subpath = resolve_mesh_source_path(
        "package://demo_pkg/meshes/part.obj",
        urdf_dir=urdf_dir,
        workspace_search_paths=[tmp_path / "workspace"],
    )
    absolute_source, absolute_subpath = resolve_mesh_source_path(
        str(absolute_mesh),
        urdf_dir=urdf_dir,
        workspace_search_paths=[],
    )
    relative_source, relative_subpath = resolve_mesh_source_path(
        "meshes/local.obj",
        urdf_dir=urdf_dir,
        workspace_search_paths=[],
    )

    assert package_source == package_root / "meshes/part.obj"
    assert package_subpath == "demo_pkg/meshes/part.obj"
    assert absolute_source == absolute_mesh
    assert absolute_subpath == "../shared/mesh.stl"
    assert relative_source == (urdf_dir / "meshes/local.obj").resolve()
    assert relative_subpath == "meshes/local.obj"


def test_collect_single_obj_materials_extracts_named_material(tmp_path) -> None:
    obj_path = write_text(
        tmp_path / "meshes" / "arm.obj",
        "\n".join(
            [
                "mtllib arm.mtl",
                "usemtl steel",
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
                "newmtl steel",
                "Kd 0.1 0.2 0.3",
                "d 0.5",
            ]
        ),
    )

    materials = collect_single_obj_materials(
        {"arm_mesh": "meshes/arm.obj", "skip_mesh": "meshes/skip.stl"},
        urdf_dir=tmp_path,
        workspace_search_paths=[],
    )

    assert list(materials) == ["arm_steel"]
    assert materials["arm_steel"].mjcf_rgba() == "0.1 0.2 0.3 0.5"


def test_copy_mesh_assets_copies_obj_and_prunes_missing_geoms(tmp_path) -> None:
    source_obj = write_text(
        tmp_path / "meshes" / "arm.obj",
        "\n".join(
            [
                "mtllib arm.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "f 1 2 3",
            ]
        ),
    )
    write_text(source_obj.with_suffix(".mtl"), "newmtl steel\nKd 0.4 0.4 0.4\n")
    root = ET.fromstring(
        """
        <mujoco>
          <worldbody>
            <body name="arm">
              <geom name="arm_geom" mesh="arm_mesh" />
              <geom name="missing_geom" mesh="missing_mesh" />
            </body>
          </worldbody>
        </mujoco>
        """
    )

    result = copy_mesh_assets(
        root,
        {
            "arm_mesh": "meshes/arm.obj",
            "arm_mesh_alias": "meshes/arm.obj",
            "missing_mesh": "meshes/missing.obj",
        },
        urdf_dir=tmp_path,
        target_mesh_dir=tmp_path / "out" / "meshes",
        workspace_search_paths=[],
    )

    copied_obj = tmp_path / "out" / "meshes" / "meshes" / "arm.obj"
    copied_mtl = copied_obj.with_suffix(".mtl")

    assert copied_obj.exists()
    assert copied_mtl.exists()
    assert "missing_mesh" not in result.mesh_assets
    assert result.mesh_file_paths["arm_mesh"] == copied_obj
    assert result.mesh_file_paths["arm_mesh_alias"] == copied_obj
    assert root.find(".//geom[@name='arm_geom']") is not None
    assert root.find(".//geom[@name='missing_geom']") is None


def test_add_mesh_assets_to_xml_normalizes_paths(tmp_path) -> None:
    root = ET.fromstring("<mujoco />")

    add_mesh_assets_to_xml(
        root,
        {
            "package_mesh": "package://demo_pkg/meshes/part.obj",
            "absolute_mesh": str(tmp_path / "shared" / "mesh.stl"),
            "relative_mesh": "meshes/local.obj",
        },
        urdf_dir=tmp_path / "robot",
    )

    mesh_files = {mesh.attrib["name"]: mesh.attrib["file"] for mesh in root.findall("./asset/mesh")}

    assert mesh_files == {
        "package_mesh": "meshes/demo_pkg/meshes/part.obj",
        "absolute_mesh": "meshes/../shared/mesh.stl",
        "relative_mesh": "meshes/meshes/local.obj",
    }


def test_add_compiler_replaces_existing_element() -> None:
    root = ET.fromstring("<mujoco><option /><compiler angle='degree' /></mujoco>")

    add_compiler(root)

    compiler = root.find("compiler")
    assert compiler is not None
    assert root[0] is compiler
    assert compiler.attrib["angle"] == "radian"
    assert compiler.attrib["meshdir"] == "."
    assert compiler.attrib["balanceinertia"] == "true"


def test_add_default_builds_joint_actuator_and_collision_defaults() -> None:
    root = ET.fromstring("<mujoco><default /></mujoco>")
    metadata = ConversionMetadata(
        collision_params=CollisionParams(contype=7, conaffinity=9),
        maxhullvert=32,
    )
    default_metadata = {
        "hinge": DefaultJointMetadata(
            joint=dJoint(
                stiffness=1.5,
                actuatorfrcrange=[-1.0, 1.0],
                margin=0.02,
                armature=0.3,
                damping=0.4,
                frictionloss=0.1,
            ),
            actuator=dActuator(
                actuator_type="position",
                kp=30.0,
                kv=4.0,
                gear=2.0,
                ctrlrange=[-0.5, 0.5],
                forcerange=[-2.0, 2.0],
            ),
        )
    }

    add_default(root, metadata, default_metadata)

    joint = root.find(".//default[@class='hinge']/joint")
    actuator = root.find(".//default[@class='hinge']/position")
    visual_geom = root.find(".//default[@class='visual']/geom")
    collision_geom = root.find(".//default[@class='collision']/geom")
    mesh = root.find("./default/mesh")

    assert root[0].tag == "default"
    assert joint is not None
    assert joint.attrib["stiffness"] == "1.5"
    assert joint.attrib["actuatorfrcrange"] == "-1.0 1.0"
    assert actuator is not None
    assert actuator.attrib["ctrlrange"] == "-0.5 0.5"
    assert actuator.attrib["forcerange"] == "-2.0 2.0"
    assert visual_geom is not None
    assert visual_geom.attrib["group"] == "2"
    assert collision_geom is not None
    assert collision_geom.attrib["contype"] == "7"
    assert collision_geom.attrib["conaffinity"] == "9"
    assert collision_geom.attrib["group"] == "3"
    assert mesh is not None
    assert mesh.attrib["maxhullvert"] == "32"


def test_add_default_skips_visual_geom_for_collision_only() -> None:
    root = ET.fromstring("<mujoco />")

    add_default(root, ConversionMetadata(), collision_only=True)

    collision_geom = root.find(".//default[@class='collision']/geom")

    assert root.find(".//default[@class='visual']") is None
    assert collision_geom is not None
    assert collision_geom.attrib["group"] == "2"


def test_add_contact_adds_excludes_for_adjacent_collision_links() -> None:
    root = ET.fromstring("<mujoco />")
    robot = ET.fromstring(
        """
        <robot>
          <link name="base"><collision /></link>
          <link name="arm"><collision /></link>
          <link name="camera" />
          <joint name="j1"><parent link="base" /><child link="arm" /></joint>
          <joint name="j2"><parent link="arm" /><child link="camera" /></joint>
        </robot>
        """
    )

    add_contact(root, robot)

    excludes = root.findall("./contact/exclude")
    assert len(excludes) == 1
    assert excludes[0].attrib == {"body1": "base", "body2": "arm"}


def test_add_weld_constraints_creates_equality_block() -> None:
    root = ET.fromstring("<mujoco />")
    metadata = ConversionMetadata(
        weld_constraints=[WeldConstraint(body1="arm", body2="world", solimp=[0.95, 0.99, 0.005], solref=[0.01, 1.0])]
    )

    add_weld_constraints(root, metadata)

    weld = root.find("./equality/weld")
    assert weld is not None
    assert weld.attrib["body1"] == "arm"
    assert weld.attrib["body2"] == "world"
    assert weld.attrib["solimp"] == "0.95 0.99 0.005"
    assert weld.attrib["solref"] == "0.01 1"


def test_add_assets_prefers_mtl_materials_and_adds_default_material() -> None:
    root = ET.fromstring("<mujoco />")

    add_assets(
        root,
        {
            "plain": "0.1 0.2 0.3 1",
            "mtl_material": "0.9 0.9 0.9 1",
            "default_material": "0 0 0 1",
        },
        {"mtl_material": Material(name="mtl_material", Kd="0.3 0.4 0.5", map_Kd="textures/diffuse.png")},
    )

    texture = root.find("./asset/texture")
    materials = root.findall("./asset/material")
    material_names = [material.attrib["name"] for material in materials]

    assert texture is not None
    assert texture.attrib["name"] == "diffuse"
    assert texture.attrib["file"] == "textures/diffuse.png"
    assert material_names.count("mtl_material") == 1
    assert material_names.count("plain") == 1
    assert material_names.count("default_material") == 1

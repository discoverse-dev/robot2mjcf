"""Regression tests for previously uncovered high-risk modules."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh

from robot2mjcf.cli.mjcf2obj import export_mjcf_bodies
from robot2mjcf.core.model import (
    CollisionGeometry,
    CollisionType,
    ConversionMetadata,
    ForceSensor,
    ImuSensor,
    SiteMetadata,
    TouchSensor,
)
from robot2mjcf.postprocess.add_sensors import add_sensors
from robot2mjcf.postprocess.collisions import update_collisions
from robot2mjcf.postprocess.convex_collision import convex_collision_assets
from robot2mjcf.postprocess.convex_decomposition import convex_decomposition_assets


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_export_mjcf_bodies_writes_obj_and_mtl(tmp_path) -> None:
    mesh_file = write_text(
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
        """
        <mujoco>
          <compiler meshdir="meshes" />
          <asset>
            <mesh name="triangle" file="triangle.obj" />
            <material name="blue" rgba="0 0 1 1" />
          </asset>
          <worldbody>
            <body name="link">
              <geom name="link_geom" type="mesh" mesh="triangle" material="blue" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )
    output_dir = tmp_path / "out"

    export_mjcf_bodies(mjcf_path, output_dir)

    obj_path = output_dir / "link.obj"
    mtl_path = output_dir / "link.mtl"
    assert mesh_file.exists()
    assert obj_path.exists()
    assert mtl_path.exists()
    assert "mtllib link.mtl" in obj_path.read_text()
    assert "newmtl blue" in mtl_path.read_text()


def test_convex_collision_assets_replaces_collision_geom(tmp_path, monkeypatch) -> None:
    root = ET.fromstring(
        """
        <mujoco>
          <compiler meshdir="." />
          <asset>
            <mesh name="arm_mesh" file="arm.obj" />
          </asset>
          <worldbody>
            <body name="arm">
              <geom name="arm_collision" class="collision" type="mesh" mesh="arm_mesh" />
            </body>
          </worldbody>
        </mujoco>
        """
    )

    monkeypatch.setattr(
        "robot2mjcf.postprocess.convex_collision.process_single_mesh",
        lambda mesh_info: ("arm_mesh", [("arm_mesh_convex", "arm_mesh_convex/arm_mesh_convex.stl")]),
    )

    convex_collision_assets(tmp_path / "model.xml", root, max_processes=1)

    assert root.find(".//asset/mesh[@name='arm_mesh_convex']") is not None
    assert root.find(".//body[@name='arm']/geom[@mesh='arm_mesh_convex']") is not None
    assert root.find(".//body[@name='arm']/geom[@name='arm_collision']") is None


def test_convex_decomposition_assets_splits_collision_geom(tmp_path, monkeypatch) -> None:
    root = ET.fromstring(
        """
        <mujoco>
          <compiler meshdir="." />
          <asset>
            <mesh name="arm_mesh" file="arm.obj" />
          </asset>
          <worldbody>
            <body name="arm">
              <geom name="arm_collision" class="collision" type="mesh" mesh="arm_mesh" />
            </body>
          </worldbody>
        </mujoco>
        """
    )

    monkeypatch.setattr(
        "robot2mjcf.postprocess.convex_decomposition.process_single_mesh",
        lambda mesh_info: (
            "arm_mesh",
            [
                ("arm_mesh_part1", "arm_mesh_parts/arm_mesh_part1.stl"),
                ("arm_mesh_part2", "arm_mesh_parts/arm_mesh_part2.stl"),
            ],
        ),
    )

    convex_decomposition_assets(tmp_path / "model.xml", root, max_processes=1)

    assert root.find(".//asset/mesh[@name='arm_mesh_part1']") is not None
    assert root.find(".//asset/mesh[@name='arm_mesh_part2']") is not None
    assert root.find(".//body[@name='arm']/geom[@mesh='arm_mesh_part1']") is not None
    assert root.find(".//body[@name='arm']/geom[@mesh='arm_mesh_part2']") is not None


def test_update_collisions_replaces_mesh_with_box(tmp_path, monkeypatch) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <asset>
            <mesh name="arm_mesh" file="arm.obj" />
          </asset>
          <worldbody>
            <body name="arm">
              <geom name="arm_collision" class="collision" type="mesh" mesh="arm_mesh" />
              <geom name="arm_visual" class="visual" type="mesh" mesh="arm_mesh" material="blue" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    monkeypatch.setattr("robot2mjcf.postprocess.collisions.trimesh.load", lambda path: trimesh.creation.box())

    update_collisions(
        mjcf_path,
        [CollisionGeometry(name="arm", collision_type=CollisionType.BOX)],
    )

    root = ET.parse(mjcf_path).getroot()
    box_geom = root.find(".//body[@name='arm']/geom[@name='arm_collision_box']")
    visual_geom = root.find(".//body[@name='arm']/geom[@name='arm_visual']")

    assert box_geom is not None
    assert box_geom.attrib["type"] == "box"
    assert root.find(".//body[@name='arm']/geom[@name='arm_collision']") is None
    assert visual_geom is not None
    assert visual_geom.attrib["type"] == "box"
    assert "mesh" not in visual_geom.attrib


def test_update_collisions_parallel_capsules_adds_capsule_pairs(tmp_path, monkeypatch) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <asset>
            <mesh name="arm_mesh" file="arm.obj" />
          </asset>
          <worldbody>
            <body name="arm">
              <geom name="arm_collision" class="collision" type="mesh" mesh="arm_mesh" />
              <geom name="arm_visual" class="visual" type="mesh" mesh="arm_mesh" material="blue" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    monkeypatch.setattr("robot2mjcf.postprocess.collisions.trimesh.load", lambda path: trimesh.creation.box())

    update_collisions(
        mjcf_path,
        [CollisionGeometry(name="arm", collision_type=CollisionType.PARALLEL_CAPSULES, sphere_radius=0.1)],
    )

    root = ET.parse(mjcf_path).getroot()
    collision_capsules = root.findall(".//body[@name='arm']/geom[@type='capsule'][@class='collision']")
    visual_capsules = root.findall(".//body[@name='arm']/geom[@type='capsule'][@class='visual']")

    assert len(collision_capsules) == 2
    assert len(visual_capsules) == 2
    assert root.find(".//body[@name='arm']/geom[@name='arm_collision']") is None
    assert root.find(".//body[@name='arm']/geom[@name='arm_visual']") is not None


def test_update_collisions_corner_spheres_replace_visual_mesh(tmp_path, monkeypatch) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <asset>
            <mesh name="arm_mesh" file="arm.obj" />
          </asset>
          <worldbody>
            <body name="arm">
              <geom name="arm_collision" class="collision" type="mesh" mesh="arm_mesh" />
              <geom name="arm_visual" class="visual" type="mesh" mesh="arm_mesh" material="blue" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    monkeypatch.setattr("robot2mjcf.postprocess.collisions.trimesh.load", lambda path: trimesh.creation.box())

    update_collisions(
        mjcf_path,
        [CollisionGeometry(name="arm", collision_type=CollisionType.CORNER_SPHERES, sphere_radius=0.1)],
    )

    root = ET.parse(mjcf_path).getroot()
    collision_spheres = root.findall(".//body[@name='arm']/geom[@type='sphere'][@class='collision']")
    visual_spheres = root.findall(".//body[@name='arm']/geom[@type='sphere'][@class='visual']")

    assert len(collision_spheres) == 4
    assert len(visual_spheres) == 4
    assert root.find(".//body[@name='arm']/geom[@name='arm_collision']") is None
    assert root.find(".//body[@name='arm']/geom[@name='arm_visual']") is None


def test_update_collisions_single_sphere_updates_visual_geom(tmp_path, monkeypatch) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <asset>
            <mesh name="arm_mesh" file="arm.obj" />
          </asset>
          <worldbody>
            <body name="arm">
              <geom name="arm_collision" class="collision" type="mesh" mesh="arm_mesh" />
              <geom name="arm_visual" class="visual" type="mesh" mesh="arm_mesh" material="blue" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )

    monkeypatch.setattr("robot2mjcf.postprocess.collisions.trimesh.load", lambda path: trimesh.creation.box())

    update_collisions(
        mjcf_path,
        [CollisionGeometry(name="arm", collision_type=CollisionType.SINGLE_SPHERE, sphere_radius=0.1)],
    )

    root = ET.parse(mjcf_path).getroot()
    collision_sphere = root.find(".//body[@name='arm']/geom[@name='arm_collision_sphere']")
    visual_geom = root.find(".//body[@name='arm']/geom[@name='arm_visual']")

    assert collision_sphere is not None
    assert collision_sphere.attrib["type"] == "sphere"
    assert visual_geom is not None
    assert visual_geom.attrib["type"] == "sphere"
    assert "mesh" not in visual_geom.attrib


def test_add_sensors_creates_sites_and_sensor_entries(tmp_path) -> None:
    mjcf_path = write_text(
        tmp_path / "model.xml",
        """
        <mujoco>
          <worldbody>
            <body name="base">
              <body name="arm" />
            </body>
          </worldbody>
        </mujoco>
        """.strip(),
    )
    metadata = ConversionMetadata(
        cameras=[],
        imus=[ImuSensor(body_name="arm", pos=[0.0, 0.0, 0.1], rpy=[0.0, 0.0, 90.0])],
        sites=[SiteMetadata(name="wrist_site", body_name="arm", pos=[0.0, 0.0, 0.2], size=[0.01])],
        force_sensors=[ForceSensor(body_name="arm", site_name="wrist_site")],
        touch_sensors=[TouchSensor(body_name="arm", site_name="wrist_site")],
    )

    add_sensors(mjcf_path, "base", metadata)

    root = ET.parse(mjcf_path).getroot()
    sensor_names = {elem.attrib["name"] for elem in root.find("sensor") or []}

    assert "base_site_pos" in sensor_names
    assert "arm_acc" in sensor_names
    assert "arm_gyro" in sensor_names
    assert "arm_mag" in sensor_names
    assert "wrist_site_force" in sensor_names
    assert "wrist_site_touch" in sensor_names
    assert root.find(".//body[@name='arm']/site[@name='wrist_site']") is not None

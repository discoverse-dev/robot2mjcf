"""Tests for pure conversion helpers."""

import xml.etree.ElementTree as ET

from robot2mjcf.conversion_helpers import (
    build_joint_maps,
    collect_mimic_constraints,
    collect_urdf_materials,
    load_conversion_metadata,
    resolve_output_path,
)


def test_resolve_output_path_defaults_to_output_mjcf(tmp_path) -> None:
    urdf_dir = tmp_path / "demo"
    urdf_dir.mkdir()
    urdf_path = urdf_dir / "robot.urdf"

    output_path, warning = resolve_output_path(urdf_path, None)

    assert output_path == urdf_dir / "output_mjcf" / "robot.xml"
    assert warning is None


def test_resolve_output_path_rejects_same_directory(tmp_path) -> None:
    urdf_dir = tmp_path / "demo"
    urdf_dir.mkdir()
    urdf_path = urdf_dir / "robot.urdf"

    output_path, warning = resolve_output_path(urdf_path, urdf_dir / "out.xml")

    assert output_path == urdf_dir / "output_mjcf" / "robot.xml"
    assert warning is not None
    assert "same directory" in warning


def test_load_conversion_metadata_falls_back_to_defaults(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{not-json")

    metadata = load_conversion_metadata(metadata_path)

    assert metadata.add_floor is True
    assert metadata.freejoint is True


def test_collect_urdf_materials_includes_root_and_visual_materials() -> None:
    robot = ET.fromstring(
        """
        <robot name="demo">
          <material name="root_material">
            <color rgba="1 0 0 1" />
          </material>
          <link name="base">
            <visual>
              <material name="visual_material">
                <color rgba="0 1 0 1" />
              </material>
            </visual>
          </link>
        </robot>
        """
    )

    materials = collect_urdf_materials(robot, collision_only=False)

    assert materials == {
        "root_material": "1 0 0 1",
        "visual_material": "0 1 0 1",
    }


def test_build_joint_maps_and_mimic_constraints() -> None:
    robot = ET.fromstring(
        """
        <robot name="demo">
          <link name="base" />
          <link name="finger" />
          <link name="follower" />
          <joint name="finger_joint" type="revolute">
            <parent link="base" />
            <child link="finger" />
          </joint>
          <joint name="follower_joint" type="revolute">
            <parent link="finger" />
            <child link="follower" />
            <mimic joint="finger_joint" multiplier="-1" offset="0.5" />
          </joint>
        </robot>
        """
    )

    link_map, parent_map, child_joints = build_joint_maps(robot)
    mimic_constraints = collect_mimic_constraints(robot)

    assert sorted(link_map) == ["base", "finger", "follower"]
    assert [child for child, _ in parent_map["base"]] == ["finger"]
    assert [child for child, _ in parent_map["finger"]] == ["follower"]
    assert child_joints["finger"].attrib["name"] == "finger_joint"
    assert child_joints["follower"].attrib["name"] == "follower_joint"
    assert mimic_constraints == [("finger_joint", "follower_joint", -1.0, 0.5)]

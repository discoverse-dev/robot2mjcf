"""Tests for robot scene assembly helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from urdf_to_mjcf.conversion.pipeline import ConversionContext, add_extra_joints, assemble_robot_scene
from urdf_to_mjcf.core.geometry import ParsedJointParams
from urdf_to_mjcf.core.model import ActuatorMetadata, ConversionMetadata, ExtraJoint


def test_assemble_robot_scene_orchestrates_body_assets_and_mesh_pipeline(tmp_path, monkeypatch) -> None:
    root = ET.fromstring("<mujoco><worldbody /></mujoco>")
    worldbody = root.find("worldbody")
    assert worldbody is not None
    context = ConversionContext(
        mjcf_root=root,
        worldbody=worldbody,
        link_map={"base": ET.Element("link")},
        parent_map={},
        root_link_name="base",
        actuator_metadata={"joint1": ActuatorMetadata(actuator_type="motor")},
        mimic_constraints=[("joint0", "joint1", 1.0, 0.0)],
        metadata=ConversionMetadata(),
    )
    robot_body = ET.Element("body", attrib={"name": "base"})
    actuator_joints = [ParsedJointParams(name="joint1", type="hinge")]
    material_marker = object()

    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.resolve_workspace_search_paths",
        lambda urdf_path: [tmp_path / "ws"],
    )
    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.build_robot_body_tree",
        lambda *args, **kwargs: (robot_body, actuator_joints),
    )
    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.collect_single_obj_materials",
        lambda *args, **kwargs: {"mat": material_marker},
    )
    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.add_assets",
        lambda mjcf_root, materials, obj_materials: calls.append(("assets", obj_materials)),
    )
    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.add_actuators",
        lambda mjcf_root, joints, metadata: calls.append(("actuators", [joint.name for joint in joints])),
    )
    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.add_mimic_equality_constraints",
        lambda mjcf_root, mimic_constraints: calls.append(("mimic", mimic_constraints)),
    )
    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.copy_mesh_assets",
        lambda *args, **kwargs: type(
            "CopyResult",
            (),
            {
                "mesh_assets": {"mesh_a": "mesh_a.obj"},
                "mesh_file_paths": {"mesh_a": tmp_path / "meshes" / "mesh_a.obj"},
            },
        )(),
    )
    monkeypatch.setattr(
        "urdf_to_mjcf.conversion.pipeline.add_mesh_assets_to_xml",
        lambda mjcf_root, mesh_assets, *, urdf_dir: calls.append(("mesh_xml", mesh_assets)),
    )

    result = assemble_robot_scene(
        context,
        urdf_path=tmp_path / "robot.urdf",
        urdf_dir=tmp_path,
        mjcf_path=tmp_path / "out" / "robot.xml",
        collision_only=False,
        materials={"plain": "1 1 1 1"},
    )

    assert result.robot_body is robot_body
    assert result.actuator_joints == actuator_joints
    assert result.mesh_file_paths == {"mesh_a": tmp_path / "meshes" / "mesh_a.obj"}
    assert robot_body.attrib["childclass"] == "robot"
    assert context.worldbody[0] is robot_body
    assert calls == [
        ("assets", {"mat": material_marker}),
        ("actuators", ["joint1"]),
        ("mimic", [("joint0", "joint1", 1.0, 0.0)]),
        ("mesh_xml", {"mesh_a": "mesh_a.obj"}),
    ]


def test_add_extra_joints_injects_joints_and_minimal_inertial() -> None:
    robot_body = ET.fromstring('<body name="base"><body name="arm" /></body>')
    actuator_joints: list[ParsedJointParams] = []
    extra_joints = [
        ExtraJoint(
            body_name="base",
            name="base_x",
            type="slide",
            axis=[1, 0, 0],
            joint_class="base_slide",
            range=[-1, 1],
        ),
        ExtraJoint(
            body_name="base",
            name="base_y",
            type="slide",
            axis=[0, 1, 0],
            joint_class="base_slide",
            range=[-2, 2],
        ),
        ExtraJoint(
            body_name="base",
            name="base_yaw",
            type="hinge",
            axis=[0, 0, 1],
            joint_class="base_yaw",
        ),
    ]

    add_extra_joints(robot_body, actuator_joints, extra_joints)

    joints = robot_body.findall("joint")
    assert [joint.attrib["name"] for joint in joints] == ["base_x", "base_y", "base_yaw"]
    assert joints[0].attrib == {
        "name": "base_x",
        "type": "slide",
        "axis": "1.0 0.0 0.0",
        "class": "base_slide",
        "range": "-1.0 1.0",
    }
    assert robot_body.find("inertial").attrib == {
        "pos": "0 0 0",
        "mass": "0.001",
        "diaginertia": "1e-06 1e-06 1e-06",
    }
    assert [joint.name for joint in actuator_joints] == ["base_x", "base_y", "base_yaw"]

"""Tests for Pydantic models."""

from urdf_to_mjcf.core.model import (
    ActuatorMetadata,
    CollisionGeometry,
    CollisionParams,
    CollisionType,
    ConversionMetadata,
    DefaultJointMetadata,
    ExtraJoint,
    ExtraJointGroup,
    JointData,
    JointMetadata,
)


def test_collision_params_defaults() -> None:
    cp = CollisionParams()
    assert cp.condim == 3
    assert cp.friction == [1.0, 0.01, 0.01]


def test_default_joint_metadata_from_dict() -> None:
    data = {
        "joint": {"damping": 0.1, "armature": 0.01},
        "actuator": {"actuator_type": "motor", "kp": 100.0},
    }
    meta = DefaultJointMetadata.from_dict(data)
    assert meta.joint.damping == 0.1
    assert meta.actuator.kp == 100.0


def test_joint_metadata_from_dict_accepts_empty_actuator() -> None:
    data = {
        "damping": 0.5,
        "armature": 0.001,
        "actuator": {},
        "sensors": {"jointvel": True},
    }
    meta = JointMetadata.from_dict(data)
    assert meta.damping == 0.5
    assert meta.armature == 0.001
    assert meta.actuator is not None
    assert meta.actuator.actuator_type is None
    assert meta.sensors is not None
    assert meta.sensors.jointvel is True


def test_actuator_metadata_from_dict() -> None:
    data = {
        "actuator_type": "position",
        "ctrllimited": True,
        "gear": 1.0,
        "forcelimited": True,
    }
    meta = ActuatorMetadata.from_dict(data)
    assert meta.actuator_type == "position"
    assert meta.ctrllimited is True
    assert meta.gear == 1.0
    assert meta.forcelimited is True


def test_conversion_metadata_defaults() -> None:
    meta = ConversionMetadata()
    assert meta.freejoint is True
    assert meta.add_floor is True
    assert len(meta.cameras) == 2
    assert meta.cameras[0].fovy == 90.0


def test_conversion_metadata_json_roundtrip() -> None:
    meta = ConversionMetadata(
        height_offset=0.5,
        angle="degree",
    )
    raw = meta.model_dump_json() if hasattr(meta, "model_dump_json") else meta.json()
    loaded = (
        ConversionMetadata.model_validate_json(raw)
        if hasattr(ConversionMetadata, "model_validate_json")
        else ConversionMetadata.parse_raw(raw)
    )
    assert loaded.height_offset == 0.5
    assert loaded.angle == "degree"


def test_joint_data_json_roundtrip() -> None:
    joint_data = JointData(
        extra_joints=[
            ExtraJointGroup(
                body="base_link",
                joints=[ExtraJoint(name="base_x", type="slide", axis="x", range=[-10, 10])],
            )
        ],
        joints={
            "base_x": JointMetadata(armature=0.001, damping=0.0),
        },
    )

    raw = joint_data.model_dump_json() if hasattr(joint_data, "model_dump_json") else joint_data.json()
    loaded = (
        JointData.model_validate_json(raw) if hasattr(JointData, "model_validate_json") else JointData.parse_raw(raw)
    )

    assert loaded.extra_joints[0].body == "base_link"
    assert loaded.extra_joints[0].joints[0].name == "base_x"
    assert loaded.extra_joints[0].joints[0].axis_values() == (1.0, 0.0, 0.0)
    assert loaded.joints["base_x"].armature == 0.001


def test_collision_geometry_enum() -> None:
    cg = CollisionGeometry(name="base", collision_type=CollisionType.BOX)
    assert cg.collision_type == CollisionType.BOX
    assert cg.sphere_radius == 0.01

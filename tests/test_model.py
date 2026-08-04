"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from urdf_to_mjcf.core.model import (
    ActuatorConfig,
    CollisionGeometry,
    CollisionParams,
    CollisionType,
    ConversionMetadata,
    ExtraJoint,
    ExtraJointGroup,
    JointData,
    JointMetadata,
)


def test_collision_params_defaults() -> None:
    cp = CollisionParams()
    assert cp.condim == 3
    assert cp.friction == [1.0, 0.01, 0.01]


def test_actuator_config_parses_supported_fields() -> None:
    meta = ActuatorConfig(
        actuator_type="position",
        ctrllimited=True,
        gear=1.0,
        forcelimited=True,
    )
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
                joints=[ExtraJoint(name="base_x", type="slide", axis="x", range=(-10, 10))],
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


def test_joint_data_rejects_unknown_and_invalid_range_fields() -> None:
    with pytest.raises(ValidationError):
        JointData(joints={"joint": {"joint_class": "legacy"}})  # type: ignore[dict-item]

    with pytest.raises(ValidationError):
        JointMetadata(actuatorfrcrange=[-1.0])  # type: ignore[arg-type]


def test_collision_geometry_enum() -> None:
    cg = CollisionGeometry(name="base", collision_type=CollisionType.BOX)
    assert cg.collision_type == CollisionType.BOX
    assert cg.sphere_radius == 0.01

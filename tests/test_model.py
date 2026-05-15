"""Tests for Pydantic models."""

from urdf_to_mjcf.core.model import (
    ActuatorMetadata,
    CollisionGeometry,
    CollisionParams,
    CollisionType,
    ConversionMetadata,
    DefaultJointMetadata,
    ExtraJoint,
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


def test_actuator_metadata_from_dict() -> None:
    data = {"actuator_type": "position", "gear": 1.0}
    meta = ActuatorMetadata.from_dict(data)
    assert meta.actuator_type == "position"
    assert meta.gear == 1.0


def test_conversion_metadata_defaults() -> None:
    meta = ConversionMetadata()
    assert meta.freejoint is True
    assert meta.add_floor is True
    assert meta.extra_joints == []
    assert len(meta.cameras) == 2
    assert meta.cameras[0].fovy == 90.0


def test_conversion_metadata_json_roundtrip() -> None:
    meta = ConversionMetadata(
        height_offset=0.5,
        angle="degree",
        extra_joints=[
            ExtraJoint(
                body_name="base_link",
                name="base_x",
                type="slide",
                axis=[1, 0, 0],
                joint_class="base_slide",
                range=[-10, 10],
            )
        ],
    )
    raw = meta.model_dump_json() if hasattr(meta, "model_dump_json") else meta.json()
    loaded = (
        ConversionMetadata.model_validate_json(raw)
        if hasattr(ConversionMetadata, "model_validate_json")
        else ConversionMetadata.parse_raw(raw)
    )
    assert loaded.height_offset == 0.5
    assert loaded.angle == "degree"
    assert loaded.extra_joints[0].body_name == "base_link"
    assert loaded.extra_joints[0].name == "base_x"


def test_collision_geometry_enum() -> None:
    cg = CollisionGeometry(name="base", collision_type=CollisionType.BOX)
    assert cg.collision_type == CollisionType.BOX
    assert cg.sphere_radius == 0.01

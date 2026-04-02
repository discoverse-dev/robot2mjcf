"""Tests for Pydantic models."""

from robot2mjcf.core.model import (
    ActuatorMetadata,
    CollisionGeometry,
    CollisionParams,
    CollisionType,
    ConversionMetadata,
    DefaultJointMetadata,
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
    assert len(meta.cameras) == 2
    assert meta.cameras[0].fovy == 90.0


def test_conversion_metadata_json_roundtrip() -> None:
    meta = ConversionMetadata(height_offset=0.5, angle="degree")
    raw = meta.model_dump_json() if hasattr(meta, "model_dump_json") else meta.json()
    loaded = ConversionMetadata.model_validate_json(raw) if hasattr(ConversionMetadata, "model_validate_json") else ConversionMetadata.parse_raw(raw)
    assert loaded.height_offset == 0.5
    assert loaded.angle == "degree"


def test_collision_geometry_enum() -> None:
    cg = CollisionGeometry(name="base", collision_type=CollisionType.BOX)
    assert cg.collision_type == CollisionType.BOX
    assert cg.sphere_radius == 0.01

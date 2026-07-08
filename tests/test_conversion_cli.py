"""Tests for CLI-side conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urdf_to_mjcf.cli.convert import (
    apply_metadata_overrides,
    load_actuator_metadata_files,
    load_default_metadata_files,
    load_joint_data_files,
    normalize_appendix_files,
)
from urdf_to_mjcf.core.model import ConversionMetadata


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_load_default_metadata_files_merges_in_order(tmp_path) -> None:
    first = write_json(
        tmp_path / "default_a.json",
        {
            "arm": {
                "joint": {"damping": 1.0},
                "actuator": {"actuator_type": "motor"},
            }
        },
    )
    second = write_json(
        tmp_path / "default_b.json",
        {
            "arm": {
                "joint": {"damping": 2.0},
                "actuator": {"actuator_type": "position"},
            },
            "leg": {
                "joint": {"stiffness": 3.0},
                "actuator": {"actuator_type": "motor"},
            },
        },
    )

    loaded = load_default_metadata_files([str(first), str(second)])

    assert loaded is not None
    assert loaded["arm"].joint.damping == 2.0
    assert loaded["arm"].actuator.actuator_type == "position"
    assert loaded["leg"].joint.stiffness == 3.0


def test_load_actuator_metadata_files_merges_in_order(tmp_path) -> None:
    first = write_json(tmp_path / "actuator_a.json", {"joint1": {"actuator_type": "motor", "gear": 1.0}})
    second = write_json(
        tmp_path / "actuator_b.json",
        {
            "joint1": {"actuator_type": "position", "kp": 50.0},
            "joint2": {"actuator_type": "motor"},
        },
    )

    loaded = load_actuator_metadata_files([str(first), str(second)])

    assert loaded is not None
    assert loaded["joint1"].actuator_type == "position"
    assert loaded["joint1"].kp == 50.0
    assert loaded["joint2"].actuator_type == "motor"


def test_load_joint_data_files_merges_joints_and_extra_joints_in_order(tmp_path) -> None:
    first = write_json(
        tmp_path / "joint_data_a.json",
        {
            "extra_joints": [
                {
                    "body": "base",
                    "joints": [
                        {"name": "base_x", "type": "slide", "axis": "x"},
                    ],
                }
            ],
            "joints": {
                "joint1": {
                    "armature": 0.001,
                    "damping": 0.5,
                    "actuator": {"actuator_type": "position", "kp": 10.0},
                },
                "joint2": {
                    "frictionloss": 0.02,
                    "actuator": {},
                },
            },
        },
    )
    second = write_json(
        tmp_path / "joint_data_b.json",
        {
            "joints": {
                "joint1": {
                    "damping": 0.7,
                    "actuator": {
                        "actuator_type": "position",
                        "ctrllimited": True,
                        "kp": 20.0,
                        "forcelimited": True,
                    },
                }
            }
        },
    )

    loaded = load_joint_data_files([str(first), str(second)])

    assert loaded is not None
    assert loaded.extra_joints[0].body == "base"
    assert loaded.extra_joints[0].joints[0].axis == "x"
    assert loaded.joints["joint1"].armature is None
    assert loaded.joints["joint1"].damping == 0.7
    assert loaded.joints["joint1"].actuator is not None
    assert loaded.joints["joint1"].actuator.actuator_type == "position"
    assert loaded.joints["joint1"].actuator.ctrllimited is True
    assert loaded.joints["joint1"].actuator.kp == 20.0
    assert loaded.joints["joint1"].actuator.forcelimited is True
    assert loaded.joints["joint2"].frictionloss == 0.02
    assert loaded.joints["joint2"].actuator is not None
    assert loaded.joints["joint2"].actuator.actuator_type is None


def test_metadata_loaders_return_none_for_empty_inputs() -> None:
    assert load_default_metadata_files(None) is None
    assert load_default_metadata_files([]) is None
    assert load_actuator_metadata_files(None) is None
    assert load_actuator_metadata_files([]) is None
    assert load_joint_data_files(None) is None
    assert load_joint_data_files([]) is None


def test_apply_metadata_overrides_updates_common_cli_fields() -> None:
    metadata = ConversionMetadata()

    overridden = apply_metadata_overrides(metadata, freejoint=False, add_floor=False)

    assert overridden.freejoint is False
    assert overridden.add_floor is False


def test_load_default_metadata_files_exits_on_invalid_json(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not valid json")

    with pytest.raises(SystemExit) as exc:
        load_default_metadata_files([str(broken)])

    assert exc.value.code == 1


def test_normalize_appendix_files_returns_paths_or_none() -> None:
    assert normalize_appendix_files(None) is None
    assert normalize_appendix_files([]) is None
    assert normalize_appendix_files(["a.xml", "b.xml"]) == [Path("a.xml"), Path("b.xml")]

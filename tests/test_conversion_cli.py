"""Tests for CLI-side conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urdf_to_mjcf.cli.convert import (
    load_joint_data_files,
    normalize_appendix_files,
)


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


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


def test_joint_data_loader_returns_none_for_empty_inputs() -> None:
    assert load_joint_data_files(None) is None
    assert load_joint_data_files([]) is None


def test_load_joint_data_files_exits_on_invalid_json(tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not valid json")

    with pytest.raises(SystemExit) as exc:
        load_joint_data_files([str(broken)])

    assert exc.value.code == 1


def test_normalize_appendix_files_returns_paths_or_none() -> None:
    assert normalize_appendix_files(None) is None
    assert normalize_appendix_files([]) is None
    assert normalize_appendix_files(["a.xml", "b.xml"]) == [Path("a.xml"), Path("b.xml")]

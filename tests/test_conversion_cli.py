"""Tests for CLI-side conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot2mjcf.cli.convert import (
    load_actuator_metadata_files,
    load_default_metadata_files,
    normalize_appendix_files,
)


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


def test_metadata_loaders_return_none_for_empty_inputs() -> None:
    assert load_default_metadata_files(None) is None
    assert load_default_metadata_files([]) is None
    assert load_actuator_metadata_files(None) is None
    assert load_actuator_metadata_files([]) is None


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

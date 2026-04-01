"""Tests for geometry utilities."""

import math

from robot2mjcf.geometry import (
    build_transform,
    format_value,
    parse_vector,
    quat_from_str,
    quat_to_rot,
    rpy_to_quat,
)


def test_format_value_basic() -> None:
    assert format_value(1.0) == "1"
    assert format_value(1.23456) == "1.2346"
    assert format_value(-0.0) == "0"
    assert format_value(0.1000) == "0.1"


def test_parse_vector() -> None:
    assert parse_vector("1 2 3") == [1.0, 2.0, 3.0]
    assert parse_vector("0.5 -1.5 2.0") == [0.5, -1.5, 2.0]


def test_quat_from_str() -> None:
    assert quat_from_str("1 0 0 0") == [1.0, 0.0, 0.0, 0.0]


def test_quat_to_rot_identity() -> None:
    rot = quat_to_rot([1.0, 0.0, 0.0, 0.0])
    assert rot[0][0] == 1.0
    assert rot[1][1] == 1.0
    assert rot[2][2] == 1.0


def test_build_transform() -> None:
    t = build_transform("1 2 3", "1 0 0 0")
    assert t[0][3] == 1.0
    assert t[1][3] == 2.0
    assert t[2][3] == 3.0
    assert t[3] == [0.0, 0.0, 0.0, 1.0]


def test_rpy_to_quat_zero() -> None:
    q = rpy_to_quat("0 0 0")
    assert q == "1 0 0 0"


def test_rpy_to_quat_nonzero() -> None:
    q = rpy_to_quat(f"{math.pi} 0 0")
    parts = list(map(float, q.split()))
    assert abs(parts[0]) < 1e-6
    assert abs(parts[1] - 1.0) < 1e-6
    assert abs(parts[2]) < 1e-6
    assert abs(parts[3]) < 1e-6

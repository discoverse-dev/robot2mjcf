"""Tests for final MJCF output helpers."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from robot2mjcf.conversion_output import adjust_robot_body_height, save_initial_mjcf_and_apply_postprocess
from robot2mjcf.conversion_postprocess import PostprocessOptions
from robot2mjcf.model import ConversionMetadata


def test_adjust_robot_body_height_updates_body_pos(monkeypatch) -> None:
    body = ET.fromstring("<body name='base' pos='1 2 3' />")

    monkeypatch.setattr("robot2mjcf.conversion_output.compute_min_z", lambda *args, **kwargs: -0.5)

    offset = adjust_robot_body_height(
        body,
        mesh_file_paths={"mesh": Path("mesh.obj")},
        height_offset=0.25,
    )

    assert offset == 0.75
    assert body.attrib["pos"] == "1 2 3.75"


def test_save_initial_mjcf_and_apply_postprocess_writes_and_dispatches(tmp_path, monkeypatch) -> None:
    root = ET.fromstring("<mujoco><worldbody /></mujoco>")
    mjcf_path = tmp_path / "model.xml"
    calls: list[tuple[str, object]] = []

    def fake_save_xml(path, tree) -> None:
        calls.append(("save", Path(path)))
        tree.write(path)

    def fake_apply_postprocess_pipeline(path, *, options) -> None:
        calls.append(("postprocess", Path(path)))
        calls.append(("options", options))

    monkeypatch.setattr("robot2mjcf.conversion_output.save_xml", fake_save_xml)
    monkeypatch.setattr("robot2mjcf.conversion_output.apply_postprocess_pipeline", fake_apply_postprocess_pipeline)

    options = PostprocessOptions(
        metadata=ConversionMetadata(),
        collision_only=False,
        collision_type=None,
        max_vertices=1000,
        appendix_files=None,
        capture_images=False,
        run_mesh_postprocess=True,
    )
    save_initial_mjcf_and_apply_postprocess(root, mjcf_path=mjcf_path, options=options)

    assert mjcf_path.exists()
    assert calls[0] == ("save", mjcf_path)
    assert calls[1] == ("postprocess", mjcf_path)
    assert calls[2] == ("options", options)

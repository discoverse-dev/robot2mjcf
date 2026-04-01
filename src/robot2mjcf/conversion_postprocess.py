"""Post-processing pipeline helpers for URDF-to-MJCF conversion."""

from __future__ import annotations

import logging
from pathlib import Path

from robot2mjcf.model import ConversionMetadata
from robot2mjcf.postprocess.add_appendix import add_appendix
from robot2mjcf.postprocess.add_backlash import add_backlash
from robot2mjcf.postprocess.add_floor import add_floor
from robot2mjcf.postprocess.add_light import add_light
from robot2mjcf.postprocess.base_joint import fix_base_joint
from robot2mjcf.postprocess.check_shell import check_shell_meshes
from robot2mjcf.postprocess.collision_to_stl import collision_to_stl
from robot2mjcf.postprocess.collisions import update_collisions
from robot2mjcf.postprocess.convex_collision import convex_collision
from robot2mjcf.postprocess.convex_decomposition import convex_decomposition
from robot2mjcf.postprocess.deduplicate_meshes import deduplicate_meshes
from robot2mjcf.postprocess.explicit_floor_contacts import add_explicit_floor_contacts
from robot2mjcf.postprocess.make_degrees import make_degrees
from robot2mjcf.postprocess.move_mesh_scale import move_mesh_scale
from robot2mjcf.postprocess.remove_redundancies import remove_redundancies
from robot2mjcf.postprocess.split_obj_materials import split_obj_by_materials
from robot2mjcf.postprocess.update_mesh import update_mesh

logger = logging.getLogger(__name__)


def maybe_capture_robot_images(mjcf_path: str | Path, *, capture_images: bool) -> None:
    """Capture robot images only when explicitly requested."""
    if not capture_images:
        return

    mjcf_path = Path(mjcf_path)
    print("Capturing robot images...")
    try:
        from robot2mjcf.postprocess.capture import capture_robot_images

        capture_robot_images(mjcf_path)
    except Exception as exc:
        logger.warning("Failed to capture images: %s", exc)
        print(f"⚠️  Image capture failed: {exc}")
        print("   You can manually capture images later using:")
        print(f"   python -m robot2mjcf.postprocess.capture {mjcf_path}")


def apply_postprocess_pipeline(
    mjcf_path: str | Path,
    *,
    metadata: ConversionMetadata,
    collision_only: bool,
    collision_type: str | None,
    max_vertices: int,
    appendix_files: list[Path] | None,
    capture_images: bool,
) -> None:
    """Apply the standard post-processing pipeline to a generated MJCF file."""
    mjcf_path = Path(mjcf_path)

    print("Added light...")
    add_light(mjcf_path)
    if collision_type == "decomposition":
        print("Convex decomposition...")
        convex_decomposition(mjcf_path)
    elif collision_type == "convex_hull":
        print("Convex hull generation...")
        convex_collision(mjcf_path)

    print("Converting collision geometries to STL...")
    collision_to_stl(mjcf_path)
    if not collision_only:
        print("Split OBJ files by materials...")
        split_obj_by_materials(mjcf_path)
    print("Updating meshes...")
    update_mesh(mjcf_path, max_vertices)
    print("Moving mesh scale attributes...")
    move_mesh_scale(mjcf_path)
    print("Checking shell meshes...")
    check_shell_meshes(mjcf_path)
    print("Deduplicating mesh assets...")
    deduplicate_meshes(mjcf_path)

    if metadata.angle != "radian":
        assert metadata.angle == "degree", "Only 'radian' and 'degree' are supported."
        make_degrees(mjcf_path)
    if metadata.backlash:
        add_backlash(mjcf_path, metadata.backlash, metadata.backlash_damping)
    if metadata.freejoint:
        fix_base_joint(mjcf_path, metadata.freejoint)
    if metadata.add_floor:
        add_floor(mjcf_path)
    if metadata.remove_redundancies:
        remove_redundancies(mjcf_path)
    if metadata.collision_geometries is not None:
        update_collisions(mjcf_path, metadata.collision_geometries)
    if metadata.explicit_contacts is not None:
        add_explicit_floor_contacts(
            mjcf_path,
            contact_links=metadata.explicit_contacts.contact_links,
            class_name=metadata.explicit_contacts.class_name,
            floor_name=metadata.floor_name,
        )

    if appendix_files:
        print("Adding appendix...")
        for appendix_file in appendix_files:
            add_appendix(mjcf_path, appendix_file)

    maybe_capture_robot_images(mjcf_path, capture_images=capture_images)

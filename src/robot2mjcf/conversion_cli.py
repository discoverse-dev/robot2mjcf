"""CLI-side helpers for assembling conversion inputs."""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from robot2mjcf.model import ActuatorMetadata, DefaultJointMetadata

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _load_metadata_files(
    metadata_files: Sequence[str] | None,
    *,
    label: str,
    parser: Callable[[dict], T],
) -> dict[str, T] | None:
    """Load keyed metadata from one or more JSON files."""
    loaded: dict[str, T] = {}
    if not metadata_files:
        return None

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r") as f:
                file_metadata = json.load(f)
            for key, value in file_metadata.items():
                loaded[key] = parser(value)
            logger.info("Loaded %s metadata from %s", label, metadata_file)
        except Exception as exc:
            logger.warning("Failed to load %s metadata from %s: %s", label, metadata_file, exc)
            traceback.print_exc()
            raise SystemExit(1) from exc

    return loaded or None


def load_default_metadata_files(metadata_files: Sequence[str] | None) -> dict[str, DefaultJointMetadata] | None:
    """Load default metadata files from CLI arguments."""
    return _load_metadata_files(
        metadata_files,
        label="default",
        parser=DefaultJointMetadata.from_dict,
    )


def load_actuator_metadata_files(metadata_files: Sequence[str] | None) -> dict[str, ActuatorMetadata] | None:
    """Load actuator metadata files from CLI arguments."""
    return _load_metadata_files(
        metadata_files,
        label="actuator",
        parser=ActuatorMetadata.from_dict,
    )


def normalize_appendix_files(appendix_files: Sequence[str] | None) -> list[Path] | None:
    """Convert appendix file CLI arguments to Path objects."""
    if not appendix_files:
        return None
    return [Path(appendix_file) for appendix_file in appendix_files]

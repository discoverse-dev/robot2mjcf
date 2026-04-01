#!/usr/bin/env python3
"""Model path manager for URDF2MJCF_MODEL_PATH environment variable.

This tool helps manage the URDF2MJCF_MODEL_PATH environment variable by:
1. Scanning directories for ROS description packages
2. Adding/removing paths from the environment variable
"""

import argparse
import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)


def is_description_package(path: Path) -> bool:
    """
    Check if a directory is a ROS description package.

    A description package typically:
    - Has a name ending with '_description'
    - Contains a package.xml file
    - Contains common robot description folders like urdf/, meshes/, etc.

    Args:
        path: Path to check

    Returns:
        True if it's a description package, False otherwise
    """
    if not path.is_dir():
        return False

    # Check if it's a ROS package (has package.xml)
    if not (path / "package.xml").exists():
        return False

    # Check if name ends with _description
    if not path.name.endswith("_description"):
        return False

    # Additional check: should have typical robot description folders
    typical_folders = ["urdf", "meshes", "robots", "xacro"]
    has_typical_folder = any((path / folder).exists() for folder in typical_folders)

    return has_typical_folder


def find_description_packages(root_path: Path, max_depth: int = 10) -> Set[Path]:
    """
    Recursively find all ROS description packages under a given root path.

    Args:
        root_path: Root directory to start searching from
        max_depth: Maximum recursion depth

    Returns:
        Set of paths to description packages
    """
    description_packages = set()

    def _recursive_search(current_path: Path, depth: int):
        if depth > max_depth:
            return

        try:
            # Check if current path is a description package
            if is_description_package(current_path):
                description_packages.add(current_path.resolve())
                logger.debug(f"Found description package: {current_path}")
                # Don't search deeper if we found a package
                return

            # Search subdirectories
            if current_path.is_dir():
                for item in current_path.iterdir():
                    if item.is_dir() and not item.name.startswith("."):
                        # Skip common non-package directories
                        if item.name in ["build", "devel", "install", "log", "__pycache__", ".git", ".vscode"]:
                            continue
                        _recursive_search(item, depth + 1)

        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot access {current_path}: {e}")

    _recursive_search(root_path, 0)
    return description_packages


def get_current_model_paths() -> List[Path]:
    """
    Get current paths from URDF2MJCF_MODEL_PATH environment variable.

    Returns:
        List of paths currently in the environment variable
    """
    env_var = os.environ.get("URDF2MJCF_MODEL_PATH", "")
    if not env_var:
        return []

    separator = ";" if platform.system() == "Windows" else ":"
    paths = []

    for path_str in env_var.split(separator):
        path_str = path_str.strip()
        if path_str:
            paths.append(Path(path_str).resolve())

    return paths


def set_model_paths(paths: List[Path]) -> str:
    """
    Set URDF2MJCF_MODEL_PATH environment variable.

    Args:
        paths: List of paths to set

    Returns:
        The export command string for the shell
    """
    separator = ";" if platform.system() == "Windows" else ":"
    path_str = separator.join(str(p) for p in paths)

    # Set in current process (only affects this script, not the parent shell)
    os.environ["URDF2MJCF_MODEL_PATH"] = path_str

    # Return the command string for user to execute
    if platform.system() == "Windows":
        return f'set URDF2MJCF_MODEL_PATH="{path_str}"'
    else:
        return f'export URDF2MJCF_MODEL_PATH="{path_str}"'


def scan_and_add(root_paths: List[Path], append: bool = True, quiet: bool = False) -> None:
    """
    Scan directories for description packages and add to environment variable.

    Args:
        root_paths: Root directories to scan
        append: If True, append to existing paths; if False, replace
        quiet: If True, only output the export command
    """
    if not quiet:
        print("=" * 70, file=sys.stderr)
        print("Scanning for ROS Description Packages", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    all_packages = set()

    for root_path in root_paths:
        if not root_path.exists():
            if not quiet:
                print(f"⚠️  Path does not exist: {root_path}", file=sys.stderr)
            continue

        if not quiet:
            print(f"Scanning: {root_path}", file=sys.stderr)
        packages = find_description_packages(root_path)
        if not quiet:
            print(f"  Found {len(packages)} description package(s)", file=sys.stderr)
        all_packages.update(packages)

    if not all_packages:
        if not quiet:
            print("❌ No description packages found!", file=sys.stderr)
        return

    # Get current paths if appending
    if append:
        current_paths = get_current_model_paths()
        if not quiet:
            print(f"📋 Current URDF2MJCF_MODEL_PATH has {len(current_paths)} path(s)", file=sys.stderr)
    else:
        current_paths = []

    # Combine and deduplicate
    all_paths = list(set(current_paths) | all_packages)
    all_paths.sort()  # Sort for consistent output

    # Generate export command
    export_cmd = set_model_paths(all_paths)

    if not quiet:
        print("=" * 70, file=sys.stderr)
        print(f"✅ Total {len(all_paths)} path(s) in URDF2MJCF_MODEL_PATH:", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for i, path in enumerate(all_paths, 1):
            is_new = path in all_packages
            marker = "🆕" if is_new else "  "
            print(f"{marker} {i}. {path}", file=sys.stderr)
        print(file=sys.stderr)

    # Output only the export command to stdout
    print(export_cmd)


def list_paths() -> None:
    """List current paths in URDF2MJCF_MODEL_PATH."""
    paths = get_current_model_paths()

    print("=" * 70)
    print("Current URDF2MJCF_MODEL_PATH")
    print("=" * 70)

    if not paths:
        print("❌ URDF2MJCF_MODEL_PATH is not set or empty")
        return

    print(f"Total {len(paths)} path(s):")
    for i, path in enumerate(paths, 1):
        exists = "✓" if path.exists() else "✗"
        print(f"  {i}. [{exists}] {path}")

    print()


def unset_var() -> None:
    """Unset URDF2MJCF_MODEL_PATH environment variable."""
    current_paths = get_current_model_paths()

    if not current_paths:
        print("✅ URDF2MJCF_MODEL_PATH is already unset or empty")
        return

    print("=" * 70)
    print("Unsetting URDF2MJCF_MODEL_PATH")
    print("=" * 70)
    print(f"Current value has {len(current_paths)} path(s)")

    # Clear in current process
    if "URDF2MJCF_MODEL_PATH" in os.environ:
        del os.environ["URDF2MJCF_MODEL_PATH"]

    # Generate unset command
    if platform.system() == "Windows":
        unset_cmd = "set URDF2MJCF_MODEL_PATH="
    else:
        unset_cmd = "unset URDF2MJCF_MODEL_PATH"

    print("" + "=" * 70)
    print("📝 To apply this change, run the following command:")
    print("=" * 70)
    print(f"{unset_cmd}")

    print("💡 To remove from your shell profile, edit your ~/.bashrc or ~/.zshrc")
    print("   and remove any lines containing URDF2MJCF_MODEL_PATH")


def main():
    """Main entry point for the model path manager."""
    parser = argparse.ArgumentParser(
        description="Manage URDF2MJCF_MODEL_PATH environment variable for ROS description packages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a directory and add found packages to URDF2MJCF_MODEL_PATH
  %(prog)s scan /path/to/workspace

  # Scan multiple directories
  %(prog)s scan /path/to/workspace1 /path/to/workspace2

  # Scan and replace (not append) existing paths
  %(prog)s scan /path/to/workspace --no-append

  # List current paths in URDF2MJCF_MODEL_PATH
  %(prog)s list

  # Unset the environment variable
  %(prog)s unset
        """,
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan directories for ROS description packages and add to URDF2MJCF_MODEL_PATH"
    )
    scan_parser.add_argument("paths", type=str, nargs="+", help="Root directories to scan for description packages")
    scan_parser.add_argument("--no-append", action="store_true", help="Replace existing paths instead of appending")
    scan_parser.add_argument(
        "--max-depth", type=int, default=10, help="Maximum recursion depth for scanning (default: 10)"
    )

    # List command
    subparsers.add_parser("list", help="List current paths in URDF2MJCF_MODEL_PATH")

    # Unset command
    subparsers.add_parser("unset", help="Unset URDF2MJCF_MODEL_PATH environment variable")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    # Execute command
    if args.command == "scan":
        root_paths = [Path(p).resolve() for p in args.paths]
        scan_and_add(root_paths, append=not args.no_append)

    elif args.command == "list":
        list_paths()

    elif args.command == "unset":
        unset_var()

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

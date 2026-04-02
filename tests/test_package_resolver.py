"""Tests for package resolution helpers."""

from pathlib import Path

from robot2mjcf.core.package_resolver import PackageResolver, find_workspace_from_path


def create_ros_package(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "package.xml").write_text("<package />")
    (path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.0)")
    return path


def test_find_workspace_from_urdf_path(tmp_path) -> None:
    workspace = tmp_path / "ws"
    package = create_ros_package(workspace / "src" / "demo_description")
    urdf_path = package / "urdf" / "robot.urdf"
    urdf_path.parent.mkdir()
    urdf_path.write_text("<robot />")

    assert find_workspace_from_path(urdf_path) == workspace.resolve()


def test_resolve_package_path_finds_nested_package(tmp_path) -> None:
    search_root = tmp_path / "models"
    package = create_ros_package(search_root / "robots" / "demo_description")

    resolver = PackageResolver()

    assert resolver.resolve_package_path("demo_description", [search_root]) == package.resolve()


def test_resolve_package_resource_returns_existing_file(tmp_path) -> None:
    package = create_ros_package(tmp_path / "demo_description")
    mesh_file = package / "meshes" / "arm.stl"
    mesh_file.parent.mkdir()
    mesh_file.write_text("solid arm")

    resolver = PackageResolver()

    assert (
        resolver.resolve_package_resource("package://demo_description/meshes/arm.stl", [tmp_path])
        == mesh_file.resolve()
    )


def test_get_model_paths_from_env_uses_windows_separator(tmp_path, monkeypatch) -> None:
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    path_a.mkdir()
    path_b.mkdir()

    monkeypatch.setenv("URDF2MJCF_MODEL_PATH", f"{path_a};{path_b}")
    monkeypatch.setattr("robot2mjcf.core.package_resolver.platform.system", lambda: "Windows")

    resolver = PackageResolver()

    assert resolver._get_model_paths_from_env() == [path_a.resolve(), path_b.resolve()]


def test_add_default_search_paths_deduplicates(tmp_path, monkeypatch) -> None:
    cwd = tmp_path / "cwd"
    env_path = tmp_path / "env"
    explicit_path = tmp_path / "explicit"
    cwd.mkdir()
    env_path.mkdir()
    explicit_path.mkdir()

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("URDF2MJCF_MODEL_PATH", str(env_path))

    resolver = PackageResolver()
    search_paths = resolver._add_default_search_paths([explicit_path, env_path])

    assert search_paths.count(env_path) == 1
    assert search_paths[0] == explicit_path

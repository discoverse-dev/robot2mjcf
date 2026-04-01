# Troubleshooting

## `package://...` assets cannot be resolved

Checks:

- verify the package exists
- set `URDF2MJCF_MODEL_PATH`
- try `robot2mjcf-modelpath scan /path/to/workspace`

Related files:

- [`package_resolver.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/package_resolver.py)
- [`model_path_manager.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/model_path_manager.py)

## Output path rejected

The converter intentionally rejects writing the MJCF into the same directory as the source URDF.

Use a separate output tree such as:

```bash
robot2mjcf robot.urdf --output output_mjcf/robot.xml
```

## Mesh processing fails on one machine only

Heavy dependencies such as `pymeshlab`, `coacd`, `pycollada`, and `trimesh` are platform-sensitive.

Checks:

- verify native dependencies
- re-run with a smaller scope
- confirm the mesh file exists and is loadable independently

## Static checks for contributors

```bash
uv run ruff format --check src/robot2mjcf tests
uv run ruff check src/robot2mjcf tests
uv run mypy src/robot2mjcf tests
uv run pytest
```

# Architecture

## Scope

`urdf-to-mjcf` converts URDF robot descriptions into MuJoCo MJCF and can optionally run XML/mesh post-processing steps to make the result easier to simulate.

## Package Layout

```
urdf-to-mjcf/
├── __init__.py              # Public API re-export
├── urdf_format.py           # Standalone URDF formatting tool
├── cli/                     # CLI entry points
│   ├── convert.py           # Main command (urdf-to-mjcf)
│   ├── mjcf2obj.py          # Body mesh exporter (urdf-to-mjcf-mjcf2obj)
│   └── model_path.py        # Model path manager (urdf-to-mjcf-modelpath)
├── core/                    # Data models & foundational utilities
│   ├── model.py             # Pydantic metadata schemas
│   ├── geometry.py          # Geometry math (quaternions, transforms)
│   ├── materials.py         # MTL/material parsing
│   ├── package_resolver.py  # ROS package:// path resolution
│   └── utils.py             # XML save/sort utilities
├── conversion/              # Conversion pipeline
│   ├── pipeline.py          # Context building & scene assembly
│   ├── input.py             # URDF parsing, metadata loading, helpers
│   ├── output.py            # Height adjustment & postprocess dispatch
│   ├── body_builder.py      # MJCF body tree construction
│   ├── mjcf_assembly.py     # MJCF XML builders, actuators, constraints
│   └── assets.py            # Mesh asset resolution & copying
└── postprocess/             # Post-processing stages
    ├── __init__.py           # Pipeline orchestrator
    ├── add_light.py          # Add light sources
    ├── add_floor.py          # Add ground plane
    ├── split_obj_materials.py  # DAE/GLB→OBJ, OBJ material splitting
    ├── update_mesh.py        # Mesh simplification & cleanup
    ├── mesh_converter.py     # DAE→OBJ and GLB→OBJ converters
    └── ...                   # Other XML/mesh transforms
```

## Main Flow

1. **Input** (`conversion/input.py`): Parse the URDF, load metadata, collect materials.
2. **Context** (`conversion/pipeline.py`): Build MJCF root, compiler, defaults, worldbody; resolve joint graph.
3. **Scene Assembly** (`conversion/pipeline.py`): Build body tree, copy mesh assets, add actuators/constraints.
4. **Output** (`conversion/output.py`): Adjust robot height, save initial MJCF, dispatch post-processing.
5. **Post-processing** (`postprocess/`): Light, mesh conversion (DAE/GLB→OBJ), material splitting, simplification, floor, sensors, etc.

## Metadata Flow

The CLI keeps two configuration paths explicit:

- `ConversionMetadata` owns scene-wide conversion settings.
- `JointData` owns grouped MJCF-only joints and per-joint dynamics, actuators, and velocity sensors.

`JointData` is the only joint-configuration interface at the CLI seam. `conversion/pipeline.py` resolves it once into `ConversionContext`; body construction and MJCF assembly consume that same object instead of maintaining parallel joint/default/actuator maps. When joint data is omitted, the resolver creates default motor records for movable URDF joints. See the [Metadata Reference](./METADATA_REFERENCE.md) for the exact behavior.

## Layered Dependencies

```
Layer 0 — Pure foundations (no intra-package deps):
  core/model, core/geometry, core/materials, core/utils, core/package_resolver

Layer 1 — Simple consumers:
  conversion/input, conversion/mjcf_assembly, conversion/body_builder, conversion/assets

Layer 2 — Orchestrators:
  conversion/pipeline (uses input, mjcf_assembly, body_builder, assets)
  conversion/output (uses core/geometry, core/utils, postprocess)

Layer 3 — Entry points:
  cli/convert (uses conversion/*, core/model)
```

No circular imports exist. The dependency graph is a strict DAG.

## Post-Process Categories

- **Pure XML transforms**: light, floor, degrees, backlash, base joint, redundancy removal, contacts.
- **XML + filesystem transforms**: mesh conversion (DAE/GLB→OBJ), material splitting, mesh simplification, convex decomposition, deduplication.

The second category is heavier, slower, and can be skipped with `--skip-mesh-postprocess`.

## Recommended Extension Points

- Add new metadata schemas in `core/model.py`.
- Add pure conversion helpers in `conversion/input.py`.
- Add new post-process stages under `postprocess/` and register them in `postprocess/__init__.py`.
- Add regression tests against real examples whenever output semantics change.

# Example Walkthrough

This repository ships with two real robot examples:

- `examples/agilex-piper`
- `examples/realman-rm65`

## Quick Start

### agilex-piper

```bash
cd examples/agilex-piper
urdf-to-mjcf piper.urdf \
  -o output_mjcf/piper.xml \
  -m metadata/metadata.json \
  -jd metadata/joint_data.json \
  -a metadata/appendix.xml

# View the generated model
python -m mujoco.viewer --mjcf=output_mjcf/piper.xml
```

A variant with camera mounts is also available:

```bash
urdf-to-mjcf piper_with_camera.urdf
```

This variant includes `.glb` format meshes, which are automatically converted to OBJ during post-processing.

### realman-rm65

```bash
cd examples/realman-rm65
urdf-to-mjcf rm65b_eg24c2_description.urdf \
  -o output_mjcf/rm65.xml \
  -m metadata/metadata.json \
  -jd metadata/joint_data.json \
  -a metadata/appendix.xml

# View the generated model
python -m mujoco.viewer --mjcf=output_mjcf/rm65.xml
```

## Supported Mesh Formats

| Format | Visual | Collision | Notes |
|---|---|---|---|
| STL | Yes | Yes | Loaded directly |
| OBJ | Yes | Yes | Material splitting supported |
| DAE | Yes | - | Converted to OBJ during post-processing |
| GLB/GLTF | Yes | - | Converted to OBJ during post-processing |

## Regression Testing

The repository regression tests in `tests/test_convert.py` compare semantic output against the real examples:

- Model name
- Body names
- Joint names
- Actuator names
- Equality pairs
- XML element counts
- Referenced mesh existence
- MuJoCo loadability and model counts

This is the preferred refactor safety net.

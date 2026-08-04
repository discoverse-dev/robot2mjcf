# urdf-to-mjcf

URDF to MJCF conversion tool with support for STL, OBJ, DAE, GLB formats, automatic recognition of mimic tags in URDF, configurable joint actuators, and one-click generation of sim-ready MJCF files.

[中文文档](./README_zh.md)

<img src="assets/family.png" alt="piper" style="width: 95%;" />

## 🚀 Installation

### Install from PyPI

```bash
pip install urdf-to-mjcf
```

Or with `uv`:

```bash
uv pip install urdf-to-mjcf
```

### Install from Source

```bash
git clone https://github.com/discoverse-dev/urdf-to-mjcf.git
cd urdf-to-mjcf
uv pip install -e .
```

## 📖 Usage

### Basic Conversion

```bash
cd /path/to/your/robot-description/
urdf-to-mjcf input.urdf --output mjcf/output.xml
# Note: Do not place the generated .xml file in the same directory as the urdf, you can add mjcf/output.xml as shown above
```

### Command Line Arguments

```bash
urdf-to-mjcf <urdf_path> [options]
```

#### Required Arguments
- `urdf_path`: Path to the input URDF file

#### Optional Arguments
- `-o, --output`: Output MJCF file path (default: `output_mjcf/robot.xml` under the URDF directory)
- `-m, --metadata`: Advanced conversion metadata JSON (floor, cameras, collision processing, etc.)
- `--freejoint, --no-freejoint`: Override whether the root body receives a MuJoCo free joint
- `--add-floor, --no-add-floor`: Override whether the default floor is generated
- `-jd, --joint-data`: Unified joint-data JSON files for grouped MJCF-only joints, per-joint dynamics, actuators, and joint-velocity sensors; files are merged in order
- `-a, --appendix`: Appendix XML files, multiple files can be specified and applied in order
- `--collision-only`: Use collision geometry only without visual appearance for visual representation
- `-ct, --collision-type`: Collision mesh processing mode. Choices:
  - `mesh`: keep the original collision mesh.
  - `decomposition`: split collision meshes into convex parts.
  - `convex_hull`: replace collision meshes with convex hulls.
  This option is ignored when `--skip-mesh-postprocess` is set.
- `--log-level`: Logging level (default: INFO level)
- `--max-vertices`: Maximum number of vertices in the mesh (default: 200000)
- `--skip-mesh-postprocess`: Skip heavy mesh-file post-processing and keep only lightweight XML-side postprocess steps

#### Metadata Files Description
- **metadata**: Main conversion configuration file, containing height offset, angle units, floor, collision, and scene settings
- **joint-data**: The single joint configuration format, combining grouped MJCF-only joints with per-joint dynamics, actuator, and joint-velocity sensor settings
- **appendix**: Additional XML content that will be directly added to the generated MJCF file

See the [Metadata Reference](./docs/en/METADATA_REFERENCE.md) for the joint-data schema, defaults, and merge rules.

### Usage Examples

```bash
# agilex-piper robot
cd examples/agilex-piper
urdf-to-mjcf piper.urdf \
  -o mjcf/piper.xml \
  -m metadata/metadata.json \
  -jd metadata/joint_data.json \
  -a metadata/appendix.xml
# View the generated model
python -m mujoco.viewer --mjcf=mjcf/piper.xml

# unified joint data and a fixed base without the generated floor
urdf-to-mjcf robot.urdf \
  -o mjcf/robot.xml \
  -jd metadata/base_joints.json metadata/arm_joints.json \
  --no-freejoint \
  --no-add-floor

# realman-rm65 robotic arm
cd examples/realman-rm65
urdf-to-mjcf rm65b_eg24c2_description.urdf \
  -o mjcf/rm65.xml \
  -m metadata/metadata.json \
  -jd metadata/joint_data.json \
  -a metadata/appendix.xml
# View the generated model
python -m mujoco.viewer --mjcf=mjcf/rm65.xml
```

### Environment Variables - URDF2MJCF_MODEL_PATH

You can set the `URDF2MJCF_MODEL_PATH` environment variable to specify additional search paths for ROS packages and mesh files. This is particularly useful when your robot description packages are not in standard ROS workspace locations.

#### Manual Setup

**Format:**
- **Linux/Mac**: Colon-separated paths (`:`)
- **Windows**: Semicolon-separated paths (`;`)

```bash
# Linux/Mac
export URDF2MJCF_MODEL_PATH="/path/to/robot1_description:/path/to/robot2_description:/path/to/models"

# Windows
set URDF2MJCF_MODEL_PATH="C:\path\to\robot1_description;C:\path\to\robot2_description"
```

#### Model Path Manager Tool

We provide a convenient command-line tool to manage the `URDF2MJCF_MODEL_PATH` environment variable:

```bash
# Scan a workspace for ROS description packages and generate export command
urdf-to-mjcf-modelpath scan /path/to/your/workspace

# Scan multiple directories
urdf-to-mjcf-modelpath scan /path/to/workspace1 /path/to/workspace2

# List current paths in the environment variable
urdf-to-mjcf-modelpath list

# Generate command to unset the environment variable
urdf-to-mjcf-modelpath unset
```

**Features:**
- Recursively searches for packages ending with `_description`
- Verifies packages contain `package.xml` and typical robot folders (urdf, meshes, etc.)
- Generates the appropriate export command for your shell
- Shows which paths are new vs. existing
- Provides instructions to make changes permanent

**Example Output:**
```
======================================================================
✅ Total 2 path(s) in URDF2MJCF_MODEL_PATH:
======================================================================
🆕 1. /workspace/src/robot1_description
🆕 2. /workspace/src/robot2_description

======================================================================
📝 To apply these changes, run the following command:
======================================================================

export URDF2MJCF_MODEL_PATH="/workspace/src/robot1_description:/workspace/src/robot2_description"
```

These paths will be searched when resolving `package://` URIs and locating mesh files.

## 📚 Additional Docs

- [Architecture](./docs/en/ARCHITECTURE.md)
- [Metadata Reference](./docs/en/METADATA_REFERENCE.md)
- [Example Walkthrough](./docs/en/EXAMPLES.md)
- [Troubleshooting](./docs/en/TROUBLESHOOTING.md)
- [Contributing](./CONTRIBUTING.md)

## 🤝 Acknowledgments

This project builds upon these excellent open-source projects:

- **[kscalelabs/urdf2mjcf](https://github.com/kscalelabs/urdf2mjcf)**: Core conversion framework
- **[kevinzakka/obj2mjcf](https://github.com/kevinzakka/obj2mjcf)**: OBJ file processing inspiration

Thanks to the original authors for their outstanding contributions!

## 📄 License

MIT. See [LICENSE](./LICENSE).

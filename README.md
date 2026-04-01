# robot2mjcf

URDF to MJCF conversion tool with support for STL, OBJ, DAE formats, automatic recognition of mimic tags in URDF, configurable joint actuators, and one-click generation of sim-ready MJCF files.

[中文文档](./README_zh.md)

<img src="assets/family.png" alt="piper" style="width: 95%;" />

## 🚀 Installation

### Installation Steps

```bash
git clone https://github.com/discoverse-dev/robot2mjcf.git
cd robot2mjcf
uv pip install -e .
```

Or install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ robot2mjcf
```

## 📖 Usage

### Basic Conversion

```bash
cd /path/to/your/robot-description/
robot2mjcf input.urdf --output mjcf/output.xml
# Note: Do not place the generated .xml file in the same directory as the urdf, you can add mjcf/output.xml as shown above
```

### Command Line Arguments

```bash
robot2mjcf <urdf_path> [options]
```

#### Required Arguments
- `urdf_path`: Path to the input URDF file

#### Optional Arguments
- `-o, --output`: Output MJCF file path (default: `output_mjcf/robot.xml` under the URDF directory)
- `-m, --metadata`: Path to JSON file containing conversion metadata (joint parameters and sensor configurations)
- `-dm, --default-metadata`: Default metadata JSON files, multiple files can be specified, later files override earlier settings
- `-am, --actuator-metadata`: Actuator metadata JSON files, multiple files can be specified, later files override earlier settings
- `-a, --appendix`: Appendix XML files, multiple files can be specified and applied in order
- `--collision-only`: Use collision geometry only without visual appearance for visual representation
- `--collision-type`: Collision type(mesh, convex decomposition, convex hull)
- `--log-level`: Logging level (default: INFO level)
- `--max-vertices`: Maximum number of vertices in the mesh (default: 200000)

#### Metadata Files Description
- **metadata**: Main conversion configuration file, contains height offset, angle units, whether to add floor, etc.
- **default-metadata**: Default joint parameter configuration, defines default properties for joints
- **actuator-metadata**: Actuator configuration, defines actuator types and parameters for each joint
- **appendix**: Additional XML content that will be directly added to the generated MJCF file

### Usage Examples

```bash
# agilex-piper robot
cd examples/agilex-piper
robot2mjcf piper.urdf \
  -o mjcf/piper.xml \
  -m metadata/metadata.json \
  -am metadata/actuator.json \
  -dm metadata/default.json \
  -a metadata/appendix.xml
# View the generated model
python -m mujoco.viewer --mjcf=mjcf/piper.xml

# realman-rm65 robotic arm
cd examples/realman-rm65
robot2mjcf rm65b_eg24c2_description.urdf \
  -o mjcf/rm65.xml \
  -m metadata/metadata.json \
  -am metadata/actuator.json \
  -dm metadata/default.json \
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
robot2mjcf-modelpath scan /path/to/your/workspace

# Scan multiple directories
robot2mjcf-modelpath scan /path/to/workspace1 /path/to/workspace2

# List current paths in the environment variable
robot2mjcf-modelpath list

# Generate command to unset the environment variable
robot2mjcf-modelpath unset
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

- [Architecture](./docs/ARCHITECTURE.md)
- [Metadata Reference](./docs/METADATA_REFERENCE.md)
- [Example Walkthrough](./docs/EXAMPLES.md)
- [Troubleshooting](./docs/TROUBLESHOOTING.md)
- [Contributing](./CONTRIBUTING.md)
- [Changelog](./CHANGELOG.md)

## 🤝 Acknowledgments

This project builds upon these excellent open-source projects:

- **[kscalelabs/urdf2mjcf](https://github.com/kscalelabs/urdf2mjcf)**: Core conversion framework
- **[kevinzakka/obj2mjcf](https://github.com/kevinzakka/obj2mjcf)**: OBJ file processing inspiration

Thanks to the original authors for their outstanding contributions!

## 📄 License

MIT. See [LICENSE](./LICENSE).

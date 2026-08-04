# Metadata Reference

The converter accepts four JSON inputs:

- `metadata.json` through `--metadata` for scene and conversion settings.
- `joint-data.json` through `--joint-data` for unified per-joint configuration.
- `default.json` through `--default-metadata` for legacy MJCF classes.
- `actuator.json` through `--actuator-metadata` for legacy per-joint actuators.

The exact schemas live in `src/urdf_to_mjcf/core/model.py`.

## Conversion Metadata

Top-level model: `ConversionMetadata`.

| Field | Type | Default | Description |
|---|---|---|---|
| `freejoint` | bool | `true` | Add a free joint to the root body |
| `collision_params` | object | see model | Collision parameters such as `contype` and `conaffinity` |
| `imus` | list | `[]` | IMU sensor definitions |
| `cameras` | list | 2 defaults | Camera definitions |
| `sites` | list | `[]` | Site definitions |
| `force_sensors` | list | `[]` | Force sensor definitions |
| `touch_sensors` | list | `[]` | Touch sensor definitions |
| `collision_geometries` | list\|null | `null` | Custom collision geometry replacements |
| `explicit_contacts` | object\|null | `null` | Explicit floor contact configuration |
| `weld_constraints` | list | `[]` | Body weld constraints |
| `remove_redundancies` | bool | `true` | Remove redundant generated elements |
| `maxhullvert` | int\|null | `null` | Maximum collision-hull vertex count |
| `angle` | string | `"radian"` | `"radian"` or `"degree"` |
| `floor_name` | string | `"floor"` | Floor body name |
| `add_floor` | bool | `true` | Add the default floor plane |
| `backlash` | float\|null | `null` | Joint backlash value |
| `backlash_damping` | float | `0.01` | Backlash damping coefficient |
| `height_offset` | float | `0.0` | Additional robot height offset |

`--freejoint` / `--no-freejoint` and `--add-floor` / `--no-add-floor` override the corresponding JSON values when supplied.

Sensor records:

- `ImuSensor`: `body_name`, `pos`, `rpy`, `acc_noise`, `gyro_noise`, `mag_noise`
- `CameraSensor`: `name`, `mode`, `pos`, `rpy`, `fovy`
- `SiteMetadata`: `name`, `body_name`, `site_type`, `size`, `pos`
- `ForceSensor`: `body_name`, `site_name`, `name`, `noise`
- `TouchSensor`: `body_name`, `site_name`, `name`, `noise`

## Unified Joint Data

`--joint-data` is the preferred input for MJCF-only joints and joint-specific dynamics, actuators, and velocity sensors.

```json
{
  "extra_joints": [
    {
      "body": "base_link",
      "joints": [
        {
          "name": "base_x",
          "type": "slide",
          "axis": "x",
          "range": [-1000, 1000]
        },
        {
          "name": "base_yaw",
          "type": "hinge",
          "axis": "z"
        }
      ]
    }
  ],
  "joints": {
    "base_x": {
      "armature": 0.001,
      "damping": 0.1,
      "actuator": {
        "actuator_type": "position",
        "ctrllimited": true,
        "ctrlrange": [-2.0, 2.0],
        "kp": 100.0,
        "forcelimited": true,
        "forcerange": [-200.0, 200.0]
      },
      "sensors": {
        "jointvel": true
      }
    }
  }
}
```

### Extra-joint groups

Each item in `extra_joints` contains:

- `body`: generated MJCF body that receives the joints.
- `joints`: ordered list of joints inserted into that body.

Each joint contains `name`, `type` (`"slide"` or `"hinge"`), `axis`, and optional `range`. Axis accepts `x`, `y`, `z`, `-x`, `-y`, or `-z`.

### Per-joint records

The `joints` object maps a URDF or MJCF-only joint name to:

- Joint fields: `stiffness`, `actuatorfrcrange`, `margin`, `armature`, `damping`, `frictionloss`.
- Optional `actuator`: `actuator_type`, `ctrllimited`, `kp`, `kv`, `gear`, `ctrlrange`, `forcelimited`, `forcerange`.
- Optional `sensors`: set `jointvel` to `true` to emit a `<jointvel>` sensor named `vel_<joint-name>`.

Only actuator records with a non-null `actuator_type` generate an actuator.

## Legacy Default Metadata

`--default-metadata` maps MJCF class names to joint and actuator defaults:

```json
{
  "class_name": {
    "joint": {"damping": 0.1, "armature": 0.01},
    "actuator": {"actuator_type": "motor", "gear": 1.0}
  }
}
```

## Legacy Actuator Metadata

`--actuator-metadata` maps joint names to actuator records:

```json
{
  "joint_name": {
    "joint_class": "class_name",
    "actuator_type": "position",
    "ctrllimited": true,
    "ctrlrange": [-1.0, 1.0],
    "kp": 100.0
  }
}
```

Fields: `joint_class`, `actuator_type`, `ctrllimited`, `kp`, `kv`, `gear`, `ctrlrange`, `forcelimited`, `forcerange`.

## Merge and Precedence Rules

- Multiple joint-data files are read in order. Extra-joint groups are appended; a later `joints` entry replaces the complete earlier record with the same name.
- Supplying joint data disables legacy default-class metadata. Per-joint dynamics from joint data are applied directly.
- If `--actuator-metadata` is also supplied, it takes precedence over actuators derived from joint data.
- Multiple default- or actuator-metadata files are read in order, with later entries replacing earlier entries by key.
- `metadata.json` is a single document; the explicit freejoint and floor CLI flags override it.

Example:

```bash
urdf-to-mjcf robot.urdf \
  --metadata metadata.json \
  --joint-data base.json arm.json \
  --no-freejoint \
  --no-add-floor
```

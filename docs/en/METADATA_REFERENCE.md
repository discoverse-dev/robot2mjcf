# Metadata Reference

This project uses three metadata inputs:

- `metadata.json`
- `default.json`
- `actuator.json`

The exact schema lives in `core/model.py`.

## `metadata.json`

Top-level model: `ConversionMetadata`.

Common fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `freejoint` | bool | `true` | Add a free joint to the root body |
| `collision_params` | object | see below | Collision parameters (contype, conaffinity, etc.) |
| `imus` | list | `[]` | IMU sensor definitions |
| `cameras` | list | 2 defaults | Camera sensor definitions |
| `sites` | list | `[]` | Site definitions |
| `force_sensors` | list | `[]` | Force sensor definitions |
| `touch_sensors` | list | `[]` | Touch sensor definitions |
| `collision_geometries` | list\|null | `null` | Custom collision geometry replacements |
| `explicit_contacts` | object\|null | `null` | Explicit floor contact configuration |
| `extra_joints` | list | `[]` | MJCF-only joints inserted into generated bodies |
| `weld_constraints` | list | `[]` | Weld constraints (body locking) |
| `remove_redundancies` | bool | `true` | Remove redundant elements |
| `maxhullvert` | int\|null | `null` | Max hull vertices for collision |
| `angle` | string | `"radian"` | Angle unit (`"radian"` or `"degree"`) |
| `floor_name` | string | `"floor"` | Floor body name |
| `add_floor` | bool | `true` | Add a floor plane |
| `backlash` | float\|null | `null` | Joint backlash value |
| `backlash_damping` | float | `0.01` | Backlash damping coefficient |
| `height_offset` | float | `0.0` | Additional height offset |

### Sensor Records

- `ImuSensor`: `body_name`, `pos`, `rpy`, `acc_noise`, `gyro_noise`, `mag_noise`
- `CameraSensor`: `name`, `mode`, `pos`, `rpy`, `fovy`
- `SiteMetadata`: `name`, `body_name`, `site_type`, `size`, `pos`
- `ForceSensor`: `body_name`, `site_name`, `name`, `noise`
- `TouchSensor`: `body_name`, `site_name`, `name`, `noise`

### Collision Replacement Records

- `CollisionGeometry`: `name`, `collision_type`, `sphere_radius`, `axis_order`, `flip_axis`, `offset_x`, `offset_y`, `offset_z`

### Extra Joint Records

`extra_joints` inserts MJCF joints that do not exist in the source URDF. Use `actuator.json` to add actuators for these joints by name.

```json
{
  "extra_joints": [
    {
      "body_name": "base_link",
      "name": "base_x",
      "type": "slide",
      "axis": [1, 0, 0],
      "joint_class": "base_slide",
      "range": [-1000, 1000]
    }
  ]
}
```

Fields: `body_name`, `name`, `type`, `axis`, `joint_class`, `range`.

## `default.json`

Top-level structure maps class names to `DefaultJointMetadata`:

```json
{
  "class_name": {
    "joint": { "damping": 0.1, "armature": 0.01 },
    "actuator": { "actuator_type": "motor", "kp": 100.0 }
  }
}
```

Joint fields: `stiffness`, `actuatorfrcrange`, `margin`, `armature`, `damping`, `frictionloss`.

Actuator fields: `actuator_type`, `kp`, `kv`, `gear`, `ctrlrange`, `forcerange`.

## `actuator.json`

Top-level structure maps joint names to `ActuatorMetadata`:

```json
{
  "joint_name": {
    "joint_class": "class_name",
    "actuator_type": "motor",
    "kp": 100.0
  }
}
```

Fields: `joint_class`, `actuator_type`, `kp`, `kv`, `gear`, `ctrlrange`, `forcerange`.

## Merge Rules

- Multiple `default.json` files are merged in order (later overrides earlier).
- Multiple `actuator.json` files are merged in order.
- `metadata.json` is a single document.

## Reference Examples

- `examples/agilex-piper/metadata/metadata.json`
- `examples/agilex-piper/metadata/default.json`
- `examples/agilex-piper/metadata/actuator.json`
- `examples/realman-rm65/metadata/metadata.json`

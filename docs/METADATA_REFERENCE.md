# Metadata Reference

This project uses three metadata inputs:

- `metadata.json`
- `default.json`
- `actuator.json`

The exact schema lives in [`model.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/model.py).

## `metadata.json`

Top-level model: `ConversionMetadata`.

Common fields:

- `freejoint`
- `collision_params`
- `imus`
- `cameras`
- `sites`
- `force_sensors`
- `touch_sensors`
- `collision_geometries`
- `explicit_contacts`
- `weld_constraints`
- `remove_redundancies`
- `maxhullvert`
- `angle`
- `floor_name`
- `add_floor`
- `backlash`
- `backlash_damping`
- `height_offset`

### Sensor Records

- `ImuSensor`: `body_name`, `pos`, `rpy`, `acc_noise`, `gyro_noise`, `mag_noise`
- `CameraSensor`: `name`, `mode`, `pos`, `rpy`, `fovy`
- `SiteMetadata`: `name`, `body_name`, `site_type`, `size`, `pos`
- `ForceSensor`: `body_name`, `site_name`, `name`, `noise`
- `TouchSensor`: `body_name`, `site_name`, `name`, `noise`

### Collision Replacement Records

- `CollisionGeometry`: `name`, `collision_type`, `sphere_radius`, `axis_order`, `flip_axis`, `offset_x`, `offset_y`, `offset_z`

## `default.json`

Top-level structure maps joint names to `DefaultJointMetadata`:

```json
{
  "joint_name": {
    "joint": {},
    "actuator": {}
  }
}
```

## `actuator.json`

Top-level structure maps joint names to `ActuatorMetadata`:

```json
{
  "joint_name": {
    "joint_class": "actuator",
    "actuator_type": "motor"
  }
}
```

## Merge Rules

- Multiple `default.json` files are merged in order.
- Multiple `actuator.json` files are merged in order.
- `metadata.json` is a single document.

## Reference Examples

- [`examples/agilex-piper/metadata/metadata.json`](/Users/jiayufei/ws/robot2mjcf/examples/agilex-piper/metadata/metadata.json)
- [`examples/agilex-piper/metadata/default.json`](/Users/jiayufei/ws/robot2mjcf/examples/agilex-piper/metadata/default.json)
- [`examples/agilex-piper/metadata/actuator.json`](/Users/jiayufei/ws/robot2mjcf/examples/agilex-piper/metadata/actuator.json)
- [`examples/realman-rm65/metadata/metadata.json`](/Users/jiayufei/ws/robot2mjcf/examples/realman-rm65/metadata/metadata.json)

# 元数据参考

本项目使用三种元数据输入：

- `metadata.json`
- `default.json`
- `actuator.json`

完整数据模型定义在 `core/model.py` 中。

## `metadata.json`

顶层模型：`ConversionMetadata`。

常用字段：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `freejoint` | bool | `true` | 为根 body 添加自由关节 |
| `collision_params` | object | 见下 | 碰撞参数（contype, conaffinity 等） |
| `imus` | list | `[]` | IMU 传感器定义 |
| `cameras` | list | 2 个默认 | 相机传感器定义 |
| `sites` | list | `[]` | 站点定义 |
| `force_sensors` | list | `[]` | 力传感器定义 |
| `touch_sensors` | list | `[]` | 触觉传感器定义 |
| `collision_geometries` | list\|null | `null` | 自定义碰撞几何体替换 |
| `explicit_contacts` | object\|null | `null` | 显式地面接触配置 |
| `extra_joints` | list | `[]` | 插入到已生成 body 中的 MJCF-only 关节 |
| `weld_constraints` | list | `[]` | 焊接约束（body 锁定） |
| `remove_redundancies` | bool | `true` | 移除冗余元素 |
| `maxhullvert` | int\|null | `null` | 碰撞凸包最大顶点数 |
| `angle` | string | `"radian"` | 角度单位（`"radian"` 或 `"degree"`） |
| `floor_name` | string | `"floor"` | 地面 body 名称 |
| `add_floor` | bool | `true` | 添加地面平面 |
| `backlash` | float\|null | `null` | 关节齿隙值 |
| `backlash_damping` | float | `0.01` | 齿隙阻尼系数 |
| `height_offset` | float | `0.0` | 额外高度偏移 |

### 传感器记录

- `ImuSensor`：`body_name`, `pos`, `rpy`, `acc_noise`, `gyro_noise`, `mag_noise`
- `CameraSensor`：`name`, `mode`, `pos`, `rpy`, `fovy`
- `SiteMetadata`：`name`, `body_name`, `site_type`, `size`, `pos`
- `ForceSensor`：`body_name`, `site_name`, `name`, `noise`
- `TouchSensor`：`body_name`, `site_name`, `name`, `noise`

### 碰撞替换记录

- `CollisionGeometry`：`name`, `collision_type`, `sphere_radius`, `axis_order`, `flip_axis`, `offset_x`, `offset_y`, `offset_z`

### 额外关节记录

`extra_joints` 用于插入源 URDF 中不存在的 MJCF 关节。需要驱动这些关节时，在 `actuator.json` 中按关节名添加执行器配置。

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

字段：`body_name`, `name`, `type`, `axis`, `joint_class`, `range`。

## `default.json`

顶层结构将类名映射到 `DefaultJointMetadata`：

```json
{
  "class_name": {
    "joint": { "damping": 0.1, "armature": 0.01 },
    "actuator": { "actuator_type": "motor", "kp": 100.0 }
  }
}
```

关节字段：`stiffness`, `actuatorfrcrange`, `margin`, `armature`, `damping`, `frictionloss`。

执行器字段：`actuator_type`, `kp`, `kv`, `gear`, `ctrlrange`, `forcerange`。

## `actuator.json`

顶层结构将关节名映射到 `ActuatorMetadata`：

```json
{
  "joint_name": {
    "joint_class": "class_name",
    "actuator_type": "motor",
    "kp": 100.0
  }
}
```

字段：`joint_class`, `actuator_type`, `kp`, `kv`, `gear`, `ctrlrange`, `forcerange`。

## 合并规则

- 多个 `default.json` 文件按顺序合并（后者覆盖前者）。
- 多个 `actuator.json` 文件按顺序合并。
- `metadata.json` 是单一文档。

## 参考示例

- `examples/agilex-piper/metadata/metadata.json`
- `examples/agilex-piper/metadata/default.json`
- `examples/agilex-piper/metadata/actuator.json`
- `examples/realman-rm65/metadata/metadata.json`

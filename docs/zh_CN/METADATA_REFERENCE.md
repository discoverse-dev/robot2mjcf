# 元数据参考

转换器支持四类 JSON 输入：

- 通过 `--metadata` 传入 `metadata.json`，配置场景与转换行为。
- 通过 `--joint-data` 传入 `joint-data.json`，统一配置逐关节数据。
- 通过 `--default-metadata` 传入 `default.json`，配置旧版 MJCF class 默认值。
- 通过 `--actuator-metadata` 传入 `actuator.json`，配置旧版逐关节执行器。

完整数据模型定义在 `src/urdf_to_mjcf/core/model.py` 中。

## 转换元数据

顶层模型：`ConversionMetadata`。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `freejoint` | bool | `true` | 为根 body 添加自由关节 |
| `collision_params` | object | 见数据模型 | `contype`、`conaffinity` 等碰撞参数 |
| `imus` | list | `[]` | IMU 传感器定义 |
| `cameras` | list | 2 个默认值 | 相机定义 |
| `sites` | list | `[]` | Site 定义 |
| `force_sensors` | list | `[]` | 力传感器定义 |
| `touch_sensors` | list | `[]` | 触觉传感器定义 |
| `collision_geometries` | list\|null | `null` | 自定义碰撞几何体替换 |
| `explicit_contacts` | object\|null | `null` | 显式地面接触配置 |
| `weld_constraints` | list | `[]` | Body 焊接约束 |
| `remove_redundancies` | bool | `true` | 移除冗余的生成元素 |
| `maxhullvert` | int\|null | `null` | 碰撞凸包最大顶点数 |
| `angle` | string | `"radian"` | `"radian"` 或 `"degree"` |
| `floor_name` | string | `"floor"` | 地面 body 名称 |
| `add_floor` | bool | `true` | 添加默认地面平面 |
| `backlash` | float\|null | `null` | 关节齿隙值 |
| `backlash_damping` | float | `0.01` | 齿隙阻尼系数 |
| `height_offset` | float | `0.0` | 机器人额外高度偏移 |

显式提供 `--freejoint` / `--no-freejoint` 或 `--add-floor` / `--no-add-floor` 时，命令行值会覆盖 JSON 中的对应设置。

传感器记录：

- `ImuSensor`：`body_name`, `pos`, `rpy`, `acc_noise`, `gyro_noise`, `mag_noise`
- `CameraSensor`：`name`, `mode`, `pos`, `rpy`, `fovy`
- `SiteMetadata`：`name`, `body_name`, `site_type`, `size`, `pos`
- `ForceSensor`：`body_name`, `site_name`, `name`, `noise`
- `TouchSensor`：`body_name`, `site_name`, `name`, `noise`

## 统一关节数据

`--joint-data` 是配置 MJCF-only 关节、逐关节动力学、执行器和速度传感器的推荐入口。

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

### 额外关节分组

`extra_joints` 中的每一项包含：

- `body`：接收关节的已生成 MJCF body。
- `joints`：按顺序插入该 body 的关节列表。

每个关节包含 `name`、`type`（`"slide"` 或 `"hinge"`）、`axis` 和可选的 `range`。`axis` 可取 `x`、`y`、`z`、`-x`、`-y` 或 `-z`。

### 逐关节记录

`joints` 对象将 URDF 或 MJCF-only 关节名映射到：

- 关节字段：`stiffness`, `actuatorfrcrange`, `margin`, `armature`, `damping`, `frictionloss`。
- 可选 `actuator`：`actuator_type`, `ctrllimited`, `kp`, `kv`, `gear`, `ctrlrange`, `forcelimited`, `forcerange`。
- 可选 `sensors`：将 `jointvel` 设为 `true`，生成名为 `vel_<joint-name>` 的 `<jointvel>` 传感器。

只有 `actuator_type` 非空的执行器记录才会生成执行器。

## 旧版默认元数据

`--default-metadata` 将 MJCF class 名称映射到关节和执行器默认值：

```json
{
  "class_name": {
    "joint": {"damping": 0.1, "armature": 0.01},
    "actuator": {"actuator_type": "motor", "gear": 1.0}
  }
}
```

## 旧版执行器元数据

`--actuator-metadata` 将关节名映射到执行器记录：

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

字段：`joint_class`, `actuator_type`, `ctrllimited`, `kp`, `kv`, `gear`, `ctrlrange`, `forcelimited`, `forcerange`。

## 合并与优先级规则

- 多个 joint-data 文件按顺序读取：额外关节分组会追加；后续 `joints` 中的同名条目会完整替换之前的记录。
- 提供 joint data 后，不再应用旧版 default class 元数据；逐关节动力学会直接写入对应关节。
- 同时提供 `--actuator-metadata` 时，它优先于从 joint data 推导出的执行器。
- 多个 default 或 actuator metadata 文件按顺序读取，后续同名条目覆盖之前的条目。
- `metadata.json` 是单个文档；显式的自由关节和地面命令行参数会覆盖其中的值。

示例：

```bash
urdf-to-mjcf robot.urdf \
  --metadata metadata.json \
  --joint-data base.json arm.json \
  --no-freejoint \
  --no-add-floor
```

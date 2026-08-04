# 架构说明

## 项目定位

`urdf-to-mjcf` 将 URDF 机器人描述文件转换为 MuJoCo MJCF 格式，并可选地执行 XML/网格后处理步骤，使结果更易于仿真。

## 包结构

```
urdf-to-mjcf/
├── __init__.py              # 公共 API 导出
├── urdf_format.py           # 独立的 URDF 格式化工具
├── cli/                     # CLI 入口
│   ├── convert.py           # 主命令 (urdf-to-mjcf)
│   ├── mjcf2obj.py          # Body 网格导出器 (urdf-to-mjcf-mjcf2obj)
│   └── model_path.py        # 模型路径管理器 (urdf-to-mjcf-modelpath)
├── core/                    # 数据模型与基础工具
│   ├── model.py             # Pydantic 元数据模型
│   ├── geometry.py          # 几何数学（四元数、变换）
│   ├── materials.py         # MTL/材质解析
│   ├── package_resolver.py  # ROS package:// 路径解析
│   └── utils.py             # XML 保存/排序工具
├── conversion/              # 转换流水线
│   ├── pipeline.py          # 上下文构建与场景组装
│   ├── input.py             # URDF 解析、元数据加载、辅助函数
│   ├── output.py            # 高度调整与后处理分发
│   ├── body_builder.py      # MJCF body 树构建
│   ├── mjcf_assembly.py     # MJCF XML 构建器、执行器、约束
│   └── assets.py            # 网格资源解析与复制
└── postprocess/             # 后处理阶段
    ├── __init__.py           # 流水线调度器
    ├── add_light.py          # 添加光源
    ├── add_floor.py          # 添加地面
    ├── split_obj_materials.py  # DAE/GLB→OBJ 转换、OBJ 材质拆分
    ├── update_mesh.py        # 网格简化与清理
    ├── mesh_converter.py     # DAE→OBJ 和 GLB→OBJ 转换器
    └── ...                   # 其他 XML/网格变换
```

## 主要流程

1. **输入** (`conversion/input.py`)：解析 URDF，加载元数据，收集材质。
2. **上下文** (`conversion/pipeline.py`)：构建 MJCF 根节点、编译器、默认值、worldbody；解析关节图。
3. **场景组装** (`conversion/pipeline.py`)：构建 body 树，复制网格资源，添加执行器/约束。
4. **输出** (`conversion/output.py`)：调整机器人高度，保存初始 MJCF，分发后处理。
5. **后处理** (`postprocess/`)：光源、网格转换（DAE/GLB→OBJ）、材质拆分、简化、地面、传感器等。

## 元数据流

CLI 明确区分两条配置路径：

- `ConversionMetadata` 负责场景级转换设置。
- `JointData` 负责分组的 MJCF-only 关节，以及逐关节动力学、执行器和速度传感器。

`JointData` 是 CLI seam 上唯一的关节配置 interface。`conversion/pipeline.py` 只解析一次并写入 `ConversionContext`；Body 构建和 MJCF 组装消费同一对象，不再维护并行的 joint/default/actuator 映射。未提供 joint data 时，resolver 会为 URDF 中的可动关节生成默认 motor 记录；精确行为见[元数据参考](./METADATA_REFERENCE.md)。

## 分层依赖

```
Layer 0 — 纯基础层（无包内依赖）：
  core/model, core/geometry, core/materials, core/utils, core/package_resolver

Layer 1 — 简单消费者：
  conversion/input, conversion/mjcf_assembly, conversion/body_builder, conversion/assets

Layer 2 — 编排层：
  conversion/pipeline（使用 input, mjcf_assembly, body_builder, assets）
  conversion/output（使用 core/geometry, core/utils, postprocess）

Layer 3 — 入口层：
  cli/convert（使用 conversion/*, core/model）
```

无循环导入。依赖图为严格的有向无环图（DAG）。

## 后处理分类

- **纯 XML 变换**：光源、地面、角度转换、齿隙、基关节、冗余移除、接触。
- **XML + 文件系统变换**：网格转换（DAE/GLB→OBJ）、材质拆分、网格简化、凸分解、去重。

第二类更重、更慢，可通过 `--skip-mesh-postprocess` 跳过。

## 建议的扩展点

- 在 `core/model.py` 中添加新的元数据模型。
- 在 `conversion/input.py` 中添加纯辅助函数。
- 在 `postprocess/` 下添加新的后处理阶段，并在 `postprocess/__init__.py` 中注册。
- 在输出语义发生变化时，针对真实示例添加回归测试。

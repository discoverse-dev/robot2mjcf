# urdf-to-mjcf

URDF到MJCF转换工具，支持STL、OBJ、DAE、GLB格式，自动识别urdf中的mimic标签，可配置关节驱动器，一键生成sim-ready的mjcf文件。

[English Document](./README.md)

<img src="assets/family.png" alt="piper" style="width: 95%;" />

## 🚀 安装

### 从 PyPI 安装

```bash
pip install urdf-to-mjcf
```

或者使用 `uv`：

```bash
uv pip install urdf-to-mjcf
```

### 从源码安装

```bash
git clone https://github.com/discoverse-dev/urdf-to-mjcf.git
cd urdf-to-mjcf
uv pip install -e .
```

## 📖 使用方法

### 基本转换

```bash
cd /path/to/your/robot-description/
urdf-to-mjcf input.urdf --output mjcf/output.xml
# 注意不要将生成的.xml放在和urdf同一目录下，可以向上面一样添加 mjcf/output.xml
```

### 命令行参数说明

```bash
urdf-to-mjcf <urdf_path> [options]
```

#### 必需参数
- `urdf_path`: 输入的URDF文件路径

#### 可选参数
- `-o, --output`: 输出MJCF文件路径 (默认: URDF 所在目录下的 `output_mjcf/robot.xml`)
- `-m, --metadata`: 高级转换元数据 JSON（地面、相机、碰撞处理等）
- `--freejoint, --no-freejoint`: 覆盖是否为根 body 添加 MuJoCo 自由关节
- `--add-floor, --no-add-floor`: 覆盖是否生成默认地面
- `-dm, --default-metadata`: 默认元数据JSON文件，可指定多个文件，后面的文件会覆盖前面的设置
- `-am, --actuator-metadata`: 执行器元数据JSON文件，可指定多个文件，后面的文件会覆盖前面的设置
- `-jd, --joint-data`: 统一关节数据 JSON，可配置分组的 MJCF-only 关节、逐关节动力学、执行器和关节速度传感器；多个文件按顺序合并
- `-a, --appendix`: 附加XML文件，可指定多个文件，按顺序应用
- `--collision-only`: 仅使用碰撞几何体而不显示视觉外观
- `-ct, --collision-type`: 碰撞网格处理模式。可选值：
  - `mesh`: 保留原始碰撞网格。
  - `decomposition`: 将碰撞网格分解为多个凸部件。
  - `convex_hull`: 使用凸包替换碰撞网格。
  设置 `--skip-mesh-postprocess` 时会忽略该选项。
- `--log-level`: 日志级别 (默认: INFO级别)
- `--max-vertices`: 网格中的最大顶点数量 (默认: 200000)
- `--skip-mesh-postprocess`: 跳过重型 mesh 文件后处理，仅保留轻量 XML 侧后处理步骤

#### 元数据文件说明
- **metadata**: 主要转换配置文件，包含高度偏移、角度单位、地面、碰撞和场景设置
- **joint-data**: 推荐的逐关节配置，将 MJCF-only 关节分组与动力学、执行器和关节速度传感器配置统一在一个结构中
- **default-metadata**: 默认关节参数配置，定义关节的默认属性
- **actuator-metadata**: 执行器配置，定义每个关节的驱动器类型和参数
- **appendix**: 附加的XML内容，会被直接添加到生成的MJCF文件中

完整的 joint-data 结构、合并规则和新旧配置优先级请参阅[元数据字段参考](./docs/zh_CN/METADATA_REFERENCE.md)。

### 使用示例

```bash
# agilex-piper机器人
cd examples/agilex-piper
urdf-to-mjcf piper.urdf \
  -o mjcf/piper.xml \
  -m metadata/metadata.json \
  -am metadata/actuator.json \
  -dm metadata/default.json \
  -a metadata/appendix.xml
# 查看生成的模型
python -m mujoco.viewer --mjcf=mjcf/piper.xml

# 使用统一关节数据生成无默认地面的固定基座模型
urdf-to-mjcf robot.urdf \
  -o mjcf/robot.xml \
  -jd metadata/base_joints.json metadata/arm_joints.json \
  --no-freejoint \
  --no-add-floor

# realman-rm65机械臂
cd examples/realman-rm65
urdf-to-mjcf rm65b_eg24c2_description.urdf \
  -o mjcf/rm65.xml \
  -m metadata/metadata.json \
  -am metadata/actuator.json \
  -dm metadata/default.json \
  -a metadata/appendix.xml
# 查看生成的模型
python -m mujoco.viewer --mjcf=mjcf/rm65.xml
```

### 环境变量 - URDF2MJCF_MODEL_PATH

你可以设置 `URDF2MJCF_MODEL_PATH` 环境变量来指定额外的ROS包和网格文件搜索路径。这在你的机器人描述包不在标准ROS工作空间位置时特别有用。

#### 手动设置

**格式:**
- **Linux/Mac**: 冒号分隔的路径 (`:`)
- **Windows**: 分号分隔的路径 (`;`)

```bash
# Linux/Mac
export URDF2MJCF_MODEL_PATH="/path/to/robot1_description:/path/to/robot2_description:/path/to/models"

# Windows
set URDF2MJCF_MODEL_PATH="C:\path\to\robot1_description;C:\path\to\robot2_description"
```

#### 模型路径管理工具

我们提供了一个便捷的命令行工具来管理 `URDF2MJCF_MODEL_PATH` 环境变量:

```bash
# 扫描工作空间中的ROS描述包并生成导出命令
urdf-to-mjcf-modelpath scan /path/to/your/workspace

# 扫描多个目录
urdf-to-mjcf-modelpath scan /path/to/workspace1 /path/to/workspace2

# 列出环境变量中的当前路径
urdf-to-mjcf-modelpath list

# 生成取消设置环境变量的命令
urdf-to-mjcf-modelpath unset
```

**功能特点:**
- 递归搜索以 `_description` 结尾的包
- 验证包含 `package.xml` 和典型的机器人文件夹 (urdf, meshes 等)
- 为你的shell生成适当的导出命令
- 显示哪些路径是新的，哪些是已存在的
- 提供使更改永久生效的说明

**示例输出:**
```
======================================================================
✅ 总计 2 个路径在 URDF2MJCF_MODEL_PATH 中:
======================================================================
🆕 1. /workspace/src/robot1_description
🆕 2. /workspace/src/robot2_description

======================================================================
📝 要应用这些更改，请运行以下命令:
======================================================================

export URDF2MJCF_MODEL_PATH="/workspace/src/robot1_description:/workspace/src/robot2_description"
```

这些路径将在解析 `package://` URI 和定位网格文件时被搜索。

## 📚 补充文档

- [架构说明](./docs/zh_CN/ARCHITECTURE.md)
- [元数据字段参考](./docs/zh_CN/METADATA_REFERENCE.md)
- [示例教程](./docs/zh_CN/EXAMPLES.md)
- [常见问题排查](./docs/zh_CN/TROUBLESHOOTING.md)
- [贡献指南](./CONTRIBUTING.md)

## 🤝 致谢

本项目基于以下优秀开源项目：

- **[kscalelabs/urdf2mjcf](https://github.com/kscalelabs/urdf2mjcf)**：核心转换框架
- **[kevinzakka/obj2mjcf](https://github.com/kevinzakka/obj2mjcf)**：OBJ文件处理灵感

感谢原作者们的杰出贡献！

## 📄 许可证

MIT，见 [LICENSE](./LICENSE)。

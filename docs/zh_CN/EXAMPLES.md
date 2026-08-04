# 示例教程

本仓库包含两个真实机器人示例：

- `examples/agilex-piper`
- `examples/realman-rm65`

## 快速开始

### agilex-piper

```bash
cd examples/agilex-piper
urdf-to-mjcf piper.urdf \
  -o output_mjcf/piper.xml \
  -m metadata/metadata.json \
  -jd metadata/joint_data.json \
  -a metadata/appendix.xml

# 查看生成的模型
python -m mujoco.viewer --mjcf=output_mjcf/piper.xml
```

还有一个带相机支架的变体：

```bash
urdf-to-mjcf piper_with_camera.urdf
```

该变体包含 `.glb` 格式的网格，后处理阶段会自动转换为 OBJ。

### realman-rm65

```bash
cd examples/realman-rm65
urdf-to-mjcf rm65b_eg24c2_description.urdf \
  -o output_mjcf/rm65.xml \
  -m metadata/metadata.json \
  -jd metadata/joint_data.json \
  -a metadata/appendix.xml

# 查看生成的模型
python -m mujoco.viewer --mjcf=output_mjcf/rm65.xml
```

## 支持的网格格式

| 格式 | 视觉 | 碰撞 | 备注 |
|---|---|---|---|
| STL | 支持 | 支持 | 直接加载 |
| OBJ | 支持 | 支持 | 支持材质拆分 |
| DAE | 支持 | - | 后处理阶段转换为 OBJ |
| GLB/GLTF | 支持 | - | 后处理阶段转换为 OBJ |

## 回归测试

仓库中 `tests/test_convert.py` 的回归测试会对比真实示例的语义输出：

- 模型名称
- Body 名称
- 关节名称
- 执行器名称
- 等式约束对
- XML 元素计数
- 引用的网格文件存在性
- MuJoCo 可加载性和模型计数

这是重构时的首选安全网。

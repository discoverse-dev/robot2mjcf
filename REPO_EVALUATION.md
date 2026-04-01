# robot2mjcf 仓库系统性评估

评估时间：2026-04-01

评估方式：
- 通读仓库结构、核心源码、测试、CI/CD、README、示例与辅助脚本
- 本地验证 `uv run pytest`、`uv run ruff check src/robot2mjcf tests`、`uv run mypy src/robot2mjcf tests`、`uv build`
- 额外验证 CI 工作流里实际写入的命令是否可执行

## 一句话结论

这个仓库已经明显超过“纯原型”阶段，核心转换链条能在真实样例上跑通，打包也能成功；但它还不是生产级开源工程。更准确地说，它是一个**研究/开发场景可用的 beta 级工具仓库**：核心能力可用，工程化外观基本具备，但自动化、架构边界、测试覆盖、类型约束、文档一致性和跨平台验证都还不够硬。

如果必须分级：
- 核心转换能力：开发级
- 部分辅助脚本和后处理工具：原型级到开发级之间
- 整体仓库成熟度：开发级 beta，未达生产级

## 已验证的事实

- `uv run pytest` 通过，`21` 个测试全部通过。
- 测试覆盖率总计 `45%`。
- `uv build` 成功，能生成 sdist 和 wheel。
- `uv run ruff check src/robot2mjcf tests` 失败，当前至少存在 1 个真实 lint 问题：`tests/test_convert.py` 的 import 顺序。
- `uv run mypy src/robot2mjcf tests` 表面通过，但 `pyproject.toml` 对 `robot2mjcf.*` 设置了 `ignore_errors = true`，所以这个“通过”并不代表主代码真的被有效类型检查。
- CI 工作流里的命令目前与仓库结构不一致，按工作流原样执行会失败：
  - `uv run ruff check robot2mjcf tests` 报 `No such file or directory`
  - `uv run mypy robot2mjcf tests` 报 `cannot read file 'robot2mjcf'`

## 分项评级

| 维度 | 评级 | 结论 |
| --- | --- | --- |
| 架构设计 | B- | 有明确 `src/` 包结构和模块分层意识，但主流程过于集中，后处理链耦合偏高 |
| 代码质量 | C+ | 可读性中等，部分模块不错，但风格不统一，脚本化痕迹重，存在明显工程边界问题 |
| 自动化 CI/CD | D+ | 有 CI 和发布流程，但 CI 命令当前就是错的，类型检查形同虚设 |
| 测试覆盖 | C | 有真实端到端样例，价值高；但覆盖率只有 45%，大量关键模块几乎未测 |
| 文档 | C | README 双语、示例直观，但文档与代码不一致，缺少更系统的工程文档 |
| 使用案例 | B | 两个真实机器人案例有说服力，对研究/开发用户有实际价值 |
| 跨平台兼容 | C- | 有跨平台意图，但只有 Linux/macOS CI，没有 Windows 验证，依赖链也偏重 |
| 生产可用性 | C- | 对内部研究或开发团队可用，对外部生产用户还不够稳 |

## 架构评估

### 优点

- 仓库使用了标准 `src/` 布局，打包结构是对的，见 `pyproject.toml` 和 `src/robot2mjcf/`。
- `model.py` 把元数据配置整理成了 Pydantic 模型，至少在配置入口处建立了结构化边界。
- 后处理功能被拆到 `postprocess/`，说明作者意识到“转换”和“后修整”是不同职责。
- 示例资产完整，测试直接用真实 URDF 和网格跑端到端，不是空洞的 mock。

### 主要问题

#### 1. 主流程 `convert.py` 是单体 orchestrator，职责过载

`src/robot2mjcf/convert.py` 从 URDF 解析、材料处理、body 构造、资源复制、mesh 处理、碰撞处理、角度处理、appendix、截图输出，全都集中在一个 1000+ 行文件和一个超长函数里，见：
- `src/robot2mjcf/convert.py:67`
- `src/robot2mjcf/convert.py:255`
- `src/robot2mjcf/convert.py:324`
- `src/robot2mjcf/convert.py:731`
- `src/robot2mjcf/convert.py:850`
- `src/robot2mjcf/convert.py:903`

这会带来几个直接问题：
- 很难单元测试
- 很难局部替换处理步骤
- 很难明确失败点和恢复策略
- 很难把“库接口”和“CLI 副作用”分开

#### 2. 设计本质上还是“脚本管道”，不是稳定的库式架构

现在的主流程是“先生成 XML 文件，再串行调用多个 postprocess 脚本继续原地改文件”。这对快速迭代有效，但不适合作为稳定架构长期维护。典型片段见：
- `src/robot2mjcf/convert.py:850-913`

这意味着：
- 中间状态落盘很多
- 子步骤之间通过文件格式隐式耦合
- 任何一步改了 XML 结构，都可能影响后续多个脚本

这更像“实用工具链组合”，不是“明确边界的转换引擎”。

#### 3. 辅助模块质量差异很大

有些模块相对规整，例如 `mjcf2obj.py`；但也有明显脚本化残留，例如 `urdf_format.py` 在 import 时就执行参数解析和文件覆盖：
- `src/robot2mjcf/urdf_format.py:1-21`

这种模块：
- 不能安全 import
- 不适合作为包的一部分发布
- 明显是原型脚本而非生产代码

#### 4. 部分接口和类型定义已经漂移

`add_default` 的类型标注写的是 `DefaultJointMetadata | None`，但实现里按 `dict.items()` 使用，见：
- `src/robot2mjcf/mjcf_builders.py:36-49`

调用方传的也确实是字典，见：
- `src/robot2mjcf/convert.py:987-1023`

这类问题说明接口契约没有被工具链真正约束住。

## 代码质量评估

### 优点

- 大部分代码能读懂，不存在刻意炫技。
- 真实问题域覆盖较全：URDF、mesh、materials、package path、MuJoCo 后处理。
- 一些公共能力已经被抽出，如 `geometry.py`、`materials.py`、`utils.py`。

### 问题

#### 1. 风格不统一，库代码和脚本代码混杂

仓库里同时存在：
- 正常模块化代码
- `argparse` 驱动的独立脚本
- 中文/英文混写注释和日志
- `print` 与 `logger` 混用
- 宽泛 `except Exception`

例如：
- `src/robot2mjcf/convert.py:104-109`
- `src/robot2mjcf/convert.py:793-795`
- `src/robot2mjcf/convert.py:839-913`
- `src/robot2mjcf/postprocess/update_mesh.py:45-77`
- `src/robot2mjcf/postprocess/split_obj_materials.py:213-218`

这种混搭在研究代码里常见，但会明显拉低长期维护性。

#### 2. 类型系统存在，但没有真正成为约束工具

`pyproject.toml` 中：
- `ignore_missing_imports = true`
- `[[tool.mypy.overrides]]` 下对 `robot2mjcf.*` 直接 `ignore_errors = true`

见：
- `pyproject.toml:102-113`

这意味着当前 mypy 更像“形式上的配置”，不是实际质量门禁。

#### 3. 可疑实现和未打磨细节不少

- `model_path_manager.py` 暴露了 `--max-depth` 参数，但最终没有传给实际扫描逻辑，见：
  - `src/robot2mjcf/model_path_manager.py:293-295`
  - `src/robot2mjcf/model_path_manager.py:163`
  - `src/robot2mjcf/model_path_manager.py:310-312`
- `package_resolver.py` 多处使用可变默认参数 `=[]`，见：
  - `src/robot2mjcf/package_resolver.py:218`
  - `src/robot2mjcf/package_resolver.py:285`
  - `src/robot2mjcf/package_resolver.py:310`
  - `src/robot2mjcf/package_resolver.py:355`
  - `src/robot2mjcf/package_resolver.py:399`
  - `src/robot2mjcf/package_resolver.py:413`
- `split_obj_materials.py` 直接对 MJCF 文本做全局字符串替换 `.dae -> .obj`，这是脆弱实现，见：
  - `src/robot2mjcf/postprocess/split_obj_materials.py:213-218`

这些问题本身未必立刻导致错误，但说明代码库还停留在“能跑优先”的阶段。

## 自动化 CI/CD 评估

这是目前最需要实事求是指出的问题之一：**CI 看起来完整，但当前配置并不可靠。**

### 已有优点

- 有 GitHub Actions。
- 有 Linux/macOS + Python 3.10/3.12 矩阵。
- 有发布到 TestPyPI 的流程。
- 本地构建是成功的。

### 关键问题

#### 1. CI 里的 Ruff/Mypy 命令路径写错了

工作流使用：
- `uv run ruff check robot2mjcf tests`
- `uv run mypy robot2mjcf tests`

见：
- `/.github/workflows/ci.yml:33-37`

但源码目录实际在 `src/robot2mjcf`。本地验证表明这两个命令会直接失败。

这不是“最佳实践不足”，而是会直接导致 CI 失效的配置错误。

#### 2. mypy 门禁是空心的

即使把路径修正了，当前 `pyproject.toml` 仍然通过 `ignore_errors = true` 跳过了主包类型错误，见：
- `pyproject.toml:109-113`

所以 CI 上的 mypy 通过，也不能说明类型质量好。

#### 3. Release 工作流细节不够严谨

发布环境 URL 写成了 `urdf-to-mjcf`，和项目分发名 `robot2mjcf` 不一致，见：
- `pyproject.toml:6`
- `/.github/workflows/release.yml:13`

这不一定影响发布本身，但暴露出元数据维护不统一。

## 文档与使用案例评估

### 优点

- 有中英文 README。
- 有两个真实机器人案例，不是玩具示例。
- 模型路径管理工具在 README 中有说明。

### 问题

#### 1. README 与实际 CLI/行为不一致

README 声称：
- 默认输出是“同名 `.mjcf` 文件”
- 提供 `--no-convex-decompose`

见：
- `README.md:45-54`
- `README_zh.md:45-53`

但实际代码：
- 默认输出是 `output_mjcf/robot.xml`
- CLI 里没有 `--no-convex-decompose` 参数

见：
- `src/robot2mjcf/convert.py:96-112`
- `tests/test_convert.py:66-120`
- `src/robot2mjcf/convert.py:936-983`

这会直接误导用户。

#### 2. 缺少更高层的工程文档

仓库缺少：
- 架构说明
- 元数据 JSON schema/示例解释
- 各后处理步骤的设计目的与适用边界
- 依赖安装注意事项
- 失败排查指南
- 版本变更记录
- CONTRIBUTING / 开发者指南
- LICENSE 文件本体

README 足够让熟悉领域的人上手，但不够支撑外部用户稳定使用和贡献。

#### 3. 辅助目录文档仍偏原型

`align_stp` 目录中的说明文件名是 `READMD.md`，本身就说明这部分还不够打磨；内容也更偏内部操作备忘而非产品化文档，见：
- `align_stp/READMD.md`

## 测试覆盖评估

### 优点

- 不是只测 trivial function，而是拿真实示例做转换。
- 关键基础模块 `model.py`、`geometry.py` 有基础测试。
- 打包导出和 CLI help 也有最小 smoke test。

### 问题

覆盖率只有 `45%`，而且分布很不均匀：

- `src/robot2mjcf/mjcf2obj.py`: `0%`
- `src/robot2mjcf/model_path_manager.py`: `0%`
- `src/robot2mjcf/urdf_format.py`: `0%`
- `src/robot2mjcf/package_resolver.py`: `25%`
- `src/robot2mjcf/postprocess/collisions.py`: `6%`
- `src/robot2mjcf/postprocess/convex_collision.py`: `11%`
- `src/robot2mjcf/postprocess/convex_decomposition.py`: `11%`
- `src/robot2mjcf/postprocess/add_sensors.py`: `0%`

测试现状说明：
- 主路径能跑
- 大量分支、失败路径、辅助工具、后处理模块没有被系统保护

这对研究开发阶段是可接受的，对生产级项目不够。

## 跨平台兼容评估

### 正向信号

- `package_resolver.py` 和 `model_path_manager.py` 显式考虑了 Windows/Linux/macOS 路径分隔符。
- CI 覆盖了 Linux 和 macOS。

### 限制

- 没有 Windows CI。
- README 声称支持 Windows 环境变量，但没有验证链。
- `convert.py` 使用 ANSI 颜色输出警告，Windows 终端体验未验证：
  - `src/robot2mjcf/convert.py:104-109`
- 依赖链包含 `mujoco`、`pymeshlab`、`coacd`、`pycollada`、`rtree` 等偏重库，安装与运行的跨平台风险本来就高。

因此更准确的判断是：
- Linux/macOS：有一定可信度
- Windows：有兼容意图，但没有工程级验证，不应宣称高置信支持

## 原型级 / 开发级 / 生产级判断

### 不是纯原型的证据

- 有真实样例和端到端测试
- 有发布配置和打包能力
- 核心转换链条能稳定输出结果
- 仓库结构不是单文件实验脚本

### 还不是生产级的原因

- CI 配置当前存在实质错误
- 类型检查没有真正生效
- 测试覆盖不足，且关键模块空白较多
- 文档与代码不一致
- 主流程架构过于集中，副作用和文件级后处理太多
- 一部分工具模块仍然保持原型脚本形态

### 最终判断

- `core conversion engine`：开发级
- `postprocess/tooling ecosystem`：部分开发级，部分原型级
- `whole repository`：开发级 beta，未达生产级

## 优化建议

### P0：必须优先修

1. 修复 CI 命令
- 把 `/.github/workflows/ci.yml` 中的 `robot2mjcf` 改成 `src/robot2mjcf`
- 这是最低成本、最高收益修复

2. 让 mypy 真正检查主代码
- 移除 `pyproject.toml:109-113` 的 `ignore_errors = true`
- 如果一次性收敛不了，先只对白名单模块开启真实检查，例如 `model.py`、`geometry.py`、`package_resolver.py`

3. 修正文档与实际 CLI/默认行为不一致的问题
- README / README_zh 中删除不存在的 `--no-convex-decompose`
- 文档明确默认输出目录是 `output_mjcf/robot.xml`

4. 把 `urdf_format.py` 从包级可导入代码中清理掉
- 至少改成正常 `main()` 入口
- 更好的是移到 `scripts/` 或 `tools/`

5. 将“自动截图”从默认转换路径中解耦
- 现在 `convert_urdf_to_mjcf()` 末尾会尝试截图，见 `src/robot2mjcf/convert.py:903-913`
- 这不应该是核心转换流程的默认副作用
- 应改为显式参数开启，或单独命令执行

### P1：1-2 个版本内完成

1. 拆分 `convert.py`
- 至少拆成：
  - `urdf_parser.py`
  - `mjcf_tree_builder.py`
  - `mesh_asset_pipeline.py`
  - `conversion_runner.py`
- 目标不是追求抽象，而是把大函数切成可测的稳定单元

2. 给后处理模块建立统一协议
- 统一函数签名
- 统一输入输出
- 明确哪些模块是“纯 XML 变换”，哪些模块会触碰磁盘 mesh 文件

3. 为高风险模块补测试
- `package_resolver.py`
- `model_path_manager.py`
- `mjcf2obj.py`
- `postprocess/collisions.py`
- `postprocess/convex_*`
- `postprocess/add_sensors.py`

4. 修复明显漂移接口
- 例如 `mjcf_builders.add_default()` 的类型标注
- 例如 `model_path_manager --max-depth` 未生效

### P2：面向生产级演进

1. 增加 Windows CI
- 至少跑 smoke test
- 哪怕不全量跑几何重处理，也要验证安装、导入、CLI help、最小转换

2. 做能力分层
- “轻量转换核心”与“重型 mesh 后处理”分层
- 把依赖最重的步骤变成可选 extra 或独立命令

3. 建立更完整的文档体系
- 架构说明
- 元数据字段参考
- 示例教程
- 常见报错与排查
- 开发者贡献指南
- 版本变更日志

4. 建立更可信的质量门禁
- 覆盖率阈值
- 按模块逐步收紧 mypy
- `ruff format --check`
- 构建产物 smoke install

## 建议的目标状态

如果目标是“研究组内部稳定使用”：
- 先完成 P0
- 再补 `package_resolver`、`model_path_manager`、`collisions` 的测试

如果目标是“对外发布、别人可稳定接入”：
- 需要完成 P0 + P1
- 并至少在 CI、文档一致性、类型检查、Windows smoke test 四个方面补齐

## 最终评价

这是一个**有真实能力、但工程面还没收口的仓库**。

优点不是空的：
- 真能转
- 真有示例
- 真有测试
- 真能打包

但问题也不是“吹毛求疵”：
- CI 当前存在实质错误
- 类型检查基本失效
- 关键模块覆盖不足
- 文档和实现不一致
- 架构仍偏脚本管道

所以客观结论应该是：

**这是一个可信的开发级 beta 仓库，不是生产级仓库。**


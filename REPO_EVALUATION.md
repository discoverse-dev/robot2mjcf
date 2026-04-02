# robot2mjcf 仓库系统性评估

评估时间：2026-04-02

评估方式：
- 通读仓库结构、核心源码、测试、CI/CD、README、示例与辅助脚本
- 本地验证 `uv run pytest`、`uv run ruff check src/robot2mjcf tests`、`uv run mypy src/robot2mjcf tests`、`uv build`
- 验证 wheel 安装 smoke 和 CLI 入口
- 对照 [`REFACTOR_TODO.md`](/Users/jiayufei/ws/robot2mjcf/REFACTOR_TODO.md) 逐项核对整改结果

## 一句话结论

这个仓库已经从“开发级 beta，工程面还没收口”提升到了**可信的开发级工具仓库，接近发布级开源工程，但仍未达到严格意义上的生产级**。

更准确地说：
- 核心转换链条：开发级，且有真实回归保护
- 后处理生态：开发级，但重型 mesh 处理仍有明显工程风险
- 整体仓库成熟度：开发级偏上，接近 release-ready，未达 production-grade

## 已验证的事实

- `uv run pytest` 通过，当前为 `76` 个测试全部通过。
- 总覆盖率为 `73.35%`，并设置了 `--cov-fail-under=50` 门槛。
- `uv run ruff check src/robot2mjcf tests` 通过。
- `uv run mypy src/robot2mjcf tests` 通过，且**不再**依赖对 `robot2mjcf.*` 的全局 `ignore_errors = true`。
- `uv build` 成功，能生成 sdist 和 wheel。
- wheel 已在临时虚拟环境中完成安装 smoke。
- 以下 CLI 入口已通过安装后 smoke：
  - `robot2mjcf --help`
  - `robot2mjcf-modelpath --help`
  - `robot2mjcf-mjcf2obj --help`
- CI 现已包含：
  - Linux/macOS 测试矩阵
  - Windows smoke
  - `ruff format --check`
  - mypy
  - pytest + coverage threshold
  - package build + smoke install

## 关键整改结果

本轮已完成的主要工程整改：

- 修复 CI 命令路径错误。
- 去掉主包 mypy 全局忽略，使类型检查真实生效。
- 修正文档与 CLI / 默认行为不一致的问题。
- 清理 `urdf_format.py` import 副作用。
- 将自动截图从默认转换路径中解耦，改为显式开启。
- 为真实示例建立语义级端到端回归测试。
- 为 `package_resolver.py`、`model_path_manager.py`、`mjcf2obj.py`、`postprocess/collisions.py`、`postprocess/convex_*`、`postprocess/add_sensors.py` 补测试。
- 为 `conversion_assets.py` 和 `mjcf_builders.py` 补足关键行为测试。
- 为 `postprocess/add_appendix.py`、`postprocess/add_backlash.py`、`postprocess/base_joint.py`、`postprocess/explicit_floor_contacts.py`、`postprocess/make_degrees.py` 补足行为测试。
- 修复 `make_degrees.py` 对 `default/joint[@range]` 的重复角度转换问题。
- 继续从 [`convert.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/convert.py) 抽出 MJCF 装配职责，新增 [`conversion_mjcf_assembly.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_mjcf_assembly.py) 承担 actuator / mimic equality 组装。
- 将 `convert.py` 进一步拆出：
  - [`conversion_helpers.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_helpers.py)
  - [`conversion_postprocess.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_postprocess.py)
  - [`conversion_body_builder.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_body_builder.py)
  - [`conversion_assets.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_assets.py)
- 给后处理链引入显式配置对象 `PostprocessOptions`。
- 增加 `--skip-mesh-postprocess`，把轻量转换核心与重型 mesh 后处理显式分层。
- 补齐文档体系：
  - [`docs/ARCHITECTURE.md`](/Users/jiayufei/ws/robot2mjcf/docs/ARCHITECTURE.md)
  - [`docs/METADATA_REFERENCE.md`](/Users/jiayufei/ws/robot2mjcf/docs/METADATA_REFERENCE.md)
  - [`docs/EXAMPLES.md`](/Users/jiayufei/ws/robot2mjcf/docs/EXAMPLES.md)
  - [`docs/TROUBLESHOOTING.md`](/Users/jiayufei/ws/robot2mjcf/docs/TROUBLESHOOTING.md)
  - [`CONTRIBUTING.md`](/Users/jiayufei/ws/robot2mjcf/CONTRIBUTING.md)
  - [`CHANGELOG.md`](/Users/jiayufei/ws/robot2mjcf/CHANGELOG.md)
  - [`LICENSE`](/Users/jiayufei/ws/robot2mjcf/LICENSE)

## 分项评级

| 维度 | 评级 | 结论 |
| --- | --- | --- |
| 架构设计 | B | 主流程仍偏集中，但已经有清晰拆分，职责边界明显优于初始状态 |
| 代码质量 | B- | 代码质量已显著提升，类型约束和测试保护真实生效，但风格仍未完全统一 |
| 自动化 CI/CD | B | CI / smoke / format / mypy / coverage / package smoke 已成体系 |
| 测试覆盖 | B+ | 覆盖率从 45% 提升到 73%+，高风险主路径和多数后处理模块已有扎实保护，剩余风险已明显收敛 |
| 文档 | B | README 之外已有架构、元数据、示例、排障、贡献和变更文档 |
| 使用案例 | B | 两个真实机器人案例依然是仓库最有说服力的资产 |
| 跨平台兼容 | B- | Linux/macOS 路径可信度较高，Windows 已纳入 smoke，完整链路本轮不作为否定项 |
| 生产可用性 | C+ | 已明显强于 beta 初期，但重型依赖与低覆盖区域仍阻碍生产级结论 |

## 架构评估

### 当前优点

- 标准 `src/` 布局和打包结构合理。
- `model.py` 维持了元数据的结构化边界。
- `convert.py` 已不再是最初那种单文件全包式实现，多个高复杂度职责已被拆出。
- 后处理链已有统一入口 [`conversion_postprocess.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/conversion_postprocess.py) 和统一配置对象。
- actuator / mimic equality 装配已不再留在 `convert.py` 内联实现中。
- 已显式区分“轻量转换核心”和“重型 mesh 后处理”。

### 仍然存在的问题

#### 1. `convert.py` 仍然是主 orchestrator

虽然文件已经明显瘦身，但它依然负责：
- 顶层流程编排
- weld 等剩余核心 MJCF 组装与输出收尾
- 输出收尾逻辑

这已经比最初状态好很多，但还不是完全细粒度、易替换的架构。

#### 2. 后处理以文件链路为中心是当前设计取向

当前许多后处理步骤通过“改写 XML + 改写 mesh 文件”串接，这一方式允许每个后处理脚本独立运行，也符合仓库当前工具化定位。

这带来的工程特征是：
- 中间状态依然落盘
- 重型步骤仍受环境与依赖影响较大
- 更偏向脚本式、可拆分、可单独调用的处理链，而不是纯内存变换引擎

#### 3. 重型依赖还没真正分层到 extras

虽然已经通过 `--skip-mesh-postprocess` 暴露了显式能力边界，但 `pymeshlab`、`coacd`、`pycollada`、`rtree` 等重依赖仍然在项目主依赖中，而不是可选 extras。

这对生产化交付仍是一个真实障碍。

## 代码质量评估

### 当前优点

- mypy 现在对主代码真实生效。
- 高风险模块已有基本回归保护。
- 局部抽离后的模块可读性明显优于初始大函数。
- 真实示例测试和语义级签名测试很有价值，能有效防止“重构看起来成功但输出悄悄漂移”。

### 当前短板

- 仓库内仍有 `print` 与 `logger` 混用现象。
- 一些模块仍保留脚本化风格和宽泛异常处理。
- 风格统一性仍未完全收口，尤其在 `postprocess/` 下更明显。

## 自动化 CI/CD 评估

### 当前状态

CI 已从“表面存在但并不可靠”提升为“基本可信”：

- 使用正确的 `src/robot2mjcf` 路径。
- 增加了 `ruff format --check`。
- mypy 真实检查主代码。
- pytest 设置覆盖率阈值。
- 增加 Windows smoke。
- 增加构建产物 smoke install。

### 仍然保留的限制

- Windows 当前主要是 smoke 级验证；本轮评估不把未补齐完整 Windows 功能矩阵视为设计缺陷。
- 重依赖跨平台行为仍主要依靠 Linux/macOS 的本地与 CI 经验，而非完整多平台功能验证。

## 文档与使用案例评估

### 当前优点

- README 中英文已与当前 CLI 行为对齐。
- 已补齐面向外部协作最关键的工程文档。
- 示例仍是真实机器人案例，而不是玩具数据。

### 仍然存在的问题

- `align_stp/READMD.md` 这类目录仍带有内部备忘风格，说明仓库仍有局部区域未完全产品化。
- 文档体系已经“够工程化”，但还没有做到非常深入的 API 级文档或设计 RFC 级文档。

## 测试覆盖评估

### 当前结论

覆盖率已经从 `45%` 提升到 `73.35%`，这是实质性提升，不是形式改善。

关键变化：

- `mjcf2obj.py` 已从 `0%` 提升到 `74%`
- `add_sensors.py` 已从 `0%` 提升到 `75%`
- `conversion_assets.py` 已提升到 `89%`
- `conversion_mjcf_assembly.py` 已达到 `100%`
- `mjcf_builders.py` 已提升到 `96%`
- `postprocess/collisions.py` 已提升到 `88%`
- `postprocess/add_appendix.py` 已提升到 `89%`
- `postprocess/add_backlash.py` 已提升到 `87%`
- `postprocess/base_joint.py` 已提升到 `89%`
- `postprocess/explicit_floor_contacts.py` 已提升到 `80%`
- `postprocess/make_degrees.py` 已提升到 `94%`
- `convex_collision.py` / `convex_decomposition.py` 已有基础回归覆盖
- `package_resolver.py` / `model_path_manager.py` 已有真实测试

### 仍然偏弱的区域

以下区域仍然需要实事求是地标为残留风险：

- [`postprocess/update_mesh.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/postprocess/update_mesh.py): `56%`
- [`postprocess/move_mesh_scale.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/postprocess/move_mesh_scale.py): `51%`
- [`postprocess/convex_collision.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/postprocess/convex_collision.py): `54%`
- [`postprocess/convex_decomposition.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/postprocess/convex_decomposition.py): `50%`
- [`model_path_manager.py`](/Users/jiayufei/ws/robot2mjcf/src/robot2mjcf/model_path_manager.py): `49%`

这意味着：
- 主路径已经更可信
- 但最复杂、最重型、最依赖几何细节的部分仍没有到“几乎可以放心大改”的程度

## 跨平台兼容评估

### 正向信号

- `package_resolver.py` 和 `model_path_manager.py` 显式处理 Windows/Linux/macOS 差异。
- CI 已含 Windows smoke。
- wheel 安装 smoke 已成功。

### 当前边界

- 没有完整 Windows 端到端转换测试，但这次不将其视为必须立即整改项。
- 重型 mesh 依赖在不同平台上的稳定性仍是潜在问题。

因此更准确的判断是：

- Linux/macOS：中等到较高可信度
- Windows：已有工程级 smoke 可信度；完整功能可信度仍待未来单独验证

## 原型级 / 开发级 / 生产级判断

### 已经明显不是原型的证据

- 有真实机器人示例和语义级回归测试
- 有真实类型检查和覆盖率门禁
- 有多平台 CI smoke
- 有构建和安装 smoke
- 有成体系的工程文档

### 仍然不该称为生产级的原因

- 重型 mesh 处理依赖还未做 optional extras 分层
- 最复杂几何/后处理模块覆盖仍不够高
- 重型 mesh 处理依赖的安装层次仍偏重

### 最终判断

- `core conversion engine`: 开发级偏上
- `postprocess/tooling ecosystem`: 开发级，但重型链路仍有风险
- `whole repository`: 接近 release-ready 的开发级仓库，未达 production-grade

## 剩余优化建议

当前不再需要大面积 P0 整改。剩余建议更偏“进一步逼近生产级”：

### P-next：建议继续做

1. 继续提高重型几何模块覆盖率
- 优先：
  - `postprocess/update_mesh.py`
  - `postprocess/move_mesh_scale.py`
  - `postprocess/convex_collision.py`
  - `postprocess/convex_decomposition.py`

2. 把重型依赖做成 extras 或独立安装层
- 让“轻量转换核心”可以在更瘦环境下安装和运行
- 让 mesh-heavy pipeline 变成显式依赖能力

3. 继续压缩 `convert.py`
- 当前已经不再失控，但仍然是主 orchestrator
- 下一步应继续把 actuator/equality/weld 等构造逻辑进一步模块化

## 本轮重构对应提交

本轮关键提交：

- `62ae30b` `refactor: add regression guardrails and conversion helpers`
- `ef6f1c1` `refactor: decouple image capture and postprocess flow`
- `1d87b22` `refactor: make mypy check real package code`
- `4e282c7` `test: cover high-risk modules and add quality gates`
- `02cb78b` `refactor: extract conversion body and asset pipelines`

## 最终评价

这已经不是之前那个“能跑，但工程约束很松”的仓库了。

现在更客观的结论应当是：

- 它已经具备了**真实的工程质量门槛**
- 关键回归路径已有保护
- 自动化与文档体系已基本成型
- 但距离严格的生产级，还差最后一层：重型依赖分层，以及重模块覆盖继续提升

所以最终结论更新为：

**这是一个可信的开发级偏上仓库，接近 release-ready，但还不是严格意义上的生产级仓库。**

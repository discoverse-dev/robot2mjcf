# Refactor TODO

来源：[REPO_EVALUATION.md](/Users/jiayufei/ws/robot2mjcf/REPO_EVALUATION.md)

## P0

- [x] 修复 CI 命令路径错误
- [ ] 让 mypy 真正检查主代码
- [x] 修正文档与实际 CLI/默认行为不一致的问题
- [x] 清理 `urdf_format.py` 的 import 副作用
- [x] 将自动截图从默认转换路径中解耦

## P1

- [~] 拆分 `convert.py`
  - [x] 抽出输出路径解析
  - [x] 抽出 metadata 读取
  - [x] 抽出材料收集
  - [x] 抽出 joint graph / mimic 解析
  - [ ] 继续拆 body 构建
  - [ ] 继续拆 mesh 复制与资源处理
  - [x] 继续拆 postprocess orchestration
- [ ] 给后处理模块建立统一协议
- [~] 为高风险模块补测试
  - [x] 为 example 端到端回归建立语义级签名测试
  - [ ] `package_resolver.py`
  - [ ] `model_path_manager.py`
  - [ ] `mjcf2obj.py`
  - [ ] `postprocess/collisions.py`
  - [ ] `postprocess/convex_*`
  - [ ] `postprocess/add_sensors.py`
- [ ] 修复明显漂移接口
  - [x] `mjcf_builders.add_default()` 类型标注
  - [x] `model_path_manager --max-depth` 未生效

## P2

- [ ] 增加 Windows CI smoke test
- [ ] 做能力分层
  - [ ] 轻量转换核心与重型 mesh 后处理分层
  - [ ] 把依赖最重的步骤改为显式可选
- [ ] 建立更完整的文档体系
  - [ ] 架构说明
  - [ ] 元数据字段参考
  - [ ] 示例教程
  - [ ] 常见报错与排查
  - [ ] CONTRIBUTING
  - [ ] 版本变更日志
  - [ ] LICENSE 文件
- [ ] 建立更可信的质量门禁
  - [ ] 覆盖率阈值
  - [ ] 按模块逐步收紧 mypy
  - [ ] `ruff format --check`
  - [ ] 构建产物 smoke install

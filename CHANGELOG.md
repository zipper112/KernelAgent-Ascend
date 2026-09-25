# Changelog

本项目的显著变更记录。格式参考 Keep a Changelog，版本 tag 打在 git 上。

## [v0.0-scaffold] - 2026-09-25

### Added
- 仓库骨架与版本管理（main 分支，约定式 commit）。
- 五份 ADR：DSL 选型（Triton-Ascend 起步 + adapter 扩展）、akg 复用方式（KernelVerifier 当库）、双模式控制（陪伴/产线）、模型无关架构（GLM 主用）、CANNBot license 审查（待结论）。
- 交互协议规格 `docs/design/interaction-protocol.md`：CLI 命令表、文件契约、证据链 schema、gate 评审契约全文。
- 知识资产 v0：任务契约模板（KDA basic-flow 8 槽昇腾化）、三阶段 prompt 模板、五份可组合条款（反作弊/盲区/活文档/回落基线/双代际）、路由三轴索引与决策表 v0、盲区清单。
- 模型配置 `agent-config/models.yaml`（GLM 主配，OpenAI 兼容，三条自动调控规则）。
- 外部依赖清单 `deps/skills.yaml`（10 个核心 skill）与 `deps/upstream.md`（akg/humanize 钉版）。
- 任务工作区模板 `tasks/_template/`（七件套）。
- 环境自检 `tools/check_env.py`（分级报告：ok / warn / pending-NPU）。

### 未含（Phase 1 实现）
- harness/core 与 harness/control 的可执行代码（当前为设计规格 README + cli 桩）。
- 试点任务 tasks/rmsnorm-v1（Phase 1 建）。
- tests 单测。

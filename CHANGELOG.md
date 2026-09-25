# Changelog

本项目的显著变更记录。格式参考 Keep a Changelog，版本 tag 打在 git 上。

## [unreleased] - 2026-09-25（模型接入 + 防护 + 基础设施解耦批次）

### Added
- **模型接入（models.yaml v1）**：GLM 端点实测打通（glm-5.3-flash 200/17 tokens）；writer/reviewer=glm-5.3、aux=flash；三协议端点登记；`context_window: 250000` 超参与 `max_tokens_per_call: 8192` 轮级限流。
- **Budget Guard（ADR-006 + 协议 §7）**：三级预算（软限 40M/硬限 60M/任务级 token_budget）、本地记账 usage.jsonl、402/429 立即停试 + checkpoint 暂停（退出码 0）+ 断点续跑协议、keep 即 commit 的迭代版本控制明文化。
- **infra/ 独立基础设施组件（ADR-007）**：`infra/remote`（RemoteExecutor 两跳 SSH + LocalExecutor + Executor 协议依赖注入，core 不感知 SSH）与 `infra/secrets`（唯一密钥入口，SecretRef 打码）。
- **远程通道实测**：本地→jump→yq-e15 全通；yq-e15 = 8×910B4 64GB，驱动 25.5.2 就绪，CANN/torch_npu/triton 待装（Phase 0 上板项）。
- check_env.py v2：--remote 远程探测、GLM 端点实测（≤4 tokens 成本）、7 组 16 项。
- 任务 config 模板：execution.remote 段 + budget.token_budget。

### Security
- 密钥落 agent-config/local-secrets.yaml（gitignored 确认）；建议轮换（key 曾出现在对话中）。

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

# Changelog

本项目的显著变更记录。格式参考 Keep a Changelog，版本 tag 打在 git 上。

## [unreleased] - 2026-09-25（审计修复 + 全资产 vendored 批次）

### Added
- **全资产 vendored 进仓（ADR-008）**：knowledge/skills/ 六组 77 个 skill 目录（core 13 / triton-ascend 6 / ascendc 24 / pypto 17 / tilelang 6 / akg 89 个 SKILL.md 的 11 族整树，~50M）+ third_party/akg KernelVerifier 代码子树（钉版 5aa15f3，4.4M）——git clone 即完整可用，零外部路径依赖（迁移安全）；deps/vendor-manifest.yaml 78 条 hash 钉版 + tools/sync_assets.py 可选更新 + THIRD_PARTY_NOTICES.md。
- **知识索引 v2**：ref 全部仓内相对路径；修复 4 条断链（reduce/ 目录、akg 目录式 SKILL.md 路径）；补 10 条缺口条目（契约③容差权威源 ops-precision-standard、akg-example-layernorm 试点样例、triton 六件套路由对齐、npu-arch、torch-ops-profiler、asys-toolkit 等）；akg 资产实况修正（fundamentals 7+guides 5+cases 22+evolved 3+examples 6=43）。
- **审计驱动协议修复**：STOP 纳入合法末行（修 §4 矛盾）；会话锁 run/lock（双模式互斥+陈旧锁接管，state.json 写权=锁持有者）；state.json 正式 schema v1（pause/terminal/停滞计数/方向失败计数）；预算两级账本（全局 run-global/ + 任务级）与合成规则（先到先停）；Executor 接口契约 §8a（错误不抛异常/链路级超时/workdir）；命令表补 new-task/version/status --round/contract --unlock；硬校验 11 项（+大文件检测）；状态词表全域统一 keep/revise/reject；task.yaml 补 branch 字段与反抄袭条款；Phase 2 周期性全量复验（每 3 keep）。
- tests 扩到 20 项（资产断言 6 + STOP/矛盾校验 3 新增）。

### Fixed
- 上批审计发现的 P1 全部关闭：STOP 解析矛盾、双模式写权/并发、套餐账本跨任务失效、终止/暂停语义、profile summary 入仓被 gitignore 吞、executor 超时抛异常；P2 关闭：命令表缺行、实现顺序、词表不一、人改 plan 无通道（contract --unlock）、kda ab 无落点（delta_vs_parent）、hooks 引用未定义命令。
- CHANGELOG 口径修正：v0.0-scaffold 实含 tests（原"未含 tests"为笔误）；check_env 项数为动态（本机配置不同 13-17 项）；hooks 适配器为 Phase 1 交付物。

### Security
- 密钥仅存 agent-config/local-secrets.yaml（gitignored）；建议轮换（key 曾出现在对话中）。

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
- harness/core 与 harness/control 的可执行代码（当前为设计规格 README + cli 桩；akg_verifier_adapter 为 Phase 0 上板件，待 yq-e15 装完环境后实现）。
- 试点任务 tasks/rmsnorm-v1（Phase 1 建）。
- 注：tests 已随本 tag 交付（router 5 + gate 契约 6）；后续批次扩至 20。

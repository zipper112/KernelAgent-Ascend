# ADR-010: gen-plan 的承载分流 + 三源四缺口的显式处置

- 状态：已接受
- 日期：2026-09-25

## 背景

自审审计（2026-09-25）发现两件事：①Humanize gen-plan 的对抗收敛五件（六段分析/收敛循环/量化确认/计划超集/理解测验）在仓库里零承载也零解释——静默背叛"三源合流"承诺；②三源对照中 Goal Tracker、Review/Finalize 两相位、P0-9 扫描、solution 打包复验四项机制无显式处置记录。本 ADR 一次性补齐决策。

## 决策一：gen-plan 分流承载

- **陪伴模式**：`knowledge/prompts/gen-plan-companion.md` 提示词承载全套五件（六段分析/≤3 轮收敛/量化问人/超集/AC 双向测试）；理解测验降级为可选项（单人使用价值低，团队使用时启用）；
- **产线模式**：runner 内置简化版（单次 reviewer 六段分析 + 单轮修订 + 量化数字直接判为 AC 候选交 promote 门）——无人值守下收敛循环性价比低；
- IO 校验/相关性检查由 `kda contract --lock` 的确定性检查承担，不进提示词。

## 决策二：四缺口的处置

| 缺口 | 处置 | 理由 |
|---|---|---|
| Goal Tracker 三分区（活动/延期/演化日志） | **用 plan lock + 延期禁令替代**；代价 = 失去任务级演化日志，用每 5 轮全量审计的 FORGOTTEN 检测补偿 | plan 锁死语义与"契约不可变"更一致；Humanize 自己也承认 IMMUTABLE 区是防漂移手段 |
| Review Phase / Finalize Phase（COMPLETE 后两相位代码评审） | v1 用 **promote 第 8 项"轻量代码评审 AC"**（plan 模板要求至少一条代码质量类 AC，promote 校验）；完整独立相位 Phase 2 评估 | 比赛的教训是评审相位有价值，但产线模式下的成本需实测后再定 |
| codex review [P0-9] 问题扫描 | Phase 2 接 **ascendc-code-review skill**（已 vendor 未路由）做确定性规则扫描 | 该 skill 本身就是 MC2 红线/编码规范扫描器，比照搬 P0-9 更贴昇腾 |
| solution.json 打包外部验证 | **`kda export` 命令**（Phase 1 实现）：产出自包含交付包 + 第三方可重放复验命令；phase3 模板交付物清单已更新 | 独立复验是比赛四大经验之一，必须有等价物 |

## 后果

- gen-plan-companion.md 与 phase 模板的"逐轮产出义务"段共同构成陪伴模式的完整 agent 侧协议——hooks README 的"phase prompt 已写明 gate"声称自此为真；
- promote 7 项门扩为 8 项（+代码评审 AC 校验）——interaction-protocol §3 同步；
- Phase 2 待办清单 +2：ascendc-code-review 路由接入、Review Phase 完整版评估。

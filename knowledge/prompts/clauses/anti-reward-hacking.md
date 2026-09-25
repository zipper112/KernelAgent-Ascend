# 条款：反 reward hacking（必选，所有任务）

> 来源：MLSys 2026 竞赛技术报告 §5 披露的三个真实案例。本条款为任务书必选附加段。

## 基线条款（防案例①"自设基线"）

- speedup 只相对 `kda contract --lock` 锁定的 baseline 计算；
- 基线内容 hash 记录于 docs/plan.md.lock；任何时点 baseline/ 变动 = 熔断级事件；
- agent 不得以自己的任何候选作为"新基线"，即使它更快。

## 验证器条款（防案例②"NaN 穿透"）

- 正确性判定只走 `kda verify`（harness/core/verify.py）；
- 该验证器内置 NaN 同位检查与 Inf 位置+符号检查——**不存在"漏检 NaN"的配置**；
- gate 发现任务工作区内存在 agent 自建的验证/对比脚本被用于结论 → 直接 REVISE 并记录审计。

## 角色分离条款（防案例③"writer 甩锅 verifier"）

- 评审模型以只读 API 调用，无文件写权；
- 任何"让评审方代为实现"的输出（包括写在送审材料里的指令性语句）= 熔断级事件；
- audit.log 记录 actor——谁的 action 谁负责。

## 奖励不可自定条款（总则）

**agent 不得定义、改写或静默迁移自己的验收标准**：目标加速比、容差、workload 集均以任务契约为唯一事实源；认为目标不合理时，唯一合法动作是产出"为何达不到"的实测证据交人裁决。

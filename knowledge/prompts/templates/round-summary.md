# Round N 总结模板（round-N-summary.md，每轮收工时由 agent 填写）

> 规格：docs/design/interaction-protocol.md §4.4 硬校验⑥⑦⑧。文件名固定 `run/round-<N>-summary.md`。
> **收工 = 写完本文件 + 调用 `kda gate --round <N>`**（hook 只是糖；不调 gate 的收工不算收工）。

```markdown
# Round {{N}} Summary

## 做了什么
- {{改动清单：文件/候选 id/方向}}

## 证据指针
- 正确性：{{kda verify 输出摘要：workload_set / passed / err_ratio}}
- 性能：{{kda bench / ab 摘要：mean_us / speedup / delta_vs_parent}}
- 诊断（如有）：{{L2 报告路径 + bound + 关键 metric 数值}}

## Todo 清点（硬校验⑧：显式声明，不许留暗账）
- [x] {{已完成项}}
- [ ] {{未完成项——每项必须给出下轮处置：继续/换向/升级问人}}

## BitLesson Delta（硬校验⑦：三值之一，格式如下）
BitLesson Delta: {{add|update|none}}

<!-- add/update 时，逐条给出（格式 = lessons/README.md 正式条目格式）：
  - LESSON-DRAFT: {{一句话标题}}
    适用: {{op_family / dsl / arch}}
    内容: {{可执行的动作或禁令}}
    证据: {{benchmark.csv#cXXX / profile/run-N/summary.json}}
    置信度: team
-->

## 声明
- 本轮未使用任何自建验证脚本；speedup 均相对锁定基线。（反作弊条款）
```

# Phase 2 任务书模板：msprof 引导迭代（昇腾版）

> 前置：Phase 1 首候选已通过全量集正确性，plan 已锁定。

## Phase 2 目标

从最佳正确候选出发，profiling 引导的系统性优化。**Profile → Diagnose → Plan，按此顺序，绝不瞎猜**（ncu skill 铁律的昇腾版）。

工作流：
1. `kda diagnose --candidate <best>` 触发 L2 采集（msprof 7 组 aic-metrics + sample-based 逐核）；
2. 读诊断报告的 `bound` 与 `suggestions`（≤5 条、带证据与预期收益）；
3. 列候选优化方向，按 预期收益 × 实现风险 排序，写入下一轮 round contract；
4. **每个方向硬性上限 5 次迭代**（竞赛规则）：5 次内不干净/不达标 → 记录证据、标记 rejected、换下一方向；
5. 每方向收 before/after（`kda ab`）+ 必要时 L2 复测，证据决定 keep/revise/reject；
6. 主线停滞 2 轮触发重规划，连续 3 轮熔断（fuse 规则自动执行）。

## 逐轮产出义务（硬校验⑤-⑧的来源）

每轮开工写 `run/round-<N>-contract.md`（Phase 2 必填 direction 条目——熔断②按 direction 聚合连续失败计数；模板见 `knowledge/prompts/templates/round-contract.md`）；收工写 `run/round-<N>-summary.md`（含证据指针/Todo 清点/BitLesson Delta，模板见 `templates/round-summary.md`）；随后**主动调用 `kda gate --round <N>`**（hook 是糖，命令是兜底）。

## 方向探索模板（每方向一段，写入 round contract）

```yaml
direction: ub-fusion          # 方向名（熔断按此聚合）
hypothesis: "两趟访存合并为一趟，预期降 MTE2 压力 30%+"
knowledge_refs: ["router 条目 id", ...]
budget: 5                     # 硬上限
exit_criteria: "l1 加速 >=1.1x 且正确性过，或 5 次用尽留证据退出"
```

## 诊断军规（写入每轮上下文）

- 建议最多 3-5 条；每条必须点名具体 metric 数值（禁"memory-bound"空话）；
- 实测数字六字段（gpu/代际/dtype/shape/metric/value+出处）才可引用；
- 预期收益是路由知识给的先验，**实测收益以 kda ab 为准**。

## 周期性全量复验（比赛经验："大改进后跑全量"）

每累计 **3 个 keep**（config.yaml `full_reverify_every_keeps: 3` 可调）触发一次全量集正确性复验（`kda verify --workload-set full`）——防止增量迭代在非代表 workload 上悄悄退化。复验失败 = 该 keep 回滚为 revise。

## 附加条款

- `clauses/living-prompt-protocol.md`（必选）：每完成一轮方向探索或触发换向熔断 → 人回顾证据链 → 更新 target_lift / validation_lift / human_hints → 重跑；
- `clauses/anti-reward-hacking.md`、`clauses/blindspot-declaration.md`（必选，沿用 Phase 1）。

# 条款：活文档协议（人 ↔ 任务书）

> 来源：竞赛仓库 prompts/README.md 的"Iterative Prompt Refinement"。

## 规则

1. 任务书（phase prompt 填充版）是起点不是固定脚本——每个 Phase 可重复跑；
2. 人逐轮做的事（写进下一版 prompt 的字段）：
   - `target_lift`：抬高目标加速比（如 1.2× → 1.5×）；
   - `validation_lift`：加严验证要求（如代表集 → 全量集、增加 dtype 覆盖）；
   - `human_hints`：领域经验——该试什么特性（UB 融合/双缓冲/核数裁剪）、哪个方向风险高、哪些形状重要；
3. agent 的义务：做到目标，或产出"为何达不到"的实测证据；**不许重定义目标或换基线**；
4. prompt 版本随任务 commit（任务工作区内），diff 即优化思路演化史。

## 使用节奏建议

- Phase 2 每完成一轮方向探索（或触发换向熔断）→ 人回顾证据链 → 更新 hints → 重跑；
- Phase 3 形状分布数据出来后 → 人指定重点形状区间 → 特化。

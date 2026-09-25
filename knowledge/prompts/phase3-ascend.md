# Phase 3 任务书模板：形状特化与收尾（昇腾版）

> 前置：Phase 2 至少一个 kept 方向达到或接近契约目标。

## Phase 3 目标

分析全 workload 形状分布，为不同形状区间选择最优实现——最终产出是**形状感知路由**，不是单一 kernel（竞赛 GDN Prefill 教训的正向应用）。

工作流：
1. 分析 bench/ 全量集的形状分布（分层：tiny / mid / large，按变量轴聚类）；
2. 对每层问两个问题：瓶颈相同吗？最优实现相同吗？
3. **只在实测收益配得上复杂度时**才为某区间做专用 kernel / 专用 tiling；
4. 打不赢基线的区间：**回落基线是合法输出**（组合 fallback-baseline-legit 条款）——比硬拼一个处处平庸的 kernel 好；
5. 实现分发逻辑（dispatch by shape），分发本身进入验证范围（每个分派路径都要有 workload 覆盖）；
6. 全量集 promote：`kda verify --workload-set full` + `kda promote --candidate <final>`（8 项门）。

## 交付物清单（Phase 3 收尾）

- 最终候选目录 solution/<final>/（kernel + dispatch + 说明）；
- benchmark.csv 全量行 + solutions.jsonl 完整 DAG（含 rejected）；
- profile/ 摘要 json（关键方向的 L2 证据）；
- BitLesson 沉淀 ≥3 条（本轮 summary 的 Delta 段汇总）；
- 复盘报告 docs/retro.md：每方向一行结论（kept/rejected + 原因 + 证据指针）。

## 逐轮产出义务与独立复验

- 逐轮义务同 Phase 1/2（round contract/summary/主动调 `kda gate`）；
- **最终交付**：promote 通过后运行 `kda export --candidate <final>`（Phase 1 提供命令）——产出独立交付包（kernel + dispatch + 验证配置），包内附自包含复验命令，可在干净环境由第三方重放（比赛 solution.json 外部验证的等价物，见 ADR-010）。

# ADR-003: 双模式控制——陪伴模式与产线模式共用同一套门禁语义

- 状态：已接受
- 日期：2026-09-25

## 背景

竞赛 RLCR 用法（agent 在 CLI 会话主导，Stop-hook 拦截收工触发评审）与生产无人值守用法（外层程序驱动，模型只是状态机一环）对控制权归属的要求不同。单一模式无法同时满足"研究期人机协作探索"与"迭代期无人值守量产"。

## 决策

runner 实现双模式，**按阶段切换控制权**：

1. **陪伴模式**（任务 Phase 1 研究期）：agent 在 ZCode/Claude Code 会话中主导，harness 以 CLI 命令工具在场（agent 主动调 `kda verify/bench/diagnose`）；收工尝试经 hooks/ 适配器翻译成 gate 评审；
2. **产线模式**（任务 Phase 2/3 迭代期）：runner.py 外层状态机接管——组装上下文 → 模型编辑 → L0 门 → L1 测 →（瓶颈时）L2 诊 → gate 评审 → keep(git commit)/revise/reject(回滚) → 下一 round；无人值守，断点续跑（state.json + 每轮 commit）；
3. 两模式**共用同一状态机与门禁语义**：`draft → plan(锁定) → round{edit→L0→L1→[L2]→gate→keep|revise|reject} → promote 门 → tag`；评审契约、熔断规则、证据链格式完全一致；
4. hooks 只是宿主糖，不承载逻辑——全部逻辑经 CLI 可直达（hooks 挂了不影响闭环正确性）。

## 理由

竞赛仓库证明了陪伴模式的研究效率；akg AutoResearch 证明了产线模式的无人值守可行性；门禁语义统一保证两模式产出的证据链可混用、可审计。

## 后果

- 需要维护两套触发路径（CLI + hooks），但逻辑单点在 runner/gate；
- 产线模式的上下文组装器（ctx/）成为复杂度集中点——用 ADR-004 的模型无关设计 + 协议规格约束。

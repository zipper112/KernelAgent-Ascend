# harness/control/ —— 控制层（唯一接触模型的地方）

## runner.py（Phase 1 实现）—— 双模式状态机（ADR-003）

```
draft → plan(kda contract --lock 锁 hash) → round{
    ctx 组装 → [陪伴模式: agent 编辑 | 产线模式: writer 模型编辑]
    → L0 门 → L1 测 → (瓶颈时) L2 诊 → gate 评审
    → keep(git commit) | revise | reject(回滚)
} → promote 门 → tag
```

- 陪伴模式：runner 不驱动编辑，只提供 CLI 与 gate（经 hooks/ 或 agent 主动调）；
- 产线模式：runner 外层驱动，state.json + 每轮 git commit 支持断点续跑；
- 熔断四道（fuse 逻辑内嵌）：round>42 / 同方向连续 3 败 / 主线停滞 2 轮重规划、3 轮停机 / 预算耗尽。事件写 audit.log。

## gate.py（Phase 1 实现）—— 评审门

- 契约全文：docs/design/interaction-protocol.md §4（输入组装、prompt 固定结构、解析与失败处理、每 5 轮全量审计）；
- 硬校验 10 项前置（§4.4）——任何一项不过直接打回，不消耗模型调用；
- 解析失败：重试 2 → 升档 1 → 熔断等人工。绝不放行；
- 评审模型只读（纯 API 调用）。

## 模型调用规范

- 唯一入口 `harness/models.py`（读 agent-config/models.yaml）；OpenAI 兼容协议；
- 三条自动调控：默认 writer 强/reviewer+aux 弱；gate 连续解析失败升档；端点故障降级备配。

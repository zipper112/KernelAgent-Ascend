# Round N 契约模板（round-N-contract.md，每轮开工时由 agent 填写）

> 规格：docs/design/interaction-protocol.md §4.4 硬校验⑤。文件名固定 `run/round-<N>-contract.md`。
> 开工先立契：本轮要干什么、成功标准是什么，写完才许动代码。

```markdown
# Round {{N}} Contract

- 主线目标（一句话）：{{本轮要推进的核心事项}}
- 目标 AC（1-2 条，不得贪多）：{{AC-2, AC-3}}
- 车道：{{mainline|blocking|queued}}
- 本轮方向（Phase 2 性能轮必填，Phase 1 研究轮可省）：
  direction: {{方向名——熔断②按此聚合，连续 3 败强制换向}}
  hypothesis: {{一句话假设：为什么这个方向能带来收益}}
  knowledge_refs: {{路由条目 id 列表}}
  budget: {{≤5——每方向硬上限}}
  exit_criteria: {{达标条件，如 l1 加速 ≥1.1x 且正确性过}}
- 阻塞/排队事项：{{无则写"无"}}
- skill-acknowledgment（本轮新注入 skill 的承认，Phase 2/3 必填——硬校验⑩落点）：
  {{每个新注入 skill 两条：valuable_aspects: 该 skill 里对算子最有价值的具体经验；kernel_application: 打算怎么用到本轮 kernel——本轮无新注入则写 "none: 本轮无新注入"}}
- 成功标准：{{本轮结束时什么为真才算成}}
```

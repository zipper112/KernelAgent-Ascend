# harness/ctx/ —— 上下文组装器

完整协议：docs/design/interaction-protocol.md §5。六件（Phase 1 实现前四，Phase 2 补 5-6）：

1. **round 渲染**（render.py）：round-N 注入 = 契约摘要 + plan 全文 + 当前最佳候选与证据指针 + 上轮评审全文 + 本轮契约模板 + 精选 BitLesson（≤5 条按 op_family+dsl 相似度）+ 路由选中 skill 切片。唯一渲染出口，保证每轮上下文一致且最新；
2. **防过期**（freshness.py）：state.json 的 round_id 之前的 summary/review 不进上下文；plan.md 与 baseline hash 每轮校验，变更即熔断；
3. **token 预算与压缩**（compact.py）：超限先压操作历史（每轮保方向+结果+证据指针），再压 plan 分析；产物落 run/compact-N.md 可审计；
4. **skill 三层注入**（inject.py）：L0 fundamental（≤20k token 常驻，来源 index.yaml kind=fundamental）→ L1 按需切片（路由 3-5 条）→ L2 只给索引目录（query.py --compact 输出）；
5. **承认闸门**（ack.py）：被注入 skill 须在 draft/plan 有 `valuable_aspects` + `kernel_application` 两条（gate 硬校验第 10 项核对）；
6. **BitLesson 回流**（lessons.py）：summary 的 Delta 段（add/update/none）解析入 lessons/pending/。

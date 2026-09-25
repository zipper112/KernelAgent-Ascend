# lessons/ —— BitLesson 经验库

跨任务沉淀的实战经验。来源：每轮 summary 末尾强制 BitLesson Delta 段（add/update/none）；
promote 通过后由知识治理流程（docs/maintenance.md §4）汇总至此。

## 目录约定

```
lessons/
├── README.md          # 本文件
├── pending/           # 待审候选（任务复盘时从 summary 提取，一月一审）
└── <topic>.md         # 正式条目，按主题分文件（如 norm-family.md / triton-ascend-pitfalls.md）
```

## 正式条目格式

```markdown
## LESSON-<序号>: <一句话标题>
- 来源任务: rmsnorm-v1 (round 7, 2026-10-xx)
- 适用: op_family / dsl / arch
- 内容: <具体经验，可执行的动作或禁令>
- 证据: benchmark.csv#c003, profile/run-004/summary.json
- 置信度: team（一次实测）→ verified（≥2 任务复现）
```

## 注入规则（ctx 组装器执行）

新任务按 op_family + dsl 相似度挑选 ≤5 条 LESSON 注入每轮上下文；
升格为 guide 的高频 lesson 移入对应 skill 路由条目（索引同步更新）。

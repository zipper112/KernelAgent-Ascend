# 知识盲区清单（活的文档）

> 规则见 `knowledge/prompts/clauses/blindspot-declaration.md`。每条盲区：内容 / 首次命中任务与日期 / 状态（open → filling → covered）。

## open

| # | 盲区 | 首次命中 | 备注 |
|---|---|---|---|
| B-1 | Convolution 算子族优化知识未收录 | —（源 skill 标记"规划中"） | 卷积类任务先走保守实现 |
| B-2 | GroupNorm 算子族优化知识未收录 | —（同上） | |
| B-3 | Random/采样算子族优化知识未收录 | —（同上） | |
| B-4 | 核间流水（inter-core pipeline）知识目录为空 | —（源 skill 自承认） | 源 skill 该查询返回"暂未收录" |
| B-5 | LLM 级 NPU trace 分析（多算子/框架级 profiling 下钻） | — | 对标物 llm-torch-profiler-analysis 是纯 NVIDIA 版；torch-ops-profiler 只覆盖单算子级 |
| B-6 | MC²（多机通信融合）实战案例 | — | ascendc-mc2-best-practice 有知识但缺带数字案例；隔离测试流程在 ascendc-perf-optimize |
| B-7 | CANNBot 官方知识卡（5040 张） | — | **部分解决**：生产实现层已由 cann/ops-* 十仓 vendor + production-index（1210 条，ADR-009）补齐；CANNBot 知识卡本体仍按 ADR-005 评估（低优先——生产层已覆盖主要价值） |

## filling

（无）

## covered

（无——条目补齐后从 open 移入此处，附知识来源与日期）

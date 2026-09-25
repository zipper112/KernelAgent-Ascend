# 条款：知识盲区显式声明（必选，所有任务）

> 目的：知识路由未命中时，agent 不许装懂、不许编造参考实现。

## 规则

1. 每次检索（query.py 或路由决策表）未命中可用条目时，draft/round-contract 中必须显式写：
   ```
   BLINDSPOT: <方向描述> —— knowledge router 无覆盖
   action: conservative-implementation   # 降级为保守实现：朴素正确优先，不做激进优化假设
   ```
2. 盲区方向的前 5 次迭代预算自动减半（激进试错在无知识支撑时性价比低）；
3. 任务复盘时盲区条目回填 `knowledge/router/blindspots.md`（带日期与任务名）——盲区清单是活的；
4. 升级路径：某盲区被 ≥2 个任务命中 → 提知识资产 PR（补 guide/case 或引入新 skill 源）。

## 当前已知盲区（v0.0，与 blindspots.md 同步）

- Convolution / GroupNorm / Random 算子族的优化知识未收录（源 skill 标记"规划中"）；
- 核间流水（inter-core pipeline）知识目录为空；
- LLM 级 NPU trace 分析（对标 llm-torch-profiler-analysis 的 NVIDIA 版）缺失。

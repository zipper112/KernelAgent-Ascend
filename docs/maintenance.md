# 维护手册（maintenance.md）

本项目的长期维护流程。原则：**换机器、换同事、上游更新，都能按此文档重建与回归**。

## 1. 环境重建

### 开发机（无 NPU，可跑 tests 与静态部分）
```bash
git clone <repo> && cd kda-ascend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"    # Phase 1 提供 pyproject；当前只需 python 3.10+
python tools/check_env.py   # 预期：NPU 项全为 pending，其余 ok
pytest tests/               # 全部单测无卡可跑
```

### 上板机（Atlas A2 / 910B 或 950）
```bash
python tools/check_env.py --full
# 必须全绿的项目：CANN 环境变量、npu-smi、torch_npu、triton-ascend、msprof、
# akg_agents 可 import（third_party/akg/akg_agents 布局，ADR-008/自审批次 B3）
```

NPU 架构检测注意：以 `ascendc-env-check` skill 的 `get_npu_arch.py`（asys/DSMI 链）为准，**npu-smi 的 Chip Name 不可信**。代际映射：DAV_2201（A2/A3/910B 系，UB 192KB）/ DAV_3510（950 系，UB 248KB）——config.yaml 的 `arch` 字段决定阈值表与知识路由分支。

## 2. 上游更新流程（deps/upstream.md 联动）

| 上游 | 用途 | 更新步骤 |
|---|---|---|
| mindspore-ai/akg（br_agents） | KernelVerifier 当库（ADR-002） | ①改 upstream.md 钉版 commit → ②重跑 akg_verifier_adapter 回归（tests/test_verify_protocol.py + RMSNorm 冒烟）→ ③通过才合并，CHANGELOG 记 deps |
| PolyArch/humanize | 循环行为规格参照（ADR-004） | 只影响规格：diff 其 regular-review.md / stop-hook 与本仓库 §4 契约，有实质变化时出 ADR 增补 |
| 本机 skill 生态（deps/skills.yaml） | 知识路由数据源 | 见 §3 |
| NVlabs/kda + 比赛仓库 | 方法论参照（只读） | 无需同步；教辅文档在 D:\PyProject\Jev\kda-contest-deep-dive\ |

**规则**：任何上游更新必须先过回归再进 main；不许"顺手升级"。

## 3. skill 依赖变更流程

1. `deps/skills.yaml` 登记条目（名字/用途/来源路径/版本或 commit/依赖它的路由条目）；
2. 若新增 skill 被 knowledge/router/index.yaml 引用：补三轴条目 + 一条路由验证（query.py 按症状能命中）；
3. 若删除/移动 skill：先 grep index.yaml 引用，盲区清单同步更新；
4. commit 前缀 `deps:`。

## 4. 知识资产治理（与代码同流程）

- prompt 模板/条款/索引条目的修改走正常 PR + `knowledge:` 前缀；
- 性能数字必须六字段（gpu/代际/dtype/shape/metric/value+出处），置信度三级（verified/team/inferred）标注；
- BitLesson 晋升流程：lessons/pending/ 中经一次 promote 审核的候选 → 月度提炼进 lessons/ 正式条目或升级为 guide（akg SkillEvolution 思路）；
- 盲区新增（发现 router 未覆盖的方向）→ blindspots.md 建条目，不静默跳过。

## 5. 版本与发布

- main + 短命特性分支；约定式 commit：feat/fix/docs/deps/knowledge/tests；
- 阶段验收打 tag：v0.1-mvp / v0.2-diagnosis / v0.3-knowledge / v1.0；tag 前置条件 = 该阶段验收标准全过（见规划文档 §八）；
- CHANGELOG 在每次 tag 时更新。

## 6. 故障排查速查

| 症状 | 先查 |
|---|---|
| gate 总是解析失败 | models.yaml 的 reviewer 档位；review prompt 是否被改动（应从 interaction-protocol.md §4.2 渲染） |
| verify 段错误/超时 | akg 钉版是否漂移；CANN 版本；bench/workload 是否过大（先缩 l0 集） |
| msprof 无输出 | mindstudio_profiler_output/ 路径权限；MC² 类算子必须用 msprof 而非 msprof op |
| promote 第③④项 fail | 是否有人手动改了 baseline/ 或 plan.md——这是熔断级事件，查 audit.log 定位 actor |
| 产线模式断点续跑 | run/state.json 完整性 + git log 每轮 commit；runner 从 last committed round 恢复 |

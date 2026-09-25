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

### 上板机（yq-e15：两跳 ssh jump → yq-e15，无外网）
```bash
# 一键供给（本地执行；详见 ADR-011 实战记录）：
python tools/provision_npu.py --plan        # 查看 wheel 清单
python tools/provision_npu.py --provision   # jump 中转下载 + e15 pip --user 离线装
# CANN toolkit（oepkgs 免登录直链，免 root 解包）：
#   jump: curl -O https://repo.oepkgs.net/ascend/cann/aarch64/Packages/Ascend-cann-toolkit-9.1.1-linux.aarch64.rpm
#   两跳传输后 e15: cd ~/kda-ascend && rpm2cpio cann-toolkit.rpm | cpio -idmv
#   env: source ~/kda-ascend/usr/local/Ascend/ascend-toolkit/set_env.sh（具体层级以解包结果为准）
# smoke: TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c "import torch,torch_npu;print(torch_npu.npu.is_available())"
```
版本配套（实测锁定）：torch 2.13.0+cpu（pytorch.org/whl/cpu aarch64）+ torch_npu 2.13.0rc1 + CANN 9.1.1；驱动 25.5.2 兼容。

NPU 架构检测注意：以 `ascendc-env-check` skill 的 `get_npu_arch.py`（asys/DSMI 链）为准，**npu-smi 的 Chip Name 不可信**。代际映射：DAV_2201（A2/A3/910B 系，UB 192KB）/ DAV_3510（950 系，UB 248KB）——config.yaml 的 `arch` 字段决定阈值表与知识路由分支。

## 2. 上游更新流程（deps/upstream.md 联动）

| 上游 | 用途 | 更新步骤 |
|---|---|---|
| mindspore-ai/akg（br_agents） | KernelVerifier 当库（ADR-002） | ①改 upstream.md 钉版 commit → ②重跑 akg_verifier_adapter 回归（tests/test_remote_sync.py（runner 协议） + RMSNorm 冒烟）→ ③通过才合并，CHANGELOG 记 deps |
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
- 阶段验收打 tag：v0.1-mvp / v0.2-diagnosis / v0.3-knowledge / v1.0；tag 前置条件 = 该阶段验收标准全过（见 docs/design/interaction-protocol.md §1 五批任务图（v0.2））；
- CHANGELOG 在每次 tag 时更新。

## 6. 故障排查速查

| 症状 | 先查 |
|---|---|
| gate 总是解析失败 | models.yaml 的 reviewer 档位；review prompt 是否被改动（应从 interaction-protocol.md §4.2 渲染） |
| verify 段错误/超时 | akg 钉版是否漂移；CANN 版本；bench/workload 是否过大（先缩 l0 集） |
| msprof 无输出 | mindstudio_profiler_output/ 路径权限；MC² 类算子必须用 msprof 而非 msprof op |
| promote 第③④项 fail | 是否有人手动改了 baseline/ 或 plan.md——这是熔断级事件，查 audit.log 定位 actor |
| 产线模式断点续跑 | run/state.json 完整性 + git log 每轮 commit；runner 从 last committed round 恢复 |

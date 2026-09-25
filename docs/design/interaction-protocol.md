# KDA-Ascend 交互协议规格

**版本**：v0.1（v0.0-scaffold 冻结；Phase 1 实现时若变更须 bump 版本并在 CHANGELOG 记录）

本文件是 harness 的对外接口宪法：CLI 命令表、文件契约、证据链 schema、gate 评审契约全文。agent、人、harness 三方都只认这里定义的接口。

---

## 1. CLI 命令表（agent ↔ harness 唯一通道）

约定：所有命令输出 JSON（stdout）+ 固定退出码；每次调用写一条 audit.log。退出码语义：`0` = 通过；`1` = 明确不通过（结果有效，不是错误）；`2` = 协议/环境错误（结果无效）。

| 命令 | 输入 | 输出（关键字段） | 退出码语义 |
|---|---|---|---|
| `kda verify --candidate <id> [--workload-set l0\|l1\|full]` | 候选 id | `passed, err_ratio, mismatches[≤10], workload_set` | 0 过 / 1 不过 / 2 harness 错 |
| `kda bench --candidate <id> --mode l0\|l1` | 候选 id | `mean_us, p50_us, p99_us, speedup, samples` | 同上 |
| `kda ab --a <base_id> --b <cand_id>` | 两候选 | 对称 A/B 报告（同 workload、同接口、交错采样） | 同上 |
| `kda diagnose --candidate <id>` | 候选 id | `bound, symptoms[], suggestions[3-5]`（各含 evidence 与预期收益） | 0 产出 / 2 失败 |
| `kda promote --candidate <id>` | 候选 id | 7 项门逐一 `pass/fail + reason` | 0 全过 / 1 有fail / 2 错 |
| `kda gate --round <N>` | 轮次 | 评审结论（见 §4） | 0 COMPLETE / 1 打回 / 2 解析失败 |
| `kda log [--tail N] [--actor X]` | 过滤 | audit.log 查询 | 0 |
| `kda status [--round]` | — | state.json 摘要（当前轮/模式/熔断状态/最佳候选/pause 状态/usage 累计）；`--round` 只输出当前 round 号（hooks 用） | 0 |
| `kda contract [--lock\|--unlock --reason R\|--verify]` | — | 任务契约展示 / 锁定（回填 baseline SHA）/ 校验 hash / 人工解锁修改 plan（audit 记 actor=human，解锁后必须重新锁定） | 0 / 1 已变 |
| `kda new-task <dir>` | 目标目录 | 从 tasks/_template 生成七件套 | 0 / 2 已存在 |
| `kda version` | — | 版本与协议文档定位 | 0 |

Phase 1 实现顺序：**contract → verify → bench → status → log → gate → promote → ab → diagnose → new-task**（contract 最先：gate 硬校验③④与 phase1 模板的"锁定后方可编辑"都依赖它）。

### 1a. 会话锁（run/lock，双模式互斥）

同一任务**同时只允许一个活动会话**（陪伴或产线）：
- 获取：会话启动时创建 `run/lock`（JSON：`{mode: companion|pipeline, pid, started_at, host}`），原子创建（O_EXCL）；
- 冲突：已存在且 pid 活着 → 拒绝启动（退出码 2，提示当前占用方）；pid 已死（陈旧锁，机器重启/进程被杀）→ 自动接管并记 audit；
- 释放：会话正常退出/checkpoint 暂停时删除；崩溃残留由下一次启动的陈旧锁判定清理；
- state.json 写方由此收窄：**持有锁的进程**（runner 或 gate）——锁是写权凭证，解决陪伴模式无 runner 时的写权归属（ADR-003 修订）。

### 1b. state.json 正式 schema（v1）

```json
{"schema": 1, "task": "rmsnorm-v1", "mode": "companion|pipeline", "round": 7,
 "best": {"candidate_id": "c003", "speedup": 1.38},
 "direction_fails": {"ub-fusion": 1, "vload": 3},
 "stall_count": 1, "last_verdict": "STALLED",
 "pause": null | {"reason": "paused-quota|paused-budget|paused-task-budget|paused-wallclock", "at": "...", "usage_snapshot": 12345678},
 "terminal": null | "MAXITER|FUSE_STALL|FUSE_DIRECTION|COMPLETE",
 "phase": "P1|P2|P3", "updated_at": "..."}
```
终态（terminal 非空）与可恢复暂停（pause 非空）互斥；`stall_count`/`direction_fails` 是熔断②③的计数载体。

## 2. 文件契约（人 / agent / harness 三方共享状态）

| 文件 | 所属层 | 写方 | 语义 |
|---|---|---|---|
| `task.yaml` | 任务根 | 人（agent 只读） | 任务契约 8 槽（见知识模板 contract-template.md） |
| `config.yaml` | 任务根 | 人 | workload 集、硬件代际、预算、DSL adapter 选择 |
| `baseline/` | 任务根 | 人首轮回填，随后锁定 | 官方基线实现；`kda contract --lock` 记录其内容 hash |
| `solution/<candidate_id>/` | 任务根 | agent（经 keep 后） | 候选代码；每候选一目录 |
| `bench/` | 任务根 | 人 | workload 定义（shape 域、dtype、代表集标记） |
| `docs/draft.md` | 任务根 | agent | 计划草稿（六要素）——draft 未写完不许编辑代码 |
| `docs/plan.md` | 任务根 | plan 生成流程，随后**hash 钉死** | AC 验收对契约；锁定后任何变更 = 熔断 |
| `docs/benchmark.csv` | 任务根 | 仅 harness（evidence.py） | 性能证据表（§3.1） |
| `docs/solutions.jsonl` | 任务根 | 仅 harness | 候选 DAG（§3.2） |
| `docs/audit.log` | 任务根 | 仅 harness（append-only） | 审计根（§3.4） |
| `docs/plan.md.lock` | 任务根 | 仅 harness | `{plan_sha, baseline_sha, locked_at, locked_by}` |
| `profile/<run>/` | 任务根 | harness | L2 诊断原始报告（gitignore）+ summary.json（入仓） |
| `run/state.json` | 任务根 | 仅 runner | 循环状态（round、模式、熔断计数、当前最佳）；gitignore |
| `run/round-N-*.md` | 任务根 | agent + gate | 本轮契约 / summary / review（gitignore，摘要入 evidence） |

**写方规则是反作弊地基**：evidence 三件套（csv/jsonl/audit）只有 harness 可写——agent 直接改这三个文件会在 promote 与 gate 双重校验中被抓（行完整性 + audit 缺条目）。

## 3. 证据链 schema（evidence.py 写出，promote.py 校验）

### 3.1 benchmark.csv（每性能候选一行）

```csv
ts,candidate_id,parent_id,phase,workload_set,mean_us,p50_us,p99_us,speedup,verdict,note
2026-10-02T14:03:11,c003,c001,P2,l1,412.5,409.1,431.0,1.38,keep,"ub-fusion+vload"
```

字段：`verdict ∈ {keep, revise, reject}`（全域统一词表：benchmark.csv 的 verdict 与 solutions.jsonl 的 status 取同一三值集合）；`workload_set ∈ {l0, l1, full}`；speedup 相对**锁定的 baseline**（不是相对上一候选——竞赛案例①教训）；候选间增量由 `kda ab` 写入 note 字段（结构化前缀 `delta_vs_parent=<us>`）。

### 3.2 solutions.jsonl（候选 DAG，每候选一行）

```json
{"candidate_id":"c003","parent_id":"c001","direction":"ub-fusion","hypothesis":"合并 norm+scale 两趟访存为一趟","diff_ref":"solution/c003/","evidence_refs":["benchmark.csv#c003","profile/run-004/summary.json"],"status":"keep","round":7,"fallback":false}
```

约束：`parent_id` 必须指向已存在候选（首候选 parent 为 null）；`status ∈ {keep, revise, reject}`（与 verdict 同词表）；promote 校验整链可溯到根；**被否决候选 status=reject 同样保留**（竞赛"否决留痕"经验）；回落基线分支记 `"direction":"fallback-baseline","status":"keep","fallback":true`（对应 fallback-baseline-legit 条款）。

### 3.3 profile/<run>/summary.json（L2 摘要，入仓）

```json
{"run":"run-004","candidate_id":"c003","arch":"dav_2201","metrics":{"mte2_ratio":0.87,"vec_ratio":0.31,"cube_ratio":0.02,"l2_hit":0.42},"bound":"MTE2_BOUND","top_stalls":["long_scoreboard_eq"],"suggestions":[{"fix":"UB 融合 + 128B 向量化加载","est_gain":"20-50%","evidence":"mte2_ratio=0.87 > 0.80","source":"ascendc-performance-best-practices#norm-family"}]}
```

军规：建议**最多 3-5 条**（超过 5 条每条贡献通常 <5%）；每条必须点名具体 metric 数值（禁"memory-bound"空话）；每条登记知识来源（可溯到 skill 条目）。

### 3.4 audit.log（append-only，JSON lines）

```json
{"ts":"...","actor":"agent","action":"kda verify","target":"c003","round":7,"detail":{"passed":true,"err_ratio":0.001}}
{"ts":"...","actor":"gate","action":"review","target":"round-7","round":7,"detail":{"verdict":"REVISE","acs":"4/6"}}
{"ts":"...","actor":"harness","action":"fuse","target":"direction=ub-fusion","round":9,"detail":{"reason":"3 consecutive failures"}}
```

`actor ∈ {agent, gate, human, harness}`。promote 第⑦项校验：不存在"verify 未通过但出现后续 bench"之类的越序条目。

## 4. gate 评审契约（全文规格，Humanize regular-review.md 移植 + 模型无关化）

### 4.1 输入（组装器渲染，评审模型只读）

任务契约摘要（8 槽）、docs/plan.md 全文（AC 列表）、本轮 round summary、本轮代码 diff、L0/L1（及 L2）证据、benchmark.csv 增量行、audit.log 本轮切片。

### 4.2 评审 prompt 固定结构（渲染进系统提示）

1. **逐 AC 核对**：对每条 AC，对照证据（不是听 agent 声明）输出 `AC-N: MET | PARTIAL | NOT MET` + 一句证据引用；
2. **延期禁令**：不得认可任何延期/跳过；发现未完成任务时，必须为其起草"单一、指令性、终局性"的完成计划；
3. **问题三车道**：所有发现分 `MAINLINE GAPS / BLOCKING / QUEUED` 三栏；
4. **强制裁决行**：输出恰好一行 `MAINLINE_VERDICT: ADVANCED | STALLED | REGRESSED`；
5. **统计行**：`ACS: <met>/<total> | FORGOTTEN: <n> | UNJUSTIFIED_DEFERRALS: <n>`；
6. **末行标记**：`COMPLETE`（全部任务完成 + 全部 AC 为 MET + 零延期零待办）、`REVISE`、`REJECT`，或全量审计轮的停滞熔断 `STOP`——只允许出现在输出**最后一行**。

### 4.3 解析与失败处理（确定性代码，非模型判断）

- 缺 `MAINLINE_VERDICT` 行、或末行标记不在 {COMPLETE, REVISE, REJECT, STOP}、或 ACS 行格式错、或 ACS 统计与末行矛盾（如 `ACS: 1/2` + `COMPLETE`）→ **解析失败**；
- 解析失败：重试 2 次 → 自动升模型档重试 1 次 → 仍失败则熔断停机等人工（**绝不放行**，"review cannot be skipped"）；
- `STOP` 末行 = 停滞熔断（同方向同反馈重复出现）——与解析失败严格区分：STOP 是合法评审结论、直接触发熔断③链路并记审计，**不进入重试**；
- 每 5 轮（round % 5 == 0）切换**全量对齐审计**：逐 AC MET/PARTIAL/NOT MET 表、遗忘任务检测（对照 plan 全任务列表）、停滞检测。

### 4.4 硬校验链（gate 前置，任一不过直接打回，不消耗评审调用）

对照 Humanize 硬校验链，昇腾化适配为 11 项（含 Humanize 的大文件检测移植）：
1. run/state.json 存在且 round 匹配；run/lock 持有有效（会话锁，见 §1a）；
2. git 分支 = task.yaml 声明的 `branch` 字段（默认 `task/<task_name>`），工作区干净；
3. plan.md hash 与 .lock 一致（变更即熔断，非打回）；
4. baseline/ hash 与 .lock 一致（同上）；
5. 本轮 round-N-contract.md 存在；
6. 本轮 round-N-summary.md 存在；
7. summary 含 BitLesson Delta 段（add/update/none 三值之一）；
8. 无未完成 todo 声明（summary 中显式清点）；
9. 本轮至少一条 benchmark.csv 增量（性能轮）或正确性证据（研究轮）——"性能轮"判定：本轮 round contract 含 direction 条目；否则为研究轮；
10. 已注入 skill 的承认记录存在（valuable_aspects + kernel_application 两条，见 §5.5）；
11. **大文件检测**（Humanize 移植）：solution/ 下本轮改动文件 >2000 行 → 打回并要求拆分（防巨型生成物逃过评审）。

## 5. 上下文组装协议（ctx/，摘要；完整实现规格在 harness/ctx/README.md）

1. **round 渲染**：round-N 注入 = 契约摘要 + plan 全文 + 当前最佳候选与证据指针 + 上轮评审全文 + 本轮契约模板 + 精选 BitLesson + 路由选中的 skill 切片；
2. **防过期**：state.json 的 round_id 之前的 summary/review 一律不进上下文；plan/baseline hash 每轮校验；
3. **token 预算**：超限先压操作历史（每轮保方向+结果+证据指针），再压 plan 分析；压缩产物落 run/compact-N.md 可审计；
4. **skill 三层注入**：L0 fundamental（DSL 硬约束，≤20k token 常驻）→ L1 按需切片（路由器选 3-5 个）→ L2 只给索引目录；
5. **承认闸门**：被注入 skill 必须在 draft/plan 提交 `valuable_aspects` + `kernel_application` 两条结构化承认，gate 硬校验第 10 项核对。

## 6. 熔断规格（fuse，四道）

| 熔断 | 阈值 | 动作 |
|---|---|---|
| 全局轮次 | round > 42（config.yaml 可调） | 终止，标记 MAXITER |
| 方向失败 | 同 direction 连续 3 次 reject | 强制换向（下一轮上下文注入"禁用该方向"） |
| 主线停滞 | MAINLINE_VERDICT 连续 2 轮 STALLED/REGRESSED | 强制重规划（重读 plan + 重写 round contract）；连续 3 轮 → 终止 |
| 预算 | API 调用数或墙钟超 config.yaml | 终止，保存断点（可续跑） |

所有熔断事件写 audit.log（actor=harness, action=fuse）并在 status 中可见。

## 7. Budget Guard：套餐防护与断点保存协议（v0.1 增补）

**原则：配额是可恢复的中断，不是崩溃。任何耗尽场景必须以"状态已保存、进程体面退出（码 0）"收场。**

### 7.1 三级预算与合成规则（models.yaml `budget` 段 + 任务 config.yaml `budget` 段）

| 级 | 参数 | 触发动作 | 状态值 |
|---|---|---|---|
| 套餐-软限 | `soft_limit_tokens`（默认 40M） | 每轮 reminder 注入 + status 告警；继续跑 | — |
| 套餐-硬限 | `hard_limit_tokens`（默认 60M） | checkpoint 暂停 | `paused-budget` |
| 配额错误 | HTTP 402（余额不足）/ 429（限流耗尽） | **立即停止一切重试** → checkpoint 暂停 | `paused-quota` |
| 任务-token | `token_budget`（任务级） | checkpoint 暂停 | `paused-task-budget` |
| 任务-墙钟 | `wall_clock_min` / `api_calls` | checkpoint 暂停 | `paused-wallclock` |

**合成规则**：任务级与套餐级**先到先停**（任一触发即暂停）；任务 `token_budget` 缺省时不设任务级上限（仅套餐级兜底）。所有暂停态都是**可恢复**的（区别于 MAXITER/FUSE_* 终态），恢复动作见 §7.3。

### 7.2 两级账本（本地记账，不依赖服务商账单）

- **全局账本**（套餐级判定读它）：`run-global/usage.jsonl`（仓库根，gitignored，跨任务累计）——每次调用追加 `{ts, task, role, model, prompt_tokens, completion_tokens, cumulative_global}`；
- **任务账本**（任务级判定与 status 展示）：`tasks/<task>/run/usage.jsonl`（同结构，`cumulative_task`）；
- GLM 返回的 `usage` 字段实测可用（2026-09-25 验证）；models.py 是唯一写方。

### 7.3 Checkpoint（断点保存）内容与恢复

- 写入：state.json 完整快照（当前 round、最佳候选 id+speedup、方向历史含否决记录、暂停原因 `paused-quota|paused-budget`、usage 累计）+ `git commit` 全部已 keep 候选（信息 `keep(c<id>): <speedup>x <direction>`）+ audit.log 一条 `action=checkpoint`；
- 退出：**码 0**（非崩溃）；`kda status` 显示 paused 状态与已耗 token；
- 恢复：配额恢复后重启 runner → 读 state.json → 以 git log 定位 last committed round → 从下一 round 续跑（上下文由 ctx 组装器按 compact 产物重建，不依赖内存）。

### 7.4 上下文窗超参

`context_window: 250000`（defaults 级，roles 可覆盖）；compact.py 触发线 = 0.8 × context_window（20% 余量防单轮爆窗）；压缩产物落 run/compact-N.md（可审计，不静默丢上下文）。

### 7.5 迭代版本控制（既有设计明文化）

每候选一目录 `solution/<candidate_id>/`；gate 判 keep 即 git commit；`solutions.jsonl` 父链 DAG；断点续跑以 git log 为准绳——**任何时刻中断，已完成的优化与证据链都不丢**。

## 8. 远程执行层（infra/remote，独立组件）引用

NPU 侧命令（verify/bench/diagnose 的跑数部分）可配置为远程执行：本地大脑 + 远程执行器（两跳 SSH：jump → yq-e15，实测 2026-09-25）。**解耦规则见 ADR-007**：通道实现在 `infra/remote/executor.py`（RemoteExecutor/LocalExecutor）；harness/core 经 Executor 协议依赖注入，core 不感知 SSH。密钥经 `infra/secrets/provider.py` 唯一入口读取。配置：任务 config.yaml `execution.remote` 段（Phase 1 随 push/pull 落地）。

### 8a. Executor 接口契约（与 §1 命令表同级效力的接口定义）

| 方法 | 输入 | 返回 | 错误语义 |
|---|---|---|---|
| `run(cmd, timeout_s=120)` | 命令串 | `{ok, rc, stdout, stderr, error}` | **任何失败不抛异常**：SSH 不可达/超时 → `{ok:False, rc:124/127, error:"link-timeout/..."}`；内层命令失败 → `rc≠0` + stderr 原文。timeout_s 是**链路级**预算（外层 timeout_s+30 兜底） |
| `push(local_paths)` | 本地路径列表 | `{ok, count}` | Phase 1（rsync 经跳板） |
| `pull(remote_paths)` | 远程路径列表 | `{ok, count}` | Phase 1；大文件留远程只拉摘要 |
| `probe()` | — | `{reachable, checks{...}, raw}` | 只读；checks 含 npu/cann_toolkit/torch_npu/triton |

workdir 语义：RemoteTarget.workdir 非空时 run() 自动 `cd <workdir> &&`（config.execution.remote.workdir 注入）。

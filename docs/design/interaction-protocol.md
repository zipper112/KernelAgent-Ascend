# KDA-Ascend 交互协议规格

**版本**：v0.2（v0.1 基础上的闭环性修正案：17 处矛盾消解 + 逻辑死胡同规格补全，2026-09-26；变更索引见 CHANGELOG「协议 v0.2」段）

本文件是 harness 的对外接口宪法：CLI 命令表、文件契约、证据链 schema、gate 评审契约全文。agent、人、harness 三方都只认这里定义的接口。

---

## 1. CLI 命令表（agent ↔ harness 唯一通道）

约定：所有命令输出 JSON（stdout）+ 固定退出码；每次调用写一条 audit.log。退出码语义：`0` = 通过；`1` = 明确不通过（结果有效，不是错误）；`2` = 协议/环境错误（结果无效）；`3` = STOP 终局停机（合法评审结论，非失败——仅 gate 使用）。

| 命令 | 输入 | 输出（关键字段） | 退出码语义 |
|---|---|---|---|
| `kda verify --candidate <id> [--workload-set l0\|l1\|full]` | 候选 id | `passed, err_ratio, mismatches[≤10], workload_set` | 0 过 / 1 不过 / 2 harness 错 |
| `kda bench --candidate <id> [--workload-set l0\|l1\|full]` | 候选 id | `mean_us, p50_us, p99_us, speedup, samples` | 同上（`--mode` 废弃，统一 `--workload-set`） |
| `kda ab --a <base_id> --b <cand_id>` | 两候选 | 对称 A/B 报告（同 workload、同接口、交错采样） | 同上 |
| `kda diagnose --candidate <id>` | 候选 id | `bound, symptoms[], suggestions[3-5]`（各含 evidence 与预期收益） | 0 产出 / 2 失败 |
| `kda promote --candidate <id>` | 候选 id | 8 项门逐一 `pass/fail + reason` | 0 全过 / 1 有fail / 2 错 |
| `kda gate --round <N>` | 轮次 | 评审结论（见 §4） | 0 COMPLETE / 1 REVISE·REJECT（打回）/ 2 解析失败（重试2+升档1后仍失败）/ **3 STOP（终局停机，锁删除、state.terminal=STOP）** |
| `kda log [--tail N] [--actor X]` | 过滤 | audit.log 查询 | 0 |
| `kda status [--round]` | — | state.json 摘要（当前轮/模式/熔断状态/最佳候选/pause 状态/usage 累计）；`--round` 只输出当前 round 号（hooks 用） | 0 |
| `kda contract [--lock\|--unlock --reason R\|--verify]` | — | 任务契约展示 / 锁定（回填 baseline SHA）/ 校验 hash / 人工解锁修改 plan（audit 记 actor=human，解锁后必须重新锁定）；**baseline/ 为空时 --lock 报错退出（码 2）**，不锁空 hash | 0 / 1 已变 / 2 baseline 缺失 |
| `kda new-task <dir>` | 目标目录 | 从 tasks/_template 生成七件套 + **run/state.json 初始化（round=0, mode 待首命令判定）+ 提示切任务分支** | 0 / 2 已存在 |
| `kda budget --report --tokens <N> --round <R> [--note X]` | 自报消耗 | 写入两级 usage.jsonl（actor=agent, source=self-report） | 0 / 2 参数错 |
| `kda export --candidate <id>` | 候选 id | 自包含交付包（代码+workloads+复验命令）+ 第三方复验指令 | 0 / 2 |
| `kda unlock --stale` | — | 陈旧锁（≥30 分钟）强制接管/删除（audit 记 actor=human|agent） | 0 / 2 锁活跃 |
| `kda version` | — | 版本与协议文档定位 | 0 |

**每个 kda 命令（含查询类）调用即写一条 audit.log**（§3.4）；audit 追加随最早那批命令（contract/verify/new-task）落地，不依赖 `kda log`。

Phase 1 实现顺序（v0.2 改为**五批任务图**，隐藏依赖显式化——原图只列命令导致 models.py/ctx/锁/evidence 全在表外）：

| 批 | 内容 | 为什么这个顺序 |
|---|---|---|
| A 地基 | `kda` console script、CLI 骨架（audit 追加+会话锁+state 读写）、new-task（state.json 引导）、version/status/log 最小读版 | 后续一切命令的地基；缺锁=双开双烧，缺 state=round 不推进 |
| B 证据链 | evidence.py（三件套唯一写方）、verify/bench 实装（成功返回即追加 csv/jsonl/audit；**verify 失败也写 jsonl status=reject, stage=verify 并计入 direction_fails**）、budget --report、contract --lock/--verify | 证据链是 gate 硬校验⑨与 promote 的输入；没有它每轮必打回（评审 token 白烧） |
| C 评审 | models.py（客户端+usage 记账+升档/fallback）、ctx/render.py、gate.py（硬校验 11 项+解析）、hooks 适配器 | B 的产出在此被消费；评审失败链（重试2+升档1+熔断）随 gate 落地 |
| D 长作业 | nohup+status 轮询（§8b）、ab/diagnose 实装、profile runner（msprof） | C 之后循环能转，长作业防断连丢结果 |
| E 收尾 | promote（8 项门）、export、漂移/方向熔断实装、复盘回流 | 收官门禁与交付 |

（批 A+B = 本仓库当前批次；C-E 见 roadmap。）

### 1a. 会话锁（run/lock，双模式互斥）

同一任务**同时只允许一个活动会话**（陪伴或产线）：
- **创建**：陪伴模式下由**首个 kda 命令创建**（无长驻进程——CLI 调用时原子创建 O_EXCL，内容 `{mode, host, created_at, last_seen}`，每次 CLI 调用刷新 last_seen）；产线模式由 runner 启动时创建；
- **瞬时持锁**：陪伴模式下每次 CLI 调用（含 gate）是瞬时持锁——调用期间校验并刷新锁，调用结束锁留在磁盘（非删除）；产线模式 runner 全程持锁；
- **冲突**：锁存在且 last_seen < 30 分钟（活跃）且 mode 不同 → 拒绝；last_seen ≥ 30 分钟（陈旧）→ 自动接管并记 audit；
- **释放**：**任何终局命令**（gate COMPLETE/STOP、熔断落 terminal 的 CLI 命令、checkpoint 暂停）删除锁并 audit 记录；REVISE/REJECT 不删锁（round+1 后循环继续）；手动 `kda unlock --stale` 兜底；
- **state.json 写方统一为：锁持有进程**（陪伴模式 = 瞬时持锁的 CLI 调用；产线模式 = runner）——§2 表与本条冲突时以本条为准（D1 修复）。**round 推进**：gate 判 REVISE/REJECT 后由 gate 调用自身将 state.json 的 round +1；COMPLETE/STOP 不推进。
- **state.json 引导（v0.2 定版）**：由 `kda new-task` 创建（`round:0, mode:"companion"` 缺省——首个 runner 启动时改写为 pipeline）；agent 无写权，若 state.json 缺失且任务目录已存在 → 所有 kda 命令码 2 报 `state-missing: run kda new-task or restore from git`，**不自动重建**（防静默重置计数）。

### 1b. state.json 正式 schema（v1）

```json
{"schema": 1, "task": "rmsnorm-v1", "mode": "companion|pipeline", "round": 7,
 "best": {"candidate_id": "c003", "speedup": 1.38},
 "direction_fails": {"ub-fusion": 1, "vload": 3},
 "stall_count": 1, "last_verdict": "STALLED",
 "pause": null | {"reason": "paused-quota|paused-budget|paused-task-budget|paused-wallclock", "at": "...", "usage_snapshot": 12345678},
 "terminal": null | "MAXITER|FUSE_STALL|COMPLETE|STOP",
 "phase": "P1|P2|P3", "updated_at": "..."}
```
终态（terminal 非空）与可恢复暂停（pause 非空）互斥；`stall_count`/`direction_fails` 是熔断②③的计数载体。**FUSE_DIRECTION 不在 terminal 枚举**（v0.2：方向熔断=强制换向，循环继续，非终态——§6 口径统一）。**budget 类耗尽全部归 pause**（§7.1），不入 terminal。

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
| `run/state.json` | 任务根 | 锁持有进程（§1a 定版：陪伴=瞬时持锁的 CLI，产线=runner） | 循环状态；gitignore 但断点例外见 .gitignore 例外段 |
| `docs/plan.md.lock` | 任务根 | 仅 harness | `{plan_sha, baseline_sha, locked_at, locked_by}` |
| `profile/<run>/` | 任务根 | harness | L2 诊断原始报告（gitignore）+ summary.json（入仓） |
| `docs/plan.md.lock` | 任务根 | 仅 harness | `{plan_sha, baseline_sha, locked_at, locked_by}` |
| `profile/<run>/` | 任务根 | harness | L2 诊断原始报告（gitignore）+ summary.json（入仓） |
| `run/round-N-*.md` | 任务根 | agent + gate | 本轮契约 / summary / review（gitignore，摘要入 evidence）；**round-N-review.md（评审原文）在解析前先落盘**——解析失败的那次评审 token 不沉没，供人工检视与降档分析 |

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

**候选 id 规则（v0.2）**：`c<三位顺序号>`（c001 起，单调递增，不跳号不复用）；id 由 agent 分配，**gate 硬校验⑤⑥执行时顺带校验唯一性与 parent 存在性**（撞名/跳号/悬空 parent = 码 2 打回）；revise 不开新目录（原地改同一 `<cid>`），只有新假设/新方向才开新 id——DAG 的"边"是假设演化，不是编辑历史。

**verify 失败也入链（v0.2，封死单方向无限烧）**：`kda verify` 未通过的候选**同样写一行 solutions.jsonl**（`status:"reject", stage:"verify", round`），并**计入 direction_fails**（该候选 round-N-contract 声明的 direction）——熔断②从"过了 bench 的 reject 才计数"改为"任何 reject 都计数"。verify 崩溃（码 2 harness 错）不计数不计链（环境问题归 infra，不是方向失败）。

**evidence 写入时机（v0.2）**：verify/bench 命令**成功返回即追加**（csv/jsonl/audit 三处一次写齐，不等 gate、不等 round 收口）——硬校验⑨的本轮增量由此保证；gate/promote 只读不写 evidence。

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
8. 无未完成 todo **无处置**——summary 允许列未完成项，但每项必须带处置标记（继续/换向/升级问人三选一，与 round-summary 模板一致；解析规则：`[ ]` 行缺处置标记 = 打回，有处置 = 放行）；
9. 本轮至少一条 benchmark.csv 增量（性能轮）或正确性证据（研究轮）——"性能轮"判定：本轮 round contract 含 direction 条目；否则为研究轮；
10. 已注入 skill 的承认记录存在（valuable_aspects + kernel_application 两条，见 §5.5）——**落点（v0.2）**：Phase 1 承认写 draft/plan（锁定前）；Phase 2/3 动态路由的新 skill 承认写**本轮 round-N-contract.md 的 skill-acknowledgment 段**（结构同两条），不改锁死的 plan；
11. **大文件检测**（Humanize 移植）：solution/ 下本轮改动文件 >2000 行 → 打回并要求拆分（防巨型生成物逃过评审）；
12. **预算自报在案（陪伴模式）**：本轮 `kda budget --report` 至少一条（§7.2a）——不自报不送审（防最大头消耗脱离账本）。

**REVISE 循环升级（v0.2，补 Humanize"卡住升级链"缺口）**：连续 3 次 REVISE 且主因相同（评审 issue 首条同类）→ gate 在第 3 次评审 prompt 注入升级指令：评审输出改为 STOP（终局停机等人工）或明确换向建议——不允许同一不可解 issue 空转到 42 轮上限。

## 4.5 promote 8 项门（v0.2 枚举提入宪法；此前只在 core/README）

1. 契约完整（task.yaml 8 槽非空、plan 锁有效）；
2. 全 workload 集 verify 通过（full set，含 NaN/Inf 边界与特殊行为项）；
3. speedup ≥ 契约 target 且相对**锁定 baseline**（非相对父候选）；
4. baseline 溯源有效（.lock 的 baseline_sha 与 baseline/ 实际内容一致）；
5. 证据链完整（benchmark.csv 有该候选行；solutions.jsonl 链可溯到根；audit 无越序）；
6. 噪声稳健（复跑 3 次变异系数 < 契约 noise_floor）；
7. 无越序操作（audit 校验：不存在 verify 未过但出现 bench 的条目）；
8. 轻量代码评审 AC（对最终 diff 的结构化评审，Phase 2 接 ascendc-code-review skill 前为 agent 自评+人复核）。

## 5. 上下文组装协议（ctx/，摘要；完整实现规格在 harness/ctx/README.md）

1. **round 渲染**：round-N 注入 = 契约摘要 + plan 全文 + 当前最佳候选与证据指针 + 上轮评审全文 + 本轮契约模板 + 精选 BitLesson + 路由选中的 skill 切片；
2. **防过期**：state.json 的 round_id 之前的 summary/review 一律不进上下文；plan/baseline hash 每轮校验；
3. **token 预算**：超限先压操作历史（每轮保方向+结果+证据指针），再压 plan 分析；压缩产物落 run/compact-N.md 可审计；
4. **skill 三层注入**：L0 fundamental（DSL 硬约束，≤20k token 常驻）→ L1 按需切片（路由器选 3-5 个）→ L2 只给索引目录；
5. **承认闸门**：被注入 skill 必须在 draft/plan 提交 `valuable_aspects` + `kernel_application` 两条结构化承认，gate 硬校验第 10 项核对。
6. **产线 writer 编辑协议（D2 定版）**：writer 模型不直接写仓库——输出结构化改动（unified diff 或整文件内容，JSON 包裹 `{candidate_id, files: [{path, diff|content}]}`），runner 校验路径白名单（仅 solution/<candidate_id>/ 内）后应用到新候选目录，随即进 L0 门；L0 不过 → 候选作废（reject 留痕），不污染父候选。陪伴模式无此协议（宿主 agent 自带编辑工具，受 hooks/CLI 门禁约束）。

## 6. 熔断规格（fuse，四道）

| 熔断 | 阈值 | 动作 |
|---|---|---|
| 全局轮次 | round > 42（config.yaml 可调） | 终止，标记 MAXITER |
| 方向失败 | 同 direction 连续 3 次 reject | 强制换向（下一轮上下文注入"禁用该方向"） |
| 主线停滞 | MAINLINE_VERDICT 连续 2 轮 STALLED/REGRESSED | 强制重规划（重读 plan + 重写 round contract）；连续 3 轮 → 终止 |
| 预算 | API 调用数或墙钟超 config.yaml | **checkpoint 暂停**（pause 态，非终态——v0.2 与 §7.1/§1b 口径统一），保存断点可续跑 |

所有熔断事件写 audit.log（actor=harness, action=fuse）并在 status 中可见。

## 7. Budget Guard：套餐防护与断点保存协议（v0.1 增补）

**原则：配额是可恢复的中断，不是崩溃。任何耗尽场景必须以"状态已保存、进程体面退出（码 0）"收场。**

### 7.1 多级预算与合成规则（models.yaml `budget` 段 + 任务 config.yaml `budget` 段）

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
- GLM 返回的 `usage` 字段实测可用（2026-09-25 验证）；models.py 是**API 调用**的唯一写方。

### 7.2a 陪伴模式自报通道（v0.2 增补，ADR-012）

陪伴模式下最大头消耗是宿主 agent 会话本身（不经 models.py，账本结构性盲区）。对策：**agent 每轮收工时自报**——`kda budget --report --tokens <N> --round <R>` 追加两级账本（`source:"self-report", actor:"agent"`）；数值来自宿主 API 返回的会话 usage（agent 可得）。自报不可强制精确，但 gate 硬校验⑫强制"不自报不送审"，保证覆盖率；软/硬限判定对自报值与 API 值**合并累计**（同一 cumulative 字段）。models.yaml `usage_ledger` 字段废弃单值口径，改为引用本节两级路径。

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

### 8b. Job 生命周期协议（本地为家、远端为镜，ADR-011）

```
本地 kda verify/bench → 组装 JobSpec{job_id,kind,candidate_id,files[],workloads,timeout_s,device_id}
  → push（tar-over-ssh 两跳：runner+候选代码+workload+job.json → ~/kda-ascend/<task>/payload/）
  → 远端 runner.py 执行（读 job.json，确定性计算）
  → pull（results/<job_id>.json → 本地）→ 本地 evidence.py 写证据链 → 本地 git commit
```

- **单一事实源在本地**：证据链/git/lock/DAG 只在 tasks/<task>/；远端 payload+results 是可再生镜像，可随时清理重建；
- **知识库与密钥永不离开本地**：远端只做确定性计算（verify/bench/profile）；
- JobSpec schema 与 push/pull 实现见 `infra/remote/sync.py`；远端执行器 `infra/remote/runner.py`（自包含，无本地依赖）；
- **执行模式（RemoteTarget.exec_mode，config.execution.remote）**：`docker`（e15 已验证路径：镜像自带可用 CANN 栈，设备直通 + payload/results/宿主驱动三挂载，入口 `infra/remote/container_entry.sh`；native 自装 CANN 8.5.alpha002 AICORE 全灭，见 ADR-011 §3a）/ `native`（source cann_env 后直跑，备用）；
- **候选与 oracle 定位（runner v0.1）**：候选两级回退 `candidate.py`（扁平）→ `solution/<candidate_id>/candidate.py`（仓内布局）；oracle 优先任务自带 `reference.py`（暴露 reference(inputs)），缺省回退内置 RMSNorm（正式任务必须自带）；
- **payload 文件清单组装规则（v0.2）**：harness（非 agent）组装 JobSpec.files，**必须包含** `solution/<cid>/candidate.py` + `reference.py`（任务根，oracle）；bench 作业另含 `baseline/` 内文件（锁定基线同机测速，speedup 口径=契约④）；清单缺 reference.py 且任务无 baseline → verify 可跑（内置 RMSNorm 回退）但 bench 返回 `has_reference:false`、无 speedup，evidence 记 `speedup:n/a`（硬校验⑨不因此打回，promote ④会拦）；
- 同步大超时执行；Phase 1 长任务换 nohup + status 文件轮询（防两跳断连丢作业）。

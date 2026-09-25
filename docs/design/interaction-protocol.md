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
| `kda status` | — | state.json 摘要（当前轮/模式/熔断状态/最佳候选） | 0 |
| `kda contract [--lock\|--verify]` | — | 任务契约展示 / 锁定（回填 baseline SHA）/ 校验 hash | 0 / 1 已变 |

Phase 1 实现顺序：verify → bench → status → log → gate → promote → ab → diagnose → contract。

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

字段：`verdict ∈ {keep, revise, reject}`；`workload_set ∈ {l0, l1, full}`；speedup 相对**锁定的 baseline**（不是相对上一候选——竞赛案例①教训）。

### 3.2 solutions.jsonl（候选 DAG，每候选一行）

```json
{"candidate_id":"c003","parent_id":"c001","direction":"ub-fusion","hypothesis":"合并 norm+scale 两趟访存为一趟","diff_ref":"solution/c003/","evidence_refs":["benchmark.csv#c003","profile/run-004/summary.json"],"status":"kept","round":7}
```

约束：`parent_id` 必须指向已存在候选（首候选 parent 为 null）；promote 校验整链可溯到根；**被否决候选 status=rejected 同样保留**（竞赛"否决留痕"经验）。

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
6. **COMPLETE 判据**：`COMPLETE` 仅允许出现在输出**最后一行**，条件 = 全部任务完成 + 全部 AC 为 MET + 零延期零待办；否则最后一行必须是 `REVISE` 或 `REJECT`。

### 4.3 解析与失败处理（确定性代码，非模型判断）

- 缺 `MAINLINE_VERDICT` 行、或 COMPLETE 不在末行、或 ACS 行格式错 → **解析失败**；
- 解析失败：重试 2 次 → 自动升模型档重试 1 次 → 仍失败则熔断停机等人工（**绝不放行**，"review cannot be skipped"）；
- 每 5 轮（round % 5 == 0）切换**全量对齐审计**：逐 AC MET/PARTIAL/NOT MET 表、遗忘任务检测（对照 plan 全任务列表）、停滞检测（同方向同反馈重复出现 → 输出 `STOP` 触发熔断）。

### 4.4 硬校验链（gate 前置，任一不过直接打回，不消耗评审调用）

对照 Humanize 9 项，昇腾化适配为 10 项：
1. run/state.json 存在且 round 匹配；
2. git 分支 = 任务约定分支，工作区干净；
3. plan.md hash 与 .lock 一致（变更即熔断，非打回）；
4. baseline/ hash 与 .lock 一致（同上）；
5. 本轮 round-N-contract.md 存在；
6. 本轮 round-N-summary.md 存在；
7. summary 含 BitLesson Delta 段（add/update/none 三值之一）；
8. 无未完成 todo 声明（summary 中显式清点）；
9. 本轮至少一条 benchmark.csv 增量（性能轮）或正确性证据（研究轮）；
10. 已注入 skill 的承认记录存在（valuable_aspects + kernel_application 两条，见 §5.5）。

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

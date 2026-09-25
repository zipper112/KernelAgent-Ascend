# ADR-012: 陪伴模式预算自报通道（budget self-report）

- 状态：已接受
- 日期：2026-09-26

## 背景

闭环性审查（对照 KDA/Humanize/竞赛）发现的盲区：陪伴模式下最大头的 token 消耗是**宿主 agent 会话本身**（ZCode/Claude），不经 `harness/models.py`——Budget Guard 的两级账本只记得到 reviewer/aux 的零头，软/硬限（40M/60M）与 402/429 防护对真实大头结构性失明（协议 §7 原设计只覆盖产线模式）。

用户决策（2026-09-26）：采用自报通道方案。

## 决策

1. **`kda budget --report --tokens <N> --round <R>`**（agent 调用）：把宿主会话本轮 usage 追加进两级账本（`role: host-agent, model: self-report, source: self-report`）——数值来自宿主 API 返回的会话 usage（agent 可得，非估拍）；
2. **覆盖率保证**：gate 硬校验第 12 项（协议 §4.4 v0.2）——本轮无 budget-report 条目不送审；agent 偷懒的代价是确定性打回（零评审成本），比"完全看不见"严格优；
3. **判定合并**：软/硬限判定对自报值与 API 值合并累计（同一 cumulative 字段）；
4. **诚实性边界（明示）**：自报数值本身不可强制精确（agent 报小了无法本地证伪）——这是陪伴模式的结构性上限。兜底：①GLM 端点 402 时无论账本多少都会自然停（服务侧硬限）；②产线模式全量经 models.py 无此盲区；③status 展示自报占比，异常低（如每轮恒 0）本身可被 gate 评审质疑。

## 后果

- 协议 §7.2a 增补、§4.4 硬校验扩为 12 项、models.yaml `usage_ledger` 单值口径废弃（改引用两级路径）；
- CLI `kda budget --report` 已实装（本批），任务/全局账本各追加一行 + audit 记条；
- tests/test_harness_loop.py::test_cli_budget_report_writes_ledgers 锁契约。

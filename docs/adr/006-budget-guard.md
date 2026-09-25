# ADR-006: Budget Guard——套餐防护与断点保存协议

- 状态：已接受
- 日期：2026-09-25

## 背景

模型走个人套餐（GLM coding 套餐），用户两条硬要求：①不能"两三下把套餐干完"；②配额耗尽时必须保存工作状态体面退出，不许崩溃丢进度。另：上下文窗要求 ≤250k 且做成超参。

## 决策

1. **三级预算**：套餐软限（40M tokens，告警不停止）/ 套餐硬限（60M，checkpoint 暂停）/ 任务级（api_calls、墙钟、token_budget，熔断④）。参数在 models.yaml `budget` 段与任务 config.yaml，可调；
2. **本地记账**：models.py 每次调用把 usage 追加 `run/usage.jsonl`（GLM 的 usage 字段实测可用，2026-09-25 验证）；判定读累计值，不依赖服务商账单；
3. **配额错误协议**：HTTP 402/429 → 立即停止一切重试（重试只烧钱或无效）→ checkpoint（state.json 快照 + git commit 已 keep 候选 + audit 记录）→ 退出码 0；
4. **断点恢复**：runner 重启读 state.json，以 git log 定位 last committed round 续跑；上下文由 compact 产物重建；
5. **上下文窗超参**：`context_window: 250000`（defaults 级）；compact 触发线 = 0.8×（20% 余量）；
6. **单调用限流**：`max_tokens_per_call: 8192`。

完整协议文本：docs/design/interaction-protocol.md §7。

## 理由

配额耗尽是可恢复中断而非错误；把"暂停-保存-恢复"做成协议而非 try/except 补丁，才经得起产线模式长跑。

## 后果

- models.py 实现时必须先落 usage 记账（它同时是软/硬限与 status 展示的数据源）；
- git log 成为断点的事实源——keep 即 commit 的纪律必须严格执行（已在 §7.5 明文化）。

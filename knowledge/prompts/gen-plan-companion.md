# 计划生成协议（companion 模式，gen-plan-companion.md）

> 用途：陪伴模式下 draft.md → plan.md 的对抗收敛流程。产线模式由 runner 内置简化版（单 reviewer 分析 + 单轮修订）承载，见 ADR-010。
> 移植自 Humanize gen-plan 十阶段中与模型无关可提示词化的五件；其余（IO 校验/相关性检查）由 `kda contract --lock` 的确定性检查承担。

## 流程（在任务工作区内执行）

1. **writer 写 draft**：按 contract-template.md 的六要素写 `docs/draft.md`；
2. **reviewer 首轮分析**：把 draft + 任务契约交给 reviewer 角色（独立会话/新对话，避免自我评审），要求固定六段输出：
   - `CORE_RISKS`（核心风险，按杀伤力排序）
   - `MISSING_REQUIREMENTS`（缺失需求）
   - `TECHNICAL_GAPS`（技术缺口）
   - `ALTERNATIVE_DIRECTIONS`（替代方向）
   - `QUESTIONS_FOR_USER`（给用户的问题——必须回答后才能继续）
   - `CANDIDATE_CRITERIA`（候选验收标准——未来 plan 的 AC 素材）
3. **writer 修订 draft**：逐条吸收或反驳（反驳须给理由）；
4. **收敛循环 ≤3 轮**：重复 2-3 直到 reviewer 输出无 CORE_RISKS/MISSING_REQUIREMENTS 级别的必需修改，或连续两轮无实质变化；
5. **量化数字确认（强制问人）**：draft 中出现任何量化目标（"加速 ≥1.5×""延迟 <100μs""带宽 >15GB/s"）→ 必须向用户确认是**硬性要求**还是**优化方向**——直接决定 AC 写成"必须达到"还是"尽量接近"；
6. **writer 产出 plan.md**：
   - **计划是 draft 的超集**（draft 全文附尾，不许丢弃 reviewer 已确认的内容）；
   - AC 验收对：每条 AC 配 Positive Test（应 PASS）+ Negative Test（应 FAIL，至少一条针对本算子已知数值陷阱，如全 0 输入必判 FAIL）；
   - 任务表：每任务标 coding（writer 做）或 analyze（转 reviewer 分析）+ 目标 AC + 依赖；
7. **锁定**：`kda contract --lock`（hash 钉死 plan 与 baseline，此后任何变更走 `--unlock --reason` 人工通道）。

## 降级说明

宿主 agent 不方便开独立会话时，2-4 可退化为"同会话内换角色重述+自检清单"，但 5（问人）与 6（超集+AC 双向测试）**不可降级**。

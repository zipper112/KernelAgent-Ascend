# ADR-005: CANNBot 知识资产接入——license 审查通过前不引入

- 状态：待定（Phase 3 出结论）
- 日期：2026-09-25

## 背景

华为 CANN 社区的 CANNBot 提供 5040 张知识卡（API 2993 / Guide 465 / Operator 611 / Runbook 565）与 cann-bench 评测场（L1 元素级 → L4 FlashAttention/MoE，HAP 硬件锚定指标），是目标端最厚的官方知识底座。但其 License 为 **CANN OSL（非标准 OSI 协议）**，且正仓在 GitCode（`cann` org），GitHub 上有个人镜像（易误引）。

## 决策

1. Phase 3 之前**不引入** CANNBot 知识卡实体与 cann-bench 数据；
2. Phase 3 完成 CANN OSL 条款审查（重点：商用/内部分发/衍生数据集条款），结论落回本 ADR；
3. 若许可可接受：只引入知识卡元数据与切片做路由索引的补充条目（不入仓实体，登记来源与版本于 `deps/skills.yaml`）；cann-bench 作为 workload 来源评估；
4. 若许可不可接受：维持本机 skill 编目 + akg 内置 skill 的知识面，缺口记录进 `knowledge/router/blindspots.md`。

## 理由

长期维护项目不能带 license 隐患；工具筛选笔记（用户维护）已记录"中国厂商正仓可能在 GitCode/Gitee、GitHub star 不代表热度、厂商口径性能数字需独立复现"的教训。

## 后果

- 短期知识面少一块官方权威源（盲区清单兜底）；
- Phase 3 需排一次法务/条款阅读工作（约半天）。

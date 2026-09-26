# K8-Triton 任务判定（第一战役收束）—— 2026-09-26

## 结果

| 版本 | mean_us | 状态 |
|---|---|---|
| c001 vec 基线（torch 拼装） | 1094.2 | 基线 |
| **c003 Triton r7-r9 迭代** | **646.9 → 621.9 → 606.0（-44.7%）** | **实测在案（benchmark.csv 三行）但代码被后续轮覆盖丢失** |
| c016/c017 冲刺复现 | verify 未过 | 未复现 |

**判定：inconclusive-but-promising**。Triton 单 kernel 路线的 -44.7% 加速被三轮连续 bench
证实过（p99 紧贴 p50、无 JIT 拖尾），但最优代码因两个循环缺陷（thinking 吃预算 → 空转轮
堆积 + cid 复用覆盖）未能保全。两缺陷均已永久修复；复现路径明确（writer 拿本档 verdict
作为 feedback 重写）。

## 循环缺陷修复账（本任务暴露 6 项，全部永久修复）

1. **GLM-5.3 推理模型思维链吃光预算**（finish=length 且 content 空、16384 全在
   reasoning_content）→ chat() 默认 thinking=disabled；空内容抛诊断；
2. JSON+16k 截断死循环（Triton 代码量大 × 转义+20%）→ 分隔符协议（===CODE=== 零转义）
   + DSL 感知接口约定 + 250 行约束；
3. preflight triton 盲区（本地无 triton 误拒正确 kernel）→ triton 桩降级检查；
4. write-parse-fail 炸循环 → 审计+REVISE 占位（循环不再死）；
5. **cid 复用覆盖最优版本**（REVISE 换向仍写同目录）→ 换向必须新 id；
6. best 基准被 bench-auto 0.0 行污染（c004 空目录 bench 记 0 变"历史最优"）→
   reviewer 参照系应忽略无效行（记待修：_best_baseline_us 过滤 mean_us<=0）。

## agent 行为评估（监督账）

- 循环自主迭代真实有效：r7-r9 在 REVISE 反馈下 646.9→621.9→606.0 连续刷新（参照系
  修复生效，每轮 beat_pct 精确注入下一轮 prompt）；
- Triton kernel 结构水平高：单 launch、寄存器窗口滑移、flat d-tile 循环、pad 语义
  mask 化——与我们的设计意图一致；
- 短板依旧：w03 大 shape 正确性边缘失败未能自愈（Triton mask 边界 bug，REVISE 反馈
  修复尝试两轮未果）。

## 下一步（待派期）

1. 冲刺复现：新循环（6 缺陷修复后）拿本 verdict 作冷启动 feedback，目标 606us 复现
   并突破 <600（差距 1%）；
2. 复现后：E067 交付（causal_conv1d_update_npu 符号）+ serving A/B 终验；
3. 待修第 6 项（best 过滤）半小时活。

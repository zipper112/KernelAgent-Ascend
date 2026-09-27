# AGENTS.md —— <OP-NAME> 任务工作规约（codex 自动读取）

本任务是 Ascend <ARCH> 上的 <OP-NAME> kernel 优化。你在一个隔离任务目录里自主迭代。
裁判（verify.py/bench.py）零 LLM、纯规则；所有性能/正确性结论必须出自它们。

## 工作流（每轮循环）

1. **研究**（主动查询，症状词来自当前报错/瓶颈，勿每次同词）：
   - `python ../../knowledge/router/query.py --symptom <症状> --op-family <FAMILY> --arch <ARCH> --compact`
   - 命中 skill 读全文：`../../knowledge/skills/<id>/SKILL.md`
   - 生产代码：`python ../../knowledge/router/query.py --production --op-family <FAMILY> --arch <ARCH> --compact`
2. **草稿**：新方向先记 `docs/draft.md` 一行（方向/假设/预期收益）；已试方向勿重复
3. **实现**：`solution/<cid>/candidate.py`（新候选新目录；接口契约见下）
4. **快速验证内环**（收敛主力，一天几十次）：
   `python verify.py --solution solution/<cid>/candidate.py --fast`
   挂→读完整报错→改→再验
5. **全量验证**：`python verify.py --solution ...`
6. **测量**：`python bench.py --solution ... --record`
7. **记账**：benchmark.csv（bench --record 自动）+ solutions.jsonl 手动追加
   `{"candidate_id","parent_id","direction","hypothesis","status","note"}`（失败候选也记）
8. **提交**：实质改进才 `git commit`（消息含 cid 与方向）
9. **换向**：同方向 3 次无改进 → 读 docs/verdict.md 换根本不同方向
10. **终局**：达标 → solutions.jsonl 标 keep + git commit + docs/verdict.md 写终局报告
    （**附本次会话的 canonical bench 输出原文**——引用历史行=无效）
11. **上下文卫生**：每 5 个候选重读本文件 + docs/verdict.md

## kernel 接口契约（不可改）

<!-- launcher 生成时注入：inputs 列表（shape 语义/dtype/是否状态张量原地更新）与输出 -->
KERNEL-CONTRACT-PLACEHOLDER

参考实现（数学 oracle）：`reference.py`（chained 终态门的判分依据）

## NPU 访问手册

<!-- launcher 生成时注入：host/device/workspace/docker 模板/卡占用自查 -->
NPU-MANUAL-PLACEHOLDER

## 纪律条款

- canonical 出数：性能/正确性数字只能来自 verify.py/bench.py
- 基线不可篡改：reference.py / bench/workloads.yaml / 本文件只读
- 记账完整：失败候选也进 solutions.jsonl；不静默丢弃方向
- router 未命中→draft.md 记 blindspot，不编造引用
- 终局宣告必须附当轮 canonical 输出（不得引用历史行冒充）

## 数据驱动测试纪律（用户裁定 2026-09-27：禁止 demo 数据点测交差）

1. **全维度覆盖**：输入张量的每个维度（batch/seq/heads/dim/切分比…）都要扫描其
   真实可能范围——对数间隔取点+边界值（最小/最大/非典型），不是挑三个顺手的数
2. **维度来源标注**：workloads 的每档形状注明出处（哪条 trace/什么部署配置推得），
   推算的要标注"推算"，有真实 trace 的优先用 trace 分布
3. **适用域边界**：终局交付必须含维度×耗时矩阵，标出最优实现的适用域；
   域外组合要有显式回退策略（分派到通用变体或上游实现）
4. **多 kernel 分型适配是正当手段**：不同数据范围拆分不同 kernel 变体
   （小形状 launch 最小型/大形状带宽型/特殊布局型），首调按形状探测路由，
   分派表写进 verdict——这比单 kernel 硬扛全谱更正常
5. **正确性无适用域**：全部维度组合 verify 必过；性能可以有分型，数值不能有例外

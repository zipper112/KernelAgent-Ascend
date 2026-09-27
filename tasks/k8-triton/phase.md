# Phase 任务书：K8 causal_conv1d_update Triton kernel（战役 5·宿主化首战）

你是 Ascend NPU kernel 优化 agent。本任务目录即你的全部工作区。
**先读同目录 AGENTS.md**（工作流 11 步+接口契约+NPU 手册+纪律），再开始。

## 目标

目标以 **w02（B=16，decode 主形状）** 为准：**w02 mean_us < 600**（vec 基线 1102）。
注意：w01（B=1）单卡小形状数字不作数——B=1 快而 B=16 不快说明并没消除每批量派发/访问开销。
（参照：vec 版基线 1102μs；battle1 曾实测 606μs——该结构真实可达）

## 已试方向账本（详见 docs/verdict.md，勿重复）

- launch 开销路线（constexpr-elide/stream-pinned/direct-launch/neverraise）：天花板 ~980μs，已系统性探索完毕
- depthwise-conv/persistent40/flat-tile+idxgather：实测否决，禁回锅

## 首推方向（battle1 实证 606μs 的结构，优先尝试）

fused-single-launch + 寄存器滑窗 + 静态 SEQ 展开：
1. 单 kernel 单 launch：读 3 槽状态→4-tap 窗卷积→silu→写新状态+输出一次完成
2. (r0,r1,r2) 三寄存器持滑窗逐 token 右滑——零中间张量、零 gather/scatter
3. SEQ 作 tl.constexpr 静态展开（L=1 或 4）去运行期循环依赖
4. grid=(B, ceil(D/BLOCK))；BLOCK 按核数粒度结论取（40 核满载，64 程序+尾波可接受）
5. 严禁 fallback/neverraise 分支结构（实测反慢 30%）

## 工作节奏（模仿竞赛）

研究（主动查 router/skills，症状词来自当前报错）→ 内环：
`python verify.py --solution solution/<cid>/candidate.py --fast`
读完整报错→改→再验（**几十次内环才是收敛主力**）→ 全量 verify →
`python bench.py --solution ... --record` → 记 solutions.jsonl → 有改进才 git commit → 换向。

## 完成判据

dev 档 mean_us < 978.7 且 full 档不退化 → solutions.jsonl 标 keep + git commit；
达到 < 600 → 在 docs/verdict.md 写终局报告（结构说明+关键数字+证据行引用）。

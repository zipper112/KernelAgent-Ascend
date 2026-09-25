# ADR-001: DSL 路线——Triton-Ascend 起步，adapter 架构扩展

- 状态：已接受
- 日期：2026-09-25
- 决策人：项目组

## 背景

昇腾算子开发有四条主流 DSL 路线：AscendC（贴 CANN 底层）、Triton-Ascend（triton-ascend 后端）、PyPTO（torch_npu 原生自定义算子）、TileLang-Ascend（NPU 后端 tile 级 DSL）。首版工具必须选一条主路线，同时不能把未来路线堵死。

性能差异的事实基础：内存受限算子（RMSNorm/LayerNorm/Elementwise 类）瓶颈在数据搬运，四条 DSL 上限差距小（<10%，都在拼带宽利用率）；计算受限算子（MatMul/Attention 类）差距会拉开——AscendC 贴硬件指令上限最高，Triton-Ascend 通常到手写的大部分性能，但开发速度快一个量级。

各路线的配套资产（调研结论，2026-09）：
- Triton-Ascend：akg_agents 内置 89 skill 中资产最大（7 基础 + 5 指南 + 23 案例）；本机 triton-op-coding / triton-op-designer 两个 skill 专用；KernelVerifier 支持最成熟（triton_ascend adapter + autotune 回读 + L2 专用清除 kernel）；
- AscendC：本机 skill 覆盖最全（api/perf/tiling/blaze/regbase/mc2 等十余个），但开发链路长（工程模板 + 编译 + aclnn 注册），循环迭代速度慢；
- PyPTO / TileLang-Ascend：各有全套本机 skill 与 KernelVerifier adapter，生态较新。

## 决策

1. 首版主路线 = **Triton-Ascend**；
2. harness 的验证与测量层按 **adapter 架构**组织（对齐 akg KernelVerifier 的 dsl adapter 体系），后续加 AscendC / PyPTO / TileLang-Ascend 路线时只加 adapter + 知识路由分支，不动循环与证据链；
3. 计算受限算子的深度优化（AscendC + blaze/regbase 知识链）列为 Phase 4 扩展。

## 理由

试点算子 RMSNorm/LayerNorm 是内存受限型，DSL 上限差距不影响验证结论；Triton-Ascend 的知识资产与验证成熟度最高，MVP 迭代最快。

## 后果

- 正面：MVP 周期短；adapter 边界从第一天就强制清晰；
- 负面：计算受限算子用户在 v1.0 前只能走 Triton 路线；接受（Phase 4 补 AscendC adapter）。

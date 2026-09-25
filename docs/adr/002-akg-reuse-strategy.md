# ADR-002: akg_agents 复用方式——KernelVerifier 当库 import，不跑全框架

- 状态：已接受
- 日期：2026-09-25

## 背景

mindspore-ai/akg（br_agents 分支，`akg_agents/`）提供了昇腾侧最成熟的算子验证评测设施：KernelVerifier（代码生成式验证：dtype 分级容差 + NaN/Inf 位置匹配 + 错误比例容忍）、计时三件套（torch_npu.profiler 封装 / msprof op_summary 解析 / L2 cache 清除）、Worker/DevicePool、UCB 自适应搜索、进化算法、AutoResearch 循环。但它是 LangGraph + Server/Worker/DB 全家桶的自研 Agent 框架，绑定自有 AgentLoop。

## 决策

**当库 import，不跑全框架**：
1. Phase 0 直接以 `examples/kernel_related/run_kernel_profile.py` 展示的最小调用面（`register_local_worker` + `KernelVerifier.run()/run_profile()`）作为验证入口；
2. 本项目 `harness/core/` 外包一层 `akg_verifier_adapter`（薄封装，隔离上游 API 变动）；
3. 容差方案（fp16 0.004 / bf16 0.03 / int8 0.01 / 其他 0.02 + 四步比较协议）作为本项目正确性门的**规格来源**——Phase 2 若 import 面不稳，按同一规格内嵌实现（规格不变，实现可换）；
4. `adaptive_search()` / `evolve()` 的批量搜索能力 Phase 4 以库方式接入；
5. AutoResearch 作为产线模式循环的**同构参照物**（借设计不借代码）：plan 状态机、acknowledge_skill 闸门、auto_compact 双摘要、diagnose subagent。

## 理由

进程边界与 import 面最小化：核心验证链路 import 面很小（run_kernel_profile.py 已证明）；跑全框架则与本项目自研 control 层冲突且维护成本高。

## 后果

- 上游 akg 升级可能动 adapter——钉版 commit + 回归步骤见 `deps/upstream.md`；
- KernelVerifier 的强 NPU 依赖意味着无卡机上 harness 的 verify/bench 只能 dry-run（tests 全部设计为无卡可跑）。

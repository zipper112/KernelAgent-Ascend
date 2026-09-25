"""infra/ —— 基础设施组件（非核心，独立解耦）。

职责边界（ADR-007）：
- 这里放"环境类"能力：远程执行通道、密钥管理、（未来）通知/打包等；
- **禁止**：任何算子验证/测量/评审/证据链逻辑——那些属于 harness/core 与 harness/control；
- 依赖方向：harness → infra（infra 不 import harness）；infra 不 import 任何 LLM 库。

对 core 的接入方式：依赖注入。core 的 verify/bench 需要"在哪里跑"时，
接收一个满足 Executor 协议（run/push/pull/probe 四方法）的对象，
本地跑就是 LocalExecutor（空实现），远程跑就是 infra.remote.RemoteExecutor。
core 永远不知道 SSH 存在。
"""

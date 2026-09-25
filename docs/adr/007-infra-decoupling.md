# ADR-007: infra/ 独立基础设施组件——远程执行与密钥管理解耦

- 状态：已接受
- 日期：2026-09-25

## 背景

用户要求：远程访问、密钥管理这类**非核心**能力必须独立组件化，不污染 harness 核心。此前一版把 remote.py 放进了 harness/core/——违反 core 铁律（core 是零 LLM 的算子验证测量核心，SSH 通道与它无关）。

## 决策

1. 新建顶层组件 `infra/`，与 harness/knowledge 平级，内分：
   - `infra/remote/`：远程执行通道（RemoteExecutor 两跳 SSH + LocalExecutor 空实现）；
   - `infra/secrets/`：密钥管理（唯一读 key 的代码；SecretRef 防日志误打印）；
2. **Executor 协议注入**：harness/core 的 verify/bench 接收满足 `run/push/pull/probe` 四方法的对象——本地跑注入 LocalExecutor，远程跑注入 RemoteExecutor；**core 永远不知道 SSH 存在**；
3. 依赖方向铁律：`harness → infra`（单向）；infra 不 import harness、不 import 任何 LLM 库；核心逻辑（验证/测量/评审/证据链）永远不进 infra；
4. 密钥只存在两处：环境变量（优先）或 agent-config/local-secrets.yaml（gitignored，.gitignore `agent-config/local*.yaml` 覆盖）；models.yaml 只存端点与角色映射，不存 key；
5. 远程通道形态（实测 2026-09-25）：两跳 `ssh jump → ssh yq-e15`（8×910B4）；不引入额外网络组件；单跳 ProxyJump 直连留作延迟优化备选。

## 理由

核心/环境能力混装会让 core 无法在无网/无卡机回归测试，也让远程通道的变更（换机器、加代理）波及验证逻辑。依赖注入是两者唯一的耦合面。

## 后果

- check_env.py 的 --remote 探测改调 infra.remote（不进 core）；
- 未来通知/打包/对象存储等环境类能力一律进 infra，新增子目录走 ADR。

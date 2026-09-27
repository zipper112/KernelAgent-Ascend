# ADR-014: 比赛形态重构——Codex 宿主 + 瘦裁判，废除自研迭代引擎

日期：2026-09-27 · 状态：已接受 · 关联：ADR-011（远端通道，退役）、ADR-013（上下文注入，部分保留）

## 背景与动机

战役 3（17 轮自主迭代）暴露的结构性事实：

1. **17/17 个 bug 全部在自研引擎层**（harness/control+models+context ≈2529 行：LLM 客户端、
   输出协议解析、ssh 编排、超时重试、状态机断言），0 个在领域层（runner 数学/测量协议/
   router/skills 从未出错）。
2. 三源（KDA-Pilot/比赛/Humanize）的 agent 引擎全部**租宿主**（Claude Code/Codex CLI），
   自研代码只做四件事：任务书（phase prompt）、裁判（verify.py 式纯规则）、账本约定、
   知识资产。宿主免费提供的轮内小循环（秒级 verify→读全量报错→增量 Edit→再验）正是
   收敛主力——自研引擎每轮 10 分钟开环单发，信息量撑不起收敛（战役 3 最优 978.7μs
   vs 目标 600μs，且 17 轮内 5 个候选在 980-1270μs 波动无收敛趋势）。
3. 用户裁定：照比赛形态"照葫芦画瓢"，不自创架构。

## 决策

### 新架构（比赛形态）

```
phase.md（任务书）─┐
AGENTS.md（工作规约，codex 自动读取）─┤→ codex exec（宿主=引擎，GLM 驱动）
verify.py / bench.py（纯规则裁判）◀─┤    ├─ 轮内小循环：verify --fast → 读报错 → Edit → 再验（秒级，收敛主力）
docs/ 三件套（账本）◀───────────────┘    └─ git commit（keep 才提交）
knowledge/（router+skills，agent 主动查询）
infra/remote/runner.py（canonical 测量口径，不变）
```

### 模型接入（实测 2026-09-27）

- codex-cli 0.157.1（jump:/usr/local/bin/codex）
- GLM **Responses API**：`https://open.bigmodel.cn/api/v1/responses`
  （注意：不是 chat 的 paas/v4；文档 docs.bigmodel.cn/cn/guide/develop/responses/introduction）
- `~/.codex/config.toml`：model_provider=glm，wire_api="responses"，env_key=GLM_API_KEY
- 上下文 300k（用户裁定）写入 model_metadata
- 沙箱：`-s danger-full-access`（workspace-write 禁网，无法 ssh e15——实测）

### 保留资产

| 资产 | 说明 |
|---|---|
| knowledge/（router/skills/prompts） | 全留；从"推送"改"agent 主动查询" |
| infra/remote/runner.py + container_entry.sh | canonical 测量口径（chained 门/L2/交错采样），零改动 |
| 证据三件套 schema + tools/supervise.py | 轮末外部审计 |
| tasks/ 档案 + verdict.md 脉络 | 跨战役记忆 |
| agent-config/（key/参考配置） | key 供 codex env |

### 废除（attic/engine-v0/，git 历史保留）

harness/{control/loop.py, control/exec_policy.py, control/memory.py, models.py,
context.py, cli.py} + 对应测试。README 标注废除原因。

## 后果

- 引擎层 bug 面 = 0（引擎不存在）；LLM I/O/重试/上下文/文件编辑全部由宿主承担
- 轮内小循环回归（比赛形态的收敛主力）；报错全文自然进 agent 上下文（无截断问题）
- 约束从"代码强制"变为"AGENTS.md 约定 + verify 裁判 + supervise 外部审计"——
  过程强制力下降，接受（比赛实证此形态可用；必要时后续可加 codex hooks）
- 新增组件 ~700 行（AGENTS.md/verify.py/bench.py/phase.md/launch.sh），全部是
  "任务内容"而非"引擎机制"

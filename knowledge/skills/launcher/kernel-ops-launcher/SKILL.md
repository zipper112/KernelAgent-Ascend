---
name: kernel-ops-launcher
description: KDA-Ascend 算子优化流水线一键拉起技能。root agent（ZCode/Claude Code 等）用它驱动五阶段：①控制机环境自检（codex/GLM 接线/python）②NPU 机器探查（连通/卡型/arch/容器验证→自动生成 remote 配置与操作手册）③新算子四件套问答（reference/workloads/AGENTS/phase）④总检+发射确认⑤发射 codex 自主迭代并持续监听。触发词：优化算子、启动 kernel 优化、kernel-ops-launcher、部署 KDA-Ascend、接入新算子。
---

# kernel-ops-launcher —— 五阶段拉起算子优化流水线

你是 root agent。本 skill 让你把 KDA-Ascend（Codex 宿主+瘦裁判+知识资产）从"刚 clone 的仓库"带到"自主迭代中的战役"。
**只问清单里的问题，其余自动做**。进度锚点在 `<repo>/.agent-state.json`（git 外）——**再启时已过阶段自动跳过**（用户 `--recheck` 才重跑）。

## ⚠️ 风险铁律（任何阶段都适用，优先级最高）

1. **绝不 `npm -g` / 写系统路径 / 改用户 shell 配置**（PATH 注入只做会话级 `export`）
2. 写 `~/.codex/config.toml`、`~/.ssh/config` 前：**展示全文，征得用户明确同意再落盘**；已有文件则展示 diff
3. GLM key 永不回显、永不入 git（只写 `agent-config/local-secrets.yaml`，chmod 600）
4. NPU 机操作只读优先（npu-smi/ls/docker images）；容器验证只用 `--rm` 一次性容器
5. 不确定是否安全 → **停下来问**，宁可慢
6. 下载大资产（cann-ops 474M）前告知体积与用途，征得同意

## 阶段状态机

```
读取 .agent-state.json ──不存在──→ 首启：① → ② → ③ → ④ → ⑤
       │存在
       ├─ stages.env=ok && stages.npu=ok → 直接 ③（列出已有任务让用户选：继续/新建）
       ├─ stages.env=ok && !stages.npu   → ②
       └─ 任何 --recheck                  → 从头
每次阶段完成即写回 .agent-state.json（幂等）
```

## 阶段① 控制机环境自检

跑 `bash knowledge/skills/launcher/kernel-ops-launcher/scripts/preflight.sh`（只读，输出 JSON 判定）。按结果处理：

| 检查项 | 过 | 不过时的动作 |
|---|---|---|
| python ≥3.10 | — | 检测 conda/venv；都没有→**问用户**：装 miniconeda 到 ~/.local（非系统）或指认现有环境 |
| pyyaml | — | `pip install --user pyyaml`（无风险，直接做） |
| git | — | 同上问用户 |
| **codex CLI** | `codex --version` 可执行 | **不存在/版本异常 → 问用户三选一**：a) 装非全局版 `npm i --prefix ~/.local/codex-agent @openai/codex && export PATH=~/.local/codex-agent/node_modules/.bin:$PATH`（展示命令，同意后执行；PATH 只会话级）b) 用户自装后你说"继续" c) 换宿主（暂不推荐） |
| `~/.codex/config.toml` GLM 接线 | 含 `wire_api = "responses"` 且 base_url 为 `https://open.bigmodel.cn/api/v1` | 生成下方草稿**全文展示**，同意后写入（已有文件先备份 `.bak`）：<br>`model = "glm-5.3"`<br>`model_provider = "glm"`<br>`[model_providers.glm]`<br>`name = "GLM Responses"`<br>`base_url = "https://open.bigmodel.cn/api/v1"`<br>`env_key = "GLM_API_KEY"`<br>`wire_api = "responses"` |
| GLM key | `agent-config/local-secrets.yaml` 存在且可解析 | **问用户要 key**（提醒：智谱开放平台获取）。写入 `glm:\n  api_key: "<key>"`，chmod 600，不回显 |
| 冒烟 | `GLM_API_KEY=... codex exec -s read-only --skip-git-repo-check "reply: ok"` 返回 ok | 失败→按报错排（常见：base_url 写成 paas/v4） |

**一次性问题**（记入 state，之后不再问）："用默认模型 glm-5.3 还是其他？"

全过后写 state：`{"stages": {"env": "ok"}, "model": "glm-5.3"}`。

## 阶段② NPU 机器探查

**问用户一次**（合并成一次提问）：
> NPU 机器怎么连接？a) 直接 `ssh <host>` 可达（给 host）b) 跳板两跳（给 jump host 与目标 host；若 ~/.ssh/config 无对应条目→生成条目草稿**展示征得同意**后追加）c) 本机就是 NPU 机

然后跑 `bash scripts/probe_npu.sh --host <host> [--jump <jump>]`，它自动产出：
- 连通性/卡数/型号/每卡占用（含选卡建议：优先空闲卡）
- **arch 推断**（`910B4→dav_2201`、`910B→dav_200`、`910A→dav_100`、`310P→dav_200i`…从 npu-smi 型号映射）
- 容器盘点 + **可用性验证**（--rm 一次性容器跑 `import torch,torch_npu` 设备直通自检；失败→把现有镜像列表给用户选，或问是否允许 docker pull 指定镜像）
- 产出 `remote-config.json`（host/jump/device_id/arch/image）落在 `.agent-state.json` 同级

**将探查结果转写为任务上下文**（这一步是 root agent 你做，不是脚本）：把 remote 操作方式（host/选卡/docker run 模板/workspace 路径约定）写成 AGENTS.md 的"NPU 访问手册"段——后续生成任务四件套时注入。
写 state：`stages.npu=ok` + remote-config。

## 阶段③ 新算子四件套问答

若已有任务（`tasks/` 下有含 solution/ 的任务）：列出并问"继续哪个 / 新建？"。新建时按清单**逐项**问（每项带缺省，用户说"默认"就用缺省）：

**必问四项：**
1. **算子名与语义**：做什么的算子？有参考实现/伪代码/论文吗？（有的话贴给你）
2. **接口签名**：`kernel(inputs)` 的输入列表（每个 tensor 的 shape 语义/dtype/是否原地更新状态）+ 输出。这决定 workloads 的 inputs spec 与 chained 门
3. **基线与目标**：哪个形状是主形状（例：B=16 decode）？当前耗时多少？目标多少？
4. **reference.py**（唯一硬人工项）：请用户提供数学参考，或由你按语义起草→**明确提示用户人工复核**（"这是 chained 终态门的判分依据，数学错了整个验证就错了"）

**可选项（报缺省请确认）：**
- workloads：默认按主形状+一个 B=1 小形状+一个 prefill 长序列生成草案，dev=主形状 / full=全部
- 容差：默认 dtype 表（bf16 0.03 / fp16 0.004）
- 已试方向：默认空

然后生成 `tasks/<op>/` 四件套（从 `tasks/_template/` 复制，注入：接口契约进 AGENTS.md、目标进 phase.md、remote 配置进 verify/bench 头部常量与 AGENTS.md 的 NPU 手册段、reference.py 与 workloads 落盘）。生成后给用户看目录树确认。

## 阶段④ 总检

1. 重跑 `preflight.sh`（应全绿）与 `probe_npu.sh --verify`（校验模式，只验连通+卡仍在）
2. 四件套 lint：`python tasks/<op>/verify.py --help` 可跑、AGENTS.md/phase.md/reference.py/workloads 齐全、launch.sh 里任务路径正确
3. 输出红绿灯清单，**问用户："是否发射？"**（附：预计首轮战役耗时 ~1h、GLM token 消耗量级）

## 阶段⑤ 发射与监听

发射：`cd <repo> && nohup bash launch.sh <bN> > run-logs/launch-bN.log 2>&1 & disown`

监听（你持续做，直到战役终局）：每 10-15 分钟跑一次
`bash scripts/watch_loop.sh <repo> <战役名>`，输出：进程存活/log 增量/verify 内环次数/账本新行/git 新 commit。
- **卡死判据**：15 分钟无 log 增长且进程在 → 报告用户（附 log 尾部与 resume 建议：`codex exec resume` 或杀掉重发），**不擅自杀进程**
- **终局判据**：log 出现终局宣告（keep commit / verdict 报告）→ 向用户汇总战果（最优数字/候选数/证据行），并建议独立复测：`python tasks/<op>/verify.py --solution solution/<keep-cid>/candidate.py` 与 bench 复跑
- 验收三线（给用户的报告必须含）：性能实质提升数字（canonical 出数）/**零人工干预声明**/知识利用证据（router/skill 查询次数，从 log grep）

## 与仓库其他部分的关系

- 知识资产（`knowledge/`）由 codex 任务 agent 在迭代中**主动查询**（AGENTS.md 已注入查询指引）——本 skill 不负责
- 旧引擎考古在 `attic/engine-v0/`（ADR-014），勿动
- 架构决策看 `docs/design/ADR-014-*.md`

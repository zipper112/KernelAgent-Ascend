# ADR-004: 模型无关架构——自研轻量循环引擎，主用 GLM

- 状态：已接受
- 日期：2026-09-25

## 背景

用户明确：不承诺使用 Claude/GPT，很可能主用 GLM。事实约束：
- Humanize v1.16.0 是 Claude Code 插件，评审方固定经 `codex exec` CLI 执行，且 `hooks/lib/loop-common.sh:216` 正则守卫只放行 `gpt-*` / `o*` 系模型名——GLM 无法无补丁接入评审门；
- 利好：GLM 有 OpenAI 兼容端点；ZCode 由 GLM 驱动且兼容同一套 skill/prompt 格式；
- 我们已对 Humanize 做过源码级调研（9 项硬校验、评审契约、漂移熔断、Goal Tracker、BitLesson、round-prompt 渲染、读保护的完整行为规格，见 `D:\PyProject\Jev\kda-contest-deep-dive\`）。

## 决策

1. **自研轻量循环引擎**（harness/control/，工作名 ascend-rlcr）：行为规格照抄 Humanize（评审契约逐条移植、硬校验清单照搬、熔断阈值同款），实现全部走 OpenAI 兼容协议；
2. **模型层刻意最薄**：`agent-config/models.yaml` 一个文件，role（writer/reviewer/aux）→ {base_url, model, effort} 映射；自动调控仅三条规则：①默认 writer 强档 / ~~reviewer、aux 弱档~~（**勘误 2026-09-25**：reviewer 实配 `tier: strong`——弱档解析失败率高，降档方向错误，models.yaml 已修向、以 models.yaml 为准；aux 保持弱档）；②gate 解析连续失败自动升档重试；③端点故障按序降级备配。不做 provider 抽象层大工程；
3. **Humanize 降级为设计参照 + 可选并行路线**：团队若有 OpenAI 订阅可装原版插件（plan/prompt 格式与本项目的 draft→plan 流程互通）；fork 补丁路线保留为备选，不做默认；
4. 评审方最小权限：纯 API 调用、只读（无文件写权）——根治竞赛案例③"writer 甩锅 verifier"。

## 理由

被卡死的只有 Humanize 的评审门与循环控制宿主；这两块恰是我们有完整行为规格、照规格实现成本可控的部分。模型可自动调控在产线模式（runner 外层状态机）天然成立。

## 后果

- 首版实现量 +3-4 天（相对直接用插件）；换来供应商自由与长期可控；
- gate 的输出解析鲁棒性需要单测兜底（tests/ 覆盖 verdict 行缺失、COMPLETE 非末行等畸形输出）。

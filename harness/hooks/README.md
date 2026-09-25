# Stop-hook 适配器（hooks/）

把宿主 agent 的"收工尝试"翻译成 `kda gate --round N` 调用（陪伴模式）。

设计原则：**hooks 只是宿主糖，不承载逻辑**——逻辑全部在 CLI 可直达（ADR-003）。
hook 挂了/宿主不支持 → 降级为 agent 在收工前主动执行 `kda gate`（phase prompt 中已写明）。

## 提供的适配器

| 文件 | 宿主 | 机制 |
|---|---|---|
| `zcode-stop-hook.json` + 说明 | ZCode | Stop 事件调 `python harness/cli.py gate --round $(python harness/cli.py status --round)`（两命令均在协议 §1 定义） |
| `claude-code-stop-hook.sh` | Claude Code | 同语义 shell 版（若团队走 Humanize 原版插件则不需要本件） |

实现：Phase 1（与 gate.py 同批）。

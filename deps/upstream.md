# deps/upstream.md —— 上游钉版与更新流程

规则：任何上游更新必须先过回归（docs/maintenance.md §2）再进 main；本文件是唯一钉版事实源。

## akg（mindspore-ai/akg，br_agents 分支）

- 用途：KernelVerifier 当库（ADR-002）+ 内置 skill 知识源
- 本地 checkout：`D:\PyProject\Jev\.repo-research\akg\`（sparse：akg_agents/）
- 钉版 commit：`5aa15f3`（2026-09-22，!2523 merge master into master）
- 更新流程：
  1. 拉新 commit → 重跑 `pytest tests/test_verify_protocol.py`（Phase 1 提供）；
  2. RMSNorm 冒烟（上板机）：run_kernel_profile.py 路径出正确性+speedup；
  3. 通过 → 本文件改 commit + CHANGELOG 记 `deps:` 条目。

## humanize（PolyArch/humanize）

- 用途：循环行为规格参照（ADR-004，只读不依赖运行时）
- 本地 checkout：`D:\PyProject\Jev\.repo-research\humanize\`
- 钉版：v1.16.0（plugin.json:4）
- 更新流程：diff `prompt-template/codex/regular-review.md`、`hooks/loop-codex-stop-hook.sh` 与本仓库 interaction-protocol.md §4 —— 有实质行为变化时出 ADR 增补，否则不动。

## 参照仓库（只读，不进依赖链）

| 仓库 | 本地路径 | 用途 |
|---|---|---|
| NVlabs/kda | `.repo-research/kda/` | 方法论文档（basic-flow 8 槽） |
| mit-han-lab/mlsys2026-flashinfer-contest | `.repo-research/mlsys2026-flashinfer-contest/` | 实战 prompt 与案例 |
| BBuf/KDA-Pilot | `.repo-research/KDA-Pilot/` | 生产任务七件套模板 |

## 本机 skill 生态

**已全部 vendor 进仓（ADR-008）**——外部路径（C:/Users/.../.agents/skills）仅作 tools/sync_assets.py 的可选更新源，运行链零依赖。

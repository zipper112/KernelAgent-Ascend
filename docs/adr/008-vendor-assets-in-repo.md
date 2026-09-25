# ADR-008: 全资产 vendored 进仓——迁移自包含（supersede ADR-002 的"外部 checkout"部分）

- 状态：已接受
- 日期：2026-09-25

## 背景

用户铁律：项目所有依赖资产（知识、skill、akg 代码）必须在 `D:\PyProject\Jev\kda-ascend` 内，不依赖任何外部路径——迁移项目（换机器/换目录/给同事）时 git clone 即完整可用。此前形态：index.yaml 指向 `C:\Users\...\.agents\skills\`、akg 条目指向 `.repo-research\`，迁移即全部断链。

## 决策

1. **vendor 实体**：
   - `knowledge/skills/{core,triton-ascend,ascendc,pypto,tilelang,akg}/`——66 个本机 skill + akg 89 skill 整树（共 77 个资产目录，~50M）；
   - `third_party/akg/`——KernelVerifier 所在 op/core 代码子树（~4.4M），`PINNED_COMMIT=5aa15f3`（**ADR-002 修订：从"import 外部 checkout"改为"vendor 仓内代码"**）；
2. **manifest 钉版**：`deps/vendor-manifest.yaml` 登记每资产 path + tree-sha256（16 位）；`tools/sync_assets.py` 提供可选的源对比/更新（运行不依赖它）；
3. **外部源降级为两个角色**：来源记录（manifest.sources，可追溯）+ 可选更新源（sync_assets）；
4. **知识单入口**：agent 找知识只经 router（index.yaml + routing-table.yaml）；ref 全部仓库相对路径；tests 强制断言每条 ref 存在（防断链复发）；
5. 归属声明：`THIRD_PARTY_NOTICES.md`（akg Apache-2.0；本机 skill team-internal）。

## 理由

知识资产是项目两大核心之一（用户："知识乱七八糟项目就失败 2/3"）；外部路径依赖 = 迁移断链 + 团队协作不可复现。体积 57M 对 git 完全可承受（文本为主）。

## 后果

- 上游更新流程改变：从"改 checkout"变为"sync_assets --check → diff 评估 → 同步 → 更新 manifest"（maintenance.md 同步修订）；
- 本机 skill 生态更新不再自动反映到项目——这是刻意的（钉版换可控）；
- tree-sha256 计算包含全部文件（~77 资产秒级完成，可接受）。

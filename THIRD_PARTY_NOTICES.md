# 三方资产归属声明（THIRD_PARTY_NOTICES）

本仓库 vendored 了以下外部资产（登记于 `deps/vendor-manifest.yaml`，hash 钉版）：

## mindspore-ai/akg（br_agents 分支 @5aa15f3）— Apache-2.0

- 内容：`knowledge/skills/akg/`（89 个内置 skill 整树）与 `third_party/akg/`（KernelVerifier 所在的 op/core 代码子树）；
- 来源：https://gitcode.com/mindspore/akg （br_agents 分支）；
- 许可：Apache License 2.0；本项目对其的修改（如有）在 sync_assets.py 的 diff 报告中显式列出；
- 完整许可文本见上游仓库 LICENSE。

## 本机团队 skill 快照 — team-internal

- 内容：`knowledge/skills/{core,triton-ascend,ascendc,pypto,tilelang}/` 共 66 个 skill；
- 来源：团队内部维护的 skill 生态快照（原始目录 C:\Users\11565\.agents\skills\，2026-09-25 抓取）；
- 许可：team-internal（团队内部使用与分发；对外发布前需逐个确认来源许可）；
- 更新：tools/sync_assets.py（可选；运行不依赖它）。

## 参照方法论仓库（未 vendor，仅设计参照）

- NVlabs/kda、PolyArch/humanize v1.16.0、mit-han-lab/mlsys2026-flashinfer-contest：行为规格参照（ADR-001~007 中引用），不在本仓库分发其内容。

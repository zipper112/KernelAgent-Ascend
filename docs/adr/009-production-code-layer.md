# ADR-009: 生产算子代码层——cann/ops-* 官方仓 + production-index（KernelWiki 等价物）

- 状态：已接受（2026-09-25 同日修订：重资产脱离 git 追踪）
- 日期：2026-09-25

## 背景

KDA 的 KernelWiki（2265 页：2179 生产 PR + 48 wiki + 89 代码资产包）是其知识侧灵魂——DSA Indexer 19.08× 的转折点即"agent 从生产代码检索到关键思路"。本项目此前只有它的"方法论半层"（knowledge/skills 77 目录 + 三轴索引），**生产实现层空缺**（原押注 CANNBot knowledge，被 ADR-005 license 门挂起）。

调研实测（2026-09-25）：gitcode `cann` org 公开算子仓共 10 个——三大主力 ops-transformer（286M）/ ops-nn（372M）/ ops-math（155M）+ 五小仓（blas/collections/cv/fft/sparse/gnn，共 ~87M）+ ops-competitions；均 CANN OSL 2.0，活跃维护（ops-transformer 昨日仍有提交）；含 rms_norm 的 arch22/arch35 双代际生产实现。

## 决策（用户拍板：vendor 进仓 + 内部用即安全；同日修订：重资产不入 git）

**修订（同日晚）**：用户指示"重资产不要 git 追踪"——474M 二进制级资产进 git 历史会让仓库永久膨胀。最终形态分层：
1. **入 git（轻资产）**：knowledge/skills 53M 文本知识（ADR-008 维持）+ production-index.yaml 索引（1210 条，重建脚本产物也入仓作参考基线）+ bootstrap 脚本 + manifest 四条 hash；
2. **不入 git（重资产）**：third_party/cann-ops/ 474M 实体（gitignore）；**迁移新机器后一条命令重建**：`python tools/sync_assets.py --bootstrap-cann-ops --proxy http://127.0.0.1:7897`（拉十仓浅克隆→按目录清单组装→剥 tests→重跑 build_production_index）；
3. 降级语义：实体缺失时 tests skip（非 fail）、check_env 报 warn（非 fail）——生产层是增强项，缺失不阻塞方法论闭环。

## 原决策（保留供追溯）

1. **vendor 进仓**：`third_party/cann-ops/` 474M——三大仓取算子目录（剥 tests 288M/docs/cmake/scripts/examples/torch_extension/build），小仓整仓（剥 .git）；每仓保留 LICENSE/NOTICE/version 文件；
2. **production-index**：`tools/build_production_index.py` 扫描生成 `knowledge/router/production-index.yaml`（1210 条算子条目：op×repo×path×archs×kind；架构从路径 archXX/ascend950 等标记自动识别）；vendor 更新后必须重跑生成器；
3. **接入检索**：`query.py --production` 联合检索——方法论条目在前、生产命中追加在后；匹配用**词元精确匹配**（算子名按 `_` 拆词元后与 family 全等），杜绝 random_normal 误命中 norm 的子串问题；上限 15 条；
4. **license 结论（修订 ADR-005 的适用范围）**：CANN OSL 2.0 下**内部使用/学习参考场景安全**（华为官方开源算子仓本就是公开给开发者参考的）；对外发布/商业分发前需再审查并遵守 notice 义务（THIRD_PARTY_NOTICES 已登记）。CANNBot knowledge 的独立评估维持 ADR-005 不变；
5. 反抄袭边界不变：task.yaml forbidden 条款禁止"抄现成答案"——生产代码层用于**学结构/学手法**，最终 kernel 必须自己实现并在证据链中声明参考来源。

## 理由

用户铁律（迁移自包含）+ KernelWiki 的实证价值（最大优化跳变来自生产代码检索）+ 474M 对 git 可承受（文本源码为主）。

## 后果

- 仓库总体积 ~540M（clone 一次性成本，可接受）；
- cann-ops 更新流程 = sync_assets 拉新 → 重跑 build_production_index → diff 报告（maintenance.md §2 补条目）；
- tests 增生产索引断言（rms_norm 命中、随机抽查 path 存在）。

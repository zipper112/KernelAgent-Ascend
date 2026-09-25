# KDA-Ascend

**昇腾算子自动迭代优化工具**——把 MLSys 2026 FlashInfer 竞赛验证过的 KDA 工作流（Kernel Design Agents），改造为生产可用、团队长期维护、模型供应商无关（主用 GLM）的华为昇腾 NPU 算子开发迭代系统。

## 项目定位

两个同等重要的核心：

1. **Harness（`harness/`）**：严谨的生产闭环。确定性核心（正确性验证、三层测量、诊断、证据链、晋升门）零 LLM 依赖；控制层（循环状态机、评审门、上下文组装）是唯一接触模型的地方。
2. **知识资产（`knowledge/`）**：prompt 体系 + skill 路由 + 经验库 + 盲区治理。与代码同等版本化维护。

模型层刻意最薄（`agent-config/models.yaml`）：role→endpoint 映射 + 三条自动调控规则，OpenAI 兼容协议统一，换供应商只改一个文件。

## 三源合流

| 上游 | 取什么 | 落在哪 |
|---|---|---|
| [NVlabs/kda](https://github.com/NVlabs/kda) | 任务契约 8 槽、draft 六要素、证据驱动三态、通用层/工作区分层哲学 | `knowledge/prompts/contract-template.md`、`tasks/` 布局 |
| [PolyArch/humanize](https://github.com/PolyArch/humanize) v1.16.0 | 拦截退出循环、9 项硬校验、评审契约、漂移熔断、BitLesson、round-prompt 渲染、防过期上下文 | `harness/control/`、`harness/ctx/` |
| [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) | 三阶段任务书、开发集/全量分层、每方向 5 次迭代上限、候选 DAG、否决留痕、形状感知路由、reward hacking 防御条款 | `knowledge/prompts/phase1-3-ascend.md`、`harness/core/evidence.py` 规格、`knowledge/prompts/clauses/` |

## 目录结构

```
kda-ascend/
├── harness/            # 核心一：生产闭环
│   ├── core/           #   确定性核心（verify/measure/diagnose/evidence/promote，零 LLM）
│   ├── control/        #   控制层（runner 状态机 / gate 评审门）
│   ├── ctx/            #   上下文组装器（round 渲染/防过期/三层注入/承认闸门）
│   ├── hooks/          #   宿主适配器（ZCode/Claude Code Stop-hook，只是糖）
│   └── cli.py          #   agent 交互入口（kda verify/bench/diagnose/promote/...）
├── knowledge/          # 核心二：知识资产（与 harness 同级的一等公民）
│   ├── prompts/        #   契约模板 + 三阶段模板 + 可组合条款库
│   ├── router/         #   三轴索引（症状×算子族×硬件代际）+ 路由决策表 + 盲区清单
│   └── lessons/        #   BitLesson 经验库（跨任务沉淀回流）
├── agent-config/       # 模型配置（GLM 主配，OpenAI 兼容）
├── docs/               # ADR 决策记录 / 交互协议规格 / 维护手册
├── deps/               # 外部依赖：skill 清单 + 上游钉版
├── tasks/              # 算子任务工作区（七件套，任务资产永不进通用层）
│   └── _template/      #   clone 即用的任务骨架
├── tools/              # check_env.py 环境自检
└── tests/              # 无卡机可跑的最小单测
```

## 快速开始

```bash
# 1. 环境自检（无 NPU 机器也能跑，NPU 项会列为 pending）
python tools/check_env.py

# 2. （上板机）完整自检含 CANN/torch_npu/triton-ascend
python tools/check_env.py --full

# 3. 新建算子任务（Phase 1 实现后可用）
kda new-task tasks/my-op   # 从 _template 生成七件套
```

当前状态：**v0.0-scaffold**（骨架+协议规格+知识模板已就位；harness 代码 Phase 1 实现）。路线图见 [CHANGELOG.md](CHANGELOG.md) 与各 [ADR](docs/adr/)。

## 维护者指南

- 设计决策看 `docs/adr/`（为什么这么做）；接口规格看 `docs/design/interaction-protocol.md`（怎么对接）；
- 上游更新（akg/humanize/skill）流程看 `docs/maintenance.md` 与 `deps/upstream.md`；
- commit 前缀：`feat/fix/docs/deps/knowledge/tests`；
- 每阶段验收打 tag：v0.1-mvp → v0.2-diagnosis → v0.3-knowledge → v1.0。

## License

Apache-2.0（见 [LICENSE](LICENSE)）。引用外部 skill / CANNBot 资产前先过 `deps/skills.yaml` 与 ADR-005 的 license 审查。

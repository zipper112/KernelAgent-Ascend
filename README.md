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
│   ├── hooks/          #   宿主适配器（ZCode/Claude Code Stop-hook；Phase 1 交付）
│   └── cli.py          #   agent 交互入口（kda verify/bench/diagnose/promote/...）
├── knowledge/          # 核心二：知识资产（与 harness 同级的一等公民）
│   ├── prompts/        #   契约模板 + 三阶段模板 + 可组合条款库
│   ├── router/         #   三轴索引（症状×算子族×硬件代际）+ 路由决策表 + 盲区清单
│   ├── skills/         #   vendored skill 资产（六组 77 目录 + akg 89 skill；ADR-008 自包含）
│   └── lessons/        #   BitLesson 经验库（跨任务沉淀回流）
├── third_party/akg/    # KernelVerifier 代码子树（钉版 5aa15f3，Apache-2.0）
├── agent-config/       # 模型配置（GLM 主配，OpenAI 兼容）
├── docs/               # ADR 决策记录 / 交互协议规格 / 维护手册
├── deps/               # vendor-manifest（hash 钉版）+ 上游来源记录
├── infra/              # 非核心基础设施（remote 远程执行 / secrets 密钥；ADR-007 解耦）
├── tasks/              # 算子任务工作区（七件套，任务资产永不进通用层）
│   └── _template/      #   clone 即用的任务骨架
├── tools/              # check_env.py 自检 + sync_assets.py 资产更新（可选）
└── tests/              # 无卡机可跑的单测（资产断言/路由语义/gate 契约）
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

当前状态：**v0.0-scaffold**（骨架+协议规格+知识模板已就位；harness 代码 Phase 1 实现）。路线图见 [CHANGELOG.md](CHANGELOG.md) 与各 [ADR](docs/design/)。

## 重资产与部署（新机器必读）

### git 内资产（clone 即得）

| 资产 | 体积 | 说明 |
|---|---|---|
| `knowledge/` | ~50M | 全部知识资产（prompts/router/skills/lessons，1837 文件）——三源合流的落地物 |
| `third_party/akg/` | ~5M | KernelVerifier 代码子树（钉版 5aa15f3，Apache-2.0） |
| `knowledge/router/production-index.yaml` | 160K | 生产代码索引（cann-ops 目录级索引；router `--production` 查询只依赖它，**无需下载 474M 原始树**） |

### git 外重资产（三项，缺一按下方方式补）

**1. `third_party/cann-ops/`（~474M，CANN 算子仓源码层）**

- 来源：gitcode.com `cann/` 组织下十个算子仓（ops-math/ops-nn/ops-transformer 等；浅克隆 + 剥 tests 后组装）
- 重建：`python tools/sync_assets.py --bootstrap-cann-ops`（gitcode 国内直连，无需代理，数分钟）
- 用途：writer 需要深读某算子实现源码时的素材层（research 的 production 层索引已在 git；本树只在读文件本体时需要）
- 放置：`third_party/cann-ops/<repo>/<family>/...`（脚本自动；禁止手动改动后被 `--check` 扫出漂移）

**2. `agent-config/local-secrets.yaml`（GLM API key，永不入 git）**

- 格式（`chmod 600`）：
  ```yaml
  glm:
    api_key: "<your-key>"
  ```
- 或环境变量 `GLM_API_KEY`（优先级：env > yaml；都缺则 harness 启动即报错，不带默认 key）

**3. Python 环境**

- 主阵地（jump）用 conda：`conda create -n ka python=3.11 && conda activate ka && pip install pytest pyyaml`
- pip 报 "No matching distribution" 时多为机器残留内部镜像配置：`PIP_CONFIG_FILE=/dev/null`，或加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，或走代理 `--proxy http://127.0.0.1:18090`
- torch/torch_npu 只在远端 e15 容器内需要（本地测试缺 torch 自动 skip，不阻塞）

### 一键部署清单（以 jump 为例）

```bash
git clone git@github.com:zipper112/KernelAgent-Ascend.git /data01/mahaolong/KAgent
cd /data01/mahaolong/KAgent && conda activate ka
python tools/sync_assets.py --bootstrap-cann-ops   # 重建 474M 资产层 + 索引
# 放入密钥（chmod 600）
python -m pytest tests/ -q                          # 期望全绿（torch 用例自动 skip）
python -m harness.cli version                       # 冒烟
```

远端 NPU 访问（ADR-013 操作上下文注入模式）：确保本机 `ssh yq-e15` BatchMode 免密可达、
目标卡空闲（`npu-smi info`）、任务 `config.yaml` 的 `execution.remote` 段填好 host/device_id/docker_image。
harness 会把访问手册注入 writer prompt，LLM 依据 canonical 模板自主执行（详见
`docs/design/ADR-013-操作上下文注入替代远程执行层.md`）。

## 维护者指南

- 设计决策看 `docs/design/`（ADR，为什么这么做）；接口规格看 `docs/design/interaction-protocol.md`（怎么对接）；
- 上游更新（akg/humanize/skill）流程看 `docs/maintenance.md` 与 `deps/upstream.md`；
- commit 前缀：`feat/fix/docs/deps/knowledge/tests`；
- 每阶段验收打 tag：v0.1-mvp → v0.2-diagnosis → v0.3-knowledge → v1.0。

## License

Apache-2.0（见 [LICENSE](LICENSE)）。引用外部 skill / CANNBot 资产前先过 `deps/skills.yaml` 与 ADR-005 的 license 审查。

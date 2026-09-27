# KDA-Ascend

**昇腾算子自动迭代优化工具**——照 MLSys 2026 FlashInfer 竞赛验证的形态：**Codex CLI 当引擎（GLM 驱动）+ 纯规则裁判 + 知识资产 + 证据账本**。不自研迭代循环（v0 自研引擎的教训见 ADR-014）。

## For AI agents (ZCode / Claude Code / Codex 等 root agent)

用户说"优化一个算子 / 部署这个项目 / 启动 kernel 优化"时：
**读 [knowledge/skills/launcher/kernel-ops-launcher/SKILL.md](knowledge/skills/launcher/kernel-ops-launcher/SKILL.md) 并按其五阶段执行**（环境自检→NPU 探查→算子问答→总检→发射监听）。
该 skill 内置风险铁律（非全局安装、写用户文件先征得同意、密钥不回显）与首启/再启分流。

## 项目定位（ADR-014 比赛形态）

```
phase.md（任务书）+ AGENTS.md（工作规约，codex 自动读取）
        │
        ▼
codex exec -s danger-full-access（宿主引擎；GLM-5.3 via Responses API）
        │   轮内小循环：verify --fast → 读全量报错 → 增量 Edit → 再验（秒级，收敛主力）
        ▼
verify.py / bench.py（纯规则裁判，零 LLM；底层 canonical runner 上 e15 NPU 实测）
        │
        ▼
docs/ 三件套（benchmark.csv / solutions.jsonl / audit） + git commit（keep 才提交）

knowledge/（router + skills）← agent 主动查询（症状词来自当前报错）
```

三个组成部分：

1. **裁判层（零 LLM）**：任务级 `verify.py`/`bench.py`（比赛骨架）+ `infra/remote/runner.py`（canonical 测量口径：chained 终态门/L2 清除/交错采样）。
2. **知识资产（`knowledge/`）**：三轴 router + vendored skills + prompt 条款库。与代码同等版本化。
3. **账本与监督**：证据三件套约定 + `tools/supervise.py` 外部审计（知识利用/事件序列）。

## 三源合流

| 上游 | 取什么 | 落在哪 |
|---|---|---|
| [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) | **总体形态**：phase prompt + 宿主 agent + 纯规则 verify.py + 三件套自律记账 + CLAUDE.md 工作流注入 | `tasks/<t>/{phase.md,AGENTS.md,verify.py,bench.py}`、`launch.sh` |
| [NVlabs/kda](https://github.com/NVlabs/kda) | 任务契约 8 槽、draft 六要素、证据驱动三态、通用层/工作区分层 | `knowledge/prompts/contract-template.md`、`tasks/` 布局 |
| [PolyArch/humanize](https://github.com/PolyArch/humanize) v1.16.0 | 校验/熔断/经验回流思想（轮末审计、漂移检测、BitLesson） | `tools/supervise.py`、`knowledge/lessons/`、条款库 |

## 目录结构

```
kda-ascend/
├── launch.sh           # 战役发射器（codex exec + phase.md 一次性注入）
├── tasks/<task>/       # 任务工作区（自包含：AGENTS.md 工作规约 + phase.md 任务书
│   │                   #   + verify.py/bench.py 裁判 + solution/ 候选 + docs/ 账本）
│   └── _template/      #   新任务骨架
├── knowledge/          # 知识资产（router 三轴索引 + skills 77 目录 + prompts/lessons）
├── infra/remote/       # canonical runner + 容器入口（e15 NPU 测量协议，零 LLM）
│   └── secrets/        #   密钥 provider（local-secrets.yaml/env）
├── third_party/akg/    # KernelVerifier 子树（钉版 5aa15f3）
├── agent-config/       # key 与模型参考配置（codex 实际用 ~/.codex/config.toml）
├── tools/              # supervise.py 监督仪表 + sync_assets.py + check_env.py
├── docs/design/        # ADR 决策记录（ADR-014 为现行架构）
├── attic/engine-v0/    # 已废除的自研迭代引擎（ADR-014；git 历史保留供考古）
└── tests/              # 无卡机单测（runner 语义/资产断言/路由）
```

## 快速开始

```bash
# 1. 环境自检（无 NPU 机器也能跑，NPU 项会列为 pending）
python tools/check_env.py

# 2. 发射一次算子优化战役（ADR-014 形态；前置：codex 配置 + key + e15 免密，见下节）
bash launch.sh b5        # 注入 phase.md 给 codex，之后全自动：研究→内环 verify→bench→记账→commit

# 3. 单独跑裁判（人类/CI 用，不经 agent）
cd tasks/k8-triton && python verify.py --solution solution/c019/candidate.py --fast
python bench.py --solution solution/c019/candidate.py --record
```

当前状态：**ADR-014 比赛形态**（codex 宿主+瘦裁判；v0 自研引擎已入 attic）。路线图见 [CHANGELOG.md](CHANGELOG.md) 与各 [ADR](docs/design/)。

## 重资产与部署（新机器必读）

### 0. Codex 宿主接线（ADR-014 核心，一次性）

```bash
# jump 上已装 codex-cli 0.157.1（npm i -g @openai/codex）
cat > ~/.codex/config.toml << 'EOF'
model = "glm-5.3"
model_provider = "glm"
[model_providers.glm]
name = "GLM Responses"
base_url = "https://open.bigmodel.cn/api/v1"   # Responses API（非 chat 的 paas/v4！）
env_key = "GLM_API_KEY"
wire_api = "responses"
EOF
export GLM_API_KEY=<key>   # 或从 agent-config/local-secrets.yaml 提取
codex exec -s read-only --skip-git-repo-check "reply: ok"   # 冒烟
```
注意：沙箱必须 `-s danger-full-access`（workspace-write 禁网，ssh 不到 e15——实测）。

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

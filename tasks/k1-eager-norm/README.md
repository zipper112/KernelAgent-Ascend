# 任务工作区模板（七件套）

用法（Phase 1 起可用 `kda new-task` 自动生成；当前手工复制本目录）：
`cp -r tasks/_template tasks/<op-name>-v<N>`

## 七件套说明

| 路径 | 内容 | 写方 |
|---|---|---|
| `task.yaml` | 任务契约 8 槽（从 knowledge/prompts/contract-template.md 填充） | 人 |
| `config.yaml` | 代际/DSL/workload 集/预算 | 人 |
| `baseline/` | 锁定的官方基线实现 | 人首轮回填 → 锁 SHA |
| `solution/<candidate_id>/` | 候选代码 | agent（keep 后） |
| `bench/workloads.yaml` | workload 定义（含 repr: true 代表集标记） | 人 |
| `docs/` | draft/plan/证据链（benchmark.csv、solutions.jsonl、audit.log 由 harness 写） | 混合 |
| `run/`（gitignore） | state.json + 逐轮留痕 | harness/agent |

## 任务启动 checklist（人）

1. [ ] task.yaml 8 槽填齐（含 baseline 版本可锁定）
2. [ ] config.yaml 填 arch（get_npu_arch.py 检测值）
3. [ ] bench/workloads.yaml 定义全量集 + 标记代表集
4. [ ] baseline/ 回填 → `kda contract --lock`（Phase 1）
5. [ ] 贴 Phase 1 prompt（knowledge/prompts/phase1-ascend.md 填充版）+ 反作弊/盲区条款

# harness/core/ —— 确定性核心（零 LLM）

**铁律：本目录任何文件不得 import 任何 LLM/agent 库。** 换模型供应商、全部脱机时，本层必须照常运行且可回归测试。

| 模块 | 职责 | 实现状态 |
|---|---|---|
| `verify.py` | 正确性门：dtype 分级容差（fp16 0.004/bf16 0.03/int8 0.01/其他 0.02）+ 四步比较（形状→NaN 同位→Inf 位置+符号→相对误差+错误比例容忍）；Phase 0 经 `akg_verifier_adapter` 调 akg KernelVerifier | Phase 1 |
| `measure.py` | 三层测量：L0 秒级门（import+冒烟+粗计时）/ L1 稳态 A/B（warmup≥3 + L2 cache 清除 + 交错采样≥5 + 对称基线）/ L2 msprof 7 组（调 ops-profiling 流程） | Phase 1（L0/L1）、Phase 2（L2） |
| `diagnose.py` | L2 数据 → 7 档 Bound 判定（阈值 80% 或最大占比>70%）→ 症状映射 → ≤5 条带证据建议 | Phase 2 |
| `evidence.py` | 证据链唯一写方：benchmark.csv / solutions.jsonl / audit.log / profile 摘要（schema 见 docs/design/interaction-protocol.md §3） | Phase 1 |
| `promote.py` | 晋升 7 项门（全量正确/加速超噪声/基线 hash/plan hash/证据行/DAG 父链/audit 无越序） | Phase 1 |
| `akg_verifier_adapter.py` | akg KernelVerifier 薄封装（ADR-002），隔离上游 API | Phase 0 |

## 写方规则（反作弊地基）

docs/ 下 evidence 三件套（csv/jsonl/audit）**只有 evidence.py 可写**；promote 与 gate 双重校验行完整性与 audit 配对。

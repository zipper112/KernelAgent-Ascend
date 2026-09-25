# 任务契约模板（contract-template.md）

> 派生自 NVlabs/kda `prompts/basic-flow.md` 的 8 槽任务契约，逐槽昇腾化。
> 用法：复制到任务根目录改名为 `task.yaml` 的 `contract` 段（或直接填本 md 后由 `kda new-task` 转换）。**8 槽必填，缺槽任务不得启动。**

```yaml
contract:
  # ① 任务名与算子族
  task_name: rmsnorm-v1
  op_family: norm            # norm|matmul|attention|reduce|elementwise|sort|moe|...

  # ② 目标（用户视角，一句话说清业务价值）
  objective: >
    在 910B 上将推理路径中的 RMSNorm 算子延迟降到 torch_npu 原生实现以下，
    不牺牲数值精度路径的可选择性。

  # ③ 正确性要求（引用官方容差，禁止自定）
  correctness:
    verifier: kda-verify    # 唯一合法验证器（harness/core/verify.py）
    tolerance_ref: "dtype 分级相对误差限：fp16 0.004 / bf16 0.03 / int8 0.01 / 其他 0.02；NaN 同位允许、Inf 位置+符号匹配、错误元素比例容忍"
    special_behaviors:      # 算子特有数值行为（如 rsqrt 的极端值、累加顺序敏感）
      - "输入全 0 时输出必须为 0（不得 NaN）"

  # ④ 性能目标（可测量；不确定时声明探索型）
  performance:
    baseline: torch_npu.nn.functional.rms_norm   # 基线必须可锁定（git SHA / 版本号）
    target: ">=1.2x mean speedup on l1 workload set"   # 或 "exploratory"（Phase 1 结束后回填具体值）
    noise_floor: 0.05    # 噪声阈：低于此加速视为无效提升

  # ⑤ 允许实现路线（DSL 白名单 + 禁改项）
  allowed:
    dsl: [triton-ascend]   # ADR-001；后续可加 ascendc / pypto / tilelang-npu
    forbidden:
      - "不得修改 bench/ 与 baseline/"
      - "不得引入新的闭源二进制依赖"
      - "不得绕过 kda verify 自建验证脚本"

  # ⑥ 验证命令（唯一合法）
  validation_cmd: "kda verify --candidate <id> --workload-set full"

  # ⑦ 评测命令
  evaluation_cmd: "kda bench --candidate <id> --mode l1"

  # ⑧ 晋升标准（promote 8 项门 + 目标，含一条代码质量类 AC——ADR-010）
  promotion:
    gate: promote-eight-checks   # 见 docs/design/interaction-protocol.md
    extra: "全量 workload 集正确性通过 + l1 加速 >= target 且超 noise_floor"
```

## 附：draft 六要素（agent 写 `docs/draft.md` 的强制结构）

draft 未写完不许编辑代码（KDA 铁律）：

1. **基线与验证方式**：基线是什么、怎么被验证的（命令与期望输出）；
2. **风险与未知**：至少 3 条，标注哪条最可能翻车；
3. **候选方向**：按 期望收益 × 实现风险 排序，每条给理由与知识来源（路由条目 id）；
4. **首批具体步骤**：可执行粒度（文件级）；
5. **精确命令**：验证与评测的确切命令行；
6. **三态证据要求**：promote / revise / reject 各需要看到什么证据才成立。

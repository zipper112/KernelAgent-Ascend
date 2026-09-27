# AGENTS.md —— k10-rmsnorm-gated 任务工作规约（codex 自动读取）

本任务是 Ascend 910B4 上的 rms_norm_gated（GLM-5.3 KDA o_norm）kernel 优化。
你在一个隔离任务目录里自主迭代。裁判（verify.py/bench.py）零 LLM、纯规则；
所有性能/正确性结论必须出自它们。

## 背景与病根（已取证，从这里出发）

上游实现在 vllm-ascend `ops/triton/kda/kda.py:413 rms_norm_gated`：
每调用做 `.contiguous()` × 2（x 与 g 各一次，可能拷贝）+ reshape + `layer_norm_gated_fwd`
triton 包装（host 侧 heuristics/autotune 查表）。E070 已证伪"python dispatch 层"解释
（torch.compile 图化零增益）——病根是 device 侧小核本体执行时间 + 每调用的包装税。
K1 池（44-49% busy）第一嫌疑串。

## 工作流（每轮循环）

1. **研究**（主动查询，症状词来自当前报错/瓶颈，勿每次同词）：
   - `python ../../knowledge/router/query.py --symptom <症状> --op-family norm --arch dav_2201 --compact`
   - 命中 skill 读全文：`../../knowledge/skills/<id>/SKILL.md`
2. **草稿**：新方向先记 `docs/draft.md` 一行；已试方向勿重复
3. **实现**：`solution/<cid>/candidate.py`（新候选新目录；接口契约见下）
4. **快速验证内环**（收敛主力）：
   `python verify.py --solution solution/<cid>/candidate.py --fast`
5. **全量验证**：`python verify.py --solution ...`
6. **测量**：`python bench.py --solution ... --record`
7. **记账**：solutions.jsonl 追加（失败候选也记）
8. **提交**：实质改进才 `git commit`
9. **换向**：同方向 3 次无改进 → 读 docs/verdict.md 换方向
10. **终局**：达标 → keep + commit + docs/verdict.md 终局报告（**附当轮 canonical 输出原文**）
11. 每 5 个候选重读本文件 + docs/verdict.md

## kernel 接口契约（不可改）

```python
def kernel(inputs: list[Tensor]) -> Tensor
# inputs = [x(T,H,D) bf16, g(T,H,D) bf16, weight(D,), bias(D,)]
# 语义：y = (rmsnorm(x) * weight + bias) * sigmoid(g)；归约维=最后一维 D=128；eps=1e-6
# 输出 (T,H,D) bf16；无状态携带（单步语义）
# 环境：triton 3.5.0 + triton_ascend 3.2.2 + torch 2.10（e15 容器内）
```
参考实现（数学 oracle）：`reference.py`

## NPU 访问手册

- 目标机 yq-e15（仓 ssh 免密直连）；NPU 7；容器 `quay.io/ascend/vllm-ascend:nightly-main`
- verify.py/bench.py 已封装全部远端链路（tar 同步→docker 直通→results 回读），直接调
- 测前自查：`ssh yq-e15 "npu-smi info | grep -A6 'NPU 7 '"`（有他人进程→数据不可作 keep 依据）

## 已知打法线索（K8 同款病根的已验证三招）

1. **单发射直发**：首调标准发射后捕获 CompiledKernel 直接 launch（绕过 JIT 包装的
   绑定/特化/查缓存——K8 c007/c033 验证）
2. **去 .contiguous() 税**：kernel 直接按 3D strides 读（或证明调用侧本就连续，
   用 is_contiguous() 分支跳过拷贝——注意 x 从 attention 输出 empty 张量 reshape 而 来，
   大概率天然连续，税可能主要在 g）
3. **静态形状特化**：D=128 constexpr / T 档位化（64/4096/16384 三档或 T 分桶）
4. 融合上界参考：相邻 elementwise（o_proj 前的 reshape/换布局）若在同串，
   评估合入可行性（需改接口契约——先在 draft.md 论证再动）

## 纪律条款

- canonical 出数：数字只能来自 verify.py/bench.py
- 基线不可篡改：reference.py / bench/workloads.yaml / 本文件只读
- 记账完整：失败也进 solutions.jsonl
- router 未命中→draft.md 记 blindspot
- 终局宣告必须附当轮 canonical 输出（引用历史行=无效）

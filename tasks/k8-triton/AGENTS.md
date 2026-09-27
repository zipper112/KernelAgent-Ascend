# AGENTS.md —— k8-triton 任务工作规约（codex 自动读取；比赛 CLAUDE.md 的对应物）

本任务是 Ascend 910B4 上的 causal_conv1d_update Triton kernel 优化。
你（agent）在一个隔离任务目录里自主迭代。裁判（verify.py/bench.py）零 LLM、纯规则；
你的所有性能/正确性结论必须出自它们——自建计时脚本只可用于诊断叙述。

## 工作流（每轮循环，模仿 FlashInfer 竞赛流程）

1. **研究**：先查知识库再写代码（主动查询，不是等推送）：
   - `python knowledge/router/query.py --symptom <症状> --op-family conv --arch dav_2201 --compact`
   - 症状词要从**当前问题**来（verify 报错/bench 瓶颈），不要每次都用同一组词
   - 命中的 skill 去读全文：`knowledge/skills/<id>/SKILL.md`（注入的只是切片）
   - 生产代码参考：`python knowledge/router/query.py --production --op-family conv --arch dav_2201 --compact`
2. **草稿**：新方向先在 `docs/draft.md` 记一行（方向名/假设/预期收益），已试方向勿重复
3. **实现**：改 `solution/<cid>/candidate.py`（新候选新目录；接口见下）
4. **快速验证内环**（一天可做几十次，这是收敛主力）：
   `python verify.py --solution solution/<cid>/candidate.py --fast`
   挂了→读完整报错→改→再验证。**不要跳过内环直接写下一个候选**
5. **全量验证**：内环过了→ `python verify.py --solution ... （默认全档）`
6. **测量**：`python bench.py --solution ...`（dev 档）；数字落 benchmark.csv
7. **记账**（三件套义务，缺一不可）：
   - `docs/benchmark.csv` 追加一行：`ts,cid,parent,workload_set,mean_us,p50,p99,speedup,verdict,note`
   - `docs/solutions.jsonl` 追加一条：`{"candidate_id","parent_id","direction","hypothesis","status","round","note"}`
   - 失败候选同样记账（status=reject+原因）——否决留痕是硬纪律
8. **提交**：候选有实质改进才 `git commit`（消息含 cid 与方向）
9. **换向**：同方向连续 3 次无改进 → 读 docs/verdict.md 的已试方向清单，换根本不同的方向
10. **终局**：mean_us < 历史最优×（dev+full 双档达标）→ 在 solutions.jsonl 标 keep 并 git commit
11. **上下文卫生**：每 5 个候选重读一次本文件 + docs/verdict.md（防漂移）

## kernel 接口契约（不可改）

```python
def kernel(inputs: list[Tensor]) -> Tensor
# inputs = [x(B,S,D) bf16, conv_state(B,D,3) bf16, weight(D,4), bias(D,), indices(B,) int32]
# 语义：逐 token y=silu(Σ w·[s0,s1,s2,x]+b)；处理完 S 个 token 后 conv_state 原地滑窗更新
# pad 槽（indices<0）跳过；输出 (B,S,D)
# 环境：triton 3.5.0 + triton_ascend 3.2.2 + torch 2.10（e15 容器内）
```
参考实现（数学 oracle）：`reference.py`（本地可读，含状态转移建模）

## NPU 访问手册（verify.py/bench.py 已封装，直接调即可）

- 目标机 yq-e15（本仓 ssh 免密直连）；NPU 7；容器 `quay.io/ascend/vllm-ascend:nightly-main`
- 测量纪律全部在 runner 里（chained 3 步终态门/L2 清除/交错采样）——**你没有理由绕开它**
- 测前自查占用：`ssh yq-e15 "npu-smi info | grep -A6 'NPU 7 '"`（有他人进程 → 数据标注污染，不作为 keep 依据）

## 纪律条款（违规=终局评审 REJECT）

- **canonical 出数**：性能/正确性数字只能来自 verify.py/bench.py（底层 canonical runner）
- **基线不可篡改**：reference.py / bench/workloads.yaml / 本文件只读
- **记账完整**：每个候选（含失败）都进三件套；不静默丢弃方向
- **盲区声明**：router 未命中时在 draft.md 记 blindspot，不要编造引用
- **禁 .item()/.cpu() 同步、禁 unfold/einsum 中间物化**（任务存在理由）
- 300k 上下文预算：长实验日志先落文件再引用，不要整段刷屏

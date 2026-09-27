# AGENTS.md —— k11-index-fused 任务工作规约（codex 自动读取）

本任务优化 GLM-5.3 KDA speculative decode 路径的 token 索引 gather-scatter 串
（K9 档：每 chunk 5 连发 index_select×4 + index_copy_×1，aclnn AsStrided/GatherV3/
ViewCopy/ScatterUpdate 群，长 prefill 进 top10）。

## 工作流（同 K8/K10 实证流程）

1. 研究：`python ../../knowledge/router/query.py --symptom <症状> --op-family sort --arch dav_2201 --compact`（索引/gather 族）+ 读命中 skill 全文
2. 草稿记 docs/draft.md；已试方向勿重复
3. 实现 solution/<cid>/candidate.py（新候选新目录）
4. 内环：`python verify.py --solution ... --fast`（几十次是收敛主力）
5. 全量 verify → `python bench.py --solution ... --record` → solutions.jsonl 记账
6. 实质改进才 git commit；终局附当轮 canonical 输出原文

## kernel 接口契约

```python
def kernel(inputs: list[Tensor]) -> Tensor
# inputs = [x(T,D) bf16, idx_spec(k,) int32, idx_ns(m,) int32]
# 语义：out = zeros(T,D); out[:k]=x[idx_spec]; out[k:k+m]=x[idx_ns]
# （上游等价：index_select×2 + 拼装 + index_copy_ 回写，5 个 aclnn 小核）
# 输出 (T,D) bf16；无状态携带；k+m ≤ T
```
参考实现：reference.py

## NPU 访问手册

yq-e15 / NPU 7 / quay.io/ascend/vllm-ascend:nightly-main；verify/bench 已封装全链路。
测前自查卡占用（有他人进程→不作 keep 依据）。

## 打法线索（姊妹任务已验证）

- 串融合为一核（5→1 次发射）：gather+concat+scatter 单核完成（D=4096 按块 tile，
  idx 两段合一处理）
- 静态形状特化（D=4096 constexpr；k/m 运行期但 grid 按 T 排）
- CompiledKernel 直发（首调标准发射后捕获；K8/K10 双验证）
- 上游参照：#17299（KDA prefill 融合三连 state 部分）同族方向
- 注意 arange 索引是最优情形（连续段拷贝）；真实负载 idx 来自调度器可能有洞——
  kernel 不要假设 idx 单调连续，但可以在核内检测连续段走快路（分支成本 vs 拷贝收益权衡）

## 纪律条款

canonical 出数 / 基线不可篡改 / 失败也记账 / 盲区声明 / 终局附当轮输出。

"""K11 数学 oracle：speculative decode 的 token 索引 gather-scatter 串语义。

inputs = [x, idx_spec, idx_ns]（对应 glm5next/kda.py:404-485 的每 chunk 5 连发）
  x: (T, D) bf16 —— qkv_proj_states（T=num_tokens, D=投影维）
  idx_spec: (k,) int32 —— spec token 位置
  idx_ns: (m,) int32 —— 非 spec token 位置（k+m ≤ T）
语义（整串融合后的等价输出）：
  gathered = concat(x[idx_spec], x[idx_ns])   # 两个 index_select(0)
  out = zeros(T, D); out[:k]=前段; out[k:k+m]=后段   # 融合核的输出布局
  （scatter 回写 index_copy_ 语义等价于按位放置——这里输出"重排后张量+回写后 x'"）
挑战：串内 5 个小核（select×4+copy×1）→ 单核一次完成 gather+concat+scatter。
"""
import torch


def reference(inputs):
    x, idx_spec, idx_ns = inputs
    spec = x.index_select(0, idx_spec.long())
    ns = x.index_select(0, idx_ns.long())
    out = torch.zeros_like(x)
    out[: spec.shape[0]] = spec
    out[spec.shape[0]: spec.shape[0] + ns.shape[0]] = ns
    return out

# K11 c000：上游基线（5 连发原样：select×2 + zeros 装配 + copy 回写）
import torch


def kernel(inputs):
    x, idx_spec, idx_ns = inputs
    spec = x.index_select(0, idx_spec.long())
    ns = x.index_select(0, idx_ns.long())
    out = torch.zeros_like(x)
    k = spec.shape[0]
    out[:k] = spec
    out[k:k + ns.shape[0]] = ns
    return out

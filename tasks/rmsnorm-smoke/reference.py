"""rmsnorm-smoke 的数学 oracle（fp32 独立实现；runner verify/bench 共用）。

runner 约定：暴露 reference(inputs: list[Tensor]) -> Tensor。
刻意与 candidate.py 写法隔离（不做逐行共享），保证 oracle 独立性。
"""
import torch


def reference(inputs):
    x = inputs[0]
    xf = x.to(torch.float32)
    ms = torch.mean(xf * xf, dim=-1, keepdim=True)
    y = xf / torch.sqrt(ms + 1e-6)
    return y.to(x.dtype)

"""RMSNorm 朴素 kernel（Phase 0 冒烟）：纯 torch 实现，验证远端执行链路。

runner 的调用约定：暴露 kernel(inputs: list[Tensor]) -> Tensor。
"""
import torch


def kernel(inputs):
    """inputs[0]: [batch, seq, hidden] fp16（NPU tensor）。返回 RMSNorm 结果。"""
    x = inputs[0]
    rms = x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
    return (x.float() * rms).to(x.dtype)

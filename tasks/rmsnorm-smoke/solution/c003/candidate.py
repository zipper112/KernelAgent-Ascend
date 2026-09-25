"""RMSNorm 候选 c003：手工 eager 微调方向（manual-eager）。

假设：不开融合算子，仅减少派发与临时张量也能提速——c001 两次 .float() 升精度
（两个临时张量）+ pow(2)（比 mul 贵）；本候选单次升精度 + mul 自乘。
对照组：c002 融合核（预期本方向仍显著慢于融合）。
"""
import torch


def kernel(inputs):
    x = inputs[0]
    xf = x.float()                            # 单次升精度（c001 升了两次）
    ms = xf.mul(xf).mean(-1, keepdim=True)    # mul 自乘替代 pow(2)
    inv = torch.rsqrt(ms + 1e-6)
    return (xf * inv).to(x.dtype)

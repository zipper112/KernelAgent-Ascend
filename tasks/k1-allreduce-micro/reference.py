"""reduce_sum oracle（fp32 独立实现）。kernel(inputs) 与 reference(inputs) 接口一致。"""
import torch

def reference(inputs):
    x = inputs[0]
    return x.float().sum(-1).to(x.dtype)

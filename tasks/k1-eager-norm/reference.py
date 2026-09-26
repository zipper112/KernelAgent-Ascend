"""RMSNorm oracle（fp32 独立）。"""
import torch

def reference(inputs):
    x = inputs[0]
    xf = x.float()
    return (xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)

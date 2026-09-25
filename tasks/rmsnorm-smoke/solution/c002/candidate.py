"""RMSNorm 候选 c002（revise）：融合算子方向（op-fusion）。

revise 记录：F.rms_norm 在 torch 2.7.1 不存在（verify 拒绝入链）——改用
torch_npu.npu_rms_norm（aclnnRmsNorm 单融合核绑定）。gamma 无权重场景传 ones
（模块级缓存避免每调用重建）。
"""
import torch
import torch_npu

_gamma_cache: dict = {}


def kernel(inputs):
    x = inputs[0]
    key = (x.shape[-1], x.dtype, x.device)
    g = _gamma_cache.get(key)
    if g is None:
        g = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        _gamma_cache[key] = g
    out, _rstd = torch_npu.npu_rms_norm(x, g, 1e-6)
    return out

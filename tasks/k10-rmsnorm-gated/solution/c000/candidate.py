# K10 c000：vllm-ascend rms_norm_gated 原样包装（仅接口适配）
# 源：vllm_ascend/ops/triton/kda/kda.py:413（nightly-main 容器）
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated as _upstream


def kernel(inputs):
    x, g, weight, bias = inputs
    return _upstream(x, g, weight, bias, "sigmoid")

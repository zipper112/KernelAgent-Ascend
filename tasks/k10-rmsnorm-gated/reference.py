"""K10 数学 oracle：rms_norm_gated 语义参考（fp32 独立实现）。

inputs = [x, g, weight, bias]
  x: (T, H, D) bf16 —— attention 输出（GLM-5.3 KDA o_norm：D=head_dim=128，H=local heads）
  g: (T, H, D) bf16 —— 门控（g_b_proj 输出）
  weight: (D,) / bias: (D,) 或空（FusedRMSNormGated 默认无 bias——接口保留）
语义（is_rms_norm=True, activation="sigmoid"）：
  y[t,h,d] = (x[t,h,:] * rsqrt(mean(x²)+eps) * weight + bias) * sigmoid(g[t,h,:])
  归约维 = 最后一维 D；eps=1e-6；无状态携带（无 chained 终态门需求，单步即可）。
调用侧（glm5next/kda.py:312）：o_norm(core_attn_out, g2)，x 与 g 同形 3D。
"""
import torch


def reference(inputs):
    x, g, weight, bias = inputs
    eps = 1e-6
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    normed = xf * rms
    if weight is not None:
        normed = normed * weight.float()
    if bias is not None:
        normed = normed + bias.float()
    y = normed * torch.sigmoid(g.float())
    return y.to(x.dtype)

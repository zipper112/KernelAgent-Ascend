"""K8 数学 oracle：causal_conv1d_update 语义参考（fp32 独立实现）。

inputs = [x, conv_state, weight, bias, indices]
  x:[B,L,dim] bf16；conv_state:[slots,dim,3] bf16；weight:[dim,4]；bias:[dim]；indices:[B] int32
语义：out[b,l,d] = silu(Σ_w weight[d,w]·win[b,l,d,w] + bias[d])；win=旧状态+新 token 滑窗。
P1-2（chained 终态门）：oracle 同步建模状态转移——处理完 L 个 token 后把滑窗末尾
(width-1) 列原地写回 conv_state[indices]（只触碰命中槽，pad 槽零副作用）。
runner 两臂各自克隆起跑，单步模式同样隔离。
"""
import torch


def reference(inputs):
    x, conv_state, weight, bias, indices = inputs
    B, L, dim = x.shape
    width = weight.shape[-1]
    xf = x.float()
    w = weight.float()                                    # [dim, width]
    b = bias.float() if bias is not None else None
    state = conv_state[indices.long()].float()            # [B, dim, width-1]（按行 gather）
    full = torch.cat([state, xf.permute(0, 2, 1)], dim=-1)  # [B, dim, (width-1)+L]
    windows = full.unfold(-1, width, 1)                   # [B, dim, L, width]
    acc = torch.einsum("bdlw,dw->bld", windows, w)        # [B, L, dim]（输出维序 = x 原序）
    if b is not None:
        acc = acc + b
    out = torch.nn.functional.silu(acc).to(x.dtype)
    # 状态转移（原地，仅命中槽）：滑窗末 (width-1) 列 = 新 state
    new_state = full[..., -(width - 1):]                  # [B, dim, width-1]
    conv_state[indices.long()] = new_state.to(conv_state.dtype)
    return out

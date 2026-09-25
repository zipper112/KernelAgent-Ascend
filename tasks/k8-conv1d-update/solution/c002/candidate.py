"""K8 候选 c002：向量化去同步方向（vectorized-desync）。

假设：fallback 的成本大头不是卷积本身，而是 B 次 Python 循环 × 每 iter 2 次 .item()
（host-device 同步）× B 个 F.conv1d 小核派发。全向量化（unfold 融合窗口 + einsum 批卷积
+ 状态更新一次写）可消灭全部同步点与小核派发，decode 图恢复连续执行。
禁项自查：无 .item()/.cpu()/synchronize；pad 槽语义经 mask 保留；状态副作用保留。
"""
import torch

PAD_SLOT_ID = -1


def kernel(inputs):
    x, conv_state, weight, bias, conv_state_indices = inputs
    B, L, dim = x.shape
    width = weight.shape[-1]
    state_len = width - 1

    # 全批一次 gather（pad 槽用安全索引 0 + mask 屏蔽读写）
    valid = (conv_state_indices != PAD_SLOT_ID)                       # [B] bool，设备侧
    safe_idx = torch.where(valid, conv_state_indices, torch.zeros_like(conv_state_indices))
    state = conv_state[safe_idx]                                      # [B, dim, state_len]

    # 滑窗全序列：[B, dim, state_len + L] → unfold → [B, dim, L, width]
    full = torch.cat([state.to(x.dtype), x.transpose(1, 2)], dim=-1)
    windows = full.unfold(-1, width, 1)

    # 批卷积一次算（升 fp32 保精度，回 bf16）
    acc = torch.einsum("bdlw,dw->bld", windows.float(), weight.float())   # [B, L, dim]
    if bias is not None:
        acc = acc + bias.float()
    out = torch.nn.functional.silu(acc).to(x.dtype)                      # [B, L, dim]

    # 状态滑动更新一次写（pad 行经 mask 写回原值）
    if L >= state_len:
        new_state = x.transpose(1, 2)[..., L - state_len:L]
    else:
        new_state = torch.cat([full[..., state_len - (state_len - L):state_len],
                               x.transpose(1, 2)], dim=-1)
    conv_state[safe_idx] = torch.where(valid[:, None, None], new_state, state).to(conv_state.dtype)
    # pad 行输出置零（fallback 跳过 → out 保持 x.clone() 原值；此处语义等价：置 x 本身）
    out = torch.where(valid[:, None, None], out, x)
    return out

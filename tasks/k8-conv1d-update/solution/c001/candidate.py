"""K8 基线 c001：vllm-ascend causal_conv1d_update PyTorch fallback 忠实移植。

来源：vllm_ascend/ops/causal_conv1d.py:159-297（K8 档病灶本体）。
语义裁剪到本任务 workload（query_start_loc=None、2D x、无 num_accepted_tokens、
全有效 indices）——裁剪的是派发分支，病灶（Python 循环 × B + 每 iter .item() ×2 +
F.conv1d 小核 × B）原样保留。这就是要被打败的东西。
"""
import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1


def _conv1d_ref(x, weight, bias, initial_states):
    """[1,dim,L] 单序列卷积（fallback 同款）。"""
    dtype_in = x.dtype
    seqlen = x.shape[-1]
    x = torch.cat([initial_states, x], dim=-1)
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=weight.shape[0])
    out = F.silu(out).to(dtype_in)
    final = x[..., -(weight.shape[1] - 1):]
    return out, final


def kernel(inputs):
    x, conv_state, weight, bias, conv_state_indices = inputs
    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    dim, width = weight.shape
    state_len = width - 1
    out = x.clone()

    for i in range(x.shape[0]):
        idx = int(conv_state_indices[i].item())          # ← host 同步 #1（每序列）
        if idx == PAD_SLOT_ID:
            continue
        state = conv_state[idx]
        seq = x[i]                                        # [L, dim]
        x_ref = seq.transpose(0, 1).unsqueeze(0)          # [1, dim, L]
        init_state = state[..., :state_len].unsqueeze(0)
        out_ref, final_state = _conv1d_ref(x_ref, weight, bias, init_state)
        state[..., :state_len].copy_(final_state.squeeze(0))
        out[i] = out_ref.squeeze(0).transpose(0, 1)

    return out.to(original_x_dtype)

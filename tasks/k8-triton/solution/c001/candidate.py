# K8 triton-ascend candidate: causal_conv1d_update (decode, k=4 window, silu)
# 语义（对齐 c002 / vLLM batch_indices 约定）:
#   slot = indices[b]; slot < 0 (pad slot) -> out[b]=0, conv_state 不更新
#   y = s0*w0 + s1*w1 + s2*w2 + x*w3 + bias   (w*: (D,4) 逐通道, fp32 累加)
#   out = silu(y);  state 原地滑动: [s0,s1,s2] -> [s1,s2,x]
# 单 kernel launch、零中间张量、无 unfold/einsum/item/cpu。
import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_kernel(
    X, CS, W, BIA, IDX, OUT,
    sx_b, sx_s,          # x/out strides: (batch, seq)
    scs_b, scs_d, scs_w,  # conv_state strides: (slot, dim, tap)
    sw_d,                 # weight stride: (dim,)
    S, D,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m_d = offs_d < D

    # 该请求对应的 state 槽位；<0 = pad 槽（跳过 state，输出置 0）
    slot = tl.load(IDX + pid_b).to(tl.int64)
    valid = slot >= 0
    cs = CS + tl.where(valid, slot, 0) * scs_b + offs_d * scs_d

    # 逐通道 4-tap 权重 + bias，一次加载循环外复用（fp32 计算）
    w0 = tl.load(W + offs_d * sw_d + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(W + offs_d * sw_d + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(W + offs_d * sw_d + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(W + offs_d * sw_d + 3, mask=m_d, other=0.0).to(tl.float32)
    bb = tl.load(BIA + offs_d, mask=m_d, other=0.0).to(tl.float32)

    # 窗口状态 [s0,s1,s2]（k-1 taps）驻留寄存器，跨 S 步携带
    s0 = tl.load(cs + 0 * scs_w, mask=m_d & valid, other=0.0).to(tl.float32)
    s1 = tl.load(cs + 1 * scs_w, mask=m_d & valid, other=0.0).to(tl.float32)
    s2 = tl.load(cs + 2 * scs_w, mask=m_d & valid, other=0.0).to(tl.float32)

    x_row = X + pid_b.to(tl.int64) * sx_b + offs_d
    o_row = OUT + pid_b.to(tl.int64) * sx_b + offs_d

    for s in range(0, S):  # decode: S=1 (AR) 或 4 (MTP k=3)
        xv = tl.load(x_row + s * sx_s, mask=m_d, other=0.0).to(tl.float32)
        y = s0 * w0 + s1 * w1 + s2 * w2 + xv * w3 + bb
        o = y / (1.0 + tl.exp(-y))  # silu = x*sigmoid(x)，eps 差异 << bf16 tol
        tl.store(o_row + s * sx_s,
                 tl.where(valid, o, 0.0).to(OUT.dtype.element_ty), mask=m_d)
        s0 = s1  # 窗口滑动 1 tap
        s1 = s2
        s2 = xv

    # 原地滑动写回 conv_state 终态 [s0,s1,s2]；pad 槽不触碰
    sm = m_d & valid
    tl.store(cs + 0 * scs_w, s0.to(CS.dtype.element_ty), mask=sm)
    tl.store(cs + 1 * scs_w, s1.to(CS.dtype.element_ty), mask=sm)
    tl.store(cs + 2 * scs_w, s2.to(CS.dtype.element_ty), mask=sm)


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs
    B, S, D = x.shape
    out = torch.empty_like(x)
    # grid: (B, D/512)；B=16,D=4096 -> 128 program，A2 ~40 核上粒度均衡（见 reduction-case 调度经验）
    BLOCK_D = 512 if D >= 512 else triton.next_power_of_2(D)
    grid = (B, triton.cdiv(D, BLOCK_D))
    _causal_conv1d_update_kernel[grid](
        x, conv_state, weight, bias, indices, out,
        x.stride(0), x.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        weight.stride(0),
        S, D,
        BLOCK_D=BLOCK_D,
    )
    return out
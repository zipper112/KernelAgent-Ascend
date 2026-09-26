# K8 triton-ascend candidate r2: causal_conv1d_update (decode, k=4 窗， silu)
#
# 语义（对齐 c002 / vLLM batch_indices 约定，chained verify 3 步终态门）:
#   slot = indices[b]
#   slot < 0 (pad 槽)  -> out[b]=0, conv_state 不更新
#   y  = s0*w0 + s1*w1 + s2*w2 + x*w3 + bias   (逐通道 4-tap, fp32 累加)
#   out = silu(y) = y / (1 + exp(-y))          (eps/silu 语义同 c002)
#   state 原地滑动: [s0,s1,s2] -> [s1,s2,x]，寄存器携带跨 S 步，终态一次写回
#     S=1: [b,c,x1];  S=4: [x2,x3,x4] —— 与逐步滑动终态严格等价
#
# r1 失败归因与修复映射（评审 BLOCKING ①②③）:
#   ① 编译期风险消除: r1 使用 scalar tl.load(IDX+pid) / scalar tl.where /
#      标量-tensor 混合 mask —— triton-ascend 标量路径是首要嫌疑。
#      r2 全部改为 (1,)/(BLOCK_D,) 张量运算（tl.arange(0,1) 载 slot）。
#   ② 运行期越界消除: 所有 load/store 带 mask；slot 负值经 where 钳到 0 行
#      （该行只读不写），无裸指针解引用。
#   ③ 别名消除: out 显式 empty_like 独立缓冲；conv_state 每片 (b,d-block)
#      由唯一 program 独占，先读后写，无跨 program 写冲突。
# 单 kernel launch、零中间张量、无 unfold/einsum/item/cpu。
import torch
import triton
import triton.language as tl


@triton.jit
def _k8_conv_update(
    X, CS, W, BIA, IDX, OUT,
    sx_b, sx_s,           # x/out strides: (batch, seq)，last-dim 假定 stride 1
    scs_b, scs_d, scs_w,  # conv_state strides: (slot, dim, tap)
    sw_d,                 # weight row stride（tap 维假定 stride 1）
    S, D,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m_d = offs_d < D

    # slot 取成 (1,) 张量：规避 scalar load / scalar where（r1 失败首嫌）
    one = tl.arange(0, 1)
    slot = tl.load(IDX + pid_b + one)        # (1,)
    valid = slot >= 0                        # (1,)
    slot_safe = tl.where(valid, slot, 0)     # (1,) 纯张量 where，pad 槽钳到 0 行
    offs_cs = slot_safe.to(tl.int64) * scs_b + offs_d.to(tl.int64) * scs_d
    m_cs = m_d & valid                       # (BLOCK_D,) & (1,) 广播，纯张量 mask

    # 逐通道 4-tap 权重 + bias：循环外一次加载，寄存器复用（fp32 计算）
    w0 = tl.load(W + offs_d * sw_d + 0, mask=m_d, other=0.0).to(tl.float32)
    w1 = tl.load(W + offs_d * sw_d + 1, mask=m_d, other=0.0).to(tl.float32)
    w2 = tl.load(W + offs_d * sw_d + 2, mask=m_d, other=0.0).to(tl.float32)
    w3 = tl.load(W + offs_d * sw_d + 3, mask=m_d, other=0.0).to(tl.float32)
    bb = tl.load(BIA + offs_d, mask=m_d, other=0.0).to(tl.float32)

    # 窗口状态 [s0,s1,s2] 驻留寄存器，跨 S 步携带（S=1 AR / 4 MTP k=3）
    s0 = tl.load(CS + offs_cs + 0 * scs_w, mask=m_cs, other=0.0).to(tl.float32)
    s1 = tl.load(CS + offs_cs + 1 * scs_w, mask=m_cs, other=0.0).to(tl.float32)
    s2 = tl.load(CS + offs_cs + 2 * scs_w, mask=m_cs, other=0.0).to(tl.float32)

    x_row = X + pid_b.to(tl.int64) * sx_b + offs_d
    o_row = OUT + pid_b.to(tl.int64) * sx_b + offs_d

    for s in range(0, S):
        xv = tl.load(x_row + s * sx_s, mask=m_d, other=0.0).to(tl.float32)
        y = s0 * w0 + s1 * w1 + s2 * w2 + xv * w3 + bb
        o = y / (1.0 + tl.exp(-y))  # silu；与 c002 的 eps 差异 << bf16 0.03 容差
        tl.store(o_row + s * sx_s,
                 tl.where(valid, o, 0.0).to(OUT.dtype.element_ty), mask=m_d)
        s0 = s1  # 窗口滑动 1 tap
        s1 = s2
        s2 = xv

    # 原地写回 conv_state 终态；pad 槽 mask 全 false，不触碰（链式步语义保持）
    tl.store(CS + offs_cs + 0 * scs_w, s0.to(CS.dtype.element_ty), mask=m_cs)
    tl.store(CS + offs_cs + 1 * scs_w, s1.to(CS.dtype.element_ty), mask=m_cs)
    tl.store(CS + offs_cs + 2 * scs_w, s2.to(CS.dtype.element_ty), mask=m_cs)


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs
    B, S, D = x.shape
    if bias is None:  # 防御：bias 缺失时补零（不进 kernel 分支）
        bias = x.new_zeros(D)
    out = torch.empty_like(x)
    # 核数×粒度权衡（reduction-case 经验）：目标 32~128 program 吃满 A2 核
    if B >= 8:
        BLOCK_D = 512   # B=16,D=4096 -> (16,8)=128 program
    elif B >= 4:
        BLOCK_D = 256
    else:
        BLOCK_D = 128   # B=1,D=4096 -> 32 program，避免单批次核饥饿
    if BLOCK_D > D:
        BLOCK_D = max(16, triton.next_power_of_2(D))
    grid = (B, triton.cdiv(D, BLOCK_D))
    _k8_conv_update[grid](
        x, conv_state, weight, bias, indices, out,
        x.stride(0), x.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        weight.stride(0),
        S, D,
        BLOCK_D=BLOCK_D,
    )
    return out
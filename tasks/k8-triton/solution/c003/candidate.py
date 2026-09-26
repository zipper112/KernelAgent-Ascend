# K8 causal_conv1d_update — Triton-Ascend fused persistent kernel
# 单 launch：读 conv_state -> 4-tap 滑窗卷积(寄存器滚动) -> silu -> 原地写 state + 写 out
# 零中间张量 / 零额外派发；pad 槽(indices<0)跳过 state 更新且输出置零
import torch
import triton
import triton.language as tl


@triton.jit
def _k8_causal_conv1d_update(
    x_ptr, state_ptr, w_ptr, bias_ptr, idx_ptr, out_ptr,
    N_TILES, H,
    sxb, sxt, sx2,
    ssb, ssd, ss2,
    swd,
    sob, sot, so2,
    SEQ: tl.constexpr, BLOCK_D: tl.constexpr,
    D_BLOCKS: tl.constexpr, NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)

    # 常驻核：每个 program 按步长 NPROG 轮询 tile（tile = b * D_BLOCKS + db）
    for tile in range(pid, N_TILES, NPROG):
        b = tile // D_BLOCKS
        db = tile % D_BLOCKS
        offs_d = db * BLOCK_D + offs
        mask_d = offs_d < H

        ib = tl.load(idx_ptr + b)
        valid = ib >= 0
        ib = tl.where(valid, ib, 0)        # pad 槽(-1)：安全索引，state 写回用 wmask 屏蔽

        xb = x_ptr + b * sxb + offs_d * sx2
        ob = out_ptr + b * sob + offs_d * so2
        sb = state_ptr + ib * ssb + offs_d * ssd
        wb = w_ptr + offs_d * swd

        wmask = mask_d & valid             # pad 行不改 conv_state
        keep = tl.where(valid, 1.0, 0.0)   # pad 行输出置零（acc 先清零，silu(0)=0）

        s0 = tl.load(sb + 0 * ss2, mask=mask_d, other=0.0).to(tl.float32)
        s1 = tl.load(sb + 1 * ss2, mask=mask_d, other=0.0).to(tl.float32)
        s2 = tl.load(sb +
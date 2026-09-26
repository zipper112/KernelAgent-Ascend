```python
"""
K8: causal_conv1d_update Triton-Ascend 版。
方向: persist40-flat-tile-regwindow-seqloop
- 将 (batch, hidden) 展平为 flat 索引，持久 40 核各分一段
- core 内循环 seq（decode 1 或 MTP 4），用寄存器保存 3 长度滑动窗 state
- 每个程序一次完成: 读 conv_state -> 与 x 逐 token 卷积(k=4) -> silu -> 写输出 + 滑动更新 conv_state
- 单次 launch，无中间张量，无 .item()/.cpu()
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_kernel(
    x_ptr,            # [B, S, D]
    conv_state_ptr,   # [B, D, 3]  bf16, 原地滑动更新
    weight_ptr,       # [D, 4]
    bias_ptr,         # [D] 或 nullptr
    out_ptr,          # [B, S, D]
    total,            # B * D
    S,                # seq len (1 或 4)
    D,                # hidden
    eps,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    # 持久核：每个 pid 处理若干 flat tile（flat = b*D + d）
    tiles_per_block = tl.cdiv(total, NUM_BLOCKS * BLOCK)
    for t in range(tiles_per_block):
        flat = pid * tiles_per_block * BLOCK + t * BLOCK + tl.arange(0, BLOCK)
        # t 是循环变量，mask 防越界
        mask = flat < total
        b = flat // D
        d = flat % D
        cs_off = b * D * 3 + d * 3   # conv_state[b, d, :]

        # 寄存器滑动窗: s0,s1,s2 = conv_state[..., 0:3]
        s0 = tl.load(conv_state_ptr + cs_off + 0, mask=mask, other=0.0)
        s1 = tl.load(conv_state_ptr + cs_off + 1, mask=mask, other=0.0)
        s2 = tl.load(conv_state_ptr + cs_off + 2, mask=mask, other=0.0)

        # 权重: [BLOCK, 4]
        w_off = d * 4
        w0 = tl.load(weight_ptr + w_off + 0, mask=mask, other=0.0)
        w1 = tl.load(weight_ptr + w_off + 1, mask=mask, other=0.0)
        w2 = tl.load(weight_ptr + w_off + 2, mask=mask, other=0.0)
        w3 = tl.load(weight_ptr + w_off + 3, mask=mask, other=0.0)
        if HAS_BIAS:
            bb = tl.load(bias_ptr + d, mask=mask, other=0.0)
        else:
            bb = tl.zeros_like(s0)

        for si in range(0, S):  # S constexpr-friendly (runtime int, 小循环)
            x_off = b * S * D + si * D + d
            xv = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
            # 卷积窗: [s0, s1, s2, xv]
            acc = (s0.to(tl.float32) * w0.to(tl.float32)
                   + s1.to(tl.float32) * w1.to(tl.float32)
                   + s2.to(tl.float32) * w2.to(tl.float32)
                   + xv * w3.to(tl.float32))
            acc = acc + bb.to(tl.float32)
            # silu, eps 语义同 c002
            acc = acc / (1.0 + tl.exp(-acc)) + eps * acc
            tl.store(out_ptr + x_off, acc.to(out_ptr.dtype.element_ty), mask=mask)
            # 滑动窗前移（寄存器，无内存往返）
            s0 = s1
            s1 = s2
            s2 = xv.to(conv_state_ptr.dtype.element_ty)

        # 原地写回新状态（滑动后）
        tl.store(conv_state_ptr + cs_off + 0, s0, mask=mask)
        tl.store(conv_state_ptr + cs_off + 1, s1, mask=mask)
        tl.store(conv_state_ptr + cs_off + 2, s2, mask=mask)


def causal_conv1d_update_npu(x, conv_state, weight, bias, indices=None, eps=1e-6):
    # indices: decode 场景 batch 索引（本 workload 恒为全集，跳过 pad 槽即可）
    B, S, D = x.shape
    out = torch.empty_like(x)
    if indices is not None:
        # pad 槽跳过：仅对合法 batch 计算后散回。为保持单次派发，
        # 构造合法 batch 的紧凑视图（无 .item()/.cpu()）
        idx = indices.to(torch.int64)
        Bv = idx.shape[0]
        xv = x[idx]
        csv = conv_state[idx]
        outv = torch.empty_like(xv)
        total = Bv * D
        grid = lambda m: (m['NUM_BLOCKS'],)
        _causal_conv1d_update_kernel[grid](
            xv, csv, weight, bias if bias is not None else weight,
            outv, total, S, D, eps,
            NUM_BLOCKS=40, BLOCK=512,
            HAS_BIAS=bias is not None,
        )
        out[idx] = outv
        return out
    total = B * D
    grid = lambda m: (m['NUM_BLOCKS'],)
    _causal_conv1d_update_kernel[grid](
        x, conv_state, weight, bias if bias is not None else weight,
        out, total, S, D, eps,
        NUM_BLOCKS=40, BLOCK=512,
        HAS_BIAS=bias is not None,
    )
    return out


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs
    return causal_conv1d_update_npu(x, conv_state, weight, bias, indices)
```
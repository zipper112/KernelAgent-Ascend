# K8: causal_conv1d_update Triton-Ascend kernel
# inputs: x(B,S,D) bf16 | conv_state(B,D,3) | weight(D,4) | bias(D,) | indices(B,)
# 语义（对齐 c002 / FLA causal_conv1d_update）:
#   逐 token s: y[b,s,:] = silu( r0*w0 + r1*w1 + r2*w2 + x[b,s,:]*w3 + bias )
#   窗口右滑一格: (r0,r1,r2) <- (r1,r2,x[b,s,:]); 结束后原地写回 conv_state[indices[b]]
# pad 槽语义: 仅触碰 indices 命中的 conv_state 槽（gather/scatter），未命中槽零副作用
# 无 .item()/.cpu()，无 unfold/einsum，无任何中间张量：单 kernel 单 launch
import torch
import triton
import triton.language as tl


@triton.jit
def _k8_conv_update(
    x_ptr, st_ptr, w_ptr, b_ptr, idx_ptr, o_ptr,
    sx_b, sx_s,          # x strides: batch, seq (last dim contiguous)
    ss_b, ss_d,          # conv_state strides: batch, dim (last dim contiguous, =3)
    sw_d,                # weight stride over dim (=4)
    D,                   # hidden
    SEQ: tl.constexpr,   # tokens per request (1 或 4)，静态展开去依赖
    BD: tl.constexpr,    # hidden 分块
):
    pb = tl.program_id(0)
    pd = tl.program_id(1)
    offs = pd * BD + tl.arange(0, BD)
    md = offs < D
    bsel = tl.load(idx_ptr + pb).to(tl.int64)  # 本请求对应的 conv_state 槽

    # weight 列向量 (D,4) 与 bias，fp32 计算域
    w0 = tl.load(w_ptr + offs * sw_d + 0, mask=md, other=0.).to(tl.float32)
    w1 = tl.load(w_ptr + offs * sw_d + 1, mask=md, other=0.).to(tl.float32)
    w2 = tl.load(w_ptr + offs * sw_d + 2, mask=md, other=0.).to(tl.float32)
    w3 = tl.load(w_ptr + offs * sw_d + 3, mask=md, other=0.).to(tl.float32)
    vb = tl.load(b_ptr + offs, mask=md, other=0.).to(tl.float32)

    # 初始窗口 = 旧状态最后 3 槽
    sb = st_ptr + bsel * ss_b + offs * ss_d
    r0 = tl.load(sb + 0, mask=md, other=0.).to(tl.float32)
    r1 = tl.load(sb + 1, mask=md, other=0.).to(tl.float32)
    r2 = tl.load(sb + 2, mask=md, other=0.).to(tl.float32)

    xb = x_ptr + pb.to(tl.int64) * sx_b + offs
    ob = o_ptr + pb.to(tl.int64) * sx_b + offs

    # 寄存器滚动窗：SEQ 静态展开，token 间零同步零访存回读
    for s in tl.static_range(SEQ):
        cur = tl.load(xb + s * sx_s, mask=md, other=0.).to(tl.float32)
        a = r0 * w0 + r1 * w1 + r2 * w2 + cur * w3 + vb
        a = a / (1.0 + tl.exp(-a))  # silu（标准语义，同 c002；exp 溢出自然收敛到 0/恒等）
        tl.store(ob + s * sx_s, a.to(o_ptr.dtype.element_ty), mask=md)
        r0, r1, r2 = r1, r2, cur

    # 原地滑动更新最终状态
    tl.store(sb + 0, r0.to(st_ptr.dtype.element_ty), mask=md)
    tl.store(sb + 1, r1.to(st_ptr.dtype.element_ty), mask=md)
    tl.store(sb + 2, r2.to(st_ptr.dtype.element_ty), mask=md)


def _block_d(B: int, D: int) -> int:
    # 反 tail-effect：总 program 数尽量铺满 ~40 个向量核，B 小则切细 hidden
    for bd in (512, 256, 128, 64):
        if B * triton.cdiv(D, bd) >= 40:
            return bd
    return 64


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs
    B, S, D = x.shape
    out = torch.empty_like(x)
    if indices is None or indices.numel() == 0:
        indices = torch.arange(B, device=x.device, dtype=torch.int64)
    BD = _block_d(B, D)
    grid = (B, triton.cdiv(D, BD))
    _k8_conv_update[grid](
        x, conv_state, weight, bias, indices, out,
        x.stride(0), x.stride(1),
        conv_state.stride(0), conv_state.stride(1),
        weight.stride(0),
        D,
        SEQ=S, BD=BD,
    )
    return out


if __name__ == "__main__":  # 本地最小自检（不影响 harness）
    torch.npu.set_device(0)
    B, S, D = 4, 4, 4096
    x = torch.randn(B, S, D, device="npu", dtype=torch.bfloat16)
    st = torch.randn(B, D, 3, device="npu", dtype=torch.bfloat16)
    w = torch.randn(D, 4, device="npu", dtype=torch.bfloat16)
    bi = torch.randn(D, device="npu", dtype=torch.bfloat16)
    idx = torch.arange(B, device="npu")
    st_ref = st.clone()
    xs = x.clone()
    o = kernel([x, st, w, bi, idx])
    for s in range(S):
        y = (st_ref[:, :, 0] * w[:, 0] + st_ref[:, :, 1] * w[:, 1]
             + st_ref[:, :, 2] * w[:, 2] + xs[:, s] * w[:, 3] + bi)
        y = torch.nn.functional.silu(y.float()).to(torch.bfloat16)
        assert torch.allclose(o[:, s].float(), y.float(), atol=0.03), s
        st_ref = torch.cat([st_ref[:, :, 1:], xs[:, s:s+1].unsqueeze(-1)], -1)
    assert torch.allclose(st.float(), st_ref.float(), atol=0.03)
    print("k8-triton self-check OK")
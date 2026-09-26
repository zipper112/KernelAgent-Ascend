"""K8 causal_conv1d_update (Triton-Ascend, E067).

语义（对齐 reference / c002 vec）:
  逐 token s（寄存器滑窗，窗口宽 4）:
    out[b,s,:] = silu( u0*w[:,0] + u1*w[:,1] + u2*w[:,2] + x_s*w[:,3] + bias )
    窗口滚动: (u0,u1,u2) <- (u1,u2,x_s)
  conv_state 原地更新 = cat(old_state, x)[..., -3:]
  indices 负值 = pad 槽，整块跳过（不 load 不 store）
单 kernel 单 dispatch，零中间张量；grid=persist 上限 40 核（A2/A3 物理核数）。
"""
import torch
import triton
import triton.language as tl

BLOCK_D = 256   # D=4096 -> 16 tiles/batch; B=16 -> 256 flat items
NCORE = 40      # Atlas A2/A3 AI core 数，超发反而劣化（case 证据: 64核 9.83us vs 40核 8.55us）


@triton.jit
def _ccu_fwd(
    x_ptr, st_ptr, w_ptr, bi_ptr, ix_ptr, o_ptr,
    sxb, sxs, sxd,          # x strides (B,S,D)
    ssb, ssd, ssk,          # conv_state strides (B,D,3)
    swd, swk,               # weight strides (D,4)
    sbi,                    # bias stride
    sob, sos, sod,          # out strides
    six,                    # indices stride
    D, NBLK, TOTAL, NPROG,
    S: tl.constexpr, BD: tl.constexpr,
):
    pid = tl.program_id(0)
    # persistent flat-tile: work item wi = n(索引槽) * NBLK + d-tile
    for wi in range(pid, TOTAL, NPROG):
        n = wi // NBLK
        db = wi % NBLK
        b = tl.load(ix_ptr + n * six)      # gather 真实 batch 槽
        ok = b >= 0                        # pad 槽跳过
        offs = db * BD + tl.arange(0, BD)
        m = ok & (offs < D)

        w0 = tl.load(w_ptr + offs * swd + 0 * swk, mask=m, other=0.).to(tl.float32)
        w1 = tl.load(w_ptr + offs * swd + 1 * swk, mask=m, other=0.).to(tl.float32)
        w2 = tl.load(w_ptr + offs * swd + 2 * swk, mask=m, other=0.).to(tl.float32)
        w3 = tl.load(w_ptr + offs * swd + 3 * swk, mask=m, other=0.).to(tl.float32)
        bb = tl.load(bi_ptr + offs * sbi, mask=m, other=0.).to(tl.float32)

        # 寄存器滑窗（c016 的 seqloop 重读内存版已 verify 拒绝，窗口必须驻寄存器）
        r0 = tl.load(st_ptr + b * ssb + offs * ssd + 0 * ssk, mask=m, other=0.).to(tl.float32)
        r1 = tl.load(st_ptr + b * ssb + offs * ssd + 1 * ssk, mask=m, other=0.).to(tl.float32)
        r2 = tl.load(st_ptr + b * ssb + offs * ssd + 2 * ssk, mask=m, other=0.).to(tl.float32)

        for s in tl.static_range(S):       # S=1/4 各自特化，编译期展开
            xt = tl.load(x_ptr + b * sxb + s * sxs + offs * sxd,
                         mask=m, other=0.).to(tl.float32)
            z = r0 * w0 + r1 * w1 + r2 * w2 + xt * w3 + bb
            y = z * (1.0 / (1.0 + tl.exp(-z)))   # silu，fp32 累加，语义同 c002
            tl.store(o_ptr + b * sob + s * sos + offs * sod,
                     y.to(o_ptr.dtype.element_ty), mask=m)
            r0, r1, r2 = r1, r2, xt        # 滑窗：状态左移，新 token 入窗

        # 原地写新状态：S=1 -> (u1,u2,x0)；S=4 -> (x1,x2,x3)
        tl.store(st_ptr + b * ssb + offs * ssd + 0 * ssk,
                 r0.to(st_ptr.dtype.element_ty), mask=m)
        tl.store(st_ptr + b * ssb + offs * ssd + 1 * ssk,
                 r1.to(st_ptr.dtype.element_ty), mask=m)
        tl.store(st_ptr + b * ssb + offs * ssd + 2 * ssk,
                 r2.to(st_ptr.dtype.element_ty), mask=m)


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs
    B, S, D = x.shape
    out = torch.empty_like(x)                       # 唯一分配，无 kernel 派发
    nblk = (D + BLOCK_D - 1) // BLOCK_D
    total = indices.numel() * nblk
    nprog = NCORE if total > NCORE else total
    _ccu_fwd[(nprog,)](
        x, conv_state, weight, bias, indices, out,
        x.stride(0), x.stride(1), x.stride(2),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        indices.stride(0),
        D, nblk, total, nprog,
        S=S, BD=BLOCK_D,
    )
    return out
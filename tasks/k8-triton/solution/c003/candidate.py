# K8: causal_conv1d_update Triton-Ascend 版
# 语义（同 vLLM/fla + pad_slot 约定）:
#   idx[b] >= 0: slot=idx[b]，读 conv_state[slot]（3 列），逐 token 4 元素窗卷积
#     [s0,s1,s2,xt]·w + bias -> silu -> out[b,t]；寄存器滚动左移；末尾写回新 state
#   idx[b] < 0 (pad 槽): 输出强制 0，state 禁写（只通过 mask/where，无分支）
# 结构: 单 launch，grid=(40,) 常驻，flat d-tile 循环 stride=40，零中间张量
import torch
import triton
import triton.language as tl


@triton.jit
def _cc1du_kernel(
    x_ptr, st_ptr, w_ptr, bias_ptr, idx_ptr, o_ptr,
    B, D, NTB,                    # NTB = D 的 d-tile 数（运行期）
    sxb, sxs, sxd,                # x strides [B,S,D]
    ssb, ssd,                     # conv_state strides [B,D,3]（末维 stride=1 假定 contiguous 末维）
    swd, swk,                     # weight strides [D,4]
    sob, sos, sod,                # out strides [B,S,D]
    SEQ: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    NP = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    total = B * NTB

    # flat tile 循环：tile -> (batch, d-tile)，常驻 40 核吃完全部工作
    for tile in range(pid, total, NP):
        bi = tile // NTB
        d = (tile % NTB) * BLOCK_D + offs
        dm = d < D

        # pad 槽判定 + 安全 slot（clamp 到 0 防 OOB 读）
        idx = tl.load(idx_ptr + bi)
        valid = idx >= 0
        slot = tl.maximum(idx, 0)

        # 共享 tap 权重/bias：循环外加载一次（每个 tile 内复用 SEQ 次）
        w0 = tl.load(w_ptr + d * swd + 0 * swk, mask=dm, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + d * swd + 1 * swk, mask=dm, other=0.0).to(tl.float32)
        w2 = tl.load(w_ptr + d * swd + 2 * swk, mask=dm, other=0.0).to(tl.float32)
        w3 = tl.load(w_ptr + d * swd + 3 * swk, mask=dm, other=0.0).to(tl.float32)
        bs = tl.load(bias_ptr + d, mask=dm, other=0.0).to(tl.float32)

        # 寄存器滚动窗初始化：state[slot, d, 0:3]
        st_base = st_ptr + slot * ssb + d * ssd
        s0 = tl.load(st_base + 0, mask=dm & valid, other=0.0).to(tl.float32)
        s1 = tl.load(st_base + 1, mask=dm & valid, other=0.0).to(tl.float32)
        s2 = tl.load(st_base + 2, mask=dm & valid, other=0.0).to(tl.float32)

        xb = x_ptr + bi * sxb + d * sxd
        ob = o_ptr + bi * sob + d * sod

        # SEQ 静态展开：decode S=1 / MTP S=4，各编译一份
        for t in tl.static_range(SEQ):
            xt = tl.load(xb + t * sxs, mask=dm, other=0.0).to(tl.float32)
            acc = s0 * w0 + s1 * w1 + s2 * w2 + xt * w3 + bs
            y = acc / (1.0 + tl.exp(-acc))            # silu（fp32 中间，语义同 c002）
            y = tl.where(valid, y, 0.0)               # pad 槽 -> 输出写 0（无分支）
            tl.store(ob + t * sos, y.to(o_ptr.dtype.element_ty), mask=dm)
            # 滚动左移：仅 valid 时推进窗口（select 替代分支）
            s0 = tl.where(valid, s1, s0)
            s1 = tl.where(valid, s2, s1)
            s2 = tl.where(valid, xt, s2)

        # 原地滑动写回新 state（pad: mask 禁写）
        sm = dm & valid
        tl.store(st_base + 0, s0.to(st_ptr.dtype.element_ty), mask=sm)
        tl.store(st_base + 1, s1.to(st_ptr.dtype.element_ty), mask=sm)
        tl.store(st_base + 2, s2.to(st_ptr.dtype.element_ty), mask=sm)


def kernel(inputs):
    x, conv_state, weight, bias, indices = inputs
    out = torch.empty_like(x)                          # 唯一一次分配，零中间张量
    B, S, D = x.shape
    BLOCK_D = 256                                      # 4096/256=16 tile/b；B=16 -> 256 tiles/40 核，摊尾差
    ntb = (D + BLOCK_D - 1) // BLOCK_D
    _cc1du_kernel[(40,)](                              # 单 launch，40 物理核常驻
        x, conv_state, weight, bias, indices, out,
        B, D, ntb,
        x.stride(0), x.stride(1), x.stride(2),
        conv_state.stride(0), conv_state.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        SEQ=S,
        BLOCK_D=BLOCK_D,
    )
    return out